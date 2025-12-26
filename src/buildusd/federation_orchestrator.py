from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from .config.manifest import (
    BasePointConfig,
    ConversionManifest,
    MasterConfig,
    ResolvedFilePlan,
)
from .io_utils import (
    ensure_parent_directory,
    is_omniverse_path,
    join_path,
    path_name,
    path_stem,
    path_suffix,
    read_text,
)
from .conversion import (
    DEFAULT_BASE_POINT,
    DEFAULT_GEODETIC_CRS,
    DEFAULT_MASTER_STAGE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SHARED_BASE_POINT,
    ROOT,
)
from .federation_builder import build_federated_stage
from .usd_context import initialize_usd, shutdown_usd_context

LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _force_usdc_filename(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "Federated Model.usdc"
    path = Path(raw)
    suffix = path.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return str(path.with_suffix(".usdc"))
    return f"{raw}.usdc"


def _load_manifest(path: Path) -> ConversionManifest:
    text = read_text(path)
    suffix = path_suffix(path)
    return ConversionManifest.from_text(text, suffix=suffix or ".json")


def _normalize_anchor_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "local":
        return "local"
    if normalized in ("basepoint", "site", "shared_site"):
        return "basepoint"
    if normalized in ("none", ""):
        return None
    LOG.debug("Unknown anchor_mode '%s'; defaulting to None", value)
    return None


def _normalise_stage_root(path: str | None) -> Path:
    if not path:
        return (ROOT / "data" / "output").resolve()
    if is_omniverse_path(path):
        # For omniverse paths we do not resolve to local Path.
        return Path(path)
    return Path(path).resolve()


def _candidate_stage_files(
    stage_root: Path, stage_filters: Sequence[str] | None
) -> list[Path]:
    if stage_filters:
        files: list[Path] = []
        for raw in stage_filters:
            if is_omniverse_path(raw):
                files.append(Path(raw))
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = stage_root / raw
            if not is_omniverse_path(candidate) and not candidate.exists():
                LOG.warning("Stage path %s does not exist; skipping.", candidate)
                continue
            files.append(candidate)
        return files
    if is_omniverse_path(str(stage_root)):
        raise ValueError(
            "Automatic discovery is not supported for omniverse stage roots; provide --stage entries."
        )
    patterns = ("*.usd", "*.usda", "*.usdc")
    stage_files: list[Path] = []
    for pattern in patterns:
        stage_files.extend(stage_root.glob(pattern))
    filtered: list[Path] = []
    for path in stage_files:
        name = path_stem(path)
        if not path.is_file():
            continue
        lowered = name.lower()
        if (
            lowered.endswith("_prototypes")
            or lowered.endswith("_materials")
            or lowered.endswith("_instances")
            or lowered.endswith("_geometry2d")
        ):
            continue
        filtered.append(path)
    filtered.sort()
    return filtered


def _resolve_plan_for_stage(
    manifest: ConversionManifest,
    stage_path: Path,
    *,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_base_point: BasePointConfig,
    fallback_master_name: str,
    fallback_shared_site_base_point: BasePointConfig,
) -> ResolvedFilePlan:
    fake_ifc_path = stage_path.with_suffix(".ifc")
    return manifest.resolve_for_path(
        fake_ifc_path,
        fallback_master_name=fallback_master_name,
        fallback_projected_crs=fallback_projected_crs,
        fallback_geodetic_crs=fallback_geodetic_crs,
        fallback_base_point=fallback_base_point,
        fallback_shared_site_base_point=fallback_shared_site_base_point,
    )


@dataclass
class FederationTask:
    stage_path: Path
    plan: ResolvedFilePlan
    anchor_mode: Optional[str] = None


def _plan_federation(
    manifest: ConversionManifest,
    stage_paths: Iterable[Path],
    *,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_base_point: BasePointConfig,
    fallback_master_name: str,
    fallback_shared_site_base_point: BasePointConfig,
    anchor_mode: Optional[str],
) -> list[FederationTask]:
    tasks: list[FederationTask] = []
    resolved_anchor_mode = _normalize_anchor_mode(anchor_mode)
    for stage_path in stage_paths:
        try:
            plan = _resolve_plan_for_stage(
                manifest,
                stage_path,
                fallback_projected_crs=fallback_projected_crs,
                fallback_geodetic_crs=fallback_geodetic_crs,
                fallback_base_point=fallback_base_point,
                fallback_master_name=fallback_master_name,
                fallback_shared_site_base_point=fallback_shared_site_base_point,
            )
        except Exception as exc:
            LOG.error("Failed to resolve manifest entry for %s: %s", stage_path, exc)
            continue
        tasks.append(
            FederationTask(
                stage_path=stage_path, plan=plan, anchor_mode=resolved_anchor_mode
            )
        )
    return tasks


def _apply_federation(
    tasks: Sequence[FederationTask],
    *,
    masters_root: Path,
    parent_prim: str,
    rebuild: bool,
) -> list[str]:
    grouped: dict[str, list[FederationTask]] = {}
    for task in tasks:
        master_filename = _force_usdc_filename(task.plan.master_stage_filename)
        master_stage_path = join_path(masters_root, master_filename)
        grouped.setdefault(master_stage_path, []).append(task)

    built_master_paths: list[str] = []
    for master_stage_path, group in grouped.items():
        if not is_omniverse_path(master_stage_path):
            ensure_parent_directory(master_stage_path)
        payload_paths = [str(task.stage_path) for task in group]
        first_plan = group[0].plan
        federation_origin = _resolve_site_federation_origin(
            first_plan, DEFAULT_SHARED_BASE_POINT
        )
        federation_projected_crs = first_plan.projected_crs or "EPSG:7855"
        geodetic_crs = first_plan.geodetic_crs or DEFAULT_GEODETIC_CRS

        for task in group[1:]:
            task_origin = _resolve_site_federation_origin(
                task.plan, DEFAULT_SHARED_BASE_POINT
            )
            if task_origin != federation_origin:
                LOG.warning(
                    "Federation origin differs for %s; using the first plan's federation base point.",
                    path_name(task.stage_path),
                )
            if task.plan.projected_crs != federation_projected_crs:
                LOG.warning(
                    "Projected CRS differs for %s (%s vs %s); reprojecting anchors.",
                    path_name(task.stage_path),
                    task.plan.projected_crs,
                    federation_projected_crs,
                )

        LOG.info(
            "Building federated stage %s with %d payloads.",
            path_name(master_stage_path),
            len(payload_paths),
        )
        build_federated_stage(
            payload_paths=payload_paths,
            out_stage_path=str(master_stage_path),
            federation_origin=federation_origin,
            federation_projected_crs=federation_projected_crs,
            geodetic_crs=geodetic_crs,
            parent_prim_path=parent_prim or "/World",
            use_payloads=True,
            rebuild=rebuild,
        )
        built_master_paths.append(str(master_stage_path))
    return built_master_paths


def _resolve_overall_master_name(manifest: ConversionManifest) -> Optional[str]:
    name = manifest.defaults.overall_master_name
    if not name:
        return None
    master = MasterConfig(id="__overall__", name=name)
    return _force_usdc_filename(master.resolved_name())


def _resolve_site_federation_origin(
    plan: ResolvedFilePlan, fallback_shared_site_base_point: BasePointConfig
) -> BasePointConfig:
    if plan.master.base_point is not None:
        return plan.master.base_point
    if plan.shared_site_base_point is not None:
        return plan.shared_site_base_point
    return fallback_shared_site_base_point


def _apply_overall_master(
    master_stage_paths: Sequence[str],
    *,
    manifest: ConversionManifest,
    masters_root: Path,
    parent_prim: str,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_shared_site_base_point: BasePointConfig,
    rebuild: bool,
) -> None:
    overall_name = _resolve_overall_master_name(manifest)
    if not overall_name:
        return
    overall_origin = (
        manifest.defaults.overall_base_point
        or manifest.defaults.overall_shared_site_base_point
        or manifest.defaults.shared_site_base_point
        or fallback_shared_site_base_point
    )
    if overall_origin is None:
        LOG.warning("Overall federation requested but no shared_site_base_point set.")
        return
    projected_crs = manifest.defaults.projected_crs or fallback_projected_crs
    geodetic_crs = manifest.defaults.geodetic_crs or fallback_geodetic_crs
    overall_stage_path = join_path(masters_root, overall_name)
    if any(str(path) == str(overall_stage_path) for path in master_stage_paths):
        LOG.warning(
            "Overall master %s conflicts with an existing master stage; skipping.",
            path_name(overall_stage_path),
        )
        return
    payload_paths: list[str] = []
    for path in sorted({str(path) for path in master_stage_paths}):
        if is_omniverse_path(path):
            payload_paths.append(path)
            continue
        if Path(path).exists():
            payload_paths.append(path)
        else:
            LOG.warning("Overall master skipped missing site master: %s", path)
    if not payload_paths:
        return
    if not is_omniverse_path(overall_stage_path):
        ensure_parent_directory(overall_stage_path)
    LOG.info(
        "Building overall federated stage %s with %d site masters.",
        path_name(overall_stage_path),
        len(payload_paths),
    )
    build_federated_stage(
        payload_paths=payload_paths,
        out_stage_path=str(overall_stage_path),
        federation_origin=overall_origin,
        federation_projected_crs=projected_crs,
        geodetic_crs=geodetic_crs,
        parent_prim_path=parent_prim or "/World",
        use_payloads=True,
        rebuild=rebuild,
    )


def federate_stages(
    stage_paths: Sequence[PathLike],
    *,
    manifest: ConversionManifest,
    masters_root: PathLike,
    parent_prim: str = "/World",
    map_coordinate_system: str = "EPSG:7855",
    fallback_base_point: BasePointConfig = DEFAULT_BASE_POINT,
    fallback_shared_site_base_point: BasePointConfig = DEFAULT_SHARED_BASE_POINT,
    fallback_master_stage: str = DEFAULT_MASTER_STAGE,
    fallback_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    anchor_mode: Optional[str] = None,
    offline: bool = False,
    rebuild: bool = False,
) -> Sequence[FederationTask]:
    """Federate the supplied stage files according to the provided manifest."""

    if not stage_paths:
        return []

    normalized_stage_paths: list[Path] = []
    for raw in stage_paths:
        if isinstance(raw, Path):
            normalized_stage_paths.append(raw)
            continue
        text = str(raw)
        if is_omniverse_path(text):
            normalized_stage_paths.append(Path(text))
        else:
            normalized_stage_paths.append(Path(text).resolve())

    if isinstance(masters_root, Path):
        masters_root_input = masters_root.as_posix()
    else:
        masters_root_input = str(masters_root)
    masters_root_path = _normalise_stage_root(masters_root_input)
    resolved_anchor_mode = _normalize_anchor_mode(anchor_mode)
    tasks = _plan_federation(
        manifest,
        normalized_stage_paths,
        fallback_projected_crs=map_coordinate_system,
        fallback_geodetic_crs=fallback_geodetic_crs,
        fallback_base_point=fallback_base_point,
        fallback_master_name=fallback_master_stage,
        fallback_shared_site_base_point=fallback_shared_site_base_point,
        anchor_mode=resolved_anchor_mode,
    )
    if not tasks:
        return []

    initialize_usd(offline=offline)
    try:
        master_stage_paths = _apply_federation(
            tasks,
            masters_root=masters_root_path,
            parent_prim=parent_prim,
            rebuild=rebuild,
        )
        _apply_overall_master(
            master_stage_paths,
            manifest=manifest,
            masters_root=masters_root_path,
            parent_prim=parent_prim,
            fallback_projected_crs=map_coordinate_system,
            fallback_geodetic_crs=fallback_geodetic_crs,
            fallback_shared_site_base_point=fallback_shared_site_base_point,
            rebuild=rebuild,
        )
    finally:
        shutdown_usd_context()
    return tasks


if __name__ == "__main__":
    from .federate_cli import main

    main()
