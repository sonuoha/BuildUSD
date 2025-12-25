from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from .config.manifest import BasePointConfig, ConversionManifest, ResolvedFilePlan
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
from .process_usd import update_federated_view
from .usd_context import initialize_usd, shutdown_usd_context

LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]


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
) -> None:
    for task in tasks:
        stage_path = task.stage_path
        master_filename = task.plan.master_stage_filename
        master_stage_path = join_path(masters_root, master_filename)
        if not is_omniverse_path(master_stage_path):
            ensure_parent_directory(master_stage_path)
        payload_name = path_stem(stage_path)
        LOG.info(
            "Federating %s -> %s (prim=%s, anchor=%s)",
            path_name(stage_path),
            path_name(master_stage_path),
            payload_name,
            task.anchor_mode or "none",
        )
        update_federated_view(
            master_stage_path,
            stage_path,
            payload_prim_name=payload_name,
            parent_prim_path=parent_prim,
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
        _apply_federation(
            tasks, masters_root=masters_root_path, parent_prim=parent_prim
        )
    finally:
        shutdown_usd_context()
    return tasks
