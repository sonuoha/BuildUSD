from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

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
from .main import (
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


def _load_manifest(path: Path) -> ConversionManifest:
    text = read_text(path)
    suffix = path_suffix(path)
    return ConversionManifest.from_text(text, suffix=suffix or ".json")


def _normalize_anchor_mode(value: Optional[str]) -> str:
    if value is None:
        return "local"
    normalized = value.strip().lower()
    alias_map = {
        "local": "local",
        "site": "site",
        "basepoint": "local",
        "shared_site": "site",
    }
    return alias_map.get(normalized, "local")


def _normalise_stage_root(path: str | None) -> Path:
    if not path:
        return (ROOT / "data" / "output").resolve()
    if is_omniverse_path(path):
        # For omniverse paths we do not resolve to local Path.
        return Path(path)
    return Path(path).resolve()


def _candidate_stage_files(stage_root: Path, stage_filters: Sequence[str] | None) -> list[Path]:
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
        raise ValueError("Automatic discovery is not supported for omniverse stage roots; provide --stage entries.")
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
        if lowered.endswith("_prototypes") or lowered.endswith("_materials") or lowered.endswith("_instances") or lowered.endswith("_geometry2d"):
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
    anchor_mode: Literal["local", "site"] = "local"


def _plan_federation(
    manifest: ConversionManifest,
    stage_paths: Iterable[Path],
    *,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_base_point: BasePointConfig,
    fallback_master_name: str,
    fallback_shared_site_base_point: BasePointConfig,
    anchor_mode: str,
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
        tasks.append(FederationTask(stage_path=stage_path, plan=plan, anchor_mode=resolved_anchor_mode))
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
            task.anchor_mode,
        )
        update_federated_view(
            master_stage_path,
            stage_path,
            payload_prim_name=payload_name,
            parent_prim_path=parent_prim,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble federated USD master stages from converted stage files.")
    parser.add_argument(
        "--stage-root",
        dest="stage_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory containing converted stage files (default: data/output under repo root)",
    )
    parser.add_argument(
        "--stage",
        dest="stage_filters",
        nargs="*",
        default=None,
        help="Specific stage files to federate (relative to --stage-root unless absolute).",
    )
    parser.add_argument(
        "--masters-root",
        dest="masters_root",
        default=None,
        help="Directory for federated master stages (default: same as --stage-root).",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        required=True,
        help="Manifest describing federation targets.",
    )
    parser.add_argument(
        "--parent-prim",
        dest="parent_prim",
        default="/World",
        help="Parent prim path under which payloads are referenced (default: %(default)s).",
    )
    parser.add_argument(
        "--map-coordinate-system",
        dest="map_coordinate_system",
        default="EPSG:7855",
        help="Fallback CRS used when manifest entries omit projected_crs (default: %(default)s).",
    )
    parser.add_argument(
        "--anchor-mode",
        dest="anchor_mode",
        choices=("local", "site"),
        default="local",
        help="Match federation alignment to stages anchored via the local base point or the shared site base point (default: %(default)s).",
    )
    parser.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help="Initialise USD in standalone mode (no Kit). All inputs/outputs must be local.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    stage_root = _normalise_stage_root(args.stage_root)
    masters_root = _normalise_stage_root(args.masters_root) if args.masters_root else stage_root
    manifest_path = Path(args.manifest_path).resolve()
    manifest = _load_manifest(manifest_path)
    stage_paths = _candidate_stage_files(stage_root, args.stage_filters)
    if not stage_paths:
        LOG.warning("No stage files discovered under %s", stage_root)
        return
    cli_anchor_mode = _normalize_anchor_mode(args.anchor_mode)
    tasks = _plan_federation(
        manifest,
        stage_paths,
        fallback_projected_crs=args.map_coordinate_system,
        fallback_geodetic_crs=DEFAULT_GEODETIC_CRS,
        fallback_base_point=DEFAULT_BASE_POINT,
        fallback_master_name=DEFAULT_MASTER_STAGE,
        fallback_shared_site_base_point=DEFAULT_SHARED_BASE_POINT,
        anchor_mode=cli_anchor_mode,
    )
    if not tasks:
        LOG.warning("No federation tasks were created; nothing to do.")
        return
    initialize_usd(offline=args.offline)
    try:
        _apply_federation(
            tasks,
            masters_root=masters_root,
            parent_prim=args.parent_prim,
        )
    finally:
        shutdown_usd_context()


if __name__ == "__main__":
    main()
