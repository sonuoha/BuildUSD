from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .conversion import (
    DEFAULT_BASE_POINT,
    DEFAULT_GEODETIC_CRS,
    DEFAULT_MASTER_STAGE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SHARED_BASE_POINT,
)
from .federate import (
    _apply_federation,
    _candidate_stage_files,
    _load_manifest,
    _normalise_stage_root,
    _normalize_anchor_mode,
    _plan_federation,
)
from .usd_context import initialize_usd, shutdown_usd_context

LOG = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble federated USD master stages from converted stage files."
    )
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
        choices=("local", "basepoint", "none"),
        default="none",
        help="Match federation alignment to stages anchored via the IFC site placement, base point (PBP/SP), or skip anchoring (default: %(default)s).",
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
    masters_root = (
        _normalise_stage_root(args.masters_root) if args.masters_root else stage_root
    )
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
