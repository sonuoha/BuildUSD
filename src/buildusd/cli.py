from __future__ import annotations

import argparse
from typing import Sequence


class _JoinPathAction(argparse.Action):
    """Join successive CLI tokens into a single path string (handles spaces gracefully)."""

    def __call__(self, parser, namespace, values, option_string=None):
        joined = " ".join(values).strip()
        setattr(namespace, self.dest, joined or None)


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_input_root,
    default_output_root,
    usd_format_choices: Sequence[str],
    default_usd_format: str,
    default_usd_auto_binary_threshold_mb: float,
) -> argparse.Namespace:
    """Parse the CLI arguments for the standalone converter."""

    parser = argparse.ArgumentParser(description="Convert IFC to USD")
    parser.add_argument(
        "--map-coordinate-system",
        "--map-epsg",
        dest="map_coordinate_system",
        default="EPSG:7855",
        help="EPSG code or CRS string for map eastings/northings (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        nargs="+",
        action=_JoinPathAction,
        default=str(default_input_root),
        help="Path to an IFC file or a directory containing IFC files",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        nargs="+",
        action=_JoinPathAction,
        default=None,
        help=f"Directory for USD artifacts (default: {default_output_root})",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        nargs="+",
        action=_JoinPathAction,
        default=None,
        help="Path to a manifest (YAML or JSON) describing base points and masters",
    )
    parser.add_argument(
        "--ifc-names",
        dest="ifc_names",
        nargs="*",
        default=None,
        help="Specific IFC file names to process when --input points to a directory",
    )
    parser.add_argument(
        "--exclude",
        dest="exclude",
        nargs="*",
        default=None,
        help="IFC file names (with or without .ifc) to skip when scanning directories",
    )
    parser.add_argument(
        "--all",
        dest="process_all",
        action="store_true",
        help="Process all .ifc files in the input directory",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        action="store_true",
        help="Create Nucleus checkpoints for each authored layer and stage (omniverse:// only)",
    )
    parser.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help="Force standalone USD (no Kit). All input/output paths must be local.",
    )
    parser.add_argument(
        "--set-stage-unit",
        dest="set_stage_unit",
        nargs="+",
        action=_JoinPathAction,
        default=None,
        help=(
            "Path to an existing USD layer/stage whose metersPerUnit metadata should be updated. "
            "When provided the command performs only this edit and skips IFC conversion."
        ),
    )
    parser.add_argument(
        "--stage-unit-value",
        dest="stage_unit_value",
        type=float,
        default=1.0,
        help="Meters-per-unit value applied when using --set-stage-unit (must be >0, default: %(default)s).",
    )
    parser.add_argument(
        "--set-stage-up-axis",
        dest="set_stage_up_axis",
        nargs="+",
        action=_JoinPathAction,
        default=None,
        help=(
            "Path to an existing USD layer/stage whose upAxis metadata should be updated. "
            "When provided the command performs only this edit and skips IFC conversion."
        ),
    )
    parser.add_argument(
        "--stage-up-axis",
        dest="stage_up_axis",
        choices=("X", "Y", "Z", "x", "y", "z"),
        default="Y",
        help="Up-axis value applied when using --set-stage-up-axis (default: %(default)s).",
    )
    parser.add_argument(
        "--annotation-width-default",
        dest="annotation_width_default",
        type=str,
        default=None,
        help="Default width for annotation curves (stage units unless suffix like mm/cm/m is provided).",
    )
    parser.add_argument(
        "--annotation-width-rule",
        dest="annotation_width_rules",
        action="append",
        default=[],
        help=(
            "Annotation width override rule as comma-separated key=value pairs. "
            "Keys: width (required), layer, curve, hierarchy, step_id, unit. "
            "Example: width=12mm,layer=Alignment*,curve=Centerline*"
        ),
    )
    parser.add_argument(
        "--annotation-width-config",
        dest="annotation_width_configs",
        action="append",
        default=[],
        help="Path to a JSON or YAML file defining annotation curve width rules (may be provided multiple times).",
    )
    parser.add_argument(
        "--anchor-mode",
        dest="anchor_mode",
        choices=("local", "site", "none"),
        default="none",
        help="Choose whether stages anchor to the file-local base point, shared site base point, or skip anchoring entirely.",
    )
    parser.add_argument(
        "--usd-format",
        dest="usd_format",
        choices=usd_format_choices,
        default=default_usd_format,
        help="Output USD format to use (default: %(default)s). Use 'auto' to heuristically pick between USDA and USDC.",
    )
    parser.add_argument(
        "--usd-auto-binary-threshold-mb",
        dest="usd_auto_binary_threshold_mb",
        type=float,
        default=default_usd_auto_binary_threshold_mb,
        help=(
            "When using --usd-format usda, re-export layers as usdc when the file exceeds this many megabytes "
            "(set to 0 to disable; default: %(default)s). Auto mode also relies on this threshold for its heuristic."
        ),
    )
    parser.add_argument(
        "--enable-material-classification",
        dest="enable_material_classification",
        action="store_true",
        help="Enable component-level material classification to reconcile IFC style colours.",
    )
    parser.add_argument(
        "--detail-mode",
        dest="detail_mode",
        action="store_true",
        help="Forward the conversion through the OCC detail pipeline (enables high-detail remeshing).",
    )
    parser.add_argument(
        "--detail-scope",
        dest="detail_scope",
        choices=("none", "all", "object"),
        default=None,
        help="Select which objects receive OCC detail meshes ('none', 'all', or 'object' IDs only).",
    )
    parser.add_argument(
        "--detail-level",
        dest="detail_level",
        choices=("subshape", "face"),
        default=None,
        help="Granularity of OCC detail meshes when enabled ('subshape' groups or per 'face').",
    )
    parser.add_argument(
        "--detail-object-ids",
        dest="detail_object_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="STEP_ID",
        help="STEP ids to remesh when --detail-scope object is used (space-separated list).",
    )
    parser.add_argument(
        "--detail-object-guids",
        dest="detail_object_guids",
        nargs="+",
        default=None,
        metavar="GUID",
        help="GUIDs to remesh when --detail-scope object is used (space-separated list).",
    )

    return parser.parse_args(argv)
