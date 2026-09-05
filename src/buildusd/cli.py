from __future__ import annotations

import argparse
from typing import Sequence


class _JoinPathAction(argparse.Action):
    """Join successive CLI tokens into a single path string (handles spaces gracefully)."""

    def __call__(self, parser, namespace, values, option_string=None):
        joined = " ".join(values).strip()
        setattr(namespace, self.dest, joined or None)


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_input_root,
    default_output_root,
    usd_format_choices: Sequence[str],
    default_usd_format: str,
    default_usd_auto_binary_threshold_mb: float,
    default_geospatial_mode: str = "auto",
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
        "--forma-project-id",
        default=None,
        help="Autodesk Forma/ACC Data Management project ID",
    )
    parser.add_argument(
        "--forma-project-url",
        "--base-url",
        dest="forma_project_url",
        default=None,
        help="Base ACC Docs project, folder, or file URL",
    )
    parser.add_argument(
        "--forma-folder-id",
        default=None,
        help="Set the base folder using an ACC folder URN",
    )
    parser.add_argument(
        "--forma-version-id",
        default=None,
        help="Immutable Autodesk Forma/ACC Data Management file version ID",
    )
    parser.add_argument(
        "--forma-browse",
        action="store_true",
        help="Crawl project folders, show RVT/IFC files, and prompt for a selection",
    )
    parser.add_argument(
        "--subfolder",
        dest="forma_subfolders",
        action="store_true",
        help="Include descendants of the folder supplied by the base URL",
    )
    parser.add_argument(
        "--depth",
        dest="forma_depth",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help=(
            "Maximum folder depth below the base folder; 0 is the base folder only, "
            "1 includes its immediate subfolders"
        ),
    )
    parser.add_argument(
        "--forma-file",
        dest="forma_files",
        action="append",
        default=[],
        help="Select a discovered RVT/IFC by name or project-relative glob (repeatable)",
    )
    parser.add_argument(
        "--forma-cache-dir",
        default=None,
        help="Local cache for downloaded or translated Forma IFC files",
    )
    parser.add_argument(
        "--forma-ifc-export-setting",
        default="IFC4 Reference View",
        help="APS RVT-to-IFC export setting (default: %(default)s)",
    )
    parser.add_argument(
        "--forma-force-translation",
        action="store_true",
        help="Ask APS Model Derivative to regenerate an RVT-to-IFC derivative",
    )
    parser.add_argument(
        "--forma-translation-timeout-seconds",
        type=float,
        default=1800.0,
        help="Maximum wait for RVT-to-IFC translation (default: %(default)s)",
    )
    parser.add_argument(
        "--forma-poll-interval-seconds",
        type=float,
        default=5.0,
        help="APS translation polling interval (default: %(default)s)",
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
        help="Process every eligible local file or discovered ACC model",
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
        "--include-2d",
        dest="include_2d",
        action="store_true",
        help="Include 2D annotation geometry in the conversion output (default: off).",
    )
    parser.add_argument(
        "--anchor-mode",
        dest="anchor_mode",
        choices=("local", "basepoint", "none"),
        default="none",
        help="Choose whether stages anchor to the IFC site placement, base point (PBP/SP), or skip anchoring entirely.",
    )
    parser.add_argument(
        "--federate",
        dest="federate",
        action="store_true",
        help="After conversion, assemble/refresh federated master stages from the converted outputs.",
    )
    parser.add_argument(
        "--federate-into",
        dest="federate_into",
        nargs="+",
        action=_JoinPathAction,
        default=None,
        help=(
            "With --federate, append converted stage payloads into this explicit target master "
            "stage path instead of manifest-routed masters."
        ),
    )
    parser.add_argument(
        "--frame",
        dest="frame",
        choices=("projected", "geodetic"),
        default="projected",
        help="Federation frame for --federate (default: %(default)s).",
    )
    parser.add_argument(
        "--unanchored",
        "--missing-anchor-policy",
        dest="missing_anchor_policy",
        choices=("skip", "same-origin"),
        default="skip",
        help=(
            "With --federate, handle stages without anchor metadata by skipping them "
            "or by placing them at zero offset when all payloads share one local origin "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--geospatial-mode",
        dest="geospatial_mode",
        choices=("auto", "usd", "omni", "none"),
        default=default_geospatial_mode,
        help=(
            "Select geospatial driver: 'usd' for self-contained metadata, "
            "'omni' for OmniGeospatial/Kit contexts, 'none' to skip, 'auto' to pick based on offline/omniverse paths."
        ),
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
        "--detail-mode",
        dest="detail_mode",
        action="store_true",
        help="Forward the conversion through the OCC detail pipeline (iterator mesh remains base).",
    )
    parser.add_argument(
        "--detail-scope",
        dest="detail_scope",
        choices=("all", "object"),
        default=None,
        help="Select which objects receive OCC detail meshes ('all' or specific objects via --detail-objects).",
    )
    parser.add_argument(
        "--detail-objects",
        dest="detail_objects",
        nargs="+",
        default=None,
        metavar="STEP_OR_GUID",
        help="STEP ids or GUIDs to remesh when --detail-scope object is used (space-separated list).",
    )
    parser.add_argument(
        "--enable-semantic-subcomponents",
        dest="enable_semantic_subcomponents",
        action="store_true",
        help="Enable semantic subcomponent splitting (e.g. Panel, Frame, Glazing).",
    )
    parser.add_argument(
        "--semantic-tokens",
        dest="semantic_tokens_path",
        default=None,
        help="Path to a JSON file containing custom semantic tokens.",
    )

    parser.add_argument(
        "--detail-engine",
        dest="detail_engine",
        choices=(
            "default",
            "occ",
            "opencascade",
            "semantic",
            "ifc-subcomponents",
            "ifc-parts",
        ),
        default="default",
        help=(
            "Select detail pipeline behavior: 'default' tries IFC subcomponents then OCC; "
            "'occ'/'opencascade' skips subcomponents and goes straight to OCC; "
            "'semantic'/'ifc-subcomponents'/'ifc-parts' runs IFC subcomponent splitting only (no OCC fallback)."
        ),
    )

    return parser.parse_args(argv)
