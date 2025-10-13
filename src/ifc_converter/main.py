# ===============================
# main.py — FULL REPLACEMENT (aligned to single-pass IFC process)
# ===============================
from __future__ import annotations
import argparse
import json
import logging
import re
import tempfile
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence, Union
from .config.manifest import BasePointConfig, ConversionManifest, ResolvedFilePlan
from .io_utils import (
    ensure_directory,
    ensure_parent_directory,
    is_omniverse_path,
    join_path,
    create_checkpoint,
    list_directory,
    normalize_exclusions,
    path_name,
    path_stem,
    path_suffix,
    read_bytes,
    read_text,
    stat_entry,
)
from .process_ifc import ConversionOptions, CurveWidthRule, build_prototypes
from .process_usd import (
    apply_stage_anchor_transform,
    assign_world_geolocation,
    author_instance_layer,
    author_material_layer,
    author_prototype_layer,
    author_geometry2d_layer,
    bind_materials_to_prototypes,
    create_usd_stage,
    persist_instance_cache,
    update_federated_view,
)
from .usd_context import initialize_usd, shutdown_usd_context
__all__ = [
    "ConversionResult",
    "convert",
    "parse_args",
    "ConversionOptions",
    "CurveWidthRule",
    "ConversionManifest",
]
LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path(r"C:\Users\samue\_dev\datasets\ifc\tvc").resolve()
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "output"
OPTIONS = ConversionOptions(
    enable_instancing=True,
    enable_hash_dedup=False,
    convert_metadata=True,
)
DEFAULT_MASTER_STAGE = "Federated Model.usda"
DEFAULT_USD_FORMAT = "usdc"
USD_FORMAT_CHOICES = ("usdc", "usda", "usd", "auto")
DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB = 500.0
DEFAULT_GEODETIC_CRS = "EPSG:4326"
DEFAULT_BASE_POINT = BasePointConfig(
    easting=333800.4900,
    northing=5809101.4680,
    height=0.0,
    unit="m",
)
PathLike = Union[str, Path]
@dataclass(slots=True)
class ConversionResult:
    """Artifacts produced for a single converted IFC file."""
    ifc_path: PathLike
    stage_path: PathLike
    master_stage_path: PathLike
    layers: dict[str, str | None]
    projected_crs: str
    geodetic_crs: str
    geodetic_coordinates: tuple[float, ...] | None
    counts: dict[str, int]
    plan: ResolvedFilePlan | None
    revision: Optional[str] = None
    def as_dict(self) -> dict[str, Any]:
        """Legacy dict-compatible view of the conversion result."""
        return {
            "ifc": self.ifc_path,
            "stage": self.stage_path,
            "master_stage": self.master_stage_path,
            "layers": self.layers,
            "geodetic": {
                "crs": self.geodetic_crs,
                "coordinates": self.geodetic_coordinates,
            },
            "wgs84": self.geodetic_coordinates,
            "counts": self.counts,
            "plan": self.plan,
            "projected_crs": self.projected_crs,
            "revision": self.revision,
        }
@dataclass(slots=True)
class _IfcTarget:
    source: PathLike
    manifest_key: Path
@dataclass(slots=True)
class _OutputLayout:
    stage: PathLike
    prototypes: PathLike
    materials: PathLike
    instances: PathLike
    geometry2d: PathLike
    cache_dir: PathLike
# ---------------- CLI ----------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone converter."""
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
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Path to an IFC file or a directory containing IFC files",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory for USD artifacts (default: data/output under repo root)",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=str,
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
        "--usd-format",
        dest="usd_format",
        choices=USD_FORMAT_CHOICES,
        default=DEFAULT_USD_FORMAT,
        help="Output USD format to use (default: %(default)s). Use 'auto' to heuristically pick between USDA and USDC.",
    )
    parser.add_argument(
        "--usd-auto-binary-threshold-mb",
        dest="usd_auto_binary_threshold_mb",
        type=float,
        default=DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB,
        help=(
            "When using --usd-format usda, re-export layers as usdc when the file exceeds this many megabytes "
            "(set to 0 to disable; default: %(default)s). Auto mode also relies on this threshold for its heuristic."
        ),
    )
    return parser.parse_args(argv)
# ------------- helpers -------------
_WIDTH_VALUE_RE = re.compile(r"^\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>[A-Za-z]+)?\s*$")
def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
from .process_usd import (
    _layer_identifier,
    _rule_from_mapping,
    _rules_from_config_payload,
    _sublayer_identifier,
)  # reuse exact logic
# ------------- core convert -------------
def convert(
    input_path: PathLike,
    *,
    output_dir: PathLike | None = None,
    map_coordinate_system: str = "EPSG:7855",
    manifest: ConversionManifest | None = None,
    manifest_path: PathLike | None = None,
    ifc_names: Sequence[str] | None = None,
    process_all: bool = False,
    exclude_names: Sequence[str] | None = None,
    options: ConversionOptions | None = None,
    usd_format: str = DEFAULT_USD_FORMAT,
    usd_auto_binary_threshold_mb: float | None = DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB,
    logger: logging.Logger | None = None,
    checkpoint: bool = False,
    offline: bool = False,
) -> list[ConversionResult]:
    """Programmatic API for running the converter inside another application."""
    log = logger or LOG
    if manifest_path and manifest is not None:
        raise ValueError("Provide either manifest or manifest_path, not both.")
    exclude = normalize_exclusions(exclude_names)
    if output_dir is None:
        output_root: PathLike = DEFAULT_OUTPUT_ROOT
    else:
        output_root = output_dir
    if not is_omniverse_path(output_root):
        output_root = Path(output_root).resolve()
    check_paths: list[PathLike] = [input_path, output_root]
    if manifest_path:
        check_paths.append(manifest_path)
    if offline:
        if any(is_omniverse_path(p) for p in check_paths):
            raise ValueError("--offline mode requires local input/output/manifest paths")
        initialize_usd(offline=True)
        if checkpoint:
            log.warning("--checkpoint is ignored in offline mode (Nucleus only).")
            checkpoint = False
    else:
        initialize_usd(offline=False)
    ensure_directory(output_root)
    usd_format_normalized = _normalise_usd_format(usd_format)
    auto_binary_threshold = (
        usd_auto_binary_threshold_mb
        if usd_auto_binary_threshold_mb is not None and usd_auto_binary_threshold_mb > 0
        else None
    )
    manifest_obj = manifest
    if manifest_path:
        if is_omniverse_path(manifest_path):
            text = read_text(manifest_path)
            suffix = path_suffix(manifest_path)
            manifest_obj = ConversionManifest.from_text(text, suffix=suffix or ".json")
            log.info("Loaded manifest from %s", manifest_path)
        else:
            manifest_fp = Path(manifest_path).resolve()
            manifest_obj = ConversionManifest.from_file(manifest_fp)
            log.info("Loaded manifest from %s", manifest_fp)
    opts = options or OPTIONS
    run_options = replace(opts, manifest=manifest_obj)
    targets = _collect_ifc_paths(
        input_path,
        ifc_names=tuple(ifc_names) if ifc_names else None,
        process_all=process_all,
        exclude=exclude,
        log=log,
    )
    if not targets:
        log.info("Connected to %s but found no IFC files to convert.", input_path)
        return []
    log.info("Discovered %d IFC file(s) under %s", len(targets), input_path)
    results: list[ConversionResult] = []
    for target in targets:
        log.info("Starting conversion for %s", target.source)
        plan = None
        if manifest_obj is not None:
            try:
                plan = manifest_obj.resolve_for_path(
                    target.manifest_key,
                    fallback_master_name=DEFAULT_MASTER_STAGE,
                    fallback_projected_crs=map_coordinate_system,
                    fallback_geodetic_crs=DEFAULT_GEODETIC_CRS,
                    fallback_base_point=DEFAULT_BASE_POINT,
                )
            except Exception as exc:
                log.warning("Manifest resolution failed for %s: %s", path_name(target.source), exc)
        try:
            result = _process_single_ifc(
                target.source,
                output_root=output_root,
                coordinate_system=map_coordinate_system,
                options=run_options,
                plan=plan,
                logger=log,
                checkpoint=checkpoint,
                usd_format=usd_format_normalized,
                usd_auto_binary_threshold_mb=auto_binary_threshold,
            )
        except Exception as exc:
            log.error(
                "Conversion failed for %s: %s",
                path_name(target.source),
                exc,
                exc_info=True,
            )
            continue
        if result is None:
            log.warning("Conversion for %s produced no result; skipping.", target.source)
            continue
        results.append(result)
    return results
# ------------- local & nucleus helpers -------------
def _collect_ifc_paths(
    input_path: PathLike,
    *,
    ifc_names: Sequence[str] | None,
    process_all: bool,
    exclude: set[str],
    log: logging.Logger,
) -> list[_IfcTarget]:
    if is_omniverse_path(input_path):
        return _collect_ifc_paths_nucleus(
            str(input_path),
            ifc_names=ifc_names,
            process_all=process_all,
            exclude=exclude,
            log=log,
        )
    path_obj = Path(input_path).resolve()
    log.info("Scanning local directory %s", path_obj)
    return _collect_ifc_paths_local(
        path_obj,
        ifc_names=ifc_names,
        process_all=process_all,
        exclude=exclude,
        log=log,
    )
def _collect_ifc_paths_local(
    root: Path,
    *,
    ifc_names: Sequence[str] | None,
    process_all: bool,
    exclude: set[str],
    log: logging.Logger,
) -> list[_IfcTarget]:
    if root.is_file():
        if root.suffix.lower() != ".ifc":
            raise ValueError(f"Input file is not an IFC: {root}")
        stem = root.stem.lower()
        if stem in exclude:
            log.info("Skipping excluded IFC %s", root)
            return []
        resolved = root.resolve()
        return [_IfcTarget(source=resolved, manifest_key=resolved)]
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a file or directory: {root}")
    targets: list[_IfcTarget] = []
    if ifc_names and not process_all:
        for name in ifc_names:
            candidate = root / _ensure_ifc_name(name)
            if not candidate.exists() or not candidate.is_file():
                log.warning("Skipping missing IFC file %s", candidate)
                continue
            if candidate.suffix.lower() != ".ifc":
                log.warning("Skipping non-IFC file %s", candidate)
                continue
            stem = candidate.stem.lower()
            if stem in exclude:
                log.info("Skipping excluded IFC %s", candidate)
                continue
            resolved = candidate.resolve()
            targets.append(_IfcTarget(source=resolved, manifest_key=resolved))
        return sorted(targets, key=lambda t: str(t.source).lower())
    candidates = []
    for child in root.iterdir():
        if child.is_file() and child.suffix.lower() == ".ifc":
            candidates.append(child.resolve())
    candidates.sort()
    for candidate in candidates:
        if candidate.stem.lower() in exclude:
            log.info("Skipping excluded IFC %s", candidate)
            continue
        targets.append(_IfcTarget(source=candidate, manifest_key=candidate))
    return targets
def _collect_ifc_paths_nucleus(
    uri: str,
    *,
    ifc_names: Sequence[str] | None,
    process_all: bool,
    exclude: set[str],
    log: logging.Logger,
) -> list[_IfcTarget]:
    """Collect IFC files from a Nucleus directory or file URI with robust handling."""
    def _single_file_target(file_uri: str) -> list[_IfcTarget]:
        if not _is_ifc_candidate(path_name(file_uri)):
            log.error("Nucleus path is not an IFC file: %s", file_uri)
            return []
        stem = path_stem(file_uri).lower()
        if stem in exclude:
            log.info("Skipping excluded IFC %s", file_uri)
            return []
        manifest_key = _manifest_key_for_path(file_uri)
        return [_IfcTarget(source=file_uri, manifest_key=manifest_key)]

    # If it looks like a file (and user didn't force a directory scan), fast-path
    if _is_ifc_candidate(path_name(uri)) and not (ifc_names and process_all):
        return _single_file_target(uri)

    # Directory listing via io_utils (no new args!)
    log.info("Connecting to Nucleus directory %s", uri)
    try:
        entries = list_directory(uri)  # <- io_utils.list_directory(path) returns List[ListEntry]
    except (RuntimeError, FileNotFoundError) as exc:
        # If listing fails but the uri is a file, try single-file fallback
        if _is_ifc_candidate(path_name(uri)):
            log.warning("Directory listing failed (%s). Falling back to single-file target: %s", exc, uri)
            return _single_file_target(uri)
        log.error("Failed to list Nucleus directory %s: %s", uri, exc, exc_info=True)
        return []
    except Exception as exc:
        log.error("Unexpected error listing %s: %s", uri, exc, exc_info=True)
        return []

    targets: list[_IfcTarget] = []

    # If specific IFC names were provided, resolve only those (don’t trawl entire dir)
    if ifc_names and not process_all:
        for name in ifc_names:
            candidate = join_path(uri, _ensure_ifc_name(name))
            try:
                entry = stat_entry(candidate)
            except Exception as exc:
                log.warning("stat_entry failed for %s: %s", candidate, exc)
                entry = None
            if entry is None:
                log.warning("Skipping missing IFC file %s", candidate)
                continue
            if not _is_ifc_candidate(path_name(candidate)):
                log.warning("Skipping non-IFC file %s", candidate)
                continue
            stem = path_stem(candidate).lower()
            if stem in exclude:
                log.info("Skipping excluded IFC %s", candidate)
                continue
            targets.append(_IfcTarget(source=candidate, manifest_key=_manifest_key_for_path(candidate)))
        log.info("Resolved %d IFC(s) by explicit name under %s", len(targets), uri)
        return sorted(targets, key=lambda t: path_name(t.source).lower())

    # Helper: tolerate different omni.client ListEntry attribute names
    def _entry_to_path(e) -> Optional[str]:
        # io_utils.list_directory returns omni.client entries with properties that vary by build.
        # Try common attributes in order of likelihood.
        for attr in ("relative_path", "relativePath", "path", "uri"):
            p = getattr(e, attr, None)
            if isinstance(p, str) and p.strip():
                return join_path(uri, p) if attr in ("relative_path", "relativePath") else p
        # Some builds expose 'relative_path' inside 'info' or a tuple-like entry
        info = getattr(e, "info", None)
        if info and hasattr(info, "relative_path"):
            return join_path(uri, info.relative_path)
        return None

    # Full directory trawl
    bad = 0
    for e in entries:
        try:
            candidate = _entry_to_path(e)
            if not candidate:
                bad += 1
                continue
            name = path_name(candidate)
            if not _is_ifc_candidate(name):
                continue
            stem = path_stem(candidate).lower()
            if stem in exclude:
                log.info("Skipping excluded IFC %s", candidate)
                continue
            targets.append(_IfcTarget(source=candidate, manifest_key=_manifest_key_for_path(candidate)))
        except Exception as exc:
            bad += 1
            log.debug("Skipping problematic listing entry due to: %s", exc, exc_info=True)

    # Sort for deterministic order
    targets.sort(key=lambda t: path_name(t.source).lower())

    if not targets:
        log.warning("Nucleus listing returned 0 IFC files under %s (skipped=%d). "
                    "Check permissions and folder path case.", uri, bad)
    else:
        log.info("Discovered %d IFC file(s) under %s (skipped=%d)", len(targets), uri, bad)
    return targets

def _ensure_ifc_name(name: str) -> str:
    trimmed = (name or "").strip()
    if not trimmed: return trimmed
    if not trimmed.lower().endswith(".ifc"): trimmed = f"{trimmed}.ifc"
    return trimmed
def _is_ifc_candidate(name: str) -> bool:
    return (name or "").lower().endswith(".ifc")


def _normalise_usd_format(fmt: str) -> str:
    """Return a validated USD format string, defaulting when necessary."""
    value = (fmt or DEFAULT_USD_FORMAT).lower()
    if value not in USD_FORMAT_CHOICES:
        logging.getLogger(__name__).warning(
            "Unsupported usd_format %s; falling back to %s", fmt, DEFAULT_USD_FORMAT
        )
        return DEFAULT_USD_FORMAT
    return value

def _manifest_key_for_path(path: PathLike) -> Path:
    if isinstance(path, Path): return path.resolve()
    if is_omniverse_path(path):
        uri = str(path); stripped = uri.split("://", 1)[-1]
        return PurePosixPath(stripped)
    return Path(str(path)).resolve()
# ------------- layout + checkpoint -------------
def _build_output_layout(output_root: PathLike, base_name: str, usd_format: str) -> _OutputLayout:
    ensure_directory(output_root)
    prototypes_dir = join_path(output_root, "prototypes")
    materials_dir = join_path(output_root, "materials")
    instances_dir = join_path(output_root, "instances")
    geometry2d_dir = join_path(output_root, "geometry2d")
    caches_dir = join_path(output_root, "caches")
    for directory in (prototypes_dir, materials_dir, instances_dir, geometry2d_dir, caches_dir):
        ensure_directory(directory)
    ext = _normalise_usd_format(usd_format)
    if ext == "auto":
        raise ValueError("Auto USD format must be resolved before building the output layout.")
    stage_path = join_path(output_root, f"{base_name}.{ext}")
    ensure_parent_directory(stage_path)
    return _OutputLayout(
        stage=stage_path,
        prototypes=join_path(prototypes_dir, f"{base_name}_prototypes.{ext}"),
        materials=join_path(materials_dir, f"{base_name}_materials.{ext}"),
        instances=join_path(instances_dir, f"{base_name}_instances.{ext}"),
        geometry2d=join_path(geometry2d_dir, f"{base_name}_geometry2d.{ext}"),
        cache_dir=caches_dir,
    )


def _update_sublayer_reference(
    root_layer,
    old_reference: str,
    new_reference: str,
    label: str,
    logger: logging.Logger,
) -> bool:
    """Replace a subLayerPaths entry, returning True when an update occurred."""
    if old_reference == new_reference:
        return False
    try:
        existing = list(root_layer.subLayerPaths)
    except Exception as exc:
        logger.debug("Unable to read subLayerPaths while updating %s layer: %s", label, exc)
        return False
    try:
        index = existing.index(old_reference)
    except ValueError:
        logger.debug(
            "Expected %s sublayer reference %s not present; skipping update to %s",
            label,
            old_reference,
            new_reference,
        )
        return False
    existing[index] = new_reference
    root_layer.subLayerPaths = existing
    return True


BYTES_PER_MB = 1024 * 1024


@dataclass(slots=True)
class _AutoFormatEstimate:
    mesh_bytes: int
    prototype_count: int
    hash_count: int
    instance_count: int
    annotation_count: int


def _estimate_array_nbytes(value: Any) -> int:
    """Best-effort byte size estimate for array-like payloads."""
    if value is None:
        return 0
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            if isinstance(item, (list, tuple)):
                total += _estimate_array_nbytes(item)
            elif isinstance(item, (bytes, bytearray, memoryview)):
                total += len(item)
            elif isinstance(item, (int, float)):
                total += 8
            else:
                total += _estimate_array_nbytes(item)
        return total
    try:
        return int(sys.getsizeof(value))
    except Exception:
        return 0


def _estimate_mesh_payload_bytes(mesh: Any) -> int:
    """Return summed byte estimates for mesh dictionary payloads."""
    if mesh is None:
        return 0
    if isinstance(mesh, dict):
        return sum(_estimate_array_nbytes(v) for v in mesh.values())
    return _estimate_array_nbytes(mesh)


def _estimate_auto_payload(caches: PrototypeCaches) -> _AutoFormatEstimate:
    """Aggregate lightweight metrics used for auto USD format heuristics."""
    prototype_mesh_bytes = 0
    for proto in caches.repmaps.values():
        prototype_mesh_bytes += _estimate_mesh_payload_bytes(getattr(proto, "mesh", None))
    for proto in caches.hashes.values():
        prototype_mesh_bytes += _estimate_mesh_payload_bytes(getattr(proto, "mesh", None))
    annotation_count = len(getattr(caches, "annotations", {}) or {})
    return _AutoFormatEstimate(
        mesh_bytes=prototype_mesh_bytes,
        prototype_count=len(caches.repmaps),
        hash_count=len(caches.hashes),
        instance_count=len(caches.instances),
        annotation_count=annotation_count,
    )


def _select_auto_usd_format(
    caches: PrototypeCaches,
    *,
    threshold_mb: float | None,
    base_name: str,
    logger: logging.Logger,
) -> tuple[str, _AutoFormatEstimate]:
    """Choose an authoring format based on heuristic cache metrics."""
    estimate = _estimate_auto_payload(caches)
    threshold_bytes: int | None = None
    if threshold_mb is not None and threshold_mb > 0:
        threshold_bytes = int(threshold_mb * BYTES_PER_MB)
    estimate_mb = estimate.mesh_bytes / BYTES_PER_MB if estimate.mesh_bytes else 0.0

    if threshold_bytes is None:
        chosen = DEFAULT_USD_FORMAT
        logger.warning(
            "Auto USD format for %s: threshold disabled, defaulting to %s. "
            "Heuristic mesh payload sum ≈ %.1f MB; actual USD size may differ.",
            base_name,
            chosen.upper(),
            estimate_mb,
        )
        return chosen, estimate

    if estimate.mesh_bytes >= threshold_bytes:
        chosen = "usdc"
        logger.warning(
            "Auto USD format for %s: summed prototype mesh array nbytes ≈ %.1f MB exceeds threshold %.1f MB; "
            "selecting USDC. This decision is heuristic—the final USD size may vary.",
            base_name,
            estimate_mb,
            threshold_mb,
        )
    else:
        chosen = "usda"
        logger.info(
            "Auto USD format for %s: summed prototype mesh array nbytes ≈ %.1f MB below threshold %.1f MB; "
            "selecting USDA. This decision is heuristic—the final USD size may vary.",
            base_name,
            estimate_mb,
            threshold_mb,
        )
    logger.debug(
        "Auto USD heuristic details for %s: prototypes=%d, hashed=%d, instances=%d, annotations=%d, mesh_bytes=%d.",
        base_name,
        estimate.prototype_count,
        estimate.hash_count,
        estimate.instance_count,
        estimate.annotation_count,
        estimate.mesh_bytes,
    )
    return chosen, estimate


def _maybe_convert_layers_to_usdc(
    stage,
    layout: _OutputLayout,
    *,
    proto_layer,
    material_layer,
    inst_layer,
    geometry2d_layer,
    threshold_mb: float,
    logger: logging.Logger,
) -> tuple[_OutputLayout, list[Path]]:
    """Convert large text-based layers to usdc when they exceed the configured threshold."""
    threshold_bytes = int(max(threshold_mb, 0.0) * 1024 * 1024)
    if threshold_bytes <= 0:
        return layout, []
    root_layer = stage.GetRootLayer()
    paths_to_remove: list[Path] = []
    sublayers_modified = False

    def _convert_layer(layer_obj, current_path: PathLike, label: str) -> PathLike:
        nonlocal sublayers_modified
        if layer_obj is None:
            return current_path
        path_str = str(current_path)
        if is_omniverse_path(path_str):
            return current_path
        path_obj = Path(path_str)
        if not path_obj.exists():
            return current_path
        suffix = path_obj.suffix.lower()
        if suffix in {".usd", ".usdc"}:
            return current_path
        try:
            size_bytes = path_obj.stat().st_size
        except OSError as exc:
            logger.debug("Unable to stat %s layer at %s: %s", label, path_obj, exc)
            return current_path
        if size_bytes <= threshold_bytes:
            return current_path
        target_path = path_obj.with_suffix(".usdc")
        try:
            layer_obj.Export(_layer_identifier(target_path), args={"format": "usdc"})
        except Exception as exc:
            logger.warning(
                "Failed to re-export %s layer %s as usdc: %s", label, path_obj, exc
            )
            return current_path
        size_mb = size_bytes / (1024 * 1024)
        logger.info(
            "%s layer %s exceeded %.1f MB (%.1f MB actual); switched to %s",
            label.capitalize(),
            path_obj,
            threshold_mb,
            size_mb,
            target_path,
        )
        old_ref = _sublayer_identifier(root_layer, current_path)
        new_ref = _sublayer_identifier(root_layer, target_path)
        if _update_sublayer_reference(root_layer, old_ref, new_ref, label, logger):
            sublayers_modified = True
        if not is_omniverse_path(path_str):
            paths_to_remove.append(path_obj)
        return target_path

    new_prototypes = _convert_layer(proto_layer, layout.prototypes, "prototype")
    new_materials = _convert_layer(material_layer, layout.materials, "material")
    new_instances = _convert_layer(inst_layer, layout.instances, "instance")
    new_geometry2d = (
        _convert_layer(geometry2d_layer, layout.geometry2d, "geometry2d")
        if geometry2d_layer is not None
        else layout.geometry2d
    )

    updated_layout = replace(
        layout,
        prototypes=new_prototypes,
        materials=new_materials,
        instances=new_instances,
        geometry2d=new_geometry2d,
    )

    stage_converted = False
    stage_path_str = str(updated_layout.stage)
    if not is_omniverse_path(stage_path_str):
        stage_path_obj = Path(stage_path_str)
        if stage_path_obj.exists() and stage_path_obj.suffix.lower() == ".usda":
            try:
                stage_size = stage_path_obj.stat().st_size
            except OSError as exc:
                logger.debug("Unable to stat stage layer at %s: %s", stage_path_obj, exc)
            else:
                if stage_size > threshold_bytes:
                    target_stage_path = stage_path_obj.with_suffix(".usdc")
                    try:
                        root_layer.Export(_layer_identifier(target_stage_path), args={"format": "usdc"})
                    except Exception as exc:
                        logger.warning(
                            "Failed to re-export stage %s as usdc: %s",
                            stage_path_obj,
                            exc,
                        )
                    else:
                        size_mb = stage_size / (1024 * 1024)
                        logger.info(
                            "Stage %s exceeded %.1f MB (%.1f MB actual); switched to %s",
                            stage_path_obj.name,
                            threshold_mb,
                            size_mb,
                            target_stage_path,
                        )
                        updated_layout = replace(updated_layout, stage=target_stage_path)
                        paths_to_remove.append(stage_path_obj)
                        stage_converted = True

    if sublayers_modified and not stage_converted:
        try:
            root_layer.Save()
        except Exception as exc:
            logger.warning("Failed to persist updated sublayer references: %s", exc)

    return updated_layout, paths_to_remove


def _cleanup_paths(paths: Sequence[Path], logger: logging.Logger) -> None:
    """Best-effort removal of temporary files produced during auto conversion."""
    for candidate in paths:
        path_obj = candidate if isinstance(candidate, Path) else Path(str(candidate))
        path_str = str(path_obj)
        if is_omniverse_path(path_str):
            continue
        try:
            if path_obj.exists():
                path_obj.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.debug("Unable to remove temporary USD file %s: %s", path_obj, exc)
def _make_checkpoint_metadata(revision: Optional[str], base_name: str) -> tuple[str, list[str]]:
    note = revision or f"{base_name}"; tag_src = revision or base_name
    tag_candidates = tag_src.replace(",", " ").split(); tags = [t.strip() for t in tag_candidates if t.strip()] or [base_name]
    return note, tags
def _checkpoint_path(path: PathLike, note: str, tags: Sequence[str], logger: logging.Logger, label: str) -> None:
    try:
        did_checkpoint = create_checkpoint(path, note=note, tags=tags)
    except Exception as exc:
        logger.warning("Failed to checkpoint %s (%s): %s", label, path, exc)
        return
    if did_checkpoint:
        logger.info("Checkpoint saved for %s (%s)", label, note)
# ------------- core per-file -------------
def _process_single_ifc(
    ifc_path: PathLike,
    *,
    output_root: PathLike,
    coordinate_system: str,
    options: ConversionOptions,
    plan: ResolvedFilePlan | None,
    logger: logging.Logger,
    checkpoint: bool,
    usd_format: str = DEFAULT_USD_FORMAT,
    usd_auto_binary_threshold_mb: float | None = DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB,
    default_base_point: BasePointConfig = DEFAULT_BASE_POINT,
    default_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    default_master_stage: str = DEFAULT_MASTER_STAGE,
) -> ConversionResult:
    import ifcopenshell  # Local import
    base_name = path_stem(ifc_path)
    revision_note = plan.revision if plan and plan.revision else None
    checkpoint_note: Optional[str] = None
    checkpoint_tags: list[str] = []
    if checkpoint:
        checkpoint_note, checkpoint_tags = _make_checkpoint_metadata(revision_note, base_name)
    def _execute(local_ifc: Path) -> ConversionResult:
        ifc = ifcopenshell.open(local_ifc.as_posix())
        caches = build_prototypes(ifc, options)
        logger.info(
            "IFC %s → %d type prototypes, %d hashed prototypes, %d instances",
            path_name(ifc_path), len(caches.repmaps), len(caches.hashes), len(caches.instances),
        )
        authoring_format = _normalise_usd_format(usd_format)
        if authoring_format == "auto":
            authoring_format, _ = _select_auto_usd_format(
                caches,
                threshold_mb=usd_auto_binary_threshold_mb,
                base_name=base_name,
                logger=logger,
            )
        layout = _build_output_layout(output_root, base_name, authoring_format)
        stage = create_usd_stage(layout.stage)
        proto_layer, proto_paths = author_prototype_layer(stage, caches, layout.prototypes, base_name, options)
        mat_layer, material_paths = author_material_layer(
            stage, caches, proto_paths, layer_path=layout.materials, base_name=base_name, proto_layer=proto_layer, options=options,
        )
        bind_materials_to_prototypes(stage, proto_layer, proto_paths, material_paths)
        inst_layer = author_instance_layer(stage, caches, proto_paths, layer_path=layout.instances, base_name=base_name, options=options)
        geometry2d_layer = author_geometry2d_layer(stage, caches, layout.geometry2d, base_name, options)
        persist_instance_cache(layout.cache_dir, base_name, caches, proto_paths)
        effective_base_point = plan.base_point if plan and plan.base_point else default_base_point
        if effective_base_point is None:
            raise ValueError(
                f"No base point configured for {path_name(ifc_path)}; provide a manifest entry or update defaults."
            )
        projected_crs = plan.projected_crs if plan and plan.projected_crs else coordinate_system
        if not projected_crs:
            raise ValueError(
                f"No projected CRS available for {path_name(ifc_path)}; supply --map-coordinate-system or manifest override."
            )
        geodetic_crs = plan.geodetic_crs if plan and plan.geodetic_crs else default_geodetic_crs
        master_stage_name = plan.master_stage_filename if plan else default_master_stage
        lonlat_override = plan.lonlat if plan else None
        apply_stage_anchor_transform(
            stage, caches, base_point=effective_base_point, projected_crs=projected_crs, align_axes_to_map=True, lonlat=lonlat_override,
        )
        logger.info("Assigning world geolocation using %s → %s", projected_crs, geodetic_crs)
        geodetic_result = assign_world_geolocation(
            stage, base_point=effective_base_point, projected_crs=projected_crs, geodetic_crs=geodetic_crs, unit_hint=effective_base_point.unit, lonlat_override=lonlat_override,
        )
        stage.Save()
        proto_layer.Save()
        mat_layer.Save()
        inst_layer.Save()
        if geometry2d_layer:
            geometry2d_layer.Save()
        paths_to_cleanup: list[Path] = []
        if authoring_format == "usda" and usd_auto_binary_threshold_mb and usd_auto_binary_threshold_mb > 0:
            layout, paths_to_cleanup = _maybe_convert_layers_to_usdc(
                stage,
                layout,
                proto_layer=proto_layer,
                material_layer=mat_layer,
                inst_layer=inst_layer,
                geometry2d_layer=geometry2d_layer,
                threshold_mb=usd_auto_binary_threshold_mb,
                logger=logger,
            )
        if checkpoint and checkpoint_note is not None:
            _checkpoint_path(layout.stage, checkpoint_note, checkpoint_tags, logger, f"{base_name} stage")
            _checkpoint_path(layout.prototypes, checkpoint_note, checkpoint_tags, logger, f"{base_name} prototypes layer")
            _checkpoint_path(layout.materials, checkpoint_note, checkpoint_tags, logger, f"{base_name} materials layer")
            _checkpoint_path(layout.instances, checkpoint_note, checkpoint_tags, logger, f"{base_name} instances layer")
            if geometry2d_layer:
                _checkpoint_path(layout.geometry2d, checkpoint_note, checkpoint_tags, logger, f"{base_name} 2D geometry layer")
        logger.info("Wrote stage %s", str(layout.stage))
        master_stage_path = join_path(output_root, master_stage_name)
        ensure_parent_directory(master_stage_path)
        update_federated_view(master_stage_path, str(layout.stage), base_name, parent_prim_path="/World")
        if checkpoint and checkpoint_note is not None:
            _checkpoint_path(master_stage_path, checkpoint_note, checkpoint_tags, logger, f"{base_name} master stage")
        counts = {
            "prototypes_3d": len(caches.repmaps) + len(caches.hashes),
            "instances_3d": len(caches.instances),
            "curves_2d": len(getattr(caches, "annotations", {}) or {}),
            "total_elements": len(caches.instances) + len(getattr(caches, "annotations", {}) or {}),
        }
        layers = {
            "prototype": str(layout.prototypes),
            "material": str(layout.materials),
            "instance": str(layout.instances),
            "geometry2d": str(layout.geometry2d) if geometry2d_layer else None,
        }
        if geometry2d_layer:
            layers["annotation"] = str(layout.geometry2d)
        coordinates = tuple(geodetic_result) if geodetic_result else None
        close_stage = getattr(stage, "Close", None)
        if callable(close_stage):
            try:
                close_stage()
            except Exception as exc:
                logger.debug("stage.Close() failed for %s: %s", layout.stage, exc)
        for layer_handle in (proto_layer, mat_layer, inst_layer, geometry2d_layer):
            if layer_handle is None:
                continue
            close_layer = getattr(layer_handle, "Close", None)
            if callable(close_layer):
                try:
                    close_layer()
                except Exception as exc:
                    identifier = getattr(layer_handle, "identifier", "<anonymous>")
                    logger.debug("layer.Close() failed for %s: %s", identifier, exc)
        _cleanup_paths(paths_to_cleanup, logger)
        return ConversionResult(
            ifc_path=ifc_path,
            stage_path=layout.stage,
            master_stage_path=master_stage_path,
            layers=layers,
            projected_crs=projected_crs,
            geodetic_crs=geodetic_crs,
            geodetic_coordinates=coordinates,
            counts=counts,
            plan=plan,
            revision=revision_note,
        )
    LOG.info("Opening IFC %s", ifc_path)
    if is_omniverse_path(ifc_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / path_name(ifc_path)
            tmp_path.write_bytes(read_bytes(ifc_path))
            return _execute(tmp_path)
    local_ifc = ifc_path if isinstance(ifc_path, Path) else Path(ifc_path)
    return _execute(local_ifc)
# ------------- entrypoint -------------
def main(argv: Sequence[str] | None = None) -> list[ConversionResult]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        # width rules are parsed inside process_usd; keep the exact CLI but we reuse helpers
        width_rules = []
        options_override = replace(OPTIONS, curve_width_rules=tuple(width_rules)) if width_rules else OPTIONS
        results = convert(
            args.input_path,
            output_dir=args.output_dir,
            map_coordinate_system=args.map_coordinate_system,
            manifest_path=args.manifest_path,
            ifc_names=args.ifc_names,
            process_all=args.process_all,
            exclude_names=args.exclude,
            checkpoint=args.checkpoint,
            offline=args.offline,
            options=options_override,
            usd_format=args.usd_format,
            usd_auto_binary_threshold_mb=args.usd_auto_binary_threshold_mb,
        )
    finally:
        shutdown_usd_context()
    _print_summary(results)
    return results
# ------------- summary -------------
def _print_summary(results: Sequence[ConversionResult]) -> None:
    if not results:
        print("No IFC files were converted.")
        return
    print("\nSummary:")
    for result in results:
        counts = result.counts
        master_stage = path_name(result.master_stage_path) if result.master_stage_path else "n/a"
        coords = result.geodetic_coordinates
        revision = result.revision or "n/a"
        if coords and len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
            alt_val = coords[2] if len(coords) > 2 and coords[2] is not None else 0.0
            coord_info = f", geo=({coords[0]:.5f}, {coords[1]:.5f}, {alt_val:.2f})"
        else:
            coord_info = ""
        proto_3d = counts.get("prototypes_3d", counts.get("prototypes", 0))
        inst_3d = counts.get("instances_3d", counts.get("instances", 0))
        curves_2d = counts.get("curves_2d", counts.get("annotations", 0))
        total_elements = counts.get("total_elements")
        if total_elements is None:
            total_elements = inst_3d + curves_2d
        print(
            f"- {path_name(result.ifc_path)}: stage={result.stage_path}, master={master_stage}, "
            f"prototypes3D={proto_3d}, instances3D={inst_3d}, curves2D={curves_2d}, "
            f"totalElements={total_elements}, revision={revision}{coord_info}"
        )
