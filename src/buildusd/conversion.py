# ===============================
# main.py — FULL REPLACEMENT (aligned to single-pass IFC process)
# ===============================
from __future__ import annotations
import argparse
import json
import logging
import tempfile
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional, Sequence, Union

from .cli import parse_args as _cli_parse_args
from .config.manifest import BasePointConfig, ConversionManifest, MasterConfig, ResolvedFilePlan
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
    create_usd_stage,
    persist_instance_cache,
)
from .usd_context import initialize_usd, shutdown_usd_context
__all__ = [
    "ConversionCancelledError",
    "ConversionResult",
    "convert",
    "parse_args",
    "ConversionOptions",
    "CurveWidthRule",
    "ConversionManifest",
    "set_stage_unit",
    "set_stage_up_axis",
]
LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path.cwd()
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "output"

# ---------------- constants ----------------
OPTIONS = ConversionOptions(
    enable_instancing=True,
    enable_hash_dedup=False,
    convert_metadata=True,
    enable_high_detail_remesh=False,
    anchor_mode=None,
    split_topology_by_material=False,
    enable_material_classification=False,
)
USD_FORMAT_CHOICES = ("usdc", "usda", "usd", "auto")
DEFAULT_USD_FORMAT = "usdc"
DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB = 50.0
_FALLBACK_MASTER_STAGE = "Federated Model.usda"
_FALLBACK_GEODETIC_CRS = "EPSG:4326"
_FALLBACK_BASE_POINT = BasePointConfig(
    easting=333800.4900,
    northing=5809101.4680,
    height=0.0,
    unit="m",
    epsg="EPSG:7855",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Delegate to the CLI parser with project defaults."""

    return _cli_parse_args(
        argv,
        default_input_root=DEFAULT_INPUT_ROOT,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        usd_format_choices=USD_FORMAT_CHOICES,
        default_usd_format=DEFAULT_USD_FORMAT,
        default_usd_auto_binary_threshold_mb=DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB,
    )


def _load_manifest_backed_defaults() -> tuple[BasePointConfig, str, str, BasePointConfig]:
    """Fetch project defaults from the local manifest when present."""

    base_point = _FALLBACK_BASE_POINT
    geodetic_crs = _FALLBACK_GEODETIC_CRS
    master_stage = _FALLBACK_MASTER_STAGE
    shared_site = _FALLBACK_BASE_POINT

    config_dir = Path(__file__).resolve().parent / "config"
    manifest_obj: ConversionManifest | None = None
    manifest_path: Path | None = None
    for candidate in ("manifest.yaml", "manifest.yml", "manifest.json"):
        path = config_dir / candidate
        if path.exists():
            manifest_path = path
            break

    if manifest_path is not None:
        try:
            manifest_obj = ConversionManifest.from_file(manifest_path)
        except Exception as exc:
            LOG.debug("Failed to load manifest defaults from %s: %s", manifest_path, exc)
            manifest_obj = None

    if manifest_obj is not None:
        defaults = manifest_obj.defaults
        if defaults.base_point is not None:
            base_point = defaults.base_point
        if defaults.geodetic_crs:
            geodetic_crs = defaults.geodetic_crs
        if defaults.shared_site_base_point is not None:
            shared_site = defaults.shared_site_base_point
        elif defaults.base_point is not None:
            shared_site = defaults.base_point
        master_candidate: Optional[MasterConfig] = None
        if defaults.master_id:
            master_candidate = manifest_obj.masters.get(defaults.master_id)
            if master_candidate is None:
                LOG.debug(
                    "Defaults reference master id '%s' but it was not defined in %s",
                    defaults.master_id,
                    manifest_path,
                )
        if master_candidate is None and defaults.master_name:
            master_candidate = MasterConfig(id="__defaults__", name=defaults.master_name)
        if master_candidate is not None:
            master_stage = master_candidate.resolved_name()

    return base_point, geodetic_crs, master_stage, shared_site


DEFAULT_BASE_POINT, DEFAULT_GEODETIC_CRS, DEFAULT_MASTER_STAGE, DEFAULT_SHARED_BASE_POINT = _load_manifest_backed_defaults()
PathLike = Union[str, Path]


def _normalize_anchor_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    alias_map: dict[str, Optional[str]] = {
        "local": "local",
        "site": "site",
        "basepoint": "local",
        "shared_site": "site",
        "none": None,
        "": None,
    }
    if normalized not in alias_map:
        LOG.debug("Unknown anchor mode '%s'; ignoring", value)
        return None
    return alias_map[normalized]


class ConversionCancelledError(Exception):
    """Raised when a conversion is cancelled by the caller."""


def _is_cancelled(cancel_event: Any | None) -> bool:
    if cancel_event is None:
        return False
    is_set = getattr(cancel_event, "is_set", None)
    if callable(is_set):
        try:
            return bool(is_set())
        except Exception:
            return False
    return bool(cancel_event)


def _ensure_not_cancelled(cancel_event: Any | None, *, message: str | None = None) -> None:
    if _is_cancelled(cancel_event):
        raise ConversionCancelledError(message or "Conversion cancelled.")
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


def _normalize_stage_unit_target(path: PathLike) -> str:
    """Return an absolute/local path or passthrough omniverse URI."""
    if path is None:
        raise ValueError("--set-stage-unit requires a target USD path.")
    raw = str(path).strip()
    if not raw:
        raise ValueError("--set-stage-unit requires a target USD path.")
    if is_omniverse_path(raw):
        return raw
    path_obj = Path(raw).expanduser().resolve()
    return str(path_obj)


def _apply_stage_unit(target_path: PathLike, meters_per_unit: float) -> None:
    """Open a USD stage/layer and update its metersPerUnit metadata."""
    from .pxr_utils import Sdf, Usd, UsdGeom

    meters = float(meters_per_unit)
    if meters <= 0.0:
        raise ValueError("--stage-unit-value must be greater than zero.")
    normalized = _normalize_stage_unit_target(target_path)
    LOG.info("Setting metersPerUnit on %s to %s", normalized, meters)

    stage = None
    try:
        stage = Usd.Stage.Open(normalized)
    except Exception as exc:  # pragma: no cover - logging only
        LOG.debug("Usd.Stage.Open failed for %s: %s", normalized, exc)
        stage = None

    if stage is not None:
        UsdGeom.SetStageMetersPerUnit(stage, meters)
        root_layer = stage.GetRootLayer()
        if root_layer is None:
            raise ValueError(f"Stage {normalized} has no root layer to edit.")
        root_layer.Save()
        close = getattr(stage, "Close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - logging only
                LOG.debug("stage.Close() failed for %s: %s", normalized, exc)
        return

    layer = Sdf.Layer.FindOrOpen(normalized)
    if layer is None:
        raise ValueError(f"Failed to open USD layer {normalized}")
    layer.metersPerUnit = meters
    layer.Save()


def set_stage_unit(target_path: PathLike, meters_per_unit: float = 1.0, *, offline: bool = False) -> None:
    """Update the meters-per-unit metadata on an existing USD layer or stage."""
    raw_path = str(target_path).strip() if target_path is not None else ""
    if not raw_path:
        raise ValueError("--set-stage-unit requires a target USD path.")
    if meters_per_unit <= 0.0:
        raise ValueError("--stage-unit-value must be greater than zero.")
    if offline and is_omniverse_path(raw_path):
        raise ValueError("--offline cannot be combined with --set-stage-unit for omniverse:// targets.")
    initialize_usd(offline=offline)
    _apply_stage_unit(raw_path, meters_per_unit)


def _normalize_up_axis(axis: str | None) -> str:
    axis_text = str(axis or "").strip().upper()
    if axis_text not in {"X", "Y", "Z"}:
        raise ValueError("--stage-up-axis must be one of X, Y, or Z.")
    return axis_text


def _apply_stage_up_axis(target_path: PathLike, axis: str) -> None:
    from .pxr_utils import Sdf, Usd, UsdGeom

    normalized_path = _normalize_stage_unit_target(target_path)
    tokens = {
        "X": UsdGeom.Tokens.x,
        "Y": UsdGeom.Tokens.y,
        "Z": UsdGeom.Tokens.z,
    }
    axis_token = tokens[axis]

    stage = None
    try:
        stage = Usd.Stage.Open(normalized_path)
    except Exception as exc:  # pragma: no cover - logging only
        LOG.debug("Usd.Stage.Open failed for %s: %s", normalized_path, exc)
        stage = None

    if stage is not None:
        UsdGeom.SetStageUpAxis(stage, axis_token)
        root_layer = stage.GetRootLayer()
        if root_layer is None:
            raise ValueError(f"Stage {normalized_path} has no root layer to edit.")
        root_layer.Save()
        close = getattr(stage, "Close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - logging only
                LOG.debug("stage.Close() failed for %s: %s", normalized_path, exc)
        return

    layer = Sdf.Layer.FindOrOpen(normalized_path)
    if layer is None:
        raise ValueError(f"Failed to open USD layer {normalized_path}")
    layer.upAxis = axis
    layer.Save()


def set_stage_up_axis(
    target_path: PathLike,
    axis: str = "Y",
    *,
    offline: bool = False,
) -> None:
    """Update the up-axis metadata on an existing USD stage/layer."""

    axis_norm = _normalize_up_axis(axis)
    raw_path = str(target_path).strip() if target_path is not None else ""
    if not raw_path:
        raise ValueError("--set-stage-up-axis requires a target USD path.")
    if offline and is_omniverse_path(raw_path):
        raise ValueError("--offline cannot be combined with --set-stage-up-axis for omniverse:// targets.")
    initialize_usd(offline=offline)
    _apply_stage_up_axis(raw_path, axis_norm)


# ---------------- CLI ----------------
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


def _parse_annotation_width_rule_spec(spec: str, *, index: int) -> CurveWidthRule:
    """Parse a --annotation-width-rule specification into a CurveWidthRule."""
    if not spec or "=" not in spec:
        raise ValueError(
            f"--annotation-width-rule #{index}: expected comma-separated key=value pairs; got {spec!r}."
        )
    mapping: dict[str, Any] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(
                f"--annotation-width-rule #{index}: segment {raw_part!r} is missing '='."
            )
        key_norm = key.strip().lower()
        if not key_norm:
            raise ValueError(
                f"--annotation-width-rule #{index}: segment {raw_part!r} has an empty key."
            )
        value_norm = _strip_quotes(value.strip())
        mapping[key_norm] = value_norm
    if "width" not in mapping:
        raise ValueError(f"--annotation-width-rule #{index}: missing required 'width=' entry.")
    return _rule_from_mapping(mapping, context=f"--annotation-width-rule #{index}")


def _load_annotation_width_config(path: PathLike) -> list[CurveWidthRule]:
    """Load annotation width rules from a JSON or YAML configuration file."""
    source = str(path)
    try:
        text = read_text(path)
    except Exception as exc:
        raise ValueError(f"Failed to read annotation width config {source}: {exc}") from exc
    suffix = (path_suffix(path) or "").lower()
    payload: Any | None = None
    errors: list[str] = []
    parse_order: tuple[str, ...]
    if suffix in {".yaml", ".yml"}:
        parse_order = ("yaml", "json")
    elif suffix == ".json":
        parse_order = ("json", "yaml")
    else:
        parse_order = ("json", "yaml")
    for fmt in parse_order:
        if fmt == "json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"JSON: {exc}")
                payload = None
            else:
                break
        elif fmt == "yaml":
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                errors.append(f"YAML: PyYAML not installed ({exc}).")
                continue
            try:
                payload = yaml.safe_load(text)
            except Exception as exc:
                errors.append(f"YAML: {exc}")
                payload = None
            else:
                break
    if payload is None:
        detail = "; ".join(errors) if errors else "no parser succeeded"
        raise ValueError(f"Failed to parse annotation width config {source}: {detail}.")
    return _rules_from_config_payload(payload, source=source)


def _gather_annotation_width_rules(args: argparse.Namespace) -> tuple[CurveWidthRule, ...]:
    """Return ordered curve-width rules based on CLI defaults, configs, and overrides."""
    rules: list[CurveWidthRule] = []
    default_spec = getattr(args, "annotation_width_default", None)
    if default_spec:
        rules.append(
            _rule_from_mapping(
                {"width": default_spec},
                context="--annotation-width-default",
            )
        )
    config_paths = tuple(getattr(args, "annotation_width_configs", None) or ())
    for config_path in config_paths:
        rules.extend(_load_annotation_width_config(config_path))
    inline_specs = tuple(getattr(args, "annotation_width_rules", None) or ())
    for index, spec in enumerate(inline_specs, start=1):
        rules.append(_parse_annotation_width_rule_spec(spec, index=index))
    return tuple(rules)
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
    cancel_event: Any | None = None,
    anchor_mode: Optional[str] = None,
) -> list[ConversionResult]:
    """Programmatic API for running the converter inside another application."""
    log = logger or LOG
    _ensure_not_cancelled(cancel_event)
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
    normalized_anchor_mode = _normalize_anchor_mode(anchor_mode)
    run_options = replace(opts, manifest=manifest_obj)
    if normalized_anchor_mode is not None and getattr(run_options, "anchor_mode", None) != normalized_anchor_mode:
        run_options = replace(run_options, anchor_mode=normalized_anchor_mode)
    targets = _collect_ifc_paths(
        input_path,
        ifc_names=tuple(ifc_names) if ifc_names else None,
        process_all=process_all,
        exclude=exclude,
        log=log,
    )
    _ensure_not_cancelled(cancel_event)
    if not targets:
        log.info("Connected to %s but found no IFC files to convert.", input_path)
        return []
    log.info("Discovered %d IFC file(s) under %s", len(targets), input_path)
    results: list[ConversionResult] = []
    for target in targets:
        _ensure_not_cancelled(cancel_event)
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
                    fallback_shared_site_base_point=DEFAULT_SHARED_BASE_POINT,
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
                cancel_event=cancel_event,
            )
        except ConversionCancelledError:
            raise
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
    """Strips and normalises an IFC file name, appending '.ifc' if necessary.

    Args:
        name: The IFC file name to normalise.

    Returns:
        A trimmed and normalised IFC file name (with '.ifc' appended if necessary).
    """
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
    cancel_event: Any | None = None,
    usd_format: str = DEFAULT_USD_FORMAT,
    usd_auto_binary_threshold_mb: float | None = DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB,
    default_base_point: BasePointConfig = DEFAULT_BASE_POINT,
    default_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    default_master_stage: str = DEFAULT_MASTER_STAGE,
) -> ConversionResult:
    import ifcopenshell  # Local import

    _ensure_not_cancelled(cancel_event)
    base_name = path_stem(ifc_path)
    revision_note = plan.revision if plan and plan.revision else None
    checkpoint_note: Optional[str] = None
    checkpoint_tags: list[str] = []
    if checkpoint:
        checkpoint_note, checkpoint_tags = _make_checkpoint_metadata(revision_note, base_name)
    def _execute(local_ifc: Path) -> ConversionResult:
        _ensure_not_cancelled(cancel_event)
        ifc = ifcopenshell.open(local_ifc.as_posix())
        _ensure_not_cancelled(cancel_event)
        logger.info("Building prototypes for %s...", path_name(ifc_path))
        caches = build_prototypes(ifc, options)
        logger.info(
            "Prototype build complete: %d type prototypes, %d hashed prototypes, %d instances",
            len(caches.repmaps),
            len(caches.hashes),
            len(caches.instances),
        )
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
        _ensure_not_cancelled(cancel_event)
        mat_layer, material_library = author_material_layer(
            stage, caches, proto_paths, layer_path=layout.materials, base_name=base_name, proto_layer=proto_layer, options=options,
        )
        _ensure_not_cancelled(cancel_event)
        inst_layer = author_instance_layer(
            stage,
            caches,
            proto_paths,
            material_library,
            layer_path=layout.instances,
            base_name=base_name,
            options=options,
        )
        _ensure_not_cancelled(cancel_event)
        geometry2d_layer = author_geometry2d_layer(stage, caches, layout.geometry2d, base_name, options)
        _ensure_not_cancelled(cancel_event)
        persist_instance_cache(layout.cache_dir, base_name, caches, proto_paths)
        _ensure_not_cancelled(cancel_event)
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
        lonlat_override = plan.lonlat if plan else None
        _ensure_not_cancelled(cancel_event)
        shared_site_point = (
            plan.shared_site_base_point if plan and getattr(plan, "shared_site_base_point", None) else DEFAULT_SHARED_BASE_POINT
        )
        anchor_mode = _normalize_anchor_mode(getattr(options, "anchor_mode", None))
        if anchor_mode is not None:
            apply_stage_anchor_transform(
                stage,
                caches,
                base_point=effective_base_point,
                shared_site_base_point=shared_site_point,
                anchor_mode=anchor_mode,
                projected_crs=projected_crs,
                align_axes_to_map=True,
                lonlat=lonlat_override,
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
            _ensure_not_cancelled(cancel_event)
        if checkpoint and checkpoint_note is not None:
            _checkpoint_path(layout.stage, checkpoint_note, checkpoint_tags, logger, f"{base_name} stage")
            _checkpoint_path(layout.prototypes, checkpoint_note, checkpoint_tags, logger, f"{base_name} prototypes layer")
            _checkpoint_path(layout.materials, checkpoint_note, checkpoint_tags, logger, f"{base_name} materials layer")
            _checkpoint_path(layout.instances, checkpoint_note, checkpoint_tags, logger, f"{base_name} instances layer")
            if geometry2d_layer:
                _checkpoint_path(layout.geometry2d, checkpoint_note, checkpoint_tags, logger, f"{base_name} 2D geometry layer")
        logger.info("Wrote stage %s", str(layout.stage))
        master_stage_path = None
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
        _ensure_not_cancelled(cancel_event)
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
            _ensure_not_cancelled(cancel_event)
            return _execute(tmp_path)
    local_ifc = ifc_path if isinstance(ifc_path, Path) else Path(ifc_path)
    _ensure_not_cancelled(cancel_event)
    return _execute(local_ifc)


# ------------- entrypoint -------------
def main(argv: Sequence[str] | None = None) -> list[ConversionResult]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if getattr(args, "set_stage_unit", None):
        try:
            set_stage_unit(
                args.set_stage_unit,
                meters_per_unit=getattr(args, "stage_unit_value", 1.0),
                offline=bool(getattr(args, "offline", False)),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            shutdown_usd_context()
            raise SystemExit(2) from exc
        shutdown_usd_context()
        return []
    if getattr(args, "set_stage_up_axis", None):
        try:
            set_stage_up_axis(
                args.set_stage_up_axis,
                axis=getattr(args, "stage_up_axis", "Y"),
                offline=bool(getattr(args, "offline", False)),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            shutdown_usd_context()
            raise SystemExit(2) from exc
        shutdown_usd_context()
        return []
    try:
        width_rules = _gather_annotation_width_rules(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        shutdown_usd_context()
        raise SystemExit(2) from exc
    if width_rules:
        LOG.info("Loaded %d annotation width rule(s).", len(width_rules))
        for idx, rule in enumerate(width_rules, start=1):
            LOG.debug("Annotation width rule %d: %s", idx, rule)
    cli_anchor_mode = _normalize_anchor_mode(getattr(args, "anchor_mode", None))
    options_override = OPTIONS
    if width_rules:
        options_override = replace(options_override, curve_width_rules=width_rules)
    if cli_anchor_mode is not None and getattr(options_override, "anchor_mode", None) != cli_anchor_mode:
        options_override = replace(options_override, anchor_mode=cli_anchor_mode)
    if getattr(args, "enable_material_classification", False):
        LOG.info("Material classification enabled: running component-based style reconciliation.")
        options_override = replace(options_override, enable_material_classification=True)
    if getattr(args, "detail_mode", False):
        LOG.info("Detail mode enabled: forwarding to OCC high-detail conversion path.")
        options_override = replace(options_override, enable_high_detail_remesh=True)
    detail_scope_arg = getattr(args, "detail_scope", None)
    if detail_scope_arg:
        LOG.info("Detail scope override: %s", detail_scope_arg)
        options_override = replace(options_override, detail_scope=detail_scope_arg)
    detail_level_arg = getattr(args, "detail_level", None)
    if detail_level_arg:
        LOG.info("Detail level override: %s", detail_level_arg)
        options_override = replace(options_override, detail_level=detail_level_arg)
    detail_ids_arg = getattr(args, "detail_object_ids", None)
    if detail_ids_arg:
        LOG.info("Detail object ids override: %s", detail_ids_arg)
        options_override = replace(options_override, detail_object_ids=tuple(int(i) for i in detail_ids_arg))
    detail_guids_arg = getattr(args, "detail_object_guids", None)
    if detail_guids_arg:
        LOG.info("Detail object GUIDs override: %s", detail_guids_arg)
        options_override = replace(
            options_override,
            detail_object_guids=tuple(str(g) for g in detail_guids_arg),
        )
    try:
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
            anchor_mode=cli_anchor_mode,
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
    print(
        "\nUse `python -m buildusd.federate --stage-root <output_dir> --manifest <manifest>` "
        "(or the legacy `python -m ifc_converter.federate`) to assemble or refresh "
        "federated master stages without rerunning conversion."
    )
