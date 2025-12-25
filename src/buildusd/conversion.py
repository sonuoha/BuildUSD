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
import math
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence, Union

from .cli import parse_args as _cli_parse_args
from .config.manifest import (
    BasePointConfig,
    ConversionManifest,
    MasterConfig,
    ResolvedFilePlan,
)
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
from .occ_detail_bootstrap import bootstrap_occ
from .process_ifc import (
    ConversionOptions,
    CurveWidthRule,
    MapConversionData,
    build_prototypes,
    extract_map_conversion,
)
from .resolve_frame import _ifc_length_to_meters, _placement_location_xyz
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
from .geospatial import resolve_geospatial_mode, maybe_stream_omnigeospatial
from .federation_contract import decide_offsets
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
    detail_mode=False,
    enable_semantic_subcomponents=False,
)
USD_FORMAT_CHOICES = ("usdc", "usda", "usd", "auto")
DEFAULT_USD_FORMAT = "usdc"
DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB = 50.0
_FALLBACK_MASTER_STAGE = "Federated Model.usda"
_FALLBACK_GEODETIC_CRS = "EPSG:4326"
_FALLBACK_BASE_POINT = BasePointConfig(
    easting=318197.2518,  # 333800.4900
    northing=5815160.4723,  # 5809101.4680,
    height=0.0,
    unit="m",
    epsg="EPSG:7855",
)

# Basic length unit factors to metres for base-point offsets
_UNIT_FACTORS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "dm": 0.1,
    "decimeter": 0.1,
    "decimeters": 0.1,
}

_PROJECTED_XY_THRESHOLD_M = 10000.0
_MAP_ORIGIN_NONZERO_THRESHOLD_M = 1.0
_ROTATION_IDENTITY_EPS = 1.0e-6
_SCALE_MATCH_EPS = 1.0e-6
_DOUBLE_GEOREF_WARN_THRESHOLD_M = 5000.0


def _bp_offset_vector(bp: BasePointConfig) -> tuple[float, float, float]:
    """Return (easting, northing, height) in metres from a BasePointConfig."""
    if bp is None:
        return (0.0, 0.0, 0.0)
    unit = (bp.unit or "m").strip().lower()
    factor = _UNIT_FACTORS.get(unit, 1.0)
    try:
        return (
            float(bp.easting) * factor,
            float(bp.northing) * factor,
            float(bp.height) * factor,
        )
    except Exception:
        return (0.0, 0.0, 0.0)


def _ifc_projected_crs_present(ifc) -> bool:
    try:
        return bool(ifc.by_type("IfcProjectedCRS"))
    except Exception:
        return False


def _ifc_projected_crs_label(ifc) -> Optional[str]:
    try:
        entries = ifc.by_type("IfcProjectedCRS") or []
    except Exception:
        return None
    for crs in entries:
        parts = []
        for key in ("Name", "Description", "GeodeticDatum", "MapProjection"):
            try:
                value = getattr(crs, key, None)
            except Exception:
                value = None
            if value:
                parts.append(str(value))
        if parts:
            return " | ".join(parts)
    return None


def _ifc_site_placement_m(
    ifc, unit_scale_m: float
) -> Optional[tuple[float, float, float]]:
    raw = _ifc_site_placement_raw(ifc)
    if raw is None:
        return None
    return (
        float(raw[0]) * unit_scale_m,
        float(raw[1]) * unit_scale_m,
        float(raw[2]) * unit_scale_m,
    )


def _ifc_site_placement_raw(
    ifc,
) -> Optional[tuple[float, float, float]]:
    try:
        sites = ifc.by_type("IfcSite") or []
    except Exception:
        return None
    for site in sites:
        place = getattr(site, "ObjectPlacement", None)
        xyz = _placement_location_xyz(place)
        if xyz is None:
            continue
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return None


def resolve_project_base_point_m(ifc) -> Optional[tuple[float, float, float]]:
    # TODO: wire to exporter-specific metadata (IfcPropertySet, etc.)
    _ = ifc
    return None


def resolve_survey_point_m(ifc) -> Optional[tuple[float, float, float]]:
    # TODO: wire to exporter-specific metadata (IfcPropertySet, etc.)
    _ = ifc
    return None


def _classify_authored_space(
    ifc,
    *,
    unit_scale_m: float,
    site_m: Optional[tuple[float, float, float]],
    global_threshold_m: float = _PROJECTED_XY_THRESHOLD_M,
    sample_limit: int = 200,
) -> str:
    if site_m is not None:
        max_abs = max(abs(site_m[0]), abs(site_m[1]), abs(site_m[2]))
        if max_abs >= global_threshold_m:
            return "global"
    max_abs = 0.0
    count = 0
    for typename in ("IfcProduct", "IfcElement", "IfcBuildingElement"):
        try:
            items = ifc.by_type(typename) or []
        except Exception:
            continue
        if not items:
            continue
        for product in items:
            if count >= sample_limit:
                break
            place = getattr(product, "ObjectPlacement", None)
            xyz = _placement_location_xyz(place)
            if xyz is None:
                continue
            max_abs = max(
                max_abs,
                abs(float(xyz[0]) * unit_scale_m),
                abs(float(xyz[1]) * unit_scale_m),
                abs(float(xyz[2]) * unit_scale_m),
            )
            count += 1
        break
    return "global" if max_abs >= global_threshold_m else "local"


def _classify_georef_strategy(
    *,
    site_m: Optional[tuple[float, float, float]],
    site_raw: Optional[tuple[float, float, float]],
    map_conv: Optional[MapConversionData],
    unit_scale_m: float,
) -> tuple[str, dict]:
    if unit_scale_m <= 0:
        unit_scale_m = 1.0
    notes: list[str] = []
    strategy = "BAKED_PROJECTED"
    info: dict[str, object] = {}

    site_xy_mag = None
    if site_m is not None:
        site_xy_mag = max(abs(site_m[0]), abs(site_m[1]))
    site_is_projected = (
        site_xy_mag is not None and site_xy_mag > _PROJECTED_XY_THRESHOLD_M
    )
    site_is_local = site_xy_mag is not None and site_xy_mag <= _PROJECTED_XY_THRESHOLD_M

    map_origin_mag = None
    map_origin_projected = False
    map_origin_nonzero = False
    map_rot_identity = False
    map_scale_identity = False
    map_identity_like = False
    rotation_deg = None
    map_scale = None
    consistency_distance = None

    if map_conv is not None:
        map_scale = float(getattr(map_conv, "scale", 1.0) or 1.0)
        eastings = float(getattr(map_conv, "eastings", 0.0) or 0.0)
        northings = float(getattr(map_conv, "northings", 0.0) or 0.0)
        map_origin_mag = max(abs(eastings), abs(northings))
        map_origin_projected = map_origin_mag > _PROJECTED_XY_THRESHOLD_M
        map_origin_nonzero = map_origin_mag > _MAP_ORIGIN_NONZERO_THRESHOLD_M
        ax, ay = map_conv.normalized_axes()
        rotation_deg = map_conv.rotation_degrees()
        map_rot_identity = (
            abs(ax - 1.0) <= _ROTATION_IDENTITY_EPS
            and abs(ay) <= _ROTATION_IDENTITY_EPS
        )
        map_scale_identity = (
            abs(map_scale - 1.0) <= _SCALE_MATCH_EPS
            or abs(map_scale - unit_scale_m) <= _SCALE_MATCH_EPS
        )
        map_identity_like = (
            not map_origin_nonzero and map_rot_identity and map_scale_identity
        )

        if site_is_projected and map_origin_projected and site_m is not None:
            if site_raw is None and unit_scale_m > 0:
                site_raw = (
                    site_m[0] / unit_scale_m,
                    site_m[1] / unit_scale_m,
                    site_m[2] / unit_scale_m,
                )
            if site_raw is not None and unit_scale_m > 0:
                scale_eff = map_scale / unit_scale_m
                pred_x = eastings + scale_eff * (ax * site_raw[0] - ay * site_raw[1])
                pred_y = northings + scale_eff * (ay * site_raw[0] + ax * site_raw[1])
                consistency_distance = math.hypot(
                    pred_x - site_m[0], pred_y - site_m[1]
                )
                if consistency_distance > _DOUBLE_GEOREF_WARN_THRESHOLD_M:
                    notes.append("double_georef_suspect")

    if map_conv is None:
        notes.append("mapconv_missing")
        strategy = "BAKED_PROJECTED"
    elif site_is_local and map_origin_projected:
        strategy = "MAPCONV_ANCHORED"
    else:
        strategy = "BAKED_PROJECTED"

    if map_identity_like:
        notes.append("mapconv_identity_like")
        strategy = "BAKED_PROJECTED"

    if site_is_projected and map_origin_projected:
        notes.append("both_projected_scale")
        strategy = "BAKED_PROJECTED"

    info.update(
        {
            "site_xy_mag_m": site_xy_mag,
            "map_origin_mag_m": map_origin_mag,
            "map_rotation_deg": rotation_deg,
            "map_scale": map_scale,
            "site_is_projected": site_is_projected,
            "map_origin_projected": map_origin_projected,
            "map_identity_like": map_identity_like,
            "consistency_distance_m": consistency_distance,
            "notes": notes,
        }
    )
    return strategy, info


def _mapconv_scale_factor(map_conv: MapConversionData, unit_scale_m: float) -> float:
    scale = float(getattr(map_conv, "scale", 1.0) or 1.0)
    if unit_scale_m <= 0:
        unit_scale_m = 1.0
    return scale / unit_scale_m


def _mapconv_world_to_local_m(
    map_conv: MapConversionData,
    world_xyz_m: tuple[float, float, float],
    *,
    unit_scale_m: float,
) -> Optional[tuple[float, float, float]]:
    scale_eff = _mapconv_scale_factor(map_conv, unit_scale_m)
    if abs(scale_eff) <= 1.0e-12:
        return None
    ax, ay = map_conv.normalized_axes()
    d_e = float(world_xyz_m[0]) - float(getattr(map_conv, "eastings", 0.0) or 0.0)
    d_n = float(world_xyz_m[1]) - float(getattr(map_conv, "northings", 0.0) or 0.0)
    d_z = float(world_xyz_m[2]) - float(
        getattr(map_conv, "orthogonal_height", 0.0) or 0.0
    )
    x_m = (ax * d_e + ay * d_n) / scale_eff
    y_m = (-ay * d_e + ax * d_n) / scale_eff
    z_m = d_z / scale_eff
    return (x_m, y_m, z_m)


def _mapconv_local_to_world_m(
    map_conv: MapConversionData,
    local_xyz_m: tuple[float, float, float],
    *,
    unit_scale_m: float,
) -> Optional[tuple[float, float, float]]:
    scale_eff = _mapconv_scale_factor(map_conv, unit_scale_m)
    if abs(scale_eff) <= 1.0e-12:
        return None
    ax, ay = map_conv.normalized_axes()
    x_m, y_m, z_m = (
        float(local_xyz_m[0]),
        float(local_xyz_m[1]),
        float(local_xyz_m[2]),
    )
    easting = float(getattr(map_conv, "eastings", 0.0) or 0.0) + scale_eff * (
        ax * x_m - ay * y_m
    )
    northing = float(getattr(map_conv, "northings", 0.0) or 0.0) + scale_eff * (
        ay * x_m + ax * y_m
    )
    height = float(getattr(map_conv, "orthogonal_height", 0.0) or 0.0) + scale_eff * (
        z_m
    )
    return (easting, northing, height)


def _base_point_from_xyz_m(
    xyz_m: tuple[float, float, float], projected_crs: Optional[str]
) -> BasePointConfig:
    return BasePointConfig(
        easting=float(xyz_m[0]),
        northing=float(xyz_m[1]),
        height=float(xyz_m[2]),
        unit="m",
        epsg=projected_crs or "",
    )


def _stamp_federation_debug(
    stage,
    *,
    anchor_mode: Optional[str],
    pbp_m,
    shared_site_m,
    model_offset_m,
    georef_origin: Optional[str],
) -> None:
    """Stamp small, crate-safe debug metadata for traceability."""
    if stage is None:
        return
    try:
        world = stage.GetPrimAtPath("/World")
        if not world:
            return
        if anchor_mode:
            world.SetCustomDataByKey("ifc:federation:anchorMode", str(anchor_mode))
        if georef_origin:
            world.SetCustomDataByKey("ifc:federation:georefOrigin", str(georef_origin))
        if pbp_m:
            world.SetCustomDataByKey(
                "ifc:federation:pbpMeters",
                {"x": pbp_m[0], "y": pbp_m[1], "z": pbp_m[2]},
            )
        if shared_site_m:
            world.SetCustomDataByKey(
                "ifc:federation:sharedSiteMeters",
                {"x": shared_site_m[0], "y": shared_site_m[1], "z": shared_site_m[2]},
            )
        if model_offset_m:
            world.SetCustomDataByKey(
                "ifc:federation:modelOffsetMeters",
                {
                    "x": model_offset_m[0],
                    "y": model_offset_m[1],
                    "z": model_offset_m[2],
                },
            )
    except Exception:
        pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Delegate to the CLI parser with project defaults."""

    return _cli_parse_args(
        argv,
        default_input_root=DEFAULT_INPUT_ROOT,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        usd_format_choices=USD_FORMAT_CHOICES,
        default_usd_format=DEFAULT_USD_FORMAT,
        default_usd_auto_binary_threshold_mb=DEFAULT_USD_AUTO_BINARY_THRESHOLD_MB,
        default_geospatial_mode="auto",
    )


def _detail_features_requested(args: argparse.Namespace) -> bool:
    """Return True when any CLI knob requires OCC detail meshing."""

    if args is None:
        return False
    return any(
        (
            getattr(args, "detail_mode", False),
            bool(getattr(args, "detail_objects", None)),
        )
    )


def _load_manifest_backed_defaults() -> (
    tuple[BasePointConfig, str, str, BasePointConfig]
):
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
            LOG.debug(
                "Failed to load manifest defaults from %s: %s", manifest_path, exc
            )
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
            master_candidate = MasterConfig(
                id="__defaults__", name=defaults.master_name
            )
        if master_candidate is not None:
            master_stage = master_candidate.resolved_name()

    return base_point, geodetic_crs, master_stage, shared_site


(
    DEFAULT_BASE_POINT,
    DEFAULT_GEODETIC_CRS,
    DEFAULT_MASTER_STAGE,
    DEFAULT_SHARED_BASE_POINT,
) = _load_manifest_backed_defaults()
PathLike = Union[str, Path]


def _normalize_anchor_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    alias_map: dict[str, Optional[str]] = {
        "local": "local",
        "basepoint": "basepoint",
        "site": "basepoint",
        "shared_site": "basepoint",
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


def _ensure_not_cancelled(
    cancel_event: Any | None, *, message: str | None = None
) -> None:
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
    geospatial_mode: str
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
            "geospatial_mode": self.geospatial_mode,
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


def set_stage_unit(
    target_path: PathLike, meters_per_unit: float = 1.0, *, offline: bool = False
) -> None:
    """Update the meters-per-unit metadata on an existing USD layer or stage."""
    raw_path = str(target_path).strip() if target_path is not None else ""
    if not raw_path:
        raise ValueError("--set-stage-unit requires a target USD path.")
    if meters_per_unit <= 0.0:
        raise ValueError("--stage-unit-value must be greater than zero.")
    if offline and is_omniverse_path(raw_path):
        raise ValueError(
            "--offline cannot be combined with --set-stage-unit for omniverse:// targets."
        )
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
        raise ValueError(
            "--offline cannot be combined with --set-stage-up-axis for omniverse:// targets."
        )
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
        raise ValueError(
            f"--annotation-width-rule #{index}: missing required 'width=' entry."
        )
    return _rule_from_mapping(mapping, context=f"--annotation-width-rule #{index}")


def _load_annotation_width_config(path: PathLike) -> list[CurveWidthRule]:
    """Load annotation width rules from a JSON or YAML configuration file."""
    source = str(path)
    try:
        text = read_text(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read annotation width config {source}: {exc}"
        ) from exc
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


def _gather_annotation_width_rules(
    args: argparse.Namespace,
) -> tuple[CurveWidthRule, ...]:
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
    geospatial_mode: str = "auto",
    cancel_event: Any | None = None,
    anchor_mode: Optional[str] = None,
) -> list[ConversionResult]:
    """Programmatic API for running the converter inside another application."""
    log = logger or LOG
    _ensure_not_cancelled(cancel_event)
    if manifest_path and manifest is not None:
        raise ValueError("Provide either manifest or manifest_path, not both.")
    exclude = normalize_exclusions(exclude_names)
    stage_path_override: Optional[Path] = None
    base_name_override: Optional[str] = None
    output_path_literal = output_dir
    if output_dir is None:
        output_root: PathLike = DEFAULT_OUTPUT_ROOT
    else:
        output_root = output_dir
    if not is_omniverse_path(output_root):
        output_root = Path(output_root).resolve()
        suffix = output_root.suffix.lower()
        if suffix in {".usd", ".usda", ".usdc"} and output_dir is not None:
            stage_path_override = output_root
            base_name_override = output_root.stem or output_root.name
            parent = output_root.parent
            output_root = parent if str(parent) else Path(".").resolve()
            if stage_path_override.exists() and stage_path_override.is_dir():
                raise ValueError(
                    f"Output path {stage_path_override} is an existing directory; "
                    "remove or pick a different file when specifying --output as a USD file."
                )
    check_paths: list[PathLike] = [input_path, output_root]
    if manifest_path:
        check_paths.append(manifest_path)
    if offline:
        if any(is_omniverse_path(p) for p in check_paths):
            raise ValueError(
                "--offline mode requires local input/output/manifest paths"
            )
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
    if (
        normalized_anchor_mode is not None
        and getattr(run_options, "anchor_mode", None) != normalized_anchor_mode
    ):
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
    target_count = len(targets)
    log.info("Discovered %d IFC file(s) under %s", target_count, input_path)
    if stage_path_override is not None:
        stage_override_ext = stage_path_override.suffix.lower().lstrip(".")
        if stage_override_ext:
            if usd_format_normalized == "auto":
                usd_format_normalized = stage_override_ext
            elif stage_override_ext != usd_format_normalized:
                log.warning(
                    "Output file extension .%s overrides --usd-format %s for stage %s",
                    stage_override_ext,
                    usd_format_normalized,
                    stage_path_override,
                )
                usd_format_normalized = stage_override_ext
        if target_count == 1:
            log.info(
                "Output will be authored to explicit stage path %s",
                stage_path_override,
            )
        else:
            log.warning(
                "Output path %s points to a USD file but %d IFC files were discovered. "
                "Each IFC will be exported into %s instead.",
                output_path_literal,
                target_count,
                output_root,
            )
            stage_path_override = None
            base_name_override = None
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
                log.warning(
                    "Manifest resolution failed for %s: %s",
                    path_name(target.source),
                    exc,
                )
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
                base_name_override=base_name_override,
                stage_path_override=stage_path_override,
                geospatial_mode=geospatial_mode,
            )
            base_name_override = None
            stage_path_override = None
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
            log.warning(
                "Conversion for %s produced no result; skipping.", target.source
            )
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
        entries = list_directory(
            uri
        )  # <- io_utils.list_directory(path) returns List[ListEntry]
    except (RuntimeError, FileNotFoundError) as exc:
        # If listing fails but the uri is a file, try single-file fallback
        if _is_ifc_candidate(path_name(uri)):
            log.warning(
                "Directory listing failed (%s). Falling back to single-file target: %s",
                exc,
                uri,
            )
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
            targets.append(
                _IfcTarget(
                    source=candidate, manifest_key=_manifest_key_for_path(candidate)
                )
            )
        log.info("Resolved %d IFC(s) by explicit name under %s", len(targets), uri)
        return sorted(targets, key=lambda t: path_name(t.source).lower())

    # Helper: tolerate different omni.client ListEntry attribute names
    def _entry_to_path(e) -> Optional[str]:
        # io_utils.list_directory returns omni.client entries with properties that vary by build.
        # Try common attributes in order of likelihood.
        for attr in ("relative_path", "relativePath", "path", "uri"):
            p = getattr(e, attr, None)
            if isinstance(p, str) and p.strip():
                return (
                    join_path(uri, p)
                    if attr in ("relative_path", "relativePath")
                    else p
                )
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
            targets.append(
                _IfcTarget(
                    source=candidate, manifest_key=_manifest_key_for_path(candidate)
                )
            )
        except Exception as exc:
            bad += 1
            log.debug(
                "Skipping problematic listing entry due to: %s", exc, exc_info=True
            )

    # Sort for deterministic order
    targets.sort(key=lambda t: path_name(t.source).lower())

    if not targets:
        log.warning(
            "Nucleus listing returned 0 IFC files under %s (skipped=%d). "
            "Check permissions and folder path case.",
            uri,
            bad,
        )
    else:
        log.info(
            "Discovered %d IFC file(s) under %s (skipped=%d)", len(targets), uri, bad
        )
    return targets


def _ensure_ifc_name(name: str) -> str:
    """Strips and normalises an IFC file name, appending '.ifc' if necessary.

    Args:
        name: The IFC file name to normalise.

    Returns:
        A trimmed and normalised IFC file name (with '.ifc' appended if necessary).
    """
    trimmed = (name or "").strip()
    if not trimmed:
        return trimmed
    if not trimmed.lower().endswith(".ifc"):
        trimmed = f"{trimmed}.ifc"
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
    if isinstance(path, Path):
        return path.resolve()
    if is_omniverse_path(path):
        uri = str(path)
        stripped = uri.split("://", 1)[-1]
        return PurePosixPath(stripped)
    return Path(str(path)).resolve()


# ------------- layout + checkpoint -------------
def _build_output_layout(
    output_root: PathLike,
    base_name: str,
    usd_format: str,
    *,
    stage_override: PathLike | None = None,
) -> _OutputLayout:
    # Esnure output directory exists
    ensure_directory(output_root)

    prototypes_dir = join_path(output_root, "prototypes")
    materials_dir = join_path(output_root, "materials")
    instances_dir = join_path(output_root, "instances")
    geometry2d_dir = join_path(output_root, "geometry2d")
    caches_dir = join_path(output_root, "caches")
    for directory in (
        prototypes_dir,
        materials_dir,
        instances_dir,
        geometry2d_dir,
        caches_dir,
    ):
        ensure_directory(directory)
    ext = _normalise_usd_format(usd_format)
    if ext == "auto":
        raise ValueError(
            "Auto USD format must be resolved before building the output layout."
        )
    if stage_override is not None:
        stage_path = Path(stage_override)
        if stage_path.suffix.lower().lstrip(".") != ext:
            raise ValueError(
                f"Output file extension {stage_path.suffix} does not match usd_format {ext}."
            )
    else:
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
        logger.debug(
            "Unable to read subLayerPaths while updating %s layer: %s", label, exc
        )
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
        prototype_mesh_bytes += _estimate_mesh_payload_bytes(
            getattr(proto, "mesh", None)
        )
    for proto in caches.hashes.values():
        prototype_mesh_bytes += _estimate_mesh_payload_bytes(
            getattr(proto, "mesh", None)
        )
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
                logger.debug(
                    "Unable to stat stage layer at %s: %s", stage_path_obj, exc
                )
            else:
                if stage_size > threshold_bytes:
                    target_stage_path = stage_path_obj.with_suffix(".usdc")
                    try:
                        root_layer.Export(
                            _layer_identifier(target_stage_path),
                            args={"format": "usdc"},
                        )
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
                        updated_layout = replace(
                            updated_layout, stage=target_stage_path
                        )
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


def _make_checkpoint_metadata(
    revision: Optional[str], base_name: str
) -> tuple[str, list[str]]:
    note = revision or f"{base_name}"
    tag_src = revision or base_name
    tag_candidates = tag_src.replace(",", " ").split()
    tags = [t.strip() for t in tag_candidates if t.strip()] or [base_name]
    return note, tags


def _checkpoint_path(
    path: PathLike, note: str, tags: Sequence[str], logger: logging.Logger, label: str
) -> None:
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
    base_name_override: Optional[str] = None,
    stage_path_override: PathLike | None = None,
    geospatial_mode: str = "auto",
    offline: bool = False,
) -> ConversionResult:
    import ifcopenshell  # Local import

    _ensure_not_cancelled(cancel_event)
    base_name = base_name_override or path_stem(ifc_path)
    revision_note = plan.revision if plan and plan.revision else None
    checkpoint_note: Optional[str] = None
    checkpoint_tags: list[str] = []
    if checkpoint:
        checkpoint_note, checkpoint_tags = _make_checkpoint_metadata(
            revision_note, base_name
        )

    def _execute(local_ifc: Path) -> ConversionResult:
        _ensure_not_cancelled(cancel_event)
        ifc = ifcopenshell.open(local_ifc.as_posix())
        _ensure_not_cancelled(cancel_event)
        unit_scale_m = _ifc_length_to_meters(ifc)
        if abs(unit_scale_m - 1.0) > 1e-6:
            logger.info("IFC unit scale: %s m/unit", unit_scale_m)
        else:
            logger.debug("IFC unit scale: %s m/unit", unit_scale_m)
        try:
            contexts = ifc.by_type("IfcGeometricRepresentationContext") or []
            for ctx in contexts:
                ops = getattr(ctx, "HasCoordinateOperation", None) or []
                for op in ops:
                    if op is None or not hasattr(op, "is_a"):
                        continue
                    if op.is_a("IfcMapConversion"):
                        logger.info(
                            "IfcMapConversion: Scale=%s Scale2=%s Scale3=%s Eastings=%s Northings=%s Height=%s",
                            getattr(op, "Scale", None),
                            getattr(op, "Scale2", None),
                            getattr(op, "Scale3", None),
                            getattr(op, "Eastings", None),
                            getattr(op, "Northings", None),
                            getattr(op, "OrthogonalHeight", None),
                        )
                    elif op.is_a("IfcRigidOperation"):
                        logger.info(
                            "IfcRigidOperation: First=%s Second=%s Height=%s",
                            getattr(op, "FirstCoordinate", None),
                            getattr(op, "SecondCoordinate", None),
                            getattr(op, "Height", None),
                        )
        except Exception:
            pass

        # Determine anchoring offsets before prototype build.
        #
        # All shifts are handled via IfcOpenShell model-offset (offset-type='negative').
        anchor_mode_norm = _normalize_anchor_mode(getattr(options, "anchor_mode", None))
        effective_base_point = (
            plan.base_point if plan and plan.base_point else default_base_point
        )
        shared_site_point = (
            plan.shared_site_base_point
            if plan and getattr(plan, "shared_site_base_point", None)
            else DEFAULT_SHARED_BASE_POINT
        )
        rotation_quat = (0.0, 0.0, 0.0, 1.0)
        options_for_build = options
        model_offset_applied: tuple[float, float, float] | None = None

        map_conv = extract_map_conversion(ifc)
        projected_crs_label = _ifc_projected_crs_label(ifc)
        has_projected_crs = bool(projected_crs_label) or _ifc_projected_crs_present(ifc)

        ifc_site_raw = _ifc_site_placement_raw(ifc)
        ifc_site_authored_m = (
            (
                float(ifc_site_raw[0]) * unit_scale_m,
                float(ifc_site_raw[1]) * unit_scale_m,
                float(ifc_site_raw[2]) * unit_scale_m,
            )
            if ifc_site_raw is not None
            else None
        )
        geom_space = _classify_authored_space(
            ifc, unit_scale_m=unit_scale_m, site_m=ifc_site_authored_m
        )
        strategy, strategy_info = _classify_georef_strategy(
            site_m=ifc_site_authored_m,
            site_raw=ifc_site_raw,
            map_conv=map_conv,
            unit_scale_m=unit_scale_m,
        )
        map_conv_active = map_conv if strategy == "MAPCONV_ANCHORED" else None
        rotation_applied = False
        rotation_theta_deg = 0.0
        if map_conv_active is not None:
            ax, ay = map_conv_active.normalized_axes()
            if (
                abs(ax - 1.0) > _ROTATION_IDENTITY_EPS
                or abs(ay) > _ROTATION_IDENTITY_EPS
            ):
                rotation_theta_deg = math.degrees(math.atan2(ay, ax))
                half = math.radians(rotation_theta_deg) * 0.5
                rotation_quat = (
                    0.0,
                    0.0,
                    math.sin(half),
                    math.cos(half),
                )
                rotation_applied = True
        options_for_build = replace(options_for_build, model_rotation=rotation_quat)
        strategy_notes = set(strategy_info.get("notes", []) or [])
        if "both_projected_scale" in strategy_notes:
            logger.warning(
                "Double-georef risk for %s: site and map origin are projected-scale; strategy=%s consistency_distance=%s threshold=%s",
                path_name(ifc_path),
                strategy,
                strategy_info.get("consistency_distance_m"),
                _DOUBLE_GEOREF_WARN_THRESHOLD_M,
            )
        if "double_georef_suspect" in strategy_notes:
            logger.warning(
                "Double-georef suspect for %s: mapconv prediction deviates from site by %s m (threshold=%s); strategy=%s",
                path_name(ifc_path),
                strategy_info.get("consistency_distance_m"),
                _DOUBLE_GEOREF_WARN_THRESHOLD_M,
                strategy,
            )
        if "mapconv_identity_like" in strategy_notes:
            logger.info(
                "MapConversion is identity-like for %s; suppressing mapconv anchor.",
                path_name(ifc_path),
            )
        if map_conv_active is not None and not has_projected_crs:
            logger.info(
                "IfcProjectedCRS is missing for %s; using heuristic MapConversion anchoring.",
                path_name(ifc_path),
            )
        if map_conv is not None:
            logger.info(
                "Model rotation %s for %s: strategy=%s xa=%s xo=%s theta_deg=%s quat=%s",
                "applied" if rotation_applied else "suppressed",
                path_name(ifc_path),
                strategy,
                float(getattr(map_conv, "x_axis_abscissa", 1.0) or 1.0),
                float(getattr(map_conv, "x_axis_ordinate", 0.0) or 0.0),
                rotation_theta_deg,
                rotation_quat,
            )

        ifc_site_world_m: tuple[float, float, float] | None = None
        if ifc_site_authored_m is not None:
            if geom_space == "global":
                ifc_site_world_m = ifc_site_authored_m
            elif map_conv_active is not None:
                ifc_site_world_m = _mapconv_local_to_world_m(
                    map_conv_active, ifc_site_authored_m, unit_scale_m=unit_scale_m
                )

        pbp_ifc_m = resolve_project_base_point_m(ifc)
        sp_ifc_m = resolve_survey_point_m(ifc)

        pbp_world_m = pbp_ifc_m or (
            _bp_offset_vector(effective_base_point)
            if effective_base_point is not None
            else None
        )
        shared_site_world_m = sp_ifc_m or (
            _bp_offset_vector(shared_site_point)
            if shared_site_point is not None
            else None
        )

        if anchor_mode_norm == "local" and ifc_site_authored_m is None:
            logger.warning(
                "Anchor mode local requested but IfcSite placement was not found; modelOffset will be (0,0,0)."
            )
        if (
            anchor_mode_norm == "basepoint"
            and pbp_world_m is None
            and shared_site_world_m is None
        ):
            logger.warning(
                "Anchor mode basepoint requested but no PBP/SP was resolved; modelOffset will be (0,0,0)."
            )
        if (
            anchor_mode_norm == "basepoint"
            and geom_space == "local"
            and map_conv_active is None
            and (pbp_world_m is not None or shared_site_world_m is not None)
        ):
            logger.warning(
                "Anchor mode basepoint requires IfcMapConversion for local geometry; modelOffset will be (0,0,0)."
            )

        pbp_offset_m: tuple[float, float, float] | None = None
        if pbp_world_m is not None:
            if geom_space == "global":
                pbp_offset_m = pbp_world_m
            elif map_conv_active is not None:
                pbp_offset_m = _mapconv_world_to_local_m(
                    map_conv_active, pbp_world_m, unit_scale_m=unit_scale_m
                )

        shared_site_offset_m: tuple[float, float, float] | None = None
        if shared_site_world_m is not None:
            if geom_space == "global":
                shared_site_offset_m = shared_site_world_m
            elif map_conv_active is not None:
                shared_site_offset_m = _mapconv_world_to_local_m(
                    map_conv_active, shared_site_world_m, unit_scale_m=unit_scale_m
                )

        decision = decide_offsets(
            anchor_mode=anchor_mode_norm,
            ifc_site_m=ifc_site_authored_m,
            pbp_m=pbp_offset_m,
            shared_site_m=shared_site_offset_m,
        )

        if anchor_mode_norm is not None:
            logger.info(
                "GeoAnchor %s unit_to_m=%s site_m=%s map_origin=%s rot_deg=%s scale=%s strategy=%s anchor_mode=%s modelOffset=%s",
                path_name(ifc_path),
                unit_scale_m,
                ifc_site_authored_m,
                (
                    (
                        float(getattr(map_conv, "eastings", 0.0) or 0.0),
                        float(getattr(map_conv, "northings", 0.0) or 0.0),
                        float(getattr(map_conv, "orthogonal_height", 0.0) or 0.0),
                    )
                    if map_conv is not None
                    else None
                ),
                (map_conv.rotation_degrees() if map_conv is not None else None),
                float(getattr(map_conv, "scale", 1.0) or 1.0)
                if map_conv is not None
                else None,
                strategy,
                anchor_mode_norm,
                decision.model_offset_m,
            )
            logger.debug(
                "GeoAnchor details: crs=%s has_projected_crs=%s geom_space=%s map_conv_active=%s info=%s",
                projected_crs_label,
                has_projected_crs,
                geom_space,
                "yes" if map_conv_active is not None else "no",
                strategy_info,
            )

        if decision.model_offset_m is not None and decision.anchor_mode is not None:
            options_for_build = replace(
                options_for_build,
                model_offset=decision.model_offset_m,
                model_offset_type="negative",
                model_rotation=rotation_quat,
            )
            model_offset_applied = decision.model_offset_m

        logger.info(
            "Anchor mode=%s -> modelOffset=%s georef_origin=%s",
            decision.anchor_mode,
            tuple(round(v, 6) for v in decision.model_offset_m)
            if decision.model_offset_m
            else None,
            decision.georef_origin,
        )

        logger.info("Building prototypes for %s...", path_name(ifc_path))
        caches = build_prototypes(ifc, options_for_build, ifc_path=str(ifc_path))
        logger.info(
            "Prototype build complete: %d type prototypes, %d hashed prototypes, %d instances",
            len(caches.repmaps),
            len(caches.hashes),
            len(caches.instances),
        )
        logger.info(
            "IFC %s → %d type prototypes, %d hashed prototypes, %d instances",
            path_name(ifc_path),
            len(caches.repmaps),
            len(caches.hashes),
            len(caches.instances),
        )
        authoring_format = _normalise_usd_format(usd_format)
        if authoring_format == "auto":
            authoring_format, _ = _select_auto_usd_format(
                caches,
                threshold_mb=usd_auto_binary_threshold_mb,
                base_name=base_name,
                logger=logger,
            )
        layout = _build_output_layout(
            output_root,
            base_name,
            authoring_format,
            stage_override=stage_path_override,
        )
        resolved_geospatial_mode = resolve_geospatial_mode(
            geospatial_mode,
            offline=offline,
            output_path=layout.stage,
        )

        # Create stage for file in process.
        stage = create_usd_stage(layout.stage)
        proto_layer, proto_paths = author_prototype_layer(
            stage, caches, layout.prototypes, base_name, options
        )
        _ensure_not_cancelled(cancel_event)
        mat_layer, material_library = author_material_layer(
            stage,
            caches,
            proto_paths,
            layer_path=layout.materials,
            base_name=base_name,
            proto_layer=proto_layer,
            options=options,
        )

        _ensure_not_cancelled(
            cancel_event
        )  # Ensure conversion process is not cancelled for the file

        # Author instance layer
        inst_layer = author_instance_layer(
            stage,
            caches,
            proto_paths,
            material_library,
            layer_path=layout.instances,
            base_name=base_name,
            options=options,
        )

        _ensure_not_cancelled(
            cancel_event
        )  # Ensure conversion process is not cancelled for the file

        # Author 2D Geometries
        geometry2d_layer = author_geometry2d_layer(
            stage, caches, layout.geometry2d, base_name, options
        )
        _ensure_not_cancelled(cancel_event)

        # Author instance layer
        persist_instance_cache(
            layout.cache_dir, base_name, caches, proto_paths
        )  # Save instnace cache to file

        _ensure_not_cancelled(
            cancel_event
        )  # Ensure conversion process is not cancelled for the file

        # Resolve Coordinate Systems and Basepoints from manifests and overrides.
        effective_base_point = (
            plan.base_point if plan and plan.base_point else default_base_point
        )

        if effective_base_point is None:
            raise ValueError(
                f"No base point configured for {path_name(ifc_path)}; provide a manifest entry or update defaults."
            )
        projected_crs = (
            plan.projected_crs if plan and plan.projected_crs else coordinate_system
        )
        if not projected_crs:
            raise ValueError(
                f"No projected CRS available for {path_name(ifc_path)}; supply --map-coordinate-system or manifest override."
            )
        geodetic_crs = (
            plan.geodetic_crs if plan and plan.geodetic_crs else default_geodetic_crs
        )
        lonlat_override = plan.lonlat if plan else None

        _ensure_not_cancelled(cancel_event)
        shared_site_point = (
            plan.shared_site_base_point
            if plan and getattr(plan, "shared_site_base_point", None)
            else DEFAULT_SHARED_BASE_POINT
        )
        anchor_mode = decision.anchor_mode
        pbp_base_point = (
            _base_point_from_xyz_m(pbp_ifc_m, projected_crs)
            if pbp_ifc_m is not None
            else effective_base_point
        )
        shared_site_base_point = (
            _base_point_from_xyz_m(sp_ifc_m, projected_crs)
            if sp_ifc_m is not None
            else shared_site_point
        )
        georef_base_point: BasePointConfig | None = None
        if decision.georef_origin == "ifc_site" and ifc_site_world_m is not None:
            georef_base_point = _base_point_from_xyz_m(ifc_site_world_m, projected_crs)
        elif decision.georef_origin == "pbp":
            georef_base_point = pbp_base_point
        elif decision.georef_origin == "shared_site":
            georef_base_point = shared_site_base_point

        if anchor_mode is not None and georef_base_point is not None:
            apply_stage_anchor_transform(
                stage,
                caches,
                base_point=georef_base_point,
                pbp_base_point=pbp_base_point,
                shared_site_base_point=shared_site_base_point,
                anchor_mode=anchor_mode,
                projected_crs=projected_crs,
                align_axes_to_map=True,
                lonlat=lonlat_override,
            )
        logger.info(
            "Assigning world geolocation using %s -> %s", projected_crs, geodetic_crs
        )
        base_point_for_geo = georef_base_point or (
            pbp_base_point if lonlat_override is not None else None
        )
        if base_point_for_geo is not None or lonlat_override is not None:
            # If anchor_mode is none but lonlat_override exists, we still stamp geospatial metadata
            # without claiming any projected origin. (Useful for "already correct" exports.)
            geodetic_result = assign_world_geolocation(
                stage,
                base_point=base_point_for_geo or pbp_base_point,
                projected_crs=projected_crs,
                geodetic_crs=geodetic_crs,
                unit_hint=(georef_base_point.unit if georef_base_point else None),
                lonlat_override=lonlat_override,
                geospatial_mode=resolved_geospatial_mode,
                offline=offline,
                model_offset=model_offset_applied,
                model_offset_type="negative" if model_offset_applied else None,
                anchor_mode=anchor_mode,
            )
        else:
            geodetic_result = None
            logger.info(
                "No geospatial base point and no lonlat_override provided; skipping geospatial stamping."
            )

        # Stamp contract/debug metadata for later inspection in USD
        _stamp_federation_debug(
            stage,
            anchor_mode=decision.anchor_mode,
            pbp_m=pbp_world_m,
            shared_site_m=shared_site_world_m,
            model_offset_m=decision.model_offset_m,
            georef_origin=decision.georef_origin,
        )

        if resolved_geospatial_mode == "omni":
            try:
                maybe_stream_omnigeospatial(
                    stage,
                    resolved_geospatial_mode,
                    projected_crs=projected_crs,
                    geodetic_crs=geodetic_crs,
                    model_offset=model_offset_applied,
                    model_offset_type="negative" if model_offset_applied else None,
                )
            except Exception as exc:
                logger.debug("Skipping OmniGeospatial prep: %s", exc)
        stage.Save()
        proto_layer.Save()
        mat_layer.Save()
        inst_layer.Save()

        if geometry2d_layer:
            geometry2d_layer.Save()

        paths_to_cleanup: list[Path] = []
        if (
            authoring_format == "usda"
            and usd_auto_binary_threshold_mb
            and usd_auto_binary_threshold_mb > 0
        ):
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

        # Layer Checkpointing for Nucleus
        if checkpoint and checkpoint_note is not None:
            _checkpoint_path(
                layout.stage,
                checkpoint_note,
                checkpoint_tags,
                logger,
                f"{base_name} stage",
            )
            _checkpoint_path(
                layout.prototypes,
                checkpoint_note,
                checkpoint_tags,
                logger,
                f"{base_name} prototypes layer",
            )
            _checkpoint_path(
                layout.materials,
                checkpoint_note,
                checkpoint_tags,
                logger,
                f"{base_name} materials layer",
            )
            _checkpoint_path(
                layout.instances,
                checkpoint_note,
                checkpoint_tags,
                logger,
                f"{base_name} instances layer",
            )
            if geometry2d_layer:
                _checkpoint_path(
                    layout.geometry2d,
                    checkpoint_note,
                    checkpoint_tags,
                    logger,
                    f"{base_name} 2D geometry layer",
                )

        # Logging Outcomes
        logger.info("Wrote stage %s", str(layout.stage))
        master_stage_path = None
        counts = {
            "prototypes_3d": len(caches.repmaps) + len(caches.hashes),
            "instances_3d": len(caches.instances),
            "curves_2d": len(getattr(caches, "annotations", {}) or {}),
            "total_elements": len(caches.instances)
            + len(getattr(caches, "annotations", {}) or {}),
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

        # Clean-Ups
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
            geospatial_mode=resolved_geospatial_mode,
            counts=counts,
            plan=plan,
            revision=revision_note,
        )

    LOG.info("Opening IFC %s", ifc_path)

    # The actuall proceesing calling _execute
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
    detail_requested = _detail_features_requested(args)
    if detail_requested:
        LOG.info("Detail pipeline requested; verifying OCC runtime...")
        try:
            bootstrap_occ()
        except ImportError as exc:
            print(f"Error: OCC detail runtime unavailable: {exc}", file=sys.stderr)
            shutdown_usd_context()
            raise SystemExit(2) from exc
        LOG.info("OCC detail runtime ready.")
    cli_anchor_mode = _normalize_anchor_mode(getattr(args, "anchor_mode", None))
    options_override = OPTIONS
    if width_rules:
        options_override = replace(options_override, curve_width_rules=width_rules)
    if (
        cli_anchor_mode is not None
        and getattr(options_override, "anchor_mode", None) != cli_anchor_mode
    ):
        options_override = replace(options_override, anchor_mode=cli_anchor_mode)
    detail_scope_arg = getattr(args, "detail_scope", None)
    if getattr(args, "detail_mode", False):
        LOG.info(
            "Detail mode enabled: forwarding to OCC detail pipeline (iterator mesh for base geometry)."
        )
        if not detail_scope_arg:
            detail_scope_arg = "all"
        if not getattr(options_override, "detail_mode", False):
            options_override = replace(options_override, detail_mode=True)
    if detail_scope_arg:
        if not getattr(args, "detail_mode", False):
            print(
                "Error: --detail-scope requires --detail-mode to be enabled.",
                file=sys.stderr,
            )
            shutdown_usd_context()
            raise SystemExit(2)
        LOG.info("Detail scope override: %s", detail_scope_arg)
        options_override = replace(options_override, detail_scope=detail_scope_arg)
    detail_objects_arg = getattr(args, "detail_objects", None)
    if detail_objects_arg:
        LOG.info("Detail objects override (mixed ids/guids): %s", detail_objects_arg)
        options_override = replace(
            options_override, detail_objects=tuple(detail_objects_arg)
        )

    if getattr(args, "enable_semantic_subcomponents", False):
        LOG.info("Semantic subcomponent splitting enabled.")
        options_override = replace(options_override, enable_semantic_subcomponents=True)

    raw_engine_arg = getattr(args, "detail_engine", "default") or "default"
    norm_engine = raw_engine_arg.lower()
    if norm_engine == "opencascade":
        norm_engine = "occ"
    elif norm_engine in ("ifc-subcomponents", "ifc-parts"):
        norm_engine = "semantic"
    detail_engine_arg = norm_engine
    # Normalize deprecated force_engine into detail_engine for compatibility
    if (
        getattr(
            options_override,
            "detail_engine",
            getattr(options_override, "force_engine", "default"),
        )
        != detail_engine_arg
    ):
        options_override = replace(options_override, detail_engine=detail_engine_arg)

    semantic_tokens_path = getattr(args, "semantic_tokens_path", None)
    if semantic_tokens_path:
        try:
            st_path = Path(semantic_tokens_path).resolve()
            st_text = read_text(st_path)
            st_payload = json.loads(st_text)
            if isinstance(st_payload, dict):
                LOG.info("Loaded semantic tokens from %s", st_path)
                options_override = replace(options_override, semantic_tokens=st_payload)
            else:
                LOG.warning(
                    "Semantic tokens file %s must contain a JSON object; ignoring.",
                    st_path,
                )
        except Exception as exc:
            print(
                f"Error: Failed to load semantic tokens from {semantic_tokens_path}: {exc}",
                file=sys.stderr,
            )
            shutdown_usd_context()
            raise SystemExit(2) from exc
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
            geospatial_mode=getattr(args, "geospatial_mode", "auto"),
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
        master_stage = (
            path_name(result.master_stage_path) if result.master_stage_path else "n/a"
        )
        coords = result.geodetic_coordinates
        revision = result.revision or "n/a"
        if (
            coords
            and len(coords) >= 2
            and coords[0] is not None
            and coords[1] is not None
        ):
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
