from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .process_ifc import MapConversionData
from .resolve_frame import _placement_location_xyz

_PROJECTED_XY_THRESHOLD_M = 10000.0
_MAP_ORIGIN_NONZERO_THRESHOLD_M = 1.0
_ROTATION_IDENTITY_EPS = 1.0e-6
_SCALE_MATCH_EPS = 1.0e-6
_DOUBLE_GEOREF_WARN_THRESHOLD_M = 5000.0


@dataclass(frozen=True)
class PlacementStats:
    sample_count: int
    max_abs_m: float
    space: str


@dataclass(frozen=True)
class RotationPlan:
    quaternion: tuple[float, float, float, float]
    applied: bool
    theta_deg: float


@dataclass(frozen=True)
class LocalizationPlan:
    requested_anchor_mode: Optional[str]
    requested_anchor_source: Optional[str]
    requested_anchor_world_m: Optional[tuple[float, float, float]]
    candidate_model_offset_m: Optional[tuple[float, float, float]]
    apply_model_offset: bool
    applied_model_offset_m: Optional[tuple[float, float, float]]
    model_offset_type: Optional[str]
    skip_reason: Optional[str]
    metrics_before: PlacementStats
    metrics_after: Optional[PlacementStats]


@dataclass(frozen=True)
class GeoreferencePlan:
    effective_anchor_mode: Optional[str]
    georef_origin: Optional[str]
    projected_anchor_world_m: Optional[tuple[float, float, float]]
    coordinate_operation_origin_world_m: Optional[tuple[float, float, float]]


@dataclass(frozen=True)
class CoordinateReference:
    source_crs: Optional[str]
    projected_crs: Optional[str]
    projected_crs_label: Optional[str]
    projected_crs_source: Optional[str]
    coordinate_operation_kind: Optional[str]
    coordinate_operation_source: Optional[str]
    coordinate_operation_origin_world_m: Optional[tuple[float, float, float]]
    has_authoritative_operation: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class StageGeoreference:
    projected_crs: Optional[str]
    projected_crs_source: Optional[str]
    stage_origin_projected_m: Optional[tuple[float, float, float]]
    georef_source: Optional[str]
    status: str
    coordinate_operation_origin_world_m: Optional[tuple[float, float, float]]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IfcGeoreferencingPlan:
    projected_crs_label: Optional[str]
    has_projected_crs: bool
    map_conv: Optional[MapConversionData]
    map_conv_active: Optional[MapConversionData]
    strategy: str
    strategy_info: dict[str, object]
    geom_space: str
    ifc_site_raw: Optional[tuple[float, float, float]]
    ifc_site_authored_m: Optional[tuple[float, float, float]]
    ifc_site_world_m: Optional[tuple[float, float, float]]
    pbp_ifc_m: Optional[tuple[float, float, float]]
    sp_ifc_m: Optional[tuple[float, float, float]]
    pbp_world_m: Optional[tuple[float, float, float]]
    shared_site_world_m: Optional[tuple[float, float, float]]
    rotation: RotationPlan
    localization: LocalizationPlan
    coordinate_operation_localization: Optional[LocalizationPlan]
    coordinate_reference: CoordinateReference
    georef: GeoreferencePlan
    stage_georef: StageGeoreference


_EPSG_PATTERN = re.compile(r"\bEPSG[:/ ]?(\d{3,6})\b", re.IGNORECASE)


def _normalize_crs_identifier(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _EPSG_PATTERN.search(text)
    if match:
        return f"EPSG:{match.group(1)}"
    if text.upper().startswith("EPSG:"):
        code = text.split(":", 1)[1].strip()
        return f"EPSG:{code}" if code else None
    return text


def _ifc_crs_identifier(crs) -> Optional[str]:
    if crs is None:
        return None
    for key in ("Name", "Description", "GeodeticDatum", "MapProjection"):
        try:
            value = getattr(crs, key, None)
        except Exception:
            value = None
        normalized = _normalize_crs_identifier(value)
        if normalized:
            return normalized
    return None


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


def _ifc_projected_crs_identifier(ifc) -> Optional[str]:
    try:
        entries = ifc.by_type("IfcProjectedCRS") or []
    except Exception:
        return None
    for crs in entries:
        identifier = _ifc_crs_identifier(crs)
        if identifier:
            return identifier
    return None


def _discover_ifc_coordinate_operation(
    ifc,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        contexts = ifc.by_type("IfcGeometricRepresentationContext") or []
    except Exception:
        return (None, None, None)
    best_kind: Optional[str] = None
    best_source_crs: Optional[str] = None
    best_target_crs: Optional[str] = None
    for ctx in contexts:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            kind = None
            if op.is_a("IfcMapConversion"):
                kind = "IfcMapConversion"
            elif op.is_a("IfcRigidOperation"):
                kind = "IfcRigidOperation"
            if kind is None:
                continue
            source_crs = _ifc_crs_identifier(getattr(op, "SourceCRS", None))
            target_crs = _ifc_crs_identifier(getattr(op, "TargetCRS", None))
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return (kind, source_crs, target_crs)
            if best_kind is None:
                best_kind = kind
                best_source_crs = source_crs
                best_target_crs = target_crs
    return (best_kind, best_source_crs, best_target_crs)


def resolve_coordinate_reference(
    ifc,
    *,
    map_conv: Optional[MapConversionData],
    external_projected_crs: Optional[str],
) -> CoordinateReference:
    ifc_projected_crs_label = _ifc_projected_crs_label(ifc)
    ifc_projected_crs = _ifc_projected_crs_identifier(ifc)
    operation_kind, source_crs, target_crs = _discover_ifc_coordinate_operation(ifc)
    coordinate_operation_origin = coordinate_operation_origin_world_m(map_conv)
    warnings: list[str] = []

    resolved_projected_crs = (
        target_crs
        or ifc_projected_crs
        or _normalize_crs_identifier(external_projected_crs)
    )
    if target_crs:
        projected_crs_source = "ifc_coordinate_operation"
    elif ifc_projected_crs:
        projected_crs_source = "ifc_projected_crs"
    elif external_projected_crs:
        projected_crs_source = "external"
    else:
        projected_crs_source = None

    normalized_external = _normalize_crs_identifier(external_projected_crs)
    if resolved_projected_crs and normalized_external:
        if target_crs and normalized_external != target_crs:
            warnings.append(
                "external_projected_crs_conflicts_with_ifc_coordinate_operation"
            )
        elif (
            not target_crs
            and ifc_projected_crs
            and normalized_external != ifc_projected_crs
        ):
            warnings.append("external_projected_crs_conflicts_with_ifc_projected_crs")

    return CoordinateReference(
        source_crs=source_crs,
        projected_crs=resolved_projected_crs,
        projected_crs_label=ifc_projected_crs_label,
        projected_crs_source=projected_crs_source,
        coordinate_operation_kind=operation_kind,
        coordinate_operation_source="ifc_coordinate_operation"
        if operation_kind is not None
        else None,
        coordinate_operation_origin_world_m=coordinate_operation_origin,
        has_authoritative_operation=map_conv is not None,
        warnings=tuple(warnings),
    )


def ifc_site_placement_raw(ifc) -> Optional[tuple[float, float, float]]:
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
    stats = sample_authored_placement_stats(
        ifc,
        unit_scale_m=unit_scale_m,
        site_m=site_m,
        global_threshold_m=global_threshold_m,
        sample_limit=sample_limit,
    )
    return stats.space


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


def mapconv_world_to_local_m(
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


def mapconv_local_to_world_m(
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


def coordinate_operation_origin_world_m(
    map_conv: Optional[MapConversionData],
) -> Optional[tuple[float, float, float]]:
    if map_conv is None:
        return None
    return (
        float(getattr(map_conv, "eastings", 0.0) or 0.0),
        float(getattr(map_conv, "northings", 0.0) or 0.0),
        float(getattr(map_conv, "orthogonal_height", 0.0) or 0.0),
    )


def sample_authored_placement_points_m(
    ifc,
    *,
    unit_scale_m: float,
    sample_limit: int = 200,
) -> tuple[tuple[float, float, float], ...]:
    points: list[tuple[float, float, float]] = []
    for typename in ("IfcProduct", "IfcElement", "IfcBuildingElement"):
        try:
            items = ifc.by_type(typename) or []
        except Exception:
            continue
        if not items:
            continue
        for product in items:
            if len(points) >= sample_limit:
                break
            place = getattr(product, "ObjectPlacement", None)
            xyz = _placement_location_xyz(place)
            if xyz is None:
                continue
            points.append(
                (
                    float(xyz[0]) * unit_scale_m,
                    float(xyz[1]) * unit_scale_m,
                    float(xyz[2]) * unit_scale_m,
                )
            )
        break
    return tuple(points)


def sample_authored_placement_stats(
    ifc,
    *,
    unit_scale_m: float,
    site_m: Optional[tuple[float, float, float]],
    global_threshold_m: float = _PROJECTED_XY_THRESHOLD_M,
    sample_limit: int = 200,
) -> PlacementStats:
    points = list(
        sample_authored_placement_points_m(
            ifc, unit_scale_m=unit_scale_m, sample_limit=sample_limit
        )
    )
    if not points and site_m is not None:
        points.append(site_m)
    max_abs = 0.0
    for point in points:
        max_abs = max(max_abs, abs(point[0]), abs(point[1]), abs(point[2]))
    space = "global" if max_abs >= global_threshold_m else "local"
    return PlacementStats(sample_count=len(points), max_abs_m=max_abs, space=space)


def build_rotation_plan(map_conv_active: Optional[MapConversionData]) -> RotationPlan:
    quaternion = (0.0, 0.0, 0.0, 1.0)
    applied = False
    theta_deg = 0.0
    if map_conv_active is not None:
        ax, ay = map_conv_active.normalized_axes()
        if abs(ax - 1.0) > _ROTATION_IDENTITY_EPS or abs(ay) > _ROTATION_IDENTITY_EPS:
            theta_deg = math.degrees(math.atan2(ay, ax))
            half = math.radians(theta_deg) * 0.5
            quaternion = (0.0, 0.0, math.sin(half), math.cos(half))
            applied = True
    return RotationPlan(quaternion=quaternion, applied=applied, theta_deg=theta_deg)


def _site_world_m(
    *,
    site_authored_m: Optional[tuple[float, float, float]],
    map_conv: Optional[MapConversionData],
    unit_scale_m: float,
) -> Optional[tuple[float, float, float]]:
    if site_authored_m is None:
        return None
    site_xy_mag = max(abs(site_authored_m[0]), abs(site_authored_m[1]))
    if site_xy_mag > _PROJECTED_XY_THRESHOLD_M:
        return site_authored_m
    if map_conv is None:
        return None
    return mapconv_local_to_world_m(
        map_conv, site_authored_m, unit_scale_m=unit_scale_m
    )


def _choose_requested_anchor_world(
    *,
    anchor_mode: Optional[str],
    ifc_site_world_m: Optional[tuple[float, float, float]],
    coordinate_operation_world_m: Optional[tuple[float, float, float]],
    pbp_world_m: Optional[tuple[float, float, float]],
    shared_site_world_m: Optional[tuple[float, float, float]],
) -> tuple[Optional[str], Optional[tuple[float, float, float]]]:
    if anchor_mode == "local":
        if ifc_site_world_m is not None:
            return ("ifc_site", ifc_site_world_m)
        if coordinate_operation_world_m is not None:
            return ("coordinate_operation", coordinate_operation_world_m)
        return (None, None)
    if anchor_mode == "basepoint":
        if pbp_world_m is not None:
            return ("pbp", pbp_world_m)
        if shared_site_world_m is not None:
            return ("shared_site", shared_site_world_m)
    return (None, None)


def _candidate_model_offset_m(
    *,
    geom_space: str,
    requested_anchor_source: Optional[str],
    requested_anchor_world_m: Optional[tuple[float, float, float]],
    ifc_site_authored_m: Optional[tuple[float, float, float]],
    map_conv_active: Optional[MapConversionData],
    unit_scale_m: float,
) -> Optional[tuple[float, float, float]]:
    if requested_anchor_world_m is None:
        return None
    if geom_space == "global":
        return requested_anchor_world_m
    if map_conv_active is not None:
        return mapconv_world_to_local_m(
            map_conv_active,
            requested_anchor_world_m,
            unit_scale_m=unit_scale_m,
        )
    if requested_anchor_source == "ifc_site" and ifc_site_authored_m is not None:
        return ifc_site_authored_m
    return None


def _stats_after_offset(
    before: PlacementStats,
    points_m: tuple[tuple[float, float, float], ...],
    offset_m: tuple[float, float, float],
) -> PlacementStats:
    max_abs = 0.0
    for point in points_m:
        px = float(point[0]) - float(offset_m[0])
        py = float(point[1]) - float(offset_m[1])
        pz = float(point[2]) - float(offset_m[2])
        max_abs = max(max_abs, abs(px), abs(py), abs(pz))
    space = "global" if max_abs >= _PROJECTED_XY_THRESHOLD_M else "local"
    return PlacementStats(
        sample_count=before.sample_count, max_abs_m=max_abs, space=space
    )


def _source_point_to_projected_m(
    point_m: tuple[float, float, float],
    *,
    geom_space: str,
    map_conv: Optional[MapConversionData],
    unit_scale_m: float,
) -> Optional[tuple[float, float, float]]:
    if geom_space == "global":
        return (
            float(point_m[0]),
            float(point_m[1]),
            float(point_m[2]),
        )
    if map_conv is None:
        return None
    return mapconv_local_to_world_m(map_conv, point_m, unit_scale_m=unit_scale_m)


def decide_localization_plan(
    *,
    points_m: tuple[tuple[float, float, float], ...],
    metrics_before: PlacementStats,
    requested_anchor_mode: Optional[str],
    requested_anchor_source: Optional[str],
    requested_anchor_world_m: Optional[tuple[float, float, float]],
    candidate_model_offset_m: Optional[tuple[float, float, float]],
) -> LocalizationPlan:
    if requested_anchor_mode is None:
        return LocalizationPlan(
            requested_anchor_mode=None,
            requested_anchor_source=requested_anchor_source,
            requested_anchor_world_m=requested_anchor_world_m,
            candidate_model_offset_m=candidate_model_offset_m,
            apply_model_offset=False,
            applied_model_offset_m=None,
            model_offset_type=None,
            skip_reason="anchor_mode_none",
            metrics_before=metrics_before,
            metrics_after=None,
        )

    if candidate_model_offset_m is None:
        return LocalizationPlan(
            requested_anchor_mode=requested_anchor_mode,
            requested_anchor_source=requested_anchor_source,
            requested_anchor_world_m=requested_anchor_world_m,
            candidate_model_offset_m=None,
            apply_model_offset=False,
            applied_model_offset_m=None,
            model_offset_type=None,
            skip_reason="candidate_missing",
            metrics_before=metrics_before,
            metrics_after=None,
        )

    if all(abs(float(v)) <= 1.0e-9 for v in candidate_model_offset_m):
        return LocalizationPlan(
            requested_anchor_mode=requested_anchor_mode,
            requested_anchor_source=requested_anchor_source,
            requested_anchor_world_m=requested_anchor_world_m,
            candidate_model_offset_m=candidate_model_offset_m,
            apply_model_offset=False,
            applied_model_offset_m=None,
            model_offset_type=None,
            skip_reason="candidate_zero",
            metrics_before=metrics_before,
            metrics_after=metrics_before,
        )

    metrics_after = _stats_after_offset(
        metrics_before, points_m, candidate_model_offset_m
    )
    improves = metrics_after.max_abs_m + 1.0e-6 < metrics_before.max_abs_m
    stays_local = metrics_after.max_abs_m < _PROJECTED_XY_THRESHOLD_M

    if not improves:
        skip_reason = "candidate_does_not_improve_localization"
    elif not stays_local:
        skip_reason = "candidate_leaves_model_projected_scale"
    else:
        skip_reason = None

    return LocalizationPlan(
        requested_anchor_mode=requested_anchor_mode,
        requested_anchor_source=requested_anchor_source,
        requested_anchor_world_m=requested_anchor_world_m,
        candidate_model_offset_m=candidate_model_offset_m,
        apply_model_offset=skip_reason is None,
        applied_model_offset_m=candidate_model_offset_m
        if skip_reason is None
        else None,
        model_offset_type="negative" if skip_reason is None else None,
        skip_reason=skip_reason,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )


def derive_stage_georeference(
    *,
    coordinate_reference: CoordinateReference,
    localization: LocalizationPlan,
    geom_space: str,
    map_conv: Optional[MapConversionData],
    unit_scale_m: float,
) -> StageGeoreference:
    warnings = list(coordinate_reference.warnings)
    stage_origin_projected_m: Optional[tuple[float, float, float]] = None
    georef_source: Optional[str] = None

    if (
        localization.apply_model_offset
        and localization.applied_model_offset_m is not None
    ):
        stage_origin_projected_m = _source_point_to_projected_m(
            localization.applied_model_offset_m,
            geom_space=geom_space,
            map_conv=map_conv,
            unit_scale_m=unit_scale_m,
        )
        georef_source = (
            f"applied_localization:{localization.requested_anchor_source}"
            if localization.requested_anchor_source
            else "applied_localization"
        )
    elif coordinate_reference.has_authoritative_operation and geom_space == "local":
        stage_origin_projected_m = _source_point_to_projected_m(
            (0.0, 0.0, 0.0),
            geom_space=geom_space,
            map_conv=map_conv,
            unit_scale_m=unit_scale_m,
        )
        georef_source = "coordinate_operation_stage_origin"
    elif coordinate_reference.projected_crs is not None and geom_space == "global":
        stage_origin_projected_m = (0.0, 0.0, 0.0)
        georef_source = "declared_global_origin"

    if (
        stage_origin_projected_m is not None
        and coordinate_reference.has_authoritative_operation
    ):
        status = "authoritative"
    elif (
        stage_origin_projected_m is not None
        and coordinate_reference.projected_crs is not None
    ):
        status = "declared"
    else:
        status = "unknown"

    if (
        coordinate_reference.has_authoritative_operation
        and coordinate_reference.projected_crs is None
    ):
        warnings.append("coordinate_operation_missing_projected_crs")
    if (
        stage_origin_projected_m is None
        and localization.requested_anchor_world_m is not None
    ):
        warnings.append("requested_anchor_did_not_resolve_stage_origin")

    return StageGeoreference(
        projected_crs=coordinate_reference.projected_crs,
        projected_crs_source=coordinate_reference.projected_crs_source,
        stage_origin_projected_m=stage_origin_projected_m,
        georef_source=georef_source,
        status=status,
        coordinate_operation_origin_world_m=coordinate_reference.coordinate_operation_origin_world_m,
        warnings=tuple(warnings),
    )


def decide_georeference_plan(
    *,
    localization: LocalizationPlan,
    coordinate_operation_localization: Optional[LocalizationPlan],
    coordinate_operation_origin_world_m: Optional[tuple[float, float, float]],
) -> GeoreferencePlan:
    if (
        localization.apply_model_offset
        and localization.requested_anchor_world_m is not None
    ):
        return GeoreferencePlan(
            effective_anchor_mode=localization.requested_anchor_mode,
            georef_origin=localization.requested_anchor_source,
            projected_anchor_world_m=localization.requested_anchor_world_m,
            coordinate_operation_origin_world_m=coordinate_operation_origin_world_m,
        )
    if coordinate_operation_origin_world_m is not None and (
        coordinate_operation_localization is None
        or not coordinate_operation_localization.apply_model_offset
    ):
        return GeoreferencePlan(
            effective_anchor_mode="local",
            georef_origin="coordinate_operation",
            projected_anchor_world_m=coordinate_operation_origin_world_m,
            coordinate_operation_origin_world_m=coordinate_operation_origin_world_m,
        )
    return GeoreferencePlan(
        effective_anchor_mode=None,
        georef_origin=None,
        projected_anchor_world_m=None,
        coordinate_operation_origin_world_m=coordinate_operation_origin_world_m,
    )


def build_ifc_georeferencing_plan(
    ifc,
    *,
    unit_scale_m: float,
    anchor_mode: Optional[str],
    default_pbp_world_m: Optional[tuple[float, float, float]],
    default_shared_site_world_m: Optional[tuple[float, float, float]],
    external_projected_crs: Optional[str],
    map_conv: Optional[MapConversionData],
) -> IfcGeoreferencingPlan:
    projected_crs_label = _ifc_projected_crs_label(ifc)
    has_projected_crs = bool(projected_crs_label) or _ifc_projected_crs_present(ifc)
    coordinate_reference = resolve_coordinate_reference(
        ifc,
        map_conv=map_conv,
        external_projected_crs=external_projected_crs,
    )

    ifc_site_raw = ifc_site_placement_raw(ifc)
    ifc_site_authored_m = (
        (
            float(ifc_site_raw[0]) * unit_scale_m,
            float(ifc_site_raw[1]) * unit_scale_m,
            float(ifc_site_raw[2]) * unit_scale_m,
        )
        if ifc_site_raw is not None
        else None
    )
    points_m = sample_authored_placement_points_m(ifc, unit_scale_m=unit_scale_m)
    metrics_before = sample_authored_placement_stats(
        ifc,
        unit_scale_m=unit_scale_m,
        site_m=ifc_site_authored_m,
    )
    geom_space = metrics_before.space
    strategy, strategy_info = _classify_georef_strategy(
        site_m=ifc_site_authored_m,
        site_raw=ifc_site_raw,
        map_conv=map_conv,
        unit_scale_m=unit_scale_m,
    )
    map_conv_active = map_conv if strategy == "MAPCONV_ANCHORED" else None
    rotation = build_rotation_plan(map_conv_active)

    ifc_site_world_m = _site_world_m(
        site_authored_m=ifc_site_authored_m,
        map_conv=map_conv,
        unit_scale_m=unit_scale_m,
    )
    pbp_ifc_m = resolve_project_base_point_m(ifc)
    sp_ifc_m = resolve_survey_point_m(ifc)
    pbp_world_m = pbp_ifc_m or default_pbp_world_m
    shared_site_world_m = sp_ifc_m or default_shared_site_world_m
    coordinate_operation_world_m = coordinate_operation_origin_world_m(map_conv)

    requested_anchor_source, requested_anchor_world_m = _choose_requested_anchor_world(
        anchor_mode=anchor_mode,
        ifc_site_world_m=ifc_site_world_m,
        coordinate_operation_world_m=coordinate_operation_world_m,
        pbp_world_m=pbp_world_m,
        shared_site_world_m=shared_site_world_m,
    )
    candidate_model_offset_m = _candidate_model_offset_m(
        geom_space=geom_space,
        requested_anchor_source=requested_anchor_source,
        requested_anchor_world_m=requested_anchor_world_m,
        ifc_site_authored_m=ifc_site_authored_m,
        map_conv_active=map_conv_active,
        unit_scale_m=unit_scale_m,
    )
    localization = decide_localization_plan(
        points_m=points_m,
        metrics_before=metrics_before,
        requested_anchor_mode=anchor_mode,
        requested_anchor_source=requested_anchor_source,
        requested_anchor_world_m=requested_anchor_world_m,
        candidate_model_offset_m=candidate_model_offset_m,
    )
    coordinate_operation_localization: Optional[LocalizationPlan] = None
    if coordinate_operation_world_m is not None:
        coordinate_operation_localization = decide_localization_plan(
            points_m=points_m,
            metrics_before=metrics_before,
            requested_anchor_mode="local",
            requested_anchor_source="coordinate_operation",
            requested_anchor_world_m=coordinate_operation_world_m,
            candidate_model_offset_m=_candidate_model_offset_m(
                geom_space=geom_space,
                requested_anchor_source="coordinate_operation",
                requested_anchor_world_m=coordinate_operation_world_m,
                ifc_site_authored_m=ifc_site_authored_m,
                map_conv_active=map_conv_active,
                unit_scale_m=unit_scale_m,
            ),
        )
    georef = decide_georeference_plan(
        localization=localization,
        coordinate_operation_localization=coordinate_operation_localization,
        coordinate_operation_origin_world_m=coordinate_operation_world_m,
    )
    stage_georef = derive_stage_georeference(
        coordinate_reference=coordinate_reference,
        localization=localization,
        geom_space=geom_space,
        map_conv=map_conv,
        unit_scale_m=unit_scale_m,
    )

    return IfcGeoreferencingPlan(
        projected_crs_label=projected_crs_label,
        has_projected_crs=has_projected_crs,
        map_conv=map_conv,
        map_conv_active=map_conv_active,
        strategy=strategy,
        strategy_info=strategy_info,
        geom_space=geom_space,
        ifc_site_raw=ifc_site_raw,
        ifc_site_authored_m=ifc_site_authored_m,
        ifc_site_world_m=ifc_site_world_m,
        pbp_ifc_m=pbp_ifc_m,
        sp_ifc_m=sp_ifc_m,
        pbp_world_m=pbp_world_m,
        shared_site_world_m=shared_site_world_m,
        rotation=rotation,
        localization=localization,
        coordinate_operation_localization=coordinate_operation_localization,
        coordinate_reference=coordinate_reference,
        georef=georef,
        stage_georef=stage_georef,
    )
