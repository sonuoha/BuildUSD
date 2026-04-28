from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

from .io_utils import is_omniverse_path
from .pxr_utils import UsdGeom, Sdf

LOG = logging.getLogger(__name__)

# OmniGeospatial (omni.usd.schema.geospatial) attribute names for WGS84 reference/local position.
# These are authored directly when the schema python classes are unavailable (e.g. offline conversion).
_OMNI_WGS84_REF_POS_ATTR = (
    "omni:geospatial:wgs84:reference:referencePosition"  # (lat, lon, alt[m])
)
_OMNI_WGS84_REF_TANGENT_ATTR = (
    "omni:geospatial:wgs84:reference:tangentPlane"  # token: ENU|NED
)
_OMNI_WGS84_REF_ORIENT_ATTR = (
    "omni:geospatial:wgs84:reference:orientation"  # (yaw, pitch, roll) deg
)


def geodetic_to_ecef(lat: float, lon: float, h: float) -> Tuple[float, float, float]:
    """
    Convert Geodetic (lat, lon, height) to ECEF (x, y, z).
    Uses WGS84 ellipsoid.
    """
    import math

    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 1 - (1 - f) * (1 - f)

    rad_lat = math.radians(lat)
    rad_lon = math.radians(lon)

    sin_lat = math.sin(rad_lat)
    cos_lat = math.cos(rad_lat)
    sin_lon = math.sin(rad_lon)
    cos_lon = math.cos(rad_lon)

    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)

    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1 - e2) + h) * sin_lat

    return (X, Y, Z)


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    lat0: float,
    lon0: float,
    h0: float,
) -> Tuple[float, float, float]:
    """
    Convert ECEF target (x, y, z) to ENU relative to origin (lat0, lon0, h0).
    """
    import math

    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, h0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    rad_lat0 = math.radians(lat0)
    rad_lon0 = math.radians(lon0)

    sin_lat = math.sin(rad_lat0)
    cos_lat = math.cos(rad_lat0)
    sin_lon = math.sin(rad_lon0)
    cos_lon = math.cos(rad_lon0)

    # ENU transformation matrix
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return (e, n, u)


def ensure_geospatial_root(stage, path: str = "/World/Geospatial"):
    """Ensure an Xform prim exists at `path` and return the UsdGeom.Xform."""
    try:
        xf = UsdGeom.Xform.Define(stage, path)
        return xf
    except Exception:
        # Best-effort fallback
        prim = stage.GetPrimAtPath(path)
        if prim:
            return UsdGeom.Xform(prim)
        raise


def _geospatial_root_paths(stage) -> list[Sdf.Path]:
    """Return geospatial prim paths to update, preferring existing prims."""
    if stage is None:
        return []

    paths: list[Sdf.Path] = []
    world_path = Sdf.Path("/World")
    world_geo = world_path.AppendChild("Geospatial")

    default_prim = stage.GetDefaultPrim()
    default_path = default_prim.GetPath() if default_prim else None
    default_geo = default_path.AppendChild("Geospatial") if default_path else None

    def _add(path: Optional[Sdf.Path]) -> None:
        if path and path not in paths:
            paths.append(path)

    if default_geo and stage.GetPrimAtPath(default_geo):
        _add(default_geo)
    if stage.GetPrimAtPath(world_geo):
        _add(world_geo)

    # If no existing geospatial prims found, prefer default prim, else /World.
    if not paths:
        if default_geo:
            _add(default_geo)
        elif stage.GetPrimAtPath(world_path):
            _add(world_geo)

    return paths


def _author_wgs84_reference_attrs_on_prim(
    prim,
    *,
    latitude: float,
    longitude: float,
    altitude_m: float,
    tangent_plane: str = "ENU",
    orientation_ypr: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Author OmniGeospatial WGS84 reference position attrs on a prim (schema optional)."""
    # Prefer the schema API when running inside Kit (it also applies the API schema metadata).
    try:
        from omni.usd.schema.geospatial import OmniGeospatial  # type: ignore

        api = OmniGeospatial.WGS84ReferencePositionAPI.Apply(prim)
        api.CreateReferencePositionAttr().Set(
            (float(latitude), float(longitude), float(altitude_m))
        )
        api.CreateTangentPlaneAttr().Set(str(tangent_plane))
        api.CreateOrientationAttr().Set(
            (
                float(orientation_ypr[0]),
                float(orientation_ypr[1]),
                float(orientation_ypr[2]),
            )
        )
        return
    except Exception:
        pass

    # Offline / no schema classes available: author attributes by name.
    try:
        prim.CreateAttribute(_OMNI_WGS84_REF_POS_ATTR, Sdf.ValueTypeNames.Double3).Set(
            (float(latitude), float(longitude), float(altitude_m))
        )
        prim.CreateAttribute(
            _OMNI_WGS84_REF_TANGENT_ATTR, Sdf.ValueTypeNames.Token
        ).Set(str(tangent_plane))
        prim.CreateAttribute(
            _OMNI_WGS84_REF_ORIENT_ATTR, Sdf.ValueTypeNames.Double3
        ).Set(
            (
                float(orientation_ypr[0]),
                float(orientation_ypr[1]),
                float(orientation_ypr[2]),
            )
        )
    except Exception:
        # Keep this non-fatal: geospatial metadata must never break conversion.
        LOG.debug(
            "Failed authoring OmniGeospatial WGS84 reference attrs on %s",
            prim.GetPath(),
        )


GeospatialMode = str
_VALID_MODES = {"auto", "usd", "omni", "none"}
_OMNI_EXTENSIONS: tuple[str, ...] = (
    "omni.services.geospatial",
    "omni.kit.geospatial.terrain",
    "omni.kit.geospatial.imagery",
)


def resolve_geospatial_mode(
    requested: Optional[str],
    *,
    offline: bool,
    output_path=None,
) -> GeospatialMode:
    """
    Pick a geospatial driver based on user input and environment.

    - offline forces usd/no-op (OmniGeospatial requires Kit).
    - omniverse:// outputs bias toward omni when not offline.
    - invalid/unknown values fall back to auto->usd.
    """
    mode_in = (requested or "auto").strip().lower()
    if mode_in not in _VALID_MODES:
        LOG.debug(
            "Unknown geospatial_mode '%s'; defaulting to auto (usd/offline).", requested
        )
        mode_in = "auto"

    if offline:
        return "usd" if mode_in in ("auto", "usd", "omni") else "none"

    if mode_in == "none":
        return "none"
    if mode_in == "usd":
        return "usd"
    if mode_in == "omni":
        return "omni"

    # auto selection
    if output_path and is_omniverse_path(output_path):
        return "omni"
    return "usd"


def _get_extension_manager():
    try:
        from omni.kit.app import get_app  # type: ignore
    except Exception:
        return None
    try:
        app = get_app()
        if not app:
            return None
        return app.get_extension_manager()
    except Exception:
        return None


def _enable_geospatial_extensions(exts: Sequence[str]) -> None:
    mgr = _get_extension_manager()
    if mgr is None:
        return
    for ext in exts:
        try:
            has_ext = False
            if hasattr(mgr, "has_extension"):
                try:
                    has_ext = bool(mgr.has_extension(ext))
                except Exception:
                    has_ext = False
            if not has_ext:
                LOG.debug(
                    "Geospatial extension %s not available in this Kit; skipping.", ext
                )
                continue
            mgr.set_extension_enabled_immediate(ext, True)
        except Exception:
            LOG.debug(
                "Unable to enable extension %s (may already be active or missing).", ext
            )


def signed_model_offset(
    model_offset: Optional[Tuple[float, float, float]],
    model_offset_type: Optional[str],
) -> Tuple[float, float, float]:
    """
    Return the signed offset that was applied to geometry.

    model_offset is the raw value; model_offset_type 'negative' means the authoring
    pipeline subtracted it from geometry (so return -offset for geospatial placement).
    """
    if not model_offset:
        return (0.0, 0.0, 0.0)
    ox, oy, oz = (
        float(model_offset[0]),
        float(model_offset[1]),
        float(model_offset[2]),
    )
    sign = -1.0 if str(model_offset_type or "").strip().lower() == "negative" else 1.0
    return (sign * ox, sign * oy, sign * oz)


def maybe_stream_omnigeospatial(
    stage,
    geospatial_mode: str,
    *,
    projected_crs: str,
    geodetic_crs: str,
    model_offset: Optional[Tuple[float, float, float]] = None,
    model_offset_type: Optional[str] = None,
) -> bool:
    """
    Best-effort initialization of OmniGeospatial when running in Kit.

    - Enables the core geospatial extensions.
    - Creates a /World/Geospatial Xform (Pattern A: no translation; stage origin is the georeference).
    - Stamps CRS metadata on the Geospatial prim for service consumers.
    - Returns True if a geospatial root was prepared (even if streaming is deferred).
    """
    if geospatial_mode != "omni":
        return False
    if stage is None:
        return False

    mgr = _get_extension_manager()
    if mgr is None:
        LOG.debug("Kit extension manager unavailable; skipping OmniGeospatial setup.")
        return False

    _enable_geospatial_extensions(_OMNI_EXTENSIONS)

    # If assign_world_geolocation ran earlier, it stamped lon/lat/height on /World as custom data.
    # Mirror that into the omni.usd.schema.geospatial reference position attrs so Kit services can discover it.
    try:
        world_prim = stage.GetPrimAtPath("/World")
        coords = (
            world_prim.GetCustomDataByKey("ifc:geodeticCoordinates")
            if world_prim
            else None
        ) or None
        if isinstance(coords, dict):
            lon = float(coords.get("lon", 0.0))
            lat = float(coords.get("lat", 0.0))
            alt = float(coords.get("height", 0.0))
            for geo_path in _geospatial_root_paths(stage):
                try:
                    geo_xf = ensure_geospatial_root(stage, geo_path.pathString)
                    geo_prim = geo_xf.GetPrim()
                    _author_wgs84_reference_attrs_on_prim(
                        geo_prim,
                        latitude=lat,
                        longitude=lon,
                        altitude_m=alt,
                        tangent_plane="ENU",
                        orientation_ypr=(0.0, 0.0, 0.0),
                    )
                    geo_prim.SetCustomDataByKey("ifc:projectedCRS", projected_crs)
                    geo_prim.SetCustomDataByKey("ifc:geodeticCRS", geodetic_crs)
                except Exception:
                    continue
    except Exception:
        pass

    # Actual tile/imagery requests depend on the runtime environment and provider config.
    # This hook prepares the scene and leaves streaming to the available services.
    validate_geospatial_contract(stage)
    return True


def validate_geospatial_contract(
    stage, *, logger: Optional[logging.Logger] = None
) -> dict:
    """
    Lightweight validation for Pattern A georeferencing.

    Pattern A contract:
      - /World origin corresponds to the WGS84 reference position.
      - /World/Geospatial exists but carries no translation offset.
      - CRS + WGS84 reference attrs are discoverable (customData or schema attrs).
    """
    log = logger or LOG
    result = {
        "pattern": "A",
        "has_world": False,
        "has_geospatial_root": False,
        "has_wgs84_reference": False,
        "has_geodetic_coordinates": False,
        "projected_crs": None,
        "geodetic_crs": None,
        "model_offset": None,
        "warnings": [],
    }

    if stage is None:
        result["warnings"].append("Stage is None; cannot validate geospatial metadata.")
        return result

    world_prim = stage.GetPrimAtPath("/World")
    if world_prim:
        result["has_world"] = True
        coords = world_prim.GetCustomDataByKey("ifc:geodeticCoordinates")
        if isinstance(coords, dict) and "lon" in coords and "lat" in coords:
            result["has_geodetic_coordinates"] = True
        model_offset = world_prim.GetCustomDataByKey("ifc:modelOffset")
        if isinstance(model_offset, dict):
            result["model_offset"] = model_offset

    geo_prim = stage.GetPrimAtPath("/World/Geospatial")
    if geo_prim:
        result["has_geospatial_root"] = True
        pos_attr = geo_prim.GetAttribute(_OMNI_WGS84_REF_POS_ATTR)
        if pos_attr and pos_attr.HasAuthoredValue():
            result["has_wgs84_reference"] = True
        else:
            # fall back to any schema-provided attributes with the same name
            result["has_wgs84_reference"] = False
        try:
            xformable = UsdGeom.Xformable(geo_prim)
            for op in xformable.GetOrderedXformOps():
                try:
                    value = op.Get()
                except Exception:
                    continue
                if value is None:
                    continue
                try:
                    if any(abs(float(v)) > 1e-6 for v in value):
                        result["warnings"].append(
                            "/World/Geospatial has a non-zero transform; Pattern A expects identity."
                        )
                        break
                except Exception:
                    continue
        except Exception:
            pass

    try:
        layer = stage.GetRootLayer()
        data = dict(getattr(layer, "customLayerData", {}) or {})
        result["projected_crs"] = data.get("projectedCRS")
        result["geodetic_crs"] = data.get("geodeticCRS")
        if result["model_offset"] is None and isinstance(data.get("modelOffset"), dict):
            result["model_offset"] = data.get("modelOffset")
    except Exception:
        pass

    if not result["has_wgs84_reference"] and not result["has_geodetic_coordinates"]:
        result["warnings"].append(
            "No WGS84 reference data found on /World or /World/Geospatial."
        )
    if not result["projected_crs"] or not result["geodetic_crs"]:
        result["warnings"].append(
            "CRS metadata missing from root layer customLayerData."
        )

    for warning in result["warnings"]:
        log.debug("Geospatial validation: %s", warning)

    return result
