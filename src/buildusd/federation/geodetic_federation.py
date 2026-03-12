from __future__ import annotations

"""
Geodetic federation utilities for Omniverse Kit.

CRS assumptions:
  - WGS84 latitude/longitude are degrees.
  - Heights are meters above the WGS84 ellipsoid (ellipsoidal height).
  - ECEF coordinates are EPSG:4978 meters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..pxr_utils import Gf, Sdf, Usd, UsdGeom

_WGS84_REF_ATTR = "omni:geospatial:wgs84:reference:referencePosition"
_ECEF_REF_ATTR = "omni:geospatial:ecef:reference:referencePosition"


def _log_info(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_info(msg)
    except Exception:
        print(msg)


def _log_warn(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_warn(msg)
    except Exception:
        print(f"Warning: {msg}")


def _log_error(msg: str) -> None:
    try:
        import carb  # type: ignore

        carb.log_error(msg)
    except Exception:
        print(f"Error: {msg}")


def _coerce_vec3(value) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    try:
        if hasattr(value, "__len__") and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        pass
    return None


def _sanitize_name(name: str, fallback: str = "Site") -> str:
    import re

    raw = (name or "").strip()
    safe = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not safe:
        return fallback
    if safe[0].isdigit():
        return f"{fallback}_{safe}"
    return safe


def _ensure_wgs84_attr(prim, wgs84: Tuple[float, float, float]) -> None:
    attr = prim.GetAttribute(_WGS84_REF_ATTR)
    if not attr:
        attr = prim.CreateAttribute(_WGS84_REF_ATTR, Sdf.ValueTypeNames.Double3)
    attr.Set((float(wgs84[0]), float(wgs84[1]), float(wgs84[2])))


def _is_omniverse_path(value: object) -> bool:
    try:
        return str(value).lower().startswith("omniverse://")
    except Exception:
        return False


def _normalize_payload_path(path: object) -> str:
    text = str(path)
    if _is_omniverse_path(text):
        return text.lower()
    try:
        return str(Path(text).resolve()).lower()
    except Exception:
        return text.lower()


def _prim_custom_data_value(prim, key: str):
    if not prim:
        return None
    try:
        value = prim.GetCustomDataByKey(key)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        data = dict(prim.GetCustomData() or {})
    except Exception:
        return None
    if key in data:
        return data.get(key)
    current = data
    for part in str(key).split(":"):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current.get(part)
    return current


def _existing_payloads(parent_prim) -> Tuple[Dict[str, str], set[str]]:
    existing_payloads: Dict[str, str] = {}
    existing_names: set[str] = set()
    if not parent_prim:
        return existing_payloads, existing_names
    try:
        children = list(parent_prim.GetChildren())
    except Exception:
        children = []
    for child in children:
        name = child.GetName()
        existing_names.add(name)
        payload_path = _prim_custom_data_value(child, "ifc:federation:payloadPath")
        if payload_path:
            existing_payloads[_normalize_payload_path(payload_path)] = name
            continue
        if child.HasAuthoredPayloads():
            payload_items = child.GetPayloads().GetAddedOrExplicitItems() or None
            if payload_items:
                try:
                    existing_payloads[
                        _normalize_payload_path(payload_items[0].assetPath)
                    ] = name
                except Exception:
                    pass
            continue
        if child.HasAuthoredReferences():
            ref_items = child.GetReferences().GetAddedOrExplicitItems() or None
            if ref_items:
                try:
                    existing_payloads[
                        _normalize_payload_path(ref_items[0].assetPath)
                    ] = name
                except Exception:
                    pass
    return existing_payloads, existing_names


def _open_stage_for_anchor(path: str):
    try:
        return Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    except Exception:
        try:
            return Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
        except Exception:
            return Usd.Stage.Open(str(path))


@dataclass(frozen=True)
class GeospatialAnchor:
    prim_path: str
    wgs84: Optional[Tuple[float, float, float]]
    ecef: Optional[Tuple[float, float, float]]


def _resolve_stage(stage_or_layer) -> Optional[Usd.Stage]:
    if stage_or_layer is None:
        return None
    if hasattr(stage_or_layer, "GetPrimAtPath"):
        return stage_or_layer
    if isinstance(stage_or_layer, Sdf.Layer):
        identifier = stage_or_layer.identifier
        if not identifier:
            return None
        return Usd.Stage.Open(identifier)
    if isinstance(stage_or_layer, (str, Path)):
        return Usd.Stage.Open(str(stage_or_layer))
    return None


def _find_geospatial_prim(stage: Usd.Stage, geoprim_path: str):
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(geoprim_path)
    if prim and prim.IsValid():
        return prim
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        candidate = default_prim.GetPath().AppendChild("Geospatial")
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            return prim
    prim = stage.GetPrimAtPath("/World/Geospatial")
    if prim and prim.IsValid():
        return prim
    try:
        for child in stage.GetPseudoRoot().GetChildren():
            if child.GetName() == "Geospatial" and child.IsValid():
                return child
            geo = child.GetChild("Geospatial")
            if geo and geo.IsValid():
                return geo
    except Exception:
        pass
    try:
        root = stage.GetPseudoRoot()
        for prim in Usd.PrimRange(root):
            if (
                prim.GetAttribute(_WGS84_REF_ATTR)
                and prim.GetAttribute(_WGS84_REF_ATTR).HasValue()
            ):
                return prim
    except Exception:
        pass
    return None


def read_site_geospatial_anchor(
    stage_or_layer, geoprim_path: str = "/World/Geospatial"
) -> GeospatialAnchor:
    """Read WGS84/ECEF anchors from a geospatial prim."""
    stage = _resolve_stage(stage_or_layer)
    if stage is None:
        raise ValueError("Stage or layer could not be opened to read geospatial data.")

    prim = _find_geospatial_prim(stage, geoprim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Geospatial prim not found at {geoprim_path}.")

    wgs84 = None
    ecef = None
    attr = prim.GetAttribute(_WGS84_REF_ATTR)
    if attr and attr.HasValue():
        wgs84 = _coerce_vec3(attr.Get())
    else:
        _log_warn(f"WGS84 reference missing on {geoprim_path}.")
    attr = prim.GetAttribute(_ECEF_REF_ATTR)
    if attr and attr.HasValue():
        ecef = _coerce_vec3(attr.Get())
    elif wgs84 is None:
        _log_warn(f"ECEF reference missing on {geoprim_path}.")

    prim_path_value = prim.GetPath().pathString if prim else geoprim_path
    return GeospatialAnchor(prim_path=prim_path_value, wgs84=wgs84, ecef=ecef)


def wgs84_to_ecef(
    lat_deg: float, lon_deg: float, h_m: float
) -> Tuple[float, float, float]:
    """Convert geodetic WGS84 to ECEF (EPSG:4978)."""
    import math

    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 1 - (1 - f) * (1 - f)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1 - e2) + h_m) * sin_lat
    return (x, y, z)


def ecef_to_enu_delta(
    dxyz: Tuple[float, float, float],
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> Tuple[float, float, float]:
    """Rotate ECEF delta into local ENU at origin."""
    import math

    dx, dy, dz = dxyz
    lat = math.radians(origin_lat_deg)
    lon = math.radians(origin_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return (e, n, u)


def enu_to_ecef(
    enu: Tuple[float, float, float],
    origin_lat_deg: float,
    origin_lon_deg: float,
) -> Tuple[float, float, float]:
    """Optional inverse rotation from ENU to ECEF (delta only)."""
    import math

    e, n, u = enu
    lat = math.radians(origin_lat_deg)
    lon = math.radians(origin_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u
    return (dx, dy, dz)


def validate_geodetic_federation_stage(
    stage: Usd.Stage,
    *,
    geoprim_path: str = "/World/Geospatial",
) -> bool:
    if stage is None:
        return False
    geo_prim = stage.GetPrimAtPath(geoprim_path)
    if not geo_prim or not geo_prim.IsValid():
        _log_warn(f"Validation failed: Geospatial prim missing at {geoprim_path}.")
        return False
    geo_attr = geo_prim.GetAttribute(_WGS84_REF_ATTR)
    if not geo_attr or not geo_attr.HasValue():
        _log_warn(f"Validation failed: WGS84 reference missing on {geoprim_path}.")
        return False
    return True


def federate_sites_geodetic(
    federation_stage: Usd.Stage,
    sites: List[Dict],
    *,
    federation_root: str = "/World",
    geoprim_path: str = "/World/Geospatial",
    federation_origin_wgs84: Optional[Tuple[float, float, float]] = None,
    use_site0_as_origin_if_none: bool = True,
    use_payloads: bool = True,
    apply_grid_convergence_rotation: bool = False,
    preserve_existing: bool = True,
) -> None:
    """
    Federate multiple site layers using geodetic ENU deltas.
    """
    if federation_stage is None:
        _log_error("Federation stage is None; aborting.")
        return

    if not sites:
        _log_warn("No sites provided for federation.")
        return

    _log_info("Phase 1/3: initialize geospatial anchor.")
    UsdGeom.Xform.Define(federation_stage, "/World")
    geo_xf = UsdGeom.Xform.Define(federation_stage, geoprim_path)
    geo_prim = geo_xf.GetPrim()

    origin_wgs84 = federation_origin_wgs84
    existing_origin = None
    if preserve_existing:
        geo_attr = geo_prim.GetAttribute(_WGS84_REF_ATTR)
        if geo_attr and geo_attr.HasValue():
            existing_origin = _coerce_vec3(geo_attr.Get())
    if existing_origin is not None:
        if origin_wgs84 is not None and any(
            abs(float(existing_origin[i]) - float(origin_wgs84[i])) > 1.0e-8
            for i in range(3)
        ):
            _log_warn(
                "Federation origin mismatch; keeping existing WGS84 origin "
                f"{existing_origin} and ignoring requested {origin_wgs84}."
            )
        origin_wgs84 = (
            float(existing_origin[0]),
            float(existing_origin[1]),
            float(existing_origin[2]),
        )
    if origin_wgs84 is None and use_site0_as_origin_if_none:
        site0 = sites[0]
        layer_path = site0.get("layer_path")
        if not layer_path:
            _log_error("Site0 missing layer_path; cannot derive federation origin.")
            return
        stage0 = Usd.Stage.Open(str(layer_path))
        if stage0 is None:
            _log_error(f"Unable to open site0 stage: {layer_path}")
            return
        try:
            anchor0 = read_site_geospatial_anchor(
                stage0, site0.get("geoprim_path", geoprim_path)
            )
        except Exception as exc:
            _log_error(f"Failed to read site0 geospatial anchor: {exc}")
            return
        if anchor0.wgs84 is None:
            _log_error("Site0 missing WGS84 anchor; provide federation_origin_wgs84.")
            return
        origin_wgs84 = anchor0.wgs84

    if origin_wgs84 is None:
        _log_error("Federation origin WGS84 not provided and not derivable.")
        return

    if existing_origin is None:
        _ensure_wgs84_attr(geo_prim, origin_wgs84)
    if not validate_geodetic_federation_stage(
        federation_stage, geoprim_path=geoprim_path
    ):
        _log_error("Geospatial prim missing WGS84 reference; aborting.")
        return

    origin_lat, origin_lon, origin_h = origin_wgs84
    origin_ecef = wgs84_to_ecef(origin_lat, origin_lon, origin_h)

    root_path = Sdf.Path(federation_root)
    if not root_path.IsAbsolutePath():
        root_path = Sdf.Path("/World")
    sites_root = UsdGeom.Xform.Define(federation_stage, root_path)

    existing_payload_map, existing_names = _existing_payloads(sites_root.GetPrim())
    used_names: set[str] = set(existing_names)
    seen_payloads: set[str] = set()
    added_count = 0
    skipped_existing_count = 0
    skipped_duplicate_input_count = 0
    skipped_missing_layer_count = 0
    skipped_open_stage_count = 0
    skipped_anchor_count = 0
    for site in sites:
        layer_path = site.get("layer_path")
        if not layer_path:
            name_raw = site.get("name") or ""
            safe_name = _sanitize_name(name_raw)
            _log_warn(f"Skipping site '{safe_name}': missing layer_path.")
            skipped_missing_layer_count += 1
            continue
        normalized_payload = _normalize_payload_path(layer_path)
        if normalized_payload in seen_payloads:
            skipped_duplicate_input_count += 1
            continue
        seen_payloads.add(normalized_payload)
        if preserve_existing and normalized_payload in existing_payload_map:
            skipped_existing_count += 1
            continue

        name_raw = site.get("name") or Path(site.get("layer_path", "")).stem
        safe_name = _sanitize_name(name_raw)
        if safe_name in used_names:
            suffix = 1
            while f"{safe_name}_{suffix}" in used_names:
                suffix += 1
            safe_name = f"{safe_name}_{suffix}"
        used_names.add(safe_name)

        site_stage = _open_stage_for_anchor(str(layer_path))
        if site_stage is None:
            _log_warn(f"Skipping site '{safe_name}': cannot open {layer_path}.")
            skipped_open_stage_count += 1
            continue

        site_geoprim_path = site.get("geoprim_path", geoprim_path)
        try:
            anchor = read_site_geospatial_anchor(site_stage, site_geoprim_path)
        except Exception as exc:
            _log_warn(f"Skipping site '{safe_name}': {exc}")
            skipped_anchor_count += 1
            continue

        site_wgs84 = anchor.wgs84
        site_ecef = anchor.ecef
        if site_ecef is None and site_wgs84 is not None:
            site_ecef = wgs84_to_ecef(site_wgs84[0], site_wgs84[1], site_wgs84[2])
        if site_ecef is None:
            _log_warn(f"Skipping site '{safe_name}': no ECEF/WGS84 anchor.")
            skipped_anchor_count += 1
            continue

        d_ecef = (
            site_ecef[0] - origin_ecef[0],
            site_ecef[1] - origin_ecef[1],
            site_ecef[2] - origin_ecef[2],
        )
        d_enu = ecef_to_enu_delta(d_ecef, origin_lat, origin_lon)

        site_path = sites_root.GetPath().AppendChild(safe_name)
        site_xf = UsdGeom.Xform.Define(federation_stage, site_path)
        site_prim = site_xf.GetPrim()
        if use_payloads:
            site_prim.GetPayloads().AddPayload(str(layer_path))
        else:
            site_prim.GetReferences().AddReference(str(layer_path))

        xf_api = UsdGeom.XformCommonAPI(site_xf)
        xf_api.SetTranslate(Gf.Vec3d(*d_enu))

        if apply_grid_convergence_rotation:
            _log_warn(
                f"Grid convergence rotation not implemented for '{safe_name}'. TODO."
            )

        try:
            site_prim.SetCustomDataByKey("ifc:federation:payloadPath", str(layer_path))
            site_prim.SetCustomDataByKey(
                "ifc:federation:delta",
                {"x": d_enu[0], "y": d_enu[1], "z": d_enu[2]},
            )
            if site_wgs84 is not None:
                site_prim.SetCustomDataByKey(
                    "ifc:federation:wgs84Anchor",
                    {
                        "latitude": site_wgs84[0],
                        "longitude": site_wgs84[1],
                        "height": site_wgs84[2],
                    },
                )
            geo_prim.SetCustomDataByKey(
                f"ifc:federation:enuDelta:{safe_name}",
                {"x": d_enu[0], "y": d_enu[1], "z": d_enu[2]},
            )
        except Exception:
            pass

        added_count += 1

    _log_info(
        "Geodetic federation complete: added={added}, skipped_existing={existing}, "
        "skipped_duplicate_input={dup}, skipped_missing_layer={missing}, "
        "skipped_open_stage={open_stage}, skipped_anchor={anchor}.".format(
            added=added_count,
            existing=skipped_existing_count,
            dup=skipped_duplicate_input_count,
            missing=skipped_missing_layer_count,
            open_stage=skipped_open_stage_count,
            anchor=skipped_anchor_count,
        )
    )


if __name__ == "__main__":
    try:
        stage = Usd.Stage.GetCurrent()
    except Exception:
        stage = None
    if stage is None:
        stage = Usd.Stage.CreateInMemory()

    demo_sites = [
        {"name": "SiteA", "layer_path": "./site_a.usd"},
        {"name": "SiteB", "layer_path": "./site_b.usd"},
    ]
    try:
        federate_sites_geodetic(stage, demo_sites)
    except Exception as exc:
        _log_warn(f"Demo federation skipped: {exc}")
