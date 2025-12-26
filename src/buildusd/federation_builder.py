from __future__ import annotations

"""
Federate existing USD stages into a single master stage.

Example:
  from buildusd.federation_builder import build_federated_stage
  build_federated_stage(
      payload_paths=["/path/site_a.usdc", "/path/site_b.usdc"],
      out_stage_path="/path/master_federated.usda",
      federation_origin=BasePointConfig(easting=335700.49, northing=5807351.468, height=0.0, unit="m", epsg="EPSG:7855"),
      federation_projected_crs="EPSG:7855",
  )
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from .config.manifest import BasePointConfig
from .geospatial import ensure_geospatial_root, _author_wgs84_reference_attrs_on_prim
from .pxr_utils import Gf, Sdf, Usd, UsdGeom
from .process_usd import sanitize_name, _unit_factor

LOG = logging.getLogger(__name__)

_WGS84_REF_POS_ATTR = "omni:geospatial:wgs84:reference:referencePosition"

try:
    from pyproj import CRS, Transformer

    _HAVE_PYPROJ = True
    _TRANSFORMER_CACHE: Dict[Tuple[str, str], Transformer] = {}
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYPROJ = False
    _TRANSFORMER_CACHE: Dict[Tuple[str, str], Any] = {}


@dataclass(frozen=True)
class AnchorInfo:
    projected_crs: Optional[str]
    anchor_e: Optional[float]
    anchor_n: Optional[float]
    anchor_h: Optional[float]
    anchor_mode: Optional[str]
    source: str


def _coerce_vec3(value: Any) -> Optional[Tuple[float, float, float]]:
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


def _coerce_anchor_dict(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    return value


def _anchor_from_dict(payload: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    if payload is None:
        return None
    keys = payload.keys()
    if {"easting", "northing"}.issubset(keys):
        e = payload.get("easting")
        n = payload.get("northing")
        h = payload.get("height", 0.0)
    elif {"x", "y"}.issubset(keys):
        e = payload.get("x")
        n = payload.get("y")
        h = payload.get("z", 0.0)
    else:
        vec = _coerce_vec3(payload)
        if vec:
            return vec
        return None
    try:
        unit = payload.get("unit") or "m"
        factor = _unit_factor(unit)
        return (float(e) * factor, float(n) * factor, float(h) * factor)
    except Exception:
        return None


def _projected_to_wgs84(
    easting: float,
    northing: float,
    height: float,
    projected_crs: str,
) -> Optional[Tuple[float, float, float]]:
    if not _HAVE_PYPROJ:
        return None
    try:
        key = (projected_crs, "EPSG:4326")
        transformer = _TRANSFORMER_CACHE.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(
                CRS.from_user_input(projected_crs),
                CRS.from_epsg(4326),
                always_xy=True,
            )
            _TRANSFORMER_CACHE[key] = transformer
        lon, lat = transformer.transform(float(easting), float(northing))
        return (float(lat), float(lon), float(height))
    except Exception as exc:
        LOG.warning("Failed to reproject to WGS84 (%s): %s", projected_crs, exc)
        return None


def _reproject(
    easting: float,
    northing: float,
    height: float,
    src_crs: str,
    dst_crs: str,
) -> Optional[Tuple[float, float, float]]:
    if not _HAVE_PYPROJ:
        return None
    try:
        key = (src_crs, dst_crs)
        transformer = _TRANSFORMER_CACHE.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(
                CRS.from_user_input(src_crs),
                CRS.from_user_input(dst_crs),
                always_xy=True,
            )
            _TRANSFORMER_CACHE[key] = transformer
        x, y = transformer.transform(float(easting), float(northing))
        return (float(x), float(y), float(height))
    except Exception as exc:
        LOG.warning("Failed to reproject %s -> %s: %s", src_crs, dst_crs, exc)
        return None


def _extract_anchor_from_layer(layer: Sdf.Layer) -> Optional[AnchorInfo]:
    data = dict(getattr(layer, "customLayerData", {}) or {})
    anchor_mode = data.get("anchorMode") or data.get("ifc:anchorMode")
    projected_crs = data.get("projectedCRS") or data.get("ifc:projectedCRS")

    anchor_payload = _coerce_anchor_dict(data.get("anchorProjected"))
    if anchor_payload:
        anchor = _anchor_from_dict(anchor_payload)
        if anchor:
            return AnchorInfo(
                projected_crs=anchor_payload.get("epsg") or projected_crs,
                anchor_e=anchor[0],
                anchor_n=anchor[1],
                anchor_h=anchor[2],
                anchor_mode=anchor_mode,
                source="layer.anchorProjected",
            )

    model_offset = _coerce_anchor_dict(data.get("modelOffset"))
    model_offset_type = data.get("modelOffsetType")
    if anchor_mode and model_offset and model_offset_type:
        anchor = _anchor_from_dict(model_offset)
        if anchor:
            return AnchorInfo(
                projected_crs=projected_crs,
                anchor_e=anchor[0],
                anchor_n=anchor[1],
                anchor_h=anchor[2],
                anchor_mode=anchor_mode,
                source="layer.modelOffset",
            )

    geodetic = _coerce_anchor_dict(data.get("geodeticCoordinates"))
    if geodetic and projected_crs:
        try:
            lon = float(geodetic.get("lon"))
            lat = float(geodetic.get("lat"))
            height = float(geodetic.get("height", 0.0))
        except Exception:
            lon = lat = None
            height = 0.0
        if lon is not None and lat is not None:
            if _HAVE_PYPROJ:
                reproj = _reproject(lon, lat, height, "EPSG:4326", projected_crs)
                if reproj:
                    return AnchorInfo(
                        projected_crs=projected_crs,
                        anchor_e=reproj[0],
                        anchor_n=reproj[1],
                        anchor_h=reproj[2],
                        anchor_mode=anchor_mode,
                        source="layer.geodeticCoordinates",
                    )
            else:
                LOG.warning("pyproj unavailable; cannot reproject geodetic anchor.")

    return None


def _extract_anchor_from_world(stage: Usd.Stage) -> Optional[AnchorInfo]:
    world = stage.GetPrimAtPath("/World")
    if not world:
        return None
    data = dict(world.GetCustomData() or {})
    anchor_mode = data.get("ifc:anchorMode")
    projected_crs = data.get("ifc:projectedCRS")

    anchor_payload = _coerce_anchor_dict(data.get("ifc:anchorProjected"))
    if anchor_payload:
        anchor = _anchor_from_dict(anchor_payload)
        if anchor:
            return AnchorInfo(
                projected_crs=anchor_payload.get("epsg") or projected_crs,
                anchor_e=anchor[0],
                anchor_n=anchor[1],
                anchor_h=anchor[2],
                anchor_mode=anchor_mode,
                source="world.anchorProjected",
            )

    model_offset = _coerce_anchor_dict(data.get("ifc:modelOffset"))
    model_offset_type = data.get("ifc:modelOffsetType")
    if anchor_mode and model_offset and model_offset_type:
        anchor = _anchor_from_dict(model_offset)
        if anchor:
            return AnchorInfo(
                projected_crs=projected_crs,
                anchor_e=anchor[0],
                anchor_n=anchor[1],
                anchor_h=anchor[2],
                anchor_mode=anchor_mode,
                source="world.modelOffset",
            )

    attr = world.GetAttribute("ifc:modelOffset")
    if anchor_mode and attr and attr.HasValue():
        vec = _coerce_vec3(attr.Get())
        if vec:
            return AnchorInfo(
                projected_crs=projected_crs,
                anchor_e=vec[0],
                anchor_n=vec[1],
                anchor_h=vec[2],
                anchor_mode=anchor_mode,
                source="world.modelOffsetAttr",
            )

    return None


def _extract_anchor_from_geospatial(stage: Usd.Stage) -> Optional[AnchorInfo]:
    geo = stage.GetPrimAtPath("/World/Geospatial")
    if not geo:
        return None
    attr = geo.GetAttribute(_WGS84_REF_POS_ATTR)
    if not attr or not attr.HasValue():
        return None
    vec = _coerce_vec3(attr.Get())
    if vec is None:
        return None
    lat, lon, height = vec[0], vec[1], vec[2]
    if not _HAVE_PYPROJ:
        LOG.warning("pyproj unavailable; cannot reproject WGS84 anchor.")
        return None
    return AnchorInfo(
        projected_crs="EPSG:4326",
        anchor_e=float(lon),
        anchor_n=float(lat),
        anchor_h=float(height),
        anchor_mode=None,
        source="geospatial.referencePosition",
    )


def extract_payload_anchor(stage: Usd.Stage) -> Optional[AnchorInfo]:
    if stage is None:
        return None
    layer = stage.GetRootLayer()
    info = _extract_anchor_from_layer(layer)
    if info:
        return info
    info = _extract_anchor_from_world(stage)
    if info:
        return info
    return _extract_anchor_from_geospatial(stage)


def _unique_names(items: Iterable[str]) -> list[str]:
    seen: Dict[str, int] = {}
    results: list[str] = []
    for raw in items:
        base = sanitize_name(raw, fallback="Payload")
        count = seen.get(base, 0)
        if count == 0:
            name = base
        else:
            name = f"{base}_{count}"
        seen[base] = count + 1
        results.append(name)
    return results


def build_federated_stage(
    payload_paths: Sequence[str],
    out_stage_path: str,
    federation_origin: BasePointConfig,
    federation_projected_crs: str,
    geodetic_crs: str = "EPSG:4326",
    *,
    federation_name: str = "Federation",
    parent_prim_path: str = "/World",
    use_payloads: bool = True,
) -> None:
    if not payload_paths:
        raise ValueError("No payload paths provided for federation.")

    stage = Usd.Stage.CreateNew(out_stage_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    if parent_prim_path != "/World":
        LOG.warning(
            "Federation parent prim forced to /World (requested %s).", parent_prim_path
        )
        parent_prim_path = "/World"
    parent_path = Sdf.Path(parent_prim_path or "/World")
    if not parent_path:
        parent_path = Sdf.Path("/World")
    world = UsdGeom.Xform.Define(stage, parent_path).GetPrim()
    stage.SetDefaultPrim(world)

    geo = ensure_geospatial_root(stage, f"{parent_path.pathString}/Geospatial")
    origin_factor = _unit_factor(federation_origin.unit or "m")
    origin_e = float(federation_origin.easting) * origin_factor
    origin_n = float(federation_origin.northing) * origin_factor
    origin_h = float(federation_origin.height) * origin_factor
    wgs84 = _projected_to_wgs84(
        origin_e,
        origin_n,
        origin_h,
        federation_origin.epsg or federation_projected_crs,
    )
    if wgs84:
        _author_wgs84_reference_attrs_on_prim(
            geo.GetPrim(),
            latitude=wgs84[0],
            longitude=wgs84[1],
            altitude_m=wgs84[2],
            tangent_plane="ENU",
            orientation_ypr=(0.0, 0.0, 0.0),
        )

    try:
        world.SetCustomDataByKey("ifc:projectedCRS", federation_projected_crs)
        world.SetCustomDataByKey("ifc:geodeticCRS", geodetic_crs)
        world.SetCustomDataByKey(
            "ifc:federationOrigin",
            {
                "easting": origin_e,
                "northing": origin_n,
                "height": origin_h,
                "epsg": federation_projected_crs,
                "unit": "m",
            },
        )
    except Exception:
        pass
    try:
        layer = stage.GetRootLayer()
        data = dict(getattr(layer, "customLayerData", {}) or {})
        data.update(
            {
                "projectedCRS": federation_projected_crs,
                "geodeticCRS": geodetic_crs,
                "federationOrigin": {
                    "easting": origin_e,
                    "northing": origin_n,
                    "height": origin_h,
                    "epsg": federation_projected_crs,
                    "unit": "m",
                },
                "federationVersion": 1,
            }
        )
        layer.customLayerData = data
    except Exception:
        pass

    federation_path = parent_path.AppendChild(sanitize_name(federation_name))
    federation_root = UsdGeom.Xform.Define(stage, federation_path)

    payload_names = _unique_names([Path(p).stem for p in payload_paths])
    for payload_path, payload_name in zip(payload_paths, payload_names):
        payload_stage = Usd.Stage.Open(str(payload_path))
        anchor = extract_payload_anchor(payload_stage)
        if not anchor or anchor.anchor_e is None or anchor.anchor_n is None:
            LOG.warning("Skipping %s: no anchor metadata found.", payload_path)
            continue

        anchor_crs = anchor.projected_crs
        if anchor_crs and anchor_crs != federation_projected_crs:
            reproj = _reproject(
                anchor.anchor_e,
                anchor.anchor_n,
                anchor.anchor_h or 0.0,
                anchor_crs,
                federation_projected_crs,
            )
            if reproj is None:
                LOG.warning(
                    "Skipping %s: unable to reproject %s -> %s.",
                    payload_path,
                    anchor_crs,
                    federation_projected_crs,
                )
                continue
            anchor_e, anchor_n, anchor_h = reproj
        elif anchor_crs is None:
            LOG.warning(
                "Skipping %s: anchor CRS missing and no WGS84 fallback.",
                payload_path,
            )
            continue
        else:
            anchor_e = anchor.anchor_e
            anchor_n = anchor.anchor_n
            anchor_h = anchor.anchor_h or 0.0

        delta = (anchor_e - origin_e, anchor_n - origin_n, anchor_h - origin_h)
        LOG.info(
            "Federate %s: anchor=%s (%s) source=%s delta=%s",
            payload_path,
            (anchor_e, anchor_n, anchor_h),
            anchor_crs or anchor.projected_crs,
            anchor.source,
            delta,
        )

        wrapper_path = federation_path.AppendChild(payload_name)
        wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
        wrapper_op = wrapper.AddTranslateOp()
        wrapper_op.Set(Gf.Vec3d(*delta))
        try:
            wrapper.GetPrim().SetCustomDataByKey(
                "ifc:federation:payloadPath", str(payload_path)
            )
            wrapper.GetPrim().SetCustomDataByKey(
                "ifc:federation:delta",
                {"x": delta[0], "y": delta[1], "z": delta[2]},
            )
            wrapper.GetPrim().SetCustomDataByKey(
                "ifc:federation:anchor",
                {"easting": anchor_e, "northing": anchor_n, "height": anchor_h},
            )
            wrapper.GetPrim().SetCustomDataByKey(
                "ifc:federation:projectedCRS", federation_projected_crs
            )
        except Exception:
            pass

        payload_prim_path = wrapper_path.AppendChild(f"{payload_name}_Payload")
        payload_prim = stage.DefinePrim(payload_prim_path, "Xform")
        if use_payloads:
            payload_prim.GetPayloads().AddPayload(str(payload_path))
        else:
            payload_prim.GetReferences().AddReference(str(payload_path))
        UsdGeom.Imageable(payload_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)

    stage.GetRootLayer().Save()


def validate_federation(stage_path: str) -> bool:
    stage = Usd.Stage.Open(stage_path)
    if not stage:
        return False
    world = stage.GetPrimAtPath("/World")
    if not world:
        return False
    geo = stage.GetPrimAtPath("/World/Geospatial")
    if not geo:
        return False
    attr = geo.GetAttribute(_WGS84_REF_POS_ATTR)
    if not attr or not attr.HasValue():
        return False
    fed = stage.GetPrimAtPath("/World/Federation")
    if not fed:
        return False
    for child in fed.GetChildren():
        xf = UsdGeom.Xformable(child)
        if not xf.GetOrderedXformOps():
            return False
        payload_child = None
        for sub in child.GetChildren():
            if sub.HasAuthoredPayloads() or sub.HasAuthoredReferences():
                payload_child = sub
                break
        if payload_child is None:
            return False
    return True
