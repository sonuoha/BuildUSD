import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom

from buildusd.config.manifest import BasePointConfig, GeodeticCoordinate
from buildusd.process_usd import assign_world_geolocation


def test_assign_world_geolocation_stamps_projected_anchor_without_lonlat(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    monkeypatch.setattr(
        "buildusd.process_usd._derive_lonlat_from_projected_m",
        lambda projected_xyz_m, projected_crs: None,
    )

    result = assign_world_geolocation(
        stage,
        base_point=BasePointConfig(
            easting=338294.0,
            northing=5805921.0,
            height=0.0,
            unit="m",
            epsg="EPSG:7855",
        ),
        projected_crs="EPSG:7855",
        geodetic_crs="EPSG:4326",
        geospatial_mode="usd",
        offline=True,
        anchor_mode="local",
    )

    assert result is None

    world = stage.GetPrimAtPath("/World")
    layer_data = dict(stage.GetRootLayer().customLayerData or {})

    assert world.GetCustomDataByKey("ifc:projectedCRS") == "EPSG:7855"
    assert world.GetCustomDataByKey("ifc:geodeticCRS") == "EPSG:4326"
    assert world.GetCustomDataByKey("ifc:anchorProjected")["easting"] == 338294.0
    assert world.GetCustomDataByKey("ifc:anchorProjected")["northing"] == 5805921.0
    assert world.GetCustomDataByKey("ifc:stageOriginProjected")["easting"] == 338294.0
    assert world.GetCustomDataByKey("ifc:stageOriginProjected")["northing"] == 5805921.0
    assert layer_data["projectedCRS"] == "EPSG:7855"
    assert layer_data["geodeticCRS"] == "EPSG:4326"
    assert layer_data["anchorProjected"]["easting"] == 338294.0
    assert layer_data["stageOriginProjected"]["easting"] == 338294.0


def test_assign_world_geolocation_uses_stage_origin_for_geodetic_reference(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage_origin = (338294.0, 5805921.0, 12.5)
    calls = []

    def _fake_stage_origin_reproject(projected_xyz_m, projected_crs):
        calls.append((projected_xyz_m, projected_crs))
        assert projected_xyz_m == stage_origin
        assert projected_crs == "EPSG:7855"
        return GeodeticCoordinate(longitude=144.9631, latitude=-37.8136, height=12.5)

    monkeypatch.setattr(
        "buildusd.process_usd._derive_lonlat_from_projected_m",
        _fake_stage_origin_reproject,
    )

    result = assign_world_geolocation(
        stage,
        base_point=BasePointConfig(
            easting=100.0,
            northing=200.0,
            height=0.0,
            unit="m",
            epsg="EPSG:7855",
        ),
        projected_crs="EPSG:7855",
        geodetic_crs="EPSG:4326",
        geospatial_mode="usd",
        offline=True,
        anchor_mode="local",
        stage_origin_projected_m=stage_origin,
    )

    assert calls == [(stage_origin, "EPSG:7855")]
    assert result == (144.9631, -37.8136, 12.5)

    world = stage.GetPrimAtPath("/World")
    world_stage_origin_geodetic = world.GetCustomDataByKey("ifc:stageOriginGeodetic")
    assert world_stage_origin_geodetic["lon"] == 144.9631
    assert world_stage_origin_geodetic["lat"] == -37.8136
    assert world_stage_origin_geodetic["height"] == 12.5

    geo = stage.GetPrimAtPath("/World/Geospatial")
    assert geo
    assert geo.GetCustomDataByKey("ifc:stageOriginProjected")["easting"] == 338294.0
    assert geo.GetCustomDataByKey("ifc:stageOriginGeodetic")["lon"] == 144.9631
    ref_attr = geo.GetAttribute("omni:geospatial:wgs84:reference:referencePosition")
    assert ref_attr
    assert tuple(ref_attr.Get()) == (-37.8136, 144.9631, 12.5)
