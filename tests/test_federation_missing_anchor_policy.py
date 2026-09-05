from pathlib import Path

import pytest

pytest.importorskip("pxr")

from pxr import Usd, UsdGeom

from buildusd.config.manifest import BasePointConfig
from buildusd.federation_builder import build_federated_stage
from buildusd.usd_context import initialize_usd


def _make_unanchored_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    stage.GetRootLayer().Save()


def _payload_children(stage: Usd.Stage):
    world = stage.GetPrimAtPath("/World")
    return [
        child
        for child in world.GetChildren()
        if child.GetPath().pathString != "/World/Geospatial"
    ]


def test_federation_skips_unanchored_payloads_by_default(tmp_path):
    initialize_usd(offline=True)
    payload_a = tmp_path / "a.usda"
    payload_b = tmp_path / "b.usda"
    _make_unanchored_stage(payload_a)
    _make_unanchored_stage(payload_b)

    out_stage = tmp_path / "master.usda"
    build_federated_stage(
        [str(payload_a), str(payload_b)],
        str(out_stage),
        federation_origin=BasePointConfig(
            easting=0.0, northing=0.0, height=0.0, unit="m", epsg="EPSG:7855"
        ),
        federation_projected_crs="EPSG:7855",
    )

    stage = Usd.Stage.Open(str(out_stage))
    assert _payload_children(stage) == []


def test_same_origin_policy_payloads_unanchored_stages_at_zero_offset(tmp_path):
    initialize_usd(offline=True)
    payload_a = tmp_path / "a.usda"
    payload_b = tmp_path / "b.usda"
    _make_unanchored_stage(payload_a)
    _make_unanchored_stage(payload_b)

    out_stage = tmp_path / "master.usda"
    build_federated_stage(
        [str(payload_a), str(payload_b)],
        str(out_stage),
        federation_origin=BasePointConfig(
            easting=0.0, northing=0.0, height=0.0, unit="m", epsg="EPSG:7855"
        ),
        federation_projected_crs="EPSG:7855",
        missing_anchor_policy="same-origin",
    )

    stage = Usd.Stage.Open(str(out_stage))
    children = _payload_children(stage)
    assert len(children) == 2
    for child in children:
        assert child.GetCustomDataByKey("ifc:federation:anchorSource") == (
            "same-origin-fallback"
        )
        ops = UsdGeom.Xformable(child).GetOrderedXformOps()
        assert len(ops) == 1
        assert tuple(ops[0].Get()) == (0.0, 0.0, 0.0)


def test_same_origin_policy_supports_geodetic_federation(tmp_path):
    initialize_usd(offline=True)
    payload = tmp_path / "a.usda"
    _make_unanchored_stage(payload)

    out_stage = tmp_path / "master.usda"
    build_federated_stage(
        [str(payload)],
        str(out_stage),
        federation_origin=BasePointConfig(
            easting=334000.0,
            northing=6250000.0,
            height=10.0,
            unit="m",
            epsg="EPSG:7855",
        ),
        federation_projected_crs="EPSG:7855",
        geodetic_crs="EPSG:4326",
        frame="geodetic",
        missing_anchor_policy="same-origin",
    )

    stage = Usd.Stage.Open(str(out_stage))
    children = _payload_children(stage)
    assert len(children) == 1
    ops = UsdGeom.Xformable(children[0]).GetOrderedXformOps()
    assert len(ops) == 1
    assert tuple(ops[0].Get()) == (0.0, 0.0, 0.0)
