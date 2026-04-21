import sys
import types

# Stub ifcopenshell to avoid heavy dependency during unit tests.
if "ifcopenshell" not in sys.modules:
    ifcopenshell_stub = types.ModuleType("ifcopenshell")
    ifcopenshell_stub.__path__ = []
    ifc_util = types.ModuleType("ifcopenshell.util")
    ifc_util.__path__ = []
    ifc_util.element = types.ModuleType("ifcopenshell.util.element")
    ifc_util.representation = types.ModuleType("ifcopenshell.util.representation")
    ifc_geom = types.ModuleType("ifcopenshell.geom")

    sys.modules["ifcopenshell"] = ifcopenshell_stub
    sys.modules["ifcopenshell.util"] = ifc_util
    sys.modules["ifcopenshell.util.element"] = ifc_util.element
    sys.modules["ifcopenshell.util.representation"] = ifc_util.representation
    sys.modules["ifcopenshell.geom"] = ifc_geom

    ifcopenshell_stub.util = ifc_util
    ifcopenshell_stub.geom = ifc_geom

from buildusd.resolve_frame import classify_ifc_geometry_space, compute_model_offset


class _Coords:
    def __init__(self, xyz):
        self.Coordinates = xyz


class _RelPlacement:
    def __init__(self, xyz):
        self.Location = _Coords(xyz)


class _Placement:
    def __init__(self, xyz):
        self.RelativePlacement = _RelPlacement(xyz)


class _Product:
    def __init__(self, xyz):
        self.ObjectPlacement = _Placement(xyz)


class _IfcStub:
    def __init__(self, products):
        self._products = products

    def by_type(self, _):
        return list(self._products)


def test_classify_geometry_space_local():
    ifc = _IfcStub([_Product((10.0, 20.0, 0.0)), _Product((50.0, -5.0, 0.0))])
    result = classify_ifc_geometry_space(ifc, global_threshold_m=1.0e5, sample_limit=10)
    assert result.space == "local"
    assert result.sample_count == 2


def test_classify_geometry_space_global():
    ifc = _IfcStub([_Product((333800.0, 5809101.0, 0.0))])
    result = classify_ifc_geometry_space(ifc, global_threshold_m=1.0e5, sample_limit=10)
    assert result.space == "global"
    assert result.sample_count == 1


def test_compute_model_offset_local():
    offset = compute_model_offset(
        geom_space="local",
        anchor_mode="basepoint",
        pbp=(100.0, 200.0, 0.0),
        shared_site=(1000.0, 2000.0, 0.0),
        bp_to_m=lambda bp: bp,
    )
    assert offset == (0.0, 0.0, 0.0)


def test_compute_model_offset_global():
    offset = compute_model_offset(
        geom_space="global",
        anchor_mode="basepoint",
        pbp=(100.0, 200.0, 0.0),
        shared_site=(1000.0, 2000.0, 0.0),
        bp_to_m=lambda bp: bp,
    )
    assert offset == (100.0, 200.0, 0.0)
