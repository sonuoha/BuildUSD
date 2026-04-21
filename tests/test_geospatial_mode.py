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

from buildusd.geospatial import resolve_geospatial_mode


def test_resolve_offline_forces_usd():
    assert resolve_geospatial_mode("omni", offline=True) == "usd"


def test_resolve_auto_prefers_omni_for_nucleus():
    assert (
        resolve_geospatial_mode(
            "auto", offline=False, output_path="omniverse://server/stage.usda"
        )
        == "omni"
    )


def test_resolve_auto_defaults_usd_for_local():
    assert (
        resolve_geospatial_mode("auto", offline=False, output_path="C:/tmp/out.usda")
        == "usd"
    )
