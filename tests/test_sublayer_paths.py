from pathlib import Path
import sys
import types

# Provide a lightweight ifcopenshell stub so we can import buildusd without the heavy dep.
if "ifcopenshell" not in sys.modules:
    ifcopenshell_stub = types.ModuleType("ifcopenshell")
    ifcopenshell_stub.__path__ = []  # mark as a package for submodule imports
    ifcopenshell_stub.entity_instance = object

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

from buildusd.process_usd import _sublayer_identifier


class _LayerStub:
    def __init__(self, path: Path):
        self.realPath = path


def test_sublayer_identifier_anchors_child_in_same_tree(tmp_path):
    parent_path = tmp_path / "stage.usda"
    child_path = tmp_path / "geometry2d" / "child.usda"
    child_path.parent.mkdir(parents=True, exist_ok=True)
    parent_path.parent.mkdir(parents=True, exist_ok=True)

    parent_layer = _LayerStub(parent_path)
    result = _sublayer_identifier(parent_layer, child_path)

    assert result == "./geometry2d/child.usda"


def test_sublayer_identifier_relativises_sibling_layers(tmp_path):
    parent_dir = tmp_path / "prototypes"
    materials_dir = tmp_path / "materials"
    parent_dir.mkdir(parents=True, exist_ok=True)
    materials_dir.mkdir(parents=True, exist_ok=True)

    parent_path = parent_dir / "model.usdc"
    child_path = materials_dir / "materials.usdc"

    parent_layer = _LayerStub(parent_path)
    result = _sublayer_identifier(parent_layer, child_path)

    assert result == "../materials/materials.usdc"
