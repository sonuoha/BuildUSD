import unittest
from unittest.mock import MagicMock
import sys
import types

# Correctly mock package structure
ifcopenshell = types.ModuleType("ifcopenshell")
sys.modules["ifcopenshell"] = ifcopenshell

ifco_util = types.ModuleType("ifcopenshell.util")
sys.modules["ifcopenshell.util"] = ifco_util

ifco_util_elem = types.ModuleType("ifcopenshell.util.element")
sys.modules["ifcopenshell.util.element"] = ifco_util_elem

ifco_util_rep = types.ModuleType("ifcopenshell.util.representation")
sys.modules["ifcopenshell.util.representation"] = ifco_util_rep

# We need to reload or import carefully if it was already imported badly
if "buildusd.ifc_visuals" in sys.modules:
    del sys.modules["buildusd.ifc_visuals"]

from buildusd.ifc_visuals import _styles_from_material_definition, PBRMaterial


class MockEntity:
    def __init__(self, id_val, is_a_type, file_ref=None, **kwargs):
        self._id = id_val
        self._is_a = is_a_type
        self.file = file_ref
        for k, v in kwargs.items():
            setattr(self, k, v)

    def id(self):
        return self._id

    def is_a(self, type_name):
        return self._is_a == type_name or (
            isinstance(type_name, tuple) and self._is_a in type_name
        )


class TestLooseBinding(unittest.TestCase):
    def test_finds_orphaned_style_by_name(self):
        # Setup: Material and Style separate, sharing name "Glass"
        mock_file = MagicMock()

        mat = MockEntity(100, "IfcMaterial", file_ref=mock_file, Name="Glass")
        style = MockEntity(200, "IfcSurfaceStyle", file_ref=mock_file, Name="Glass")

        # Mock file.by_type("IfcSurfaceStyle") returning our orphaned style
        mock_file.by_type.return_value = [style]

        # Ensure HasRepresentation is empty (orphaned)
        mat.HasRepresentation = []

        # Execution
        # _styles_from_material_definition should fall back to global search
        results = _styles_from_material_definition(mat)

        # Assertion
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0], style)

    def test_ignores_mismatched_names(self):
        mock_file = MagicMock()
        mat = MockEntity(100, "IfcMaterial", file_ref=mock_file, Name="Glass")
        style = MockEntity(200, "IfcSurfaceStyle", file_ref=mock_file, Name="Concrete")

        mock_file.by_type.return_value = [style]
        mat.HasRepresentation = []

        # Should NOT match Concrete
        results = _styles_from_material_definition(mat)

        # Should fall back to generated PBR (list length 1, but NOT the style entity)
        self.assertEqual(len(results), 1)
        # It should be the generated PBR, unrelated to style entity
        self.assertNotEqual(results[0].id() if hasattr(results[0], "id") else None, 200)


if __name__ == "__main__":
    unittest.main()
