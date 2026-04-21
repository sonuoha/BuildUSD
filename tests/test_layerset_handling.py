import unittest
from unittest.mock import MagicMock
import sys
import types

# Correctly mock package structure BEFORE importing target
ifcopenshell = types.ModuleType("ifcopenshell")
sys.modules["ifcopenshell"] = ifcopenshell

ifco_util = types.ModuleType("ifcopenshell.util")
sys.modules["ifcopenshell.util"] = ifco_util
ifcopenshell.util = ifco_util

ifco_util_elem = types.ModuleType("ifcopenshell.util.element")
# Define the function on the module
ifco_util_elem.get_type = MagicMock(return_value=None)
sys.modules["ifcopenshell.util.element"] = ifco_util_elem
ifcopenshell.util.element = ifco_util_elem

ifco_util_rep = types.ModuleType("ifcopenshell.util.representation")
sys.modules["ifcopenshell.util.representation"] = ifco_util_rep
ifcopenshell.util.representation = ifco_util_rep

if "buildusd.ifc_visuals" in sys.modules:
    del sys.modules["buildusd.ifc_visuals"]
from buildusd.ifc_visuals import _constituent_styles_by_name


class MockEntity:
    def __init__(self, id_val, is_a_type, **kwargs):
        self._id = id_val
        self._is_a = is_a_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    def id(self):
        return self._id

    def is_a(self, t):
        return self._is_a == t or (isinstance(t, tuple) and self._is_a in t)


class TestLayerSetHandling(unittest.TestCase):
    def test_collects_layer_styles(self):
        # Setup: Product -> RelAssociates -> LayerSetUsage -> LayerSet -> Layer -> Material
        mock_mat = MockEntity(
            10,
            "IfcMaterial",
            Name="Brick",
            HasRepresentation=[
                MockEntity(
                    11,
                    "IfcMaterialDefinitionRepresentation",
                    Representations=[
                        MockEntity(
                            12,
                            "IfcStyledRepresentation",
                            Items=[
                                MockEntity(
                                    13,
                                    "IfcStyledItem",
                                    Styles=[
                                        MockEntity(
                                            14,
                                            "IfcPresentationStyleAssignment",
                                            Styles=[
                                                MockEntity(
                                                    15,
                                                    "IfcSurfaceStyle",
                                                    Name="BrickStyle",
                                                )
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        mock_layer = MockEntity(
            20, "IfcMaterialLayer", Material=mock_mat, Name="LayerName"
        )
        mock_lset = MockEntity(30, "IfcMaterialLayerSet", MaterialLayers=[mock_layer])
        mock_usage = MockEntity(40, "IfcMaterialLayerSetUsage", ForLayerSet=mock_lset)

        mock_rel = MockEntity(
            50, "IfcRelAssociatesMaterial", RelatingMaterial=mock_usage
        )
        mock_product = MockEntity(60, "IfcWall", HasAssociations=[mock_rel])

        # Ensure get_type returns None (no type product)
        ifcopenshell.util.element.get_type.return_value = None

        # Execution
        mapping = _constituent_styles_by_name(mock_product)

        # Assertions
        # Expecting key "LayerName"
        self.assertIn("LayerName", mapping)
        self.assertEqual(len(mapping["LayerName"]), 1)
        self.assertEqual(mapping["LayerName"][0].Name, "BrickStyle")


if __name__ == "__main__":
    unittest.main()
