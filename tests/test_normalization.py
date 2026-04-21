import unittest
from unittest.mock import MagicMock
import sys
import logging

# Mock imports
sys.modules["ifcopenshell"] = MagicMock()
sys.modules["ifcopenshell.util"] = MagicMock()
sys.modules["ifcopenshell.util.element"] = MagicMock()
sys.modules["ifcopenshell.util.representation"] = MagicMock()

from buildusd.ifc_visuals import PBRMaterial

# We need to access the function which is internal.
# It is not in __all__, so we import module
import buildusd.process_ifc as process_ifc


class TestMaterialNormalization(unittest.TestCase):
    def test_normalize_preserves_named_styles(self):
        # Setup
        mat_a = PBRMaterial(name="MatA")
        mat_b = PBRMaterial(name="MatB")
        mat_glass = PBRMaterial(name="Glass")  # Constituent style

        materials_list = [mat_a, mat_b]

        # Constituent typically has no style_id
        face_style_groups = {
            "some_group": {"material": mat_glass, "style_id": None, "faces": [1, 2, 3]}
        }

        ifc_file_mock = MagicMock()

        # ACT
        result = process_ifc._normalize_geom_materials(
            materials_list, face_style_groups, ifc_file_mock
        )

        # ASSERT
        self.assertIsInstance(result, dict)

        # Index lookups
        self.assertEqual(result.get(0), mat_a)
        self.assertEqual(result.get(1), mat_b)

        # Named lookup
        self.assertEqual(result.get("Glass"), mat_glass)

        # Name lookup for indexed items (bonus feature implemented)
        self.assertEqual(result.get("MatA"), mat_a)


if __name__ == "__main__":
    unittest.main()
