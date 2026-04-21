import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock ifcopenshell before importing buildusd
mock_ifc = MagicMock()
sys.modules["ifcopenshell"] = mock_ifc
sys.modules["ifcopenshell.util"] = MagicMock()
sys.modules["ifcopenshell.util.element"] = MagicMock()
sys.modules["ifcopenshell.util.representation"] = MagicMock()

# Ensure we can import buildusd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from buildusd.process_ifc import IFCMaterialResolver


class TestIFCMaterialResolver(unittest.TestCase):
    def setUp(self):
        self.mock_ifc_file = MagicMock()
        self.mock_product = MagicMock()
        self.mock_type_product = MagicMock()
        self.style_token_map = {100: "Token_100", 200: "Token_200", 300: "Token_300"}

        # Setup common mock behavior
        self.mock_product.id.return_value = 1
        self.mock_type_product.id.return_value = 2

        # By default no associations
        self.mock_product.HasAssociations = []
        self.mock_type_product.HasAssociations = []

    def create_resolver(self, **kwargs):
        defaults = {
            "ifc_file": self.mock_ifc_file,
            "product": self.mock_product,
            "style_token_by_style_id": self.style_token_map,
            "has_instance_style": True,
            "default_material_key": "DEFAULT",
            "type_product": self.mock_type_product,
            "face_style_groups": None,
            "item_material_token_map": None,
            "item_material_id_map": None,
        }
        defaults.update(kwargs)
        return IFCMaterialResolver(**defaults)

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_explicit_item_style(self, mock_aspect, mock_cons, mock_get_face_styles):
        # Setup: Item 50 has explicit style 100
        mock_get_face_styles.return_value = {50: [MagicMock(id=lambda: 100)]}
        mock_aspect.return_value = {}
        mock_cons.return_value = {}

        resolver = self.create_resolver()
        item = MagicMock(id=lambda: 50)
        item.Name = None  # Prevent MagicMock from autocreating Name

        self.assertEqual(resolver.resolve(item), "Token_100")

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_product_constituent_match(
        self, mock_aspect, mock_cons, mock_get_face_styles
    ):
        # Setup: Item 50 has aspect "Glass", Product has "Glass" -> Style 200
        mock_get_face_styles.return_value = {}
        mock_aspect.return_value = {50: "Glass"}

        def cons_side_effect(p):
            if p == self.mock_product:
                return {"Glass": [MagicMock(id=lambda: 200)]}
            return {}

        mock_cons.side_effect = cons_side_effect

        resolver = self.create_resolver()
        item = MagicMock(id=lambda: 50)
        item.Name = None

        self.assertEqual(resolver.resolve(item), "Token_200")

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_type_constituent_match(self, mock_aspect, mock_cons, mock_get_face_styles):
        # Setup: Item 50 has aspect "Glass", Product has NONE, Type has "Glass" -> Style 300
        mock_get_face_styles.return_value = {}
        mock_aspect.return_value = {50: "Glass"}

        def cons_side_effect(p):
            if p == self.mock_type_product:
                return {"Glass": [MagicMock(id=lambda: 300)]}
            return {}

        mock_cons.side_effect = cons_side_effect

        resolver = self.create_resolver()
        item = MagicMock(id=lambda: 50)
        item.Name = None

        self.assertEqual(resolver.resolve(item), "Token_300")

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_real_instance_material_override(
        self, mock_aspect, mock_cons, mock_get_face_styles
    ):
        # Setup: Real Material on Product. Item 50 matches Type Constituent.
        # Expect: __style__instance (Product overrides Type Constituent for unmatched parts... wait)
        # IF Item matches Aspect, and Product does NOT match aspect.
        # Does Product Material override Type Constituent?
        # Yes, Instance Material Association (Global) overrides Type (Specific).
        # (Though debatable, my implementation enforces this: Real Instance > Type Constituent).

        mock_get_face_styles.return_value = {}
        mock_aspect.return_value = {50: "Glass"}

        def cons_side_effect(p):
            if p == self.mock_type_product:
                return {"Glass": [MagicMock(id=lambda: 300)]}
            return {}

        mock_cons.side_effect = cons_side_effect

        # Add real association to product
        assoc = MagicMock()
        assoc.is_a.return_value = True  # is_a("IfcRelAssociatesMaterial")
        assoc.is_a.side_effect = lambda x: x == "IfcRelAssociatesMaterial"
        self.mock_product.HasAssociations = [assoc]

        resolver = self.create_resolver()
        item = MagicMock(id=lambda: 50)
        item.Name = None

        # Should return instance fallback because global instance material overrides type specifics
        self.assertEqual(resolver.resolve(item), "__style__instance")

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_real_type_fall_back(self, mock_aspect, mock_cons, mock_get_face_styles):
        # Setup: No Instance Material. No Constituents. Real Type Material.
        mock_get_face_styles.return_value = {}
        mock_aspect.return_value = {}
        mock_cons.return_value = {}

        # Add real association to TYPE product
        assoc = MagicMock()
        assoc.is_a.side_effect = lambda x: x == "IfcRelAssociatesMaterial"
        self.mock_type_product.HasAssociations = [assoc]

        resolver = self.create_resolver()
        item = MagicMock(id=lambda: 50)
        item.Name = None

        self.assertEqual(resolver.resolve(item), "__style__prototype")

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_instance_fallback_default(
        self, mock_aspect, mock_cons, mock_get_face_styles
    ):
        # Setup: No Real Instance, No Type Material. Just fallback instance style.
        mock_get_face_styles.return_value = {}
        mock_aspect.return_value = {}
        mock_cons.return_value = {}

        resolver = self.create_resolver(has_instance_style=True)
        item = MagicMock(id=lambda: 50)
        item.Name = None

        self.assertEqual(resolver.resolve(item), "__style__instance")

    @patch("buildusd.process_ifc.get_face_styles")
    @patch("buildusd.process_ifc._constituent_styles_by_name")
    @patch("buildusd.process_ifc._shape_aspect_name_map")
    def test_instance_fallback_false_default(
        self, mock_aspect, mock_cons, mock_get_face_styles
    ):
        # Setup: has_instance_style=False (no fallback even). No Type Material.
        mock_get_face_styles.return_value = {}
        mock_aspect.return_value = {}
        mock_cons.return_value = {}

        resolver = self.create_resolver(has_instance_style=False)
        item = MagicMock(id=lambda: 50)
        item.Name = None

        self.assertEqual(resolver.resolve(item), "DEFAULT")


if __name__ == "__main__":
    unittest.main()
