import unittest
import sys
import types

# Mock minimal ifcopenshell modules before import
ifcopenshell = types.ModuleType("ifcopenshell")
sys.modules["ifcopenshell"] = ifcopenshell
ifco_util = types.ModuleType("ifcopenshell.util")
sys.modules["ifcopenshell.util"] = ifco_util
ifcopenshell.util = ifco_util
ifco_util.element = types.ModuleType("ifcopenshell.util.element")
ifco_util.element.get_type = lambda product: None
sys.modules["ifcopenshell.util.element"] = ifco_util.element
ifco_util.representation = types.ModuleType("ifcopenshell.util.representation")
sys.modules["ifcopenshell.util.representation"] = ifco_util.representation

from buildusd.ifc_visuals import extract_face_style_groups


class FakeColourTable:
    def __init__(self, colours):
        self.ColourList = colours


class FakeColourMap:
    def __init__(self, mapped_to, colours, indices):
        self.is_a = lambda t=None: t == "IfcIndexedColourMap" or (
            isinstance(t, tuple) and "IfcIndexedColourMap" in t
        )
        self.MappedTo = mapped_to
        self.Colours = FakeColourTable(colours)
        self.ColourIndex = indices


class FakePolyFace:
    def __init__(self, face_count, colours):
        self._faces = list(range(face_count))
        self.HasColours = colours

    def is_a(self, t=None):
        if t is None:
            return "IfcPolygonalFaceSet"
        return t == "IfcPolygonalFaceSet" or (
            isinstance(t, tuple) and "IfcPolygonalFaceSet" in t
        )

    def id(self):
        return 1

    @property
    def Faces(self):
        return self._faces


class FakeRep:
    def __init__(self, items):
        self.Items = items
        self.RepresentationIdentifier = "Body"

    def is_a(self, t):
        return t == "IfcShapeRepresentation" or (
            isinstance(t, tuple) and "IfcShapeRepresentation" in t
        )


class FakeRepContainer:
    def __init__(self, reps):
        self.Representations = reps


class FakeProduct:
    def __init__(self, rep):
        self.Representation = rep
        self.HasAssociations = []
        self.HasShapeAspects = []
        self.file = None

    def id(self):
        return 1000


class TestColorMapResolution(unittest.TestCase):
    def test_color_map_produces_face_groups(self):
        # Polygonal face set with a single colour map entry applied to all faces
        poly = FakePolyFace(
            face_count=5,
            colours=[
                FakeColourMap(None, colours=[(0.1, 0.2, 0.3)], indices=[1, 1, 1, 1, 1])
            ],
        )
        rep = FakeRep([poly])
        product = FakeProduct(FakeRepContainer([rep]))

        groups = extract_face_style_groups(product)
        self.assertEqual(len(groups), 1)
        token, entry = next(iter(groups.items()))
        self.assertTrue(token.startswith("Color"))
        self.assertEqual(len(entry.get("faces", [])), 5)
        mat = entry.get("material")
        self.assertIsNotNone(mat)
        self.assertAlmostEqual(mat.base_color[0], 0.1)
        self.assertAlmostEqual(mat.base_color[1], 0.2)
        self.assertAlmostEqual(mat.base_color[2], 0.3)


if __name__ == "__main__":
    unittest.main()
