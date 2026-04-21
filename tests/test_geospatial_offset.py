import sys
import types

# Stub ifcopenshell to avoid heavy dependency during unit tests.
if "ifcopenshell" not in sys.modules:
    ifcopenshell_stub = types.ModuleType("ifcopenshell")
    ifcopenshell_stub.__path__ = []
    sys.modules["ifcopenshell"] = ifcopenshell_stub

from buildusd.geospatial import signed_model_offset


def test_signed_model_offset_negative():
    assert signed_model_offset((1.0, 2.0, 3.0), "negative") == (-1.0, -2.0, -3.0)


def test_signed_model_offset_positive():
    assert signed_model_offset((1.0, -2.0, 3.0), "positive") == (1.0, -2.0, 3.0)
