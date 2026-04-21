import sys
from pathlib import Path

# Ensure the buildusd package (under src/) is importable when running tests from repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ensure ifcopenshell.geom is imported (some builds don't attach it to the top-level package)
try:
    import ifcopenshell
    import ifcopenshell.geom as _ifc_geom  # type: ignore

    if not hasattr(ifcopenshell, "geom"):
        ifcopenshell.geom = _ifc_geom  # type: ignore[attr-defined]
except Exception:
    # Keep silent; individual tests may skip if geom is unavailable
    print(
        "Warning: ifcopenshell.geom could not be imported; some tests may be skipped."
    )
    pass
