import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path
from src.utils.ifc import debug_dump_contexts

data_path = Path(r"\Users\samue\_dev\usd_root\usdex\data\SRL-WPD-TVC-UTU8-MOD-CTU-BUW-000001.ifc").resolve()

print(f"Loading IFC from {data_path}")


if __name__ == "__main__":
    import ifcopenshell

    ifc = ifcopenshell.open(data_path.as_posix())
    print(f"IFC schema: {ifc.schema}")

    print("Dumping IfcGeometricRepresentationContext entities:")
    debug_dump_contexts(ifc)