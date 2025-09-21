from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.process_ifc import build_prototypes
from src.process_usd import (
    create_usd_stage,
    author_prototype_layer,
    author_material_layer,
    bind_materials_to_prototypes,
    author_instance_layer,
    apply_stage_anchor_transform,
)

DATA_PATH = Path(r"\Users\samue\_dev\usd_root\usdex\data\SRL-WPD-TVC-UTU8-MOD-CTU-BUW-000001.ifc").resolve()
OUTPUT_DIR = DATA_PATH.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_NAME = DATA_PATH.stem
STAGE_PATH = OUTPUT_DIR / f"{BASE_NAME}.usda"

def main():
    import ifcopenshell

    print(f"Loading IFC from {DATA_PATH}")
    ifc = ifcopenshell.open(DATA_PATH.as_posix())

    caches = build_prototypes(ifc)
    print(f"Discovered {len(caches.repmaps)} repmap prototypes and {len(caches.hashes)} fallback meshes")
    print(f"Captured {len(caches.instances)} instances")

    stage = create_usd_stage(STAGE_PATH)
    proto_layer, proto_paths = author_prototype_layer(stage, caches, OUTPUT_DIR, BASE_NAME)
    mat_layer, material_paths = author_material_layer(stage, caches, proto_paths, OUTPUT_DIR, BASE_NAME, proto_layer)
    bind_materials_to_prototypes(stage, proto_layer, proto_paths, material_paths)
    inst_layer = author_instance_layer(stage, caches, proto_paths, OUTPUT_DIR, BASE_NAME)

    apply_stage_anchor_transform(
        stage,
        caches,
        map_easting=333800.4900,
        map_northing=5809101.4680,
        map_height=0.0,
        unit_hint="m",
        align_axes_to_map=True,
    )

    stage.Save()
    proto_layer.Save()
    mat_layer.Save()
    inst_layer.Save()
    print(f"Stage written: {STAGE_PATH}")
    print(f"Prototype layer: {proto_layer.identifier}")
    print(f"Material layer: {mat_layer.identifier}")
    print(f"Instance layer: {inst_layer.identifier}")


if __name__ == "__main__":
    main()

