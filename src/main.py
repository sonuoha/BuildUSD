from pathlib import Path
import sys
import argparse
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.process_ifc import build_prototypes, ConversionOptions
from src.process_usd import (
    create_usd_stage,
    author_prototype_layer,
    author_material_layer,
    bind_materials_to_prototypes,
    author_instance_layer,
    apply_stage_anchor_transform,
    assign_world_geolocation,
    update_federated_view,
)

OPTIONS = ConversionOptions(
    enable_instancing=True,
    enable_hash_dedup=False,
    convert_metadata=True,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IFC to USD")
    parser.add_argument(
        "--map-coordinate-system",
        "--map-epsg",
        dest="map_coordinate_system",
        default="EPSG:7855",
        help="EPSG code or CRS string for map_easting/map_northing (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default=str((ROOT / "data" / "inputs").resolve()),
        help="Path to an IFC file or a directory containing IFC files (default: data/inputs)",
    )
    parser.add_argument(
        "--ifc-names",
        dest="ifc_names",
        nargs="*",
        default=None,
        help="Specific IFC file names to process from the input directory (ignored if --input points to a file)",
    )
    parser.add_argument(
        "--all",
        dest="process_all",
        action="store_true",
        help="Process all .ifc files in the input directory",
    )
    return parser.parse_args()


def _collect_ifc_paths(input_path: Path, ifc_names: list[str] | None, process_all: bool=True) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a file or directory: {input_path}")

    if ifc_names and not process_all:
        paths: list[Path] = []
        for name in ifc_names:
            p = input_path / name
            if not p.suffix:
                p = p.with_suffix(".ifc")
            if p.exists() and p.is_file():
                paths.append(p.resolve())
            else:
                print(f"?? Skipping missing IFC: {p}")
        return paths

    if process_all:
        return sorted([p.resolve() for p in input_path.glob("*.ifc") if p.is_file()])

    # Default when a directory is provided without explicit selection: process all .ifc
    return sorted([p.resolve() for p in input_path.glob("*.ifc") if p.is_file()])


def _process_single_ifc(
    ifc_path: Path,
    output_root: Path,
    coordinate_system: str,
    options: ConversionOptions,
):
    import ifcopenshell

    output_root.mkdir(parents=True, exist_ok=True)
    base_name = ifc_path.stem
    stage_path = output_root / f"{base_name}.usda"

    print(f"Loading IFC from {ifc_path}")
    ifc = ifcopenshell.open(ifc_path.as_posix())

    caches = build_prototypes(ifc, options)
    print(f"Discovered {len(caches.repmaps)} repmap prototypes and {len(caches.hashes)} fallback meshes")
    print(f"Captured {len(caches.instances)} instances")

    stage = create_usd_stage(stage_path)
    proto_layer, proto_paths = author_prototype_layer(stage, caches, output_root, base_name, options)
    mat_layer, material_paths = author_material_layer(stage, caches, proto_paths, output_root, base_name, proto_layer, options)
    bind_materials_to_prototypes(stage, proto_layer, proto_paths, material_paths)
    inst_layer = author_instance_layer(stage, caches, proto_paths, output_root, base_name, options)

    apply_stage_anchor_transform(
        stage,
        caches,
        map_easting=333800.4900,
        map_northing=5809101.4680,
        map_height=0.0,
        unit_hint="m",
        align_axes_to_map=True,
    )

    print(f"Assigning world geolocation using {coordinate_system}")

    lon, lat, alt = assign_world_geolocation(
        stage,
        easting=333800.4900,
        northing=5809101.4680,
        height=0.0,
        coordinate_system=coordinate_system,
        unit_hint="m",
    )

    stage.Save()
    proto_layer.Save()
    mat_layer.Save()
    inst_layer.Save()
    print(f"Stage written: {stage_path}")
    print(f"Prototype layer: {proto_layer.identifier}")
    print(f"Material layer: {mat_layer.identifier}")
    print(f"Instance layer: {inst_layer.identifier}")

    # Update federated master stage (do not overwrite; add payload inactive by default)
    master_stage = output_root / "federated_view.usda"
    update_federated_view(master_stage, stage_path, base_name)

    return {
        "ifc": ifc_path,
        "stage": stage_path,
        "layers": {
            "prototype": proto_layer.identifier,
            "material": mat_layer.identifier,
            "instance": inst_layer.identifier,
        },
        "wgs84": (lon, lat, alt),
        "counts": {
            "prototypes": len(caches.repmaps) + len(caches.hashes),
            "instances": len(caches.instances),
        },
    }


def main(map_coordinate_system: str = "EPSG:7855"):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("ifc_usd")

    args = parse_args()
    coordinate_system = args.map_coordinate_system or map_coordinate_system
    input_path = Path(args.input_path).resolve()
    output_dir = (ROOT / "data" / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = _collect_ifc_paths(input_path, args.ifc_names, args.process_all)
    if not targets:
        print(f"Nothing to process: no IFC files found in {input_path}")
        return

    results = []
    for ifc_fp in targets:
        res = _process_single_ifc(ifc_fp, output_dir, coordinate_system, OPTIONS)
        if res and res.get("wgs84"):
            lon, lat, alt = res["wgs84"]
            log.info(f"{ifc_fp.name}: WGS84 lon={lon:.8f}, lat={lat:.8f}, alt={alt:.3f}")
        results.append(res)

    print("\nSummary:")
    for r in results:
        c = r.get("counts", {})
        print(f"- {r['ifc'].name}: stage={r['stage']}, prototypes={c.get('prototypes', 0)}, instances={c.get('instances', 0)}")


if __name__ == "__main__":
    main()
