from pathlib import Path
import sys
import argparse
from dataclasses import replace
import logging

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.manifest import BasePointConfig, ConversionManifest, ResolvedFilePlan
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



DEFAULT_MASTER_STAGE = "Federated Model.usda"
DEFAULT_GEODETIC_CRS = "EPSG:4326"
DEFAULT_BASE_POINT = BasePointConfig(easting=333800.4900, northing=5809101.4680, height=0.0, unit="m")

def parse_args(argv: list[str] | None = None):
    """Parse CLI arguments.

    Args:
        argv: Optional list of command-line tokens (defaults to sys.argv when None).

    Returns:
        argparse.Namespace with fields: map_coordinate_system, input_path,
        ifc_names, process_all.
    """
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
        default=str((Path(r"C:\Users\samue\_dev\datasets\ifc\tvc")).resolve()),
        help="Path to an IFC file or a directory containing IFC files (default: data/inputs)",
    )

    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=str,
        default=None,
        help="Path to a manifest (YAML or JSON) describing base points and federated masters",
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
    return parser.parse_args(argv)


def _collect_ifc_paths(input_path: Path, ifc_names: list[str] | None, process_all: bool=True) -> list[Path]:
    """Resolve IFC paths from a file or directory based on selection flags.

    - If input_path is a file: return just that file.
    - If a directory: return selected names (if provided), else all *.ifc.

    Args:
        input_path: File or directory path.
        ifc_names: Optional names within the directory to process (suffix .ifc added if missing).
        process_all: When True and directory provided, processes all *.ifc. The
            function defaults to processing all ifc files when no names provided.

    Returns:
        A list of resolved IFC file paths.
    """
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
    plan: ResolvedFilePlan | None = None,
    default_base_point: BasePointConfig = DEFAULT_BASE_POINT,
    default_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    default_master_stage: str = DEFAULT_MASTER_STAGE,
):
    """Convert a single IFC file to USD and update the federated stage."""

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
    mat_layer, material_paths = author_material_layer(stage, caches, proto_paths, output_dir=output_root, base_name=base_name, proto_layer=proto_layer, options=options)
    bind_materials_to_prototypes(stage, proto_layer, proto_paths, material_paths)
    inst_layer = author_instance_layer(stage, caches, proto_paths, output_root, base_name, options)

    effective_base_point = plan.base_point if plan and plan.base_point else default_base_point
    if effective_base_point is None:
        raise ValueError(f"No base point configured for {ifc_path.name}; provide a manifest entry or update defaults")

    projected_crs = plan.projected_crs if plan and plan.projected_crs else coordinate_system
    if not projected_crs:
        raise ValueError(f"No projected CRS available for {ifc_path.name}; provide --map-coordinate-system or manifest override")
    geodetic_crs = plan.geodetic_crs if plan and plan.geodetic_crs else default_geodetic_crs
    master_stage_name = plan.master_stage_filename if plan else default_master_stage
    lonlat_override = plan.lonlat if plan else None

    apply_stage_anchor_transform(
        stage,
        caches,
        base_point=effective_base_point,
        projected_crs=projected_crs,
        align_axes_to_map=True,
        lonlat=lonlat_override,
    )

    print(f"Assigning world geolocation using {projected_crs} -> {geodetic_crs}")
    geodetic_result = assign_world_geolocation(
        stage,
        base_point=effective_base_point,
        projected_crs=projected_crs,
        geodetic_crs=geodetic_crs,
        unit_hint=effective_base_point.unit,
        lonlat_override=lonlat_override,
    )

    stage.Save()
    proto_layer.Save()
    mat_layer.Save()
    inst_layer.Save()
    print(f"Stage written: {stage_path}")
    print(f"Prototype layer: {proto_layer.identifier}")
    print(f"Material layer: {mat_layer.identifier}")
    print(f"Instance layer: {inst_layer.identifier}")

    master_stage_path = (output_root / master_stage_name).resolve()
    update_federated_view(master_stage_path, stage_path, base_name, parent_prim_path="/World")

    return {
        "ifc": ifc_path,
        "stage": stage_path,
        "master_stage": master_stage_path,
        "layers": {
            "prototype": proto_layer.identifier,
            "material": mat_layer.identifier,
            "instance": inst_layer.identifier,
        },
        "geodetic": {
            "crs": geodetic_crs,
            "coordinates": geodetic_result,
        },
        "wgs84": geodetic_result,
        "counts": {
            "prototypes": len(caches.repmaps) + len(caches.hashes),
            "instances": len(caches.instances),
        },
        "plan": plan,
        "projected_crs": projected_crs,
    }


def main(argv: list[str] | None = None, map_coordinate_system: str = "EPSG:7855") -> None:
    """Main entrypoint supporting CLI and programmatic invocation."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("ifc_usd")

    args = parse_args(argv)
    coordinate_system = args.map_coordinate_system or map_coordinate_system
    input_path = Path(args.input_path).resolve()
    output_dir = (ROOT / "data" / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: ConversionManifest | None = None
    if args.manifest_path:
        manifest_path = Path(args.manifest_path).resolve()
        try:
            manifest = ConversionManifest.from_file(manifest_path)
            print(f"Loaded manifest from {manifest_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load manifest '{manifest_path}': {exc}") from exc

    run_options = replace(OPTIONS, manifest=manifest)

    targets = _collect_ifc_paths(input_path, args.ifc_names, args.process_all)
    if not targets:
        print(f"Nothing to process: no IFC files found in {input_path}")
        return

    results = []
    for ifc_fp in targets:
        plan = None
        if manifest is not None:
            try:
                plan = manifest.resolve_for_path(
                    ifc_fp,
                    fallback_master_name=DEFAULT_MASTER_STAGE,
                    fallback_projected_crs=coordinate_system,
                    fallback_geodetic_crs=DEFAULT_GEODETIC_CRS,
                    fallback_base_point=DEFAULT_BASE_POINT,
                )
            except Exception as exc:
                log.warning("Manifest resolution failed for %s: %s", ifc_fp.name, exc)
        res = _process_single_ifc(ifc_fp, output_dir, coordinate_system, run_options, plan=plan)
        if res:
            geo = res.get("geodetic") or {}
            coords = geo.get("coordinates")
            crs = geo.get("crs", DEFAULT_GEODETIC_CRS)
            if coords and coords[0] is not None and coords[1] is not None:
                lon, lat = coords[0], coords[1]
                alt_val = coords[2] if len(coords) > 2 and coords[2] is not None else 0.0
                log.info(f"{ifc_fp.name}: {crs} lon={lon:.8f}, lat={lat:.8f}, alt={alt_val:.3f}")
        results.append(res)

    print("\nSummary:")
    for r in results:
        if not r:
            continue
        counts = r.get("counts", {})
        master_stage = r.get("master_stage")
        master_display = master_stage.name if isinstance(master_stage, Path) else (str(master_stage) if master_stage else "n/a")
        coords = (r.get("geodetic") or {}).get("coordinates")
        if coords and coords[0] is not None and coords[1] is not None:
            alt_val = coords[2] if len(coords) > 2 and coords[2] is not None else 0.0
            coord_info = f", geo=({coords[0]:.5f}, {coords[1]:.5f}, {alt_val:.2f})"
        else:
            coord_info = ""
        print(
            f"- {r['ifc'].name}: stage={r['stage']}, master={master_display}, "
            f"prototypes={counts.get('prototypes', 0)}, instances={counts.get('instances', 0)}{coord_info}"
        )


if __name__ == "__main__":
    main()
"""IFC?USD conversion entrypoint.

This module provides a CLI and VS Code-friendly main for converting one or
more IFC files into USD stages. For each IFC:
  - Builds prototype and instance caches from IFC (process_ifc).
  - Authors a USD stage with prototypes, materials, and instances (process_usd).
  - Applies geospatial anchoring using manifest/default base points and writes
    geodetic metadata (WGS84 by default).
  - Updates the resolved federated master stage with an unloaded payload per file,
    placed under /World/<file_stem>.

It supports:
  - Single-file input or a directory of IFCs.
  - Selection via --ifc-names or process-all via --all.
  - Programmatic invocation by passing argv to main()/parse_args().
"""


