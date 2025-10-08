from __future__ import annotations
import argparse
import logging
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Sequence, Union
from .config.manifest import BasePointConfig, ConversionManifest, ResolvedFilePlan
from .io_utils import (
    ensure_directory,
    ensure_parent_directory,
    is_omniverse_path,
    join_path,
    create_checkpoint,
    list_directory,
    normalize_exclusions,
    path_name,
    path_stem,
    path_suffix,
    read_bytes,
    read_text,
    stat_entry,
)
from .process_ifc import ConversionOptions, build_prototypes
from .process_usd import (
    apply_stage_anchor_transform,
    assign_world_geolocation,
    author_instance_layer,
    author_material_layer,
    author_prototype_layer,
    author_geometry2d_layer,
    bind_materials_to_prototypes,
    create_usd_stage,
    persist_instance_cache,
    update_federated_view,
)
from .usd_context import initialize_usd, shutdown_usd_context
__all__ = [
    "ConversionResult",
    "convert",
    "parse_args",
    "ConversionOptions",
    "ConversionManifest",
]
LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path(r"C:\Users\samue\_dev\datasets\ifc\tvc").resolve()
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "output"
OPTIONS = ConversionOptions(
    enable_instancing=True,
    enable_hash_dedup=False,
    convert_metadata=True,
)
DEFAULT_MASTER_STAGE = "Federated Model.usda"
DEFAULT_GEODETIC_CRS = "EPSG:4326"
DEFAULT_BASE_POINT = BasePointConfig(
    easting=333800.4900,
    northing=5809101.4680,
    height=0.0,
    unit="m",
)
PathLike = Union[str, Path]
@dataclass(slots=True)
class ConversionResult:
    """Artifacts produced for a single converted IFC file."""
    ifc_path: PathLike
    stage_path: PathLike
    master_stage_path: PathLike
    layers: dict[str, str | None]
    projected_crs: str
    geodetic_crs: str
    geodetic_coordinates: tuple[float, ...] | None
    counts: dict[str, int]
    plan: ResolvedFilePlan | None
    revision: Optional[str] = None
    def as_dict(self) -> dict[str, Any]:
        """Legacy dict-compatible view of the conversion result."""
        return {
            "ifc": self.ifc_path,
            "stage": self.stage_path,
            "master_stage": self.master_stage_path,
            "layers": self.layers,
            "geodetic": {
                "crs": self.geodetic_crs,
                "coordinates": self.geodetic_coordinates,
            },
            "wgs84": self.geodetic_coordinates,
            "counts": self.counts,
            "plan": self.plan,
            "projected_crs": self.projected_crs,
            "revision": self.revision,
        }
@dataclass(slots=True)
class _IfcTarget:
    source: PathLike
    manifest_key: Path
@dataclass(slots=True)
class _OutputLayout:
    stage: PathLike
    prototypes: PathLike
    materials: PathLike
    instances: PathLike
    geometry2d: PathLike
    cache_dir: PathLike
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone converter."""
    parser = argparse.ArgumentParser(description="Convert IFC to USD")
    parser.add_argument(
        "--map-coordinate-system",
        "--map-epsg",
        dest="map_coordinate_system",
        default="EPSG:7855",
        help="EPSG code or CRS string for map eastings/northings (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default=str(DEFAULT_INPUT_ROOT),
        help="Path to an IFC file or a directory containing IFC files",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory for USD artifacts (default: data/output under repo root)",
    )
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        type=str,
        default=None,
        help="Path to a manifest (YAML or JSON) describing base points and masters",
    )
    parser.add_argument(
        "--ifc-names",
        dest="ifc_names",
        nargs="*",
        default=None,
        help="Specific IFC file names to process when --input points to a directory",
    )
    parser.add_argument(
        "--exclude",
        dest="exclude",
        nargs="*",
        default=None,
        help="IFC file names (with or without .ifc) to skip when scanning directories",
    )
    parser.add_argument(
        "--all",
        dest="process_all",
        action="store_true",
        help="Process all .ifc files in the input directory",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        action="store_true",
        help="Create Nucleus checkpoints for each authored layer and stage (omniverse:// only)",
    )
    parser.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help="Force standalone USD (no Kit). All input/output paths must be local.",
    )
    return parser.parse_args(argv)
def convert(
    input_path: PathLike,
    *,
    output_dir: PathLike | None = None,
    map_coordinate_system: str = "EPSG:7855",
    manifest: ConversionManifest | None = None,
    manifest_path: PathLike | None = None,
    ifc_names: Sequence[str] | None = None,
    process_all: bool = False,
    exclude_names: Sequence[str] | None = None,
    options: ConversionOptions | None = None,
    logger: logging.Logger | None = None,
    checkpoint: bool = False,
    offline: bool = False,
) -> list[ConversionResult]:
    """Programmatic API for running the converter inside another application."""
    log = logger or LOG
    if manifest_path and manifest is not None:
        raise ValueError("Provide either manifest or manifest_path, not both.")

    exclude = normalize_exclusions(exclude_names)

    if output_dir is None:
        output_root: PathLike = DEFAULT_OUTPUT_ROOT
    else:
        output_root = output_dir
    if not is_omniverse_path(output_root):
        output_root = Path(output_root).resolve()

    check_paths: list[PathLike] = [input_path, output_root]
    if manifest_path:
        check_paths.append(manifest_path)

    if offline:
        if any(is_omniverse_path(p) for p in check_paths):
            raise ValueError("--offline mode requires local input/output/manifest paths")
        initialize_usd(offline=True)
        if checkpoint:
            log.warning("--checkpoint is ignored in offline mode (Nucleus only).")
            checkpoint = False
    else:
        initialize_usd(offline=False)

    ensure_directory(output_root)

    manifest_obj = manifest
    if manifest_path:
        if is_omniverse_path(manifest_path):
            text = read_text(manifest_path)
            suffix = path_suffix(manifest_path)
            manifest_obj = ConversionManifest.from_text(text, suffix=suffix or ".json")
            log.info("Loaded manifest from %s", manifest_path)
        else:
            manifest_fp = Path(manifest_path).resolve()
            manifest_obj = ConversionManifest.from_file(manifest_fp)
            log.info("Loaded manifest from %s", manifest_fp)

    opts = options or OPTIONS
    run_options = replace(opts, manifest=manifest_obj)

    targets = _collect_ifc_paths(
        input_path,
        ifc_names=tuple(ifc_names) if ifc_names else None,
        process_all=process_all,
        exclude=exclude,
        log=log,
    )
    if not targets:
        log.info("Connected to %s but found no IFC files to convert.", input_path)
        return []

    log.info("Discovered %d IFC file(s) under %s", len(targets), input_path)

    results: list[ConversionResult] = []
    for target in targets:
        log.info("Starting conversion for %s", target.source)
        plan = None
        if manifest_obj is not None:
            try:
                plan = manifest_obj.resolve_for_path(
                    target.manifest_key,
                    fallback_master_name=DEFAULT_MASTER_STAGE,
                    fallback_projected_crs=map_coordinate_system,
                    fallback_geodetic_crs=DEFAULT_GEODETIC_CRS,
                    fallback_base_point=DEFAULT_BASE_POINT,
                )
            except Exception as exc:
                log.warning("Manifest resolution failed for %s: %s", path_name(target.source), exc)
        try:
            result = _process_single_ifc(
                target.source,
                output_root=output_root,
                coordinate_system=map_coordinate_system,
                options=run_options,
                plan=plan,
                logger=log,
                checkpoint=checkpoint,
            )
        except Exception as exc:
            log.error(
                "Conversion failed for %s: %s",
                path_name(target.source),
                exc,
                exc_info=True,
            )
            continue
        if result is None:
            log.warning("Conversion for %s produced no result; skipping.", target.source)
            continue
        results.append(result)
    return results
def main(argv: Sequence[str] | None = None) -> list[ConversionResult]:
    """CLI entrypoint with logging and a conversion summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        results = convert(
            args.input_path,
            output_dir=args.output_dir,
            map_coordinate_system=args.map_coordinate_system,
            manifest_path=args.manifest_path,
            ifc_names=args.ifc_names,
            process_all=args.process_all,
            exclude_names=args.exclude,
            checkpoint=args.checkpoint,
            offline=args.offline,
        )
    finally:
        shutdown_usd_context()
    _print_summary(results)
    return results
def _collect_ifc_paths(
    input_path: PathLike,
    *,
    ifc_names: Sequence[str] | None,
    process_all: bool,
    exclude: set[str],
    log: logging.Logger,
) -> list[_IfcTarget]:
    """Resolve the set of IFC targets from a local or Nucleus path."""
    if is_omniverse_path(input_path):
        return _collect_ifc_paths_nucleus(
            str(input_path),
            ifc_names=ifc_names,
            process_all=process_all,
            exclude=exclude,
            log=log,
        )
    path_obj = Path(input_path).resolve()
    log.info("Scanning local directory %s", path_obj)
    return _collect_ifc_paths_local(
        path_obj,
        ifc_names=ifc_names,
        process_all=process_all,
        exclude=exclude,
        log=log,
    )
def _collect_ifc_paths_local(
    root: Path,
    *,
    ifc_names: Sequence[str] | None,
    process_all: bool,
    exclude: set[str],
    log: logging.Logger,
) -> list[_IfcTarget]:
    """Collect IFC files from the local filesystem."""
    if root.is_file():
        if root.suffix.lower() != ".ifc":
            raise ValueError(f"Input file is not an IFC: {root}")
        stem = root.stem.lower()
        if stem in exclude:
            log.info("Skipping excluded IFC %s", root)
            return []
        resolved = root.resolve()
        return [_IfcTarget(source=resolved, manifest_key=resolved)]
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a file or directory: {root}")
    targets: list[_IfcTarget] = []
    if ifc_names and not process_all:
        for name in ifc_names:
            candidate = root / _ensure_ifc_name(name)
            if not candidate.exists() or not candidate.is_file():
                log.warning("Skipping missing IFC file %s", candidate)
                continue
            if candidate.suffix.lower() != ".ifc":
                log.warning("Skipping non-IFC file %s", candidate)
                continue
            stem = candidate.stem.lower()
            if stem in exclude:
                log.info("Skipping excluded IFC %s", candidate)
                continue
            resolved = candidate.resolve()
            targets.append(_IfcTarget(source=resolved, manifest_key=resolved))
        return sorted(targets, key=lambda t: str(t.source).lower())
    candidates = []
    for child in root.iterdir():
        if child.is_file() and child.suffix.lower() == ".ifc":
            candidates.append(child.resolve())
    candidates.sort()
    for candidate in candidates:
        if candidate.stem.lower() in exclude:
            log.info("Skipping excluded IFC %s", candidate)
            continue
        targets.append(_IfcTarget(source=candidate, manifest_key=candidate))
    return targets
def _collect_ifc_paths_nucleus(
    uri: str,
    *,
    ifc_names: Sequence[str] | None,
    process_all: bool,
    exclude: set[str],
    log: logging.Logger,
) -> list[_IfcTarget]:
    """Collect IFC files from a Nucleus directory or file URI."""
    # Try treating input as a single file if it looks like one
    def _single_file_target(file_uri: str) -> list[_IfcTarget]:
        if not _is_ifc_candidate(path_name(file_uri)):
            raise ValueError(f"Nucleus path is not an IFC file: {file_uri}")
        stem = path_stem(file_uri).lower()
        if stem in exclude:
            log.info("Skipping excluded IFC %s", file_uri)
            return []
        manifest_key = _manifest_key_for_path(file_uri)
        return [_IfcTarget(source=file_uri, manifest_key=manifest_key)]
    if _is_ifc_candidate(path_name(uri)) and not (ifc_names and process_all):
        return _single_file_target(uri)
    entries = None
    try:
        log.info("Connecting to Nucleus directory %s", uri)
        entries = list_directory(uri)
    except (RuntimeError, FileNotFoundError):
        if _is_ifc_candidate(path_name(uri)):
            return _single_file_target(uri)
        raise
    targets: list[_IfcTarget] = []
    if ifc_names and not process_all:
        for name in ifc_names:
            candidate = join_path(uri, _ensure_ifc_name(name))
            entry = stat_entry(candidate)
            if entry is None:
                log.warning("Skipping missing IFC file %s", candidate)
                continue
            if not _is_ifc_candidate(path_name(candidate)):
                log.warning("Skipping non-IFC file %s", candidate)
                continue
            stem = path_stem(candidate).lower()
            if stem in exclude:
                log.info("Skipping excluded IFC %s", candidate)
                continue
            targets.append(_IfcTarget(source=candidate, manifest_key=_manifest_key_for_path(candidate)))
        return sorted(targets, key=lambda t: path_name(t.source).lower())
    for entry in entries:
        rel = getattr(entry, "relative_path", None)
        if rel:
            candidate = join_path(uri, rel)
            name = rel
        else:
            candidate = getattr(entry, "path", None)
            if candidate is None:
                continue
            name = path_name(candidate)
        if not _is_ifc_candidate(name):
            continue
        stem = path_stem(candidate).lower()
        if stem in exclude:
            log.info("Skipping excluded IFC %s", candidate)
            continue
        targets.append(_IfcTarget(source=candidate, manifest_key=_manifest_key_for_path(candidate)))
    targets.sort(key=lambda t: path_name(t.source).lower())
    return targets
def _ensure_ifc_name(name: str) -> str:
    """Normalize a user-specified IFC filename, appending .ifc when missing."""
    trimmed = (name or "").strip()
    if not trimmed:
        return trimmed
    if not trimmed.lower().endswith(".ifc"):
        trimmed = f"{trimmed}.ifc"
    return trimmed
def _is_ifc_candidate(name: str) -> bool:
    """Return True when `name` looks like an IFC file."""
    return (name or "").lower().endswith(".ifc")
def _manifest_key_for_path(path: PathLike) -> Path:
    """Provide a stable manifest lookup key for either local or omniverse paths."""
    if isinstance(path, Path):
        return path.resolve()
    if is_omniverse_path(path):
        uri = str(path)
        stripped = uri.split("://", 1)[-1]
        return PurePosixPath(stripped)
    return Path(str(path)).resolve()
def _build_output_layout(output_root: PathLike, base_name: str) -> _OutputLayout:
    """Return the canonical file layout for USD artifacts derived from an IFC."""
    ensure_directory(output_root)
    prototypes_dir = join_path(output_root, "prototypes")
    materials_dir = join_path(output_root, "materials")
    instances_dir = join_path(output_root, "instances")
    geometry2d_dir = join_path(output_root, "geometry2d")
    caches_dir = join_path(output_root, "caches")
    for directory in (prototypes_dir, materials_dir, instances_dir, geometry2d_dir, caches_dir):
        ensure_directory(directory)
    stage_path = join_path(output_root, f"{base_name}.usda")
    ensure_parent_directory(stage_path)
    return _OutputLayout(
        stage=stage_path,
        prototypes=join_path(prototypes_dir, f"{base_name}_prototypes.usda"),
        materials=join_path(materials_dir, f"{base_name}_materials.usda"),
        instances=join_path(instances_dir, f"{base_name}_instances.usda"),
        geometry2d=join_path(geometry2d_dir, f"{base_name}_geometry2d.usda"),
        cache_dir=caches_dir,
    )

def _make_checkpoint_metadata(revision: Optional[str], base_name: str) -> tuple[str, list[str]]:
    note = revision or f"{base_name}"
    tag_src = revision or base_name
    tag_candidates = tag_src.replace(",", " ").split()
    tags = [t.strip() for t in tag_candidates if t.strip()] or [base_name]
    return note, tags


def _checkpoint_path(path: PathLike, note: str, tags: Sequence[str], logger: logging.Logger, label: str) -> None:
    """Create a checkpoint for `path` when it resides on Nucleus."""
    try:
        did_checkpoint = create_checkpoint(path, note=note, tags=tags)
    except Exception as exc:
        logger.warning("Failed to checkpoint %s (%s): %s", label, path, exc)
        return
    if did_checkpoint:
        logger.info("Checkpoint saved for %s (%s)", label, note)
def _process_single_ifc(
    ifc_path: PathLike,
    *,
    output_root: PathLike,
    coordinate_system: str,
    options: ConversionOptions,
    plan: ResolvedFilePlan | None,
    logger: logging.Logger,
    checkpoint: bool,
    default_base_point: BasePointConfig = DEFAULT_BASE_POINT,
    default_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    default_master_stage: str = DEFAULT_MASTER_STAGE,
) -> ConversionResult:
    """Convert a single IFC file into USD layers and optional checkpoints."""
    import ifcopenshell  # Local import to avoid heavy dependency at module load
    base_name = path_stem(ifc_path)
    layout = _build_output_layout(output_root, base_name)
    revision_note = plan.revision if plan and plan.revision else None
    checkpoint_note: Optional[str] = None
    checkpoint_tags: list[str] = []
    if checkpoint:
        checkpoint_note, checkpoint_tags = _make_checkpoint_metadata(revision_note, base_name)
    def _execute(local_ifc: Path) -> ConversionResult:
        ifc = ifcopenshell.open(local_ifc.as_posix())
        caches = build_prototypes(ifc, options)
        logger.info(
            "IFC %s â†’ %d repmap prototypes, %d hashed prototypes, %d instances",
            path_name(ifc_path),
            len(caches.repmaps),
            len(caches.hashes),
            len(caches.instances),
        )
        stage = create_usd_stage(layout.stage)
        proto_layer, proto_paths = author_prototype_layer(stage, caches, layout.prototypes, base_name, options)
        mat_layer, material_paths = author_material_layer(
            stage,
            caches,
            proto_paths,
            layer_path=layout.materials,
            base_name=base_name,
            proto_layer=proto_layer,
            options=options,
        )
        bind_materials_to_prototypes(stage, proto_layer, proto_paths, material_paths)
        inst_layer = author_instance_layer(
            stage,
            caches,
            proto_paths,
            layer_path=layout.instances,
            base_name=base_name,
            options=options,
        )
        geometry2d_layer = author_geometry2d_layer(stage, caches, layout.geometry2d, base_name)
        persist_instance_cache(layout.cache_dir, base_name, caches, proto_paths)
        effective_base_point = plan.base_point if plan and plan.base_point else default_base_point
        if effective_base_point is None:
            raise ValueError(
                f"No base point configured for {path_name(ifc_path)}; provide a manifest entry or update defaults."
            )
        projected_crs = plan.projected_crs if plan and plan.projected_crs else coordinate_system
        if not projected_crs:
            raise ValueError(
                f"No projected CRS available for {path_name(ifc_path)}; supply --map-coordinate-system or manifest override."
            )
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
        logger.info("Assigning world geolocation using %s â†’ %s", projected_crs, geodetic_crs)
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
        if geometry2d_layer:
            geometry2d_layer.Save()
        if checkpoint and checkpoint_note is not None:
            _checkpoint_path(layout.stage, checkpoint_note, checkpoint_tags, logger, f"{base_name} stage")
            _checkpoint_path(layout.prototypes, checkpoint_note, checkpoint_tags, logger, f"{base_name} prototypes layer")
            _checkpoint_path(layout.materials, checkpoint_note, checkpoint_tags, logger, f"{base_name} materials layer")
            _checkpoint_path(layout.instances, checkpoint_note, checkpoint_tags, logger, f"{base_name} instances layer")
            if geometry2d_layer:
                _checkpoint_path(layout.geometry2d, checkpoint_note, checkpoint_tags, logger, f"{base_name} 2D geometry layer")
        logger.info("Wrote stage %s", layout.stage)
        master_stage_path = join_path(output_root, master_stage_name)
        ensure_parent_directory(master_stage_path)
        update_federated_view(master_stage_path, layout.stage, base_name, parent_prim_path="/World")
        if checkpoint and checkpoint_note is not None:
            _checkpoint_path(master_stage_path, checkpoint_note, checkpoint_tags, logger, f"{base_name} master stage")
        counts = {
            "prototypes_3d": len(caches.repmaps) + len(caches.hashes),
            "instances_3d": len(caches.instances),
            "curves_2d": len(getattr(caches, "annotations", {}) or {}),
            "total_elements": len(caches.instances) + len(getattr(caches, "annotations", {}) or {}),
        }
        layers = {
            "prototype": proto_layer.identifier,
            "material": mat_layer.identifier,
            "instance": inst_layer.identifier,
            "geometry2d": geometry2d_layer.identifier if geometry2d_layer else None,
        }
        if geometry2d_layer:
            layers["annotation"] = geometry2d_layer.identifier
        coordinates = tuple(geodetic_result) if geodetic_result else None
        return ConversionResult(
            ifc_path=ifc_path,
            stage_path=layout.stage,
            master_stage_path=master_stage_path,
            layers=layers,
            projected_crs=projected_crs,
            geodetic_crs=geodetic_crs,
            geodetic_coordinates=coordinates,
            counts=counts,
            plan=plan,
            revision=revision_note,
        )
    logger.info("Opening IFC %s", ifc_path)
    if is_omniverse_path(ifc_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / path_name(ifc_path)
            tmp_path.write_bytes(read_bytes(ifc_path))
            return _execute(tmp_path)
    # Local filesystem path: use as-is (converting to Path for str inputs).
    local_ifc = ifc_path if isinstance(ifc_path, Path) else Path(ifc_path)
    return _execute(local_ifc)
def _print_summary(results: Sequence[ConversionResult]) -> None:
    if not results:
        print("No IFC files were converted.")
        return
    print("\nSummary:")
    for result in results:
        counts = result.counts
        master_stage = path_name(result.master_stage_path) if result.master_stage_path else "n/a"
        coords = result.geodetic_coordinates
        revision = result.revision or "n/a"
        if coords and len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
            alt_val = coords[2] if len(coords) > 2 and coords[2] is not None else 0.0
            coord_info = f", geo=({coords[0]:.5f}, {coords[1]:.5f}, {alt_val:.2f})"
        else:
            coord_info = ""
        proto_3d = counts.get("prototypes_3d", counts.get("prototypes", 0))
        inst_3d = counts.get("instances_3d", counts.get("instances", 0))
        curves_2d = counts.get("curves_2d", counts.get("annotations", 0))
        total_elements = counts.get("total_elements")
        if total_elements is None:
            total_elements = inst_3d + curves_2d
        print(
            f"- {path_name(result.ifc_path)}: stage={result.stage_path}, master={master_stage}, "
            f"prototypes3D={proto_3d}, instances3D={inst_3d}, curves2D={curves_2d}, "
            f"totalElements={total_elements}, revision={revision}{coord_info}"
        )
if __name__ == "__main__":
    main()
