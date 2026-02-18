from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from .config.manifest import (
    BasePointConfig,
    ConversionManifest,
    MasterConfig,
    ResolvedFilePlan,
)
from .io_utils import (
    ensure_parent_directory,
    is_omniverse_path,
    join_path,
    path_name,
    path_stem,
    path_suffix,
    read_text,
)
from .conversion import (
    DEFAULT_BASE_POINT,
    DEFAULT_GEODETIC_CRS,
    DEFAULT_MASTER_STAGE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SHARED_BASE_POINT,
    ROOT,
)
from .federation_builder import build_federated_stage
from .federation.geodetic_federation import (
    federate_sites_geodetic,
    validate_geodetic_federation_stage,
)
from .usd_context import initialize_usd, shutdown_usd_context
from .pxr_utils import Sdf, Usd, UsdGeom

LOG = logging.getLogger(__name__)

PathLike = Union[str, Path]
_WGS84_REF_ATTR = "omni:geospatial:wgs84:reference:referencePosition"
_UNIT_FACTORS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
}


def _force_usdc_filename(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "Federated Model.usdc"
    path = Path(raw)
    suffix = path.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return str(path.with_suffix(".usdc"))
    return f"{raw}.usdc"


def _load_manifest(path: Path) -> ConversionManifest:
    text = read_text(path)
    suffix = path_suffix(path)
    return ConversionManifest.from_text(text, suffix=suffix or ".json")


def _normalize_anchor_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "local":
        return "local"
    if normalized in ("basepoint", "site", "shared_site"):
        return "basepoint"
    if normalized in ("none", ""):
        return None
    LOG.debug("Unknown anchor_mode '%s'; defaulting to None", value)
    return None


def _normalize_frame(value: Optional[str]) -> str:
    if value is None:
        return "projected"
    normalized = value.strip().lower()
    if normalized in ("projected", "geodetic"):
        return normalized
    LOG.debug("Unknown frame '%s'; defaulting to projected", value)
    return "projected"


def _geodetic_tuple(value: Optional[object]) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    try:
        lon = float(getattr(value, "longitude"))
        lat = float(getattr(value, "latitude"))
        height = getattr(value, "height", None)
        h = float(height) if height is not None else 0.0
        return (lat, lon, h)
    except Exception:
        return None


def _resolve_wgs84_origin(
    plan: ResolvedFilePlan,
) -> Optional[tuple[float, float, float]]:
    if plan.master and plan.master.lonlat is not None:
        origin = _geodetic_tuple(plan.master.lonlat)
        if origin:
            return origin
    if plan.lonlat is not None:
        origin = _geodetic_tuple(plan.lonlat)
        if origin:
            return origin
    return None


def _resolve_overall_wgs84_origin(
    manifest: ConversionManifest,
) -> Optional[tuple[float, float, float]]:
    defaults = manifest.defaults
    if defaults.master_id:
        master = manifest.masters.get(defaults.master_id)
        if master and master.lonlat:
            origin = _geodetic_tuple(master.lonlat)
            if origin:
                return origin
    if defaults.master_name:
        for master in manifest.masters.values():
            if master.name == defaults.master_name and master.lonlat:
                origin = _geodetic_tuple(master.lonlat)
                if origin:
                    return origin
    return None


def _normalize_pathlike(value: PathLike) -> str:
    if isinstance(value, Path):
        text = str(value)
    else:
        text = str(value)
    if is_omniverse_path(text):
        return text
    return str(Path(text).resolve())


def _normalize_payload_identifier(path: str) -> str:
    if is_omniverse_path(path):
        return str(path).lower()
    try:
        return str(Path(path).resolve()).lower()
    except Exception:
        return str(path).lower()


def _dedupe_payload_paths(
    payload_paths: Sequence[PathLike], *, out_stage_path: PathLike
) -> list[str]:
    normalized_out = _normalize_payload_identifier(_normalize_pathlike(out_stage_path))
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in payload_paths:
        path_text = _normalize_pathlike(raw)
        norm = _normalize_payload_identifier(path_text)
        if norm == normalized_out:
            LOG.warning(
                "Skipping payload %s because it matches the target federated stage.",
                path_text,
            )
            continue
        if norm in seen:
            continue
        seen.add(norm)
        normalized.append(path_text)
    return normalized


def _coerce_dict(value: object) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    return None


def _coerce_vec3(value: object) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    try:
        if hasattr(value, "__len__") and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        pass
    return None


def _unit_factor(unit: Optional[str]) -> float:
    text = str(unit or "m").strip().lower()
    return _UNIT_FACTORS.get(text, 1.0)


def _coerce_base_point(
    payload: object, *, default_epsg: Optional[str]
) -> Optional[BasePointConfig]:
    mapping = _coerce_dict(payload)
    if not mapping:
        return None
    try:
        if "easting" in mapping and "northing" in mapping:
            easting = float(mapping.get("easting", 0.0))
            northing = float(mapping.get("northing", 0.0))
            height = float(mapping.get("height", 0.0))
        elif "x" in mapping and "y" in mapping:
            easting = float(mapping.get("x", 0.0))
            northing = float(mapping.get("y", 0.0))
            height = float(mapping.get("z", 0.0))
        else:
            return None
    except Exception:
        return None
    factor = _unit_factor(mapping.get("unit"))
    epsg = mapping.get("epsg") or default_epsg
    return BasePointConfig(
        easting=easting * factor,
        northing=northing * factor,
        height=height * factor,
        unit="m",
        epsg=str(epsg).strip() if epsg else None,
    )


def _string_or_none(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _get_custom_data(prim, key: str) -> object:
    if not prim:
        return None
    try:
        return prim.GetCustomDataByKey(key)
    except Exception:
        return None


def _resolve_existing_stage_hints(
    out_stage_path: str,
) -> tuple[
    Optional[BasePointConfig],
    Optional[str],
    Optional[str],
    Optional[tuple[float, float, float]],
]:
    try:
        stage = Usd.Stage.Open(out_stage_path, load=Usd.Stage.LoadNone)
    except Exception:
        try:
            stage = Usd.Stage.Open(out_stage_path, Usd.Stage.LoadNone)
        except Exception:
            stage = Usd.Stage.Open(out_stage_path)
    if stage is None:
        return (None, None, None, None)

    layer = stage.GetRootLayer()
    layer_data = dict(getattr(layer, "customLayerData", {}) or {})
    world = stage.GetPrimAtPath("/World")
    geo = stage.GetPrimAtPath("/World/Geospatial")

    projected_crs = _string_or_none(layer_data.get("projectedCRS"))
    geodetic_crs = _string_or_none(layer_data.get("geodeticCRS"))
    if projected_crs is None:
        projected_crs = _string_or_none(_get_custom_data(world, "ifc:projectedCRS"))
    if projected_crs is None:
        projected_crs = _string_or_none(_get_custom_data(geo, "ifc:projectedCRS"))
    if geodetic_crs is None:
        geodetic_crs = _string_or_none(_get_custom_data(world, "ifc:geodeticCRS"))
    if geodetic_crs is None:
        geodetic_crs = _string_or_none(_get_custom_data(geo, "ifc:geodeticCRS"))

    origin_payload = (
        _coerce_dict(layer_data.get("federationOrigin"))
        or _coerce_dict(_get_custom_data(world, "ifc:federationOrigin"))
        or _coerce_dict(layer_data.get("anchorProjected"))
        or _coerce_dict(_get_custom_data(world, "ifc:anchorProjected"))
        or _coerce_dict(_get_custom_data(geo, "ifc:anchorProjected"))
    )
    origin = _coerce_base_point(origin_payload, default_epsg=projected_crs)

    wgs84_origin = None
    if geo:
        try:
            wgs84_attr = geo.GetAttribute(_WGS84_REF_ATTR)
            if wgs84_attr and wgs84_attr.HasValue():
                wgs84_origin = _coerce_vec3(wgs84_attr.Get())
        except Exception:
            wgs84_origin = None

    return (origin, projected_crs, geodetic_crs, wgs84_origin)


def _stamp_geodetic_federation_metadata(
    stage: Usd.Stage,
    *,
    projected_crs: str,
    geodetic_crs: str,
    anchor_mode: Optional[str],
) -> None:
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    geo = UsdGeom.Xform.Define(stage, "/World/Geospatial").GetPrim()
    if world:
        stage.SetDefaultPrim(world)
    try:
        layer = stage.GetRootLayer()
        layer_data = dict(getattr(layer, "customLayerData", {}) or {})
        layer_data.update(
            {
                "projectedCRS": projected_crs,
                "geodeticCRS": geodetic_crs,
                "federationVersion": 1,
            }
        )
        layer.customLayerData = layer_data
    except Exception:
        pass
    for prim in (world, geo):
        if not prim:
            continue
        try:
            prim.SetCustomDataByKey("ifc:projectedCRS", projected_crs)
            prim.SetCustomDataByKey("ifc:geodeticCRS", geodetic_crs)
            if anchor_mode:
                prim.SetCustomDataByKey("ifc:anchorMode", anchor_mode)
        except Exception:
            pass
        try:
            prim.CreateAttribute(
                "ifc:projectedCRS", Sdf.ValueTypeNames.String, custom=True
            ).Set(str(projected_crs))
            prim.CreateAttribute(
                "ifc:geodeticCRS", Sdf.ValueTypeNames.String, custom=True
            ).Set(str(geodetic_crs))
            if anchor_mode:
                prim.CreateAttribute(
                    "ifc:anchorMode", Sdf.ValueTypeNames.String, custom=True
                ).Set(str(anchor_mode))
        except Exception:
            pass


def _build_geodetic_federated_stage(
    *,
    out_stage_path: str,
    payload_paths: Sequence[str],
    origin_wgs84: Optional[tuple[float, float, float]],
    projected_crs: str,
    geodetic_crs: str,
    rebuild: bool,
    use_payloads: bool,
    anchor_mode: Optional[str],
) -> None:
    if rebuild:
        stage = Usd.Stage.CreateNew(out_stage_path)
    else:
        try:
            stage = Usd.Stage.Open(out_stage_path, load=Usd.Stage.LoadNone)
        except Exception:
            try:
                stage = Usd.Stage.Open(out_stage_path, Usd.Stage.LoadNone)
            except Exception:
                stage = Usd.Stage.Open(out_stage_path)
        if stage is None:
            stage = Usd.Stage.CreateNew(out_stage_path)
    if stage is None:
        raise RuntimeError(f"Failed to open or create stage: {out_stage_path}")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    sites = [
        {"name": path_stem(path), "layer_path": str(path)} for path in payload_paths
    ]
    federate_sites_geodetic(
        stage,
        sites,
        federation_origin_wgs84=origin_wgs84,
        use_payloads=use_payloads,
        preserve_existing=not rebuild,
    )
    if not validate_geodetic_federation_stage(stage):
        LOG.warning(
            "Geodetic federation validation failed for %s.", path_name(out_stage_path)
        )
    _stamp_geodetic_federation_metadata(
        stage,
        projected_crs=projected_crs,
        geodetic_crs=geodetic_crs,
        anchor_mode=anchor_mode,
    )
    stage.GetRootLayer().Save()


def _normalise_stage_root(path: str | None) -> Path:
    if not path:
        return (ROOT / "data" / "output").resolve()
    if is_omniverse_path(path):
        # For omniverse paths we do not resolve to local Path.
        return Path(path)
    return Path(path).resolve()


def _candidate_stage_files(
    stage_root: Path, stage_filters: Sequence[str] | None
) -> list[Path]:
    if stage_filters:
        files: list[Path] = []
        for raw in stage_filters:
            if is_omniverse_path(raw):
                files.append(Path(raw))
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = stage_root / raw
            if not is_omniverse_path(candidate) and not candidate.exists():
                LOG.warning("Stage path %s does not exist; skipping.", candidate)
                continue
            files.append(candidate)
        return files
    if is_omniverse_path(str(stage_root)):
        raise ValueError(
            "Automatic discovery is not supported for omniverse stage roots; provide --stage entries."
        )
    patterns = ("*.usd", "*.usda", "*.usdc")
    stage_files: list[Path] = []
    for pattern in patterns:
        stage_files.extend(stage_root.glob(pattern))
    filtered: list[Path] = []
    for path in stage_files:
        name = path_stem(path)
        if not path.is_file():
            continue
        lowered = name.lower()
        if (
            lowered.endswith("_prototypes")
            or lowered.endswith("_materials")
            or lowered.endswith("_instances")
            or lowered.endswith("_geometry2d")
        ):
            continue
        filtered.append(path)
    filtered.sort()
    return filtered


def _resolve_plan_for_stage(
    manifest: ConversionManifest,
    stage_path: Path,
    *,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_base_point: BasePointConfig,
    fallback_master_name: str,
    fallback_shared_site_base_point: BasePointConfig,
) -> ResolvedFilePlan:
    fake_ifc_path = stage_path.with_suffix(".ifc")
    return manifest.resolve_for_path(
        fake_ifc_path,
        fallback_master_name=fallback_master_name,
        fallback_projected_crs=fallback_projected_crs,
        fallback_geodetic_crs=fallback_geodetic_crs,
        fallback_base_point=fallback_base_point,
        fallback_shared_site_base_point=fallback_shared_site_base_point,
    )


@dataclass
class FederationTask:
    stage_path: Path
    plan: ResolvedFilePlan
    anchor_mode: Optional[str] = None


def _plan_federation(
    manifest: ConversionManifest,
    stage_paths: Iterable[Path],
    *,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_base_point: BasePointConfig,
    fallback_master_name: str,
    fallback_shared_site_base_point: BasePointConfig,
    anchor_mode: Optional[str],
) -> list[FederationTask]:
    tasks: list[FederationTask] = []
    skip_names = _collect_master_stage_names(manifest, fallback_master_name)
    resolved_anchor_mode = _normalize_anchor_mode(anchor_mode)
    for stage_path in stage_paths:
        if stage_path.name.lower() in skip_names:
            LOG.info("Skipping master stage input %s.", path_name(stage_path))
            continue
        try:
            plan = _resolve_plan_for_stage(
                manifest,
                stage_path,
                fallback_projected_crs=fallback_projected_crs,
                fallback_geodetic_crs=fallback_geodetic_crs,
                fallback_base_point=fallback_base_point,
                fallback_master_name=fallback_master_name,
                fallback_shared_site_base_point=fallback_shared_site_base_point,
            )
        except Exception as exc:
            LOG.error("Failed to resolve manifest entry for %s: %s", stage_path, exc)
            continue
        tasks.append(
            FederationTask(
                stage_path=stage_path, plan=plan, anchor_mode=resolved_anchor_mode
            )
        )
    return tasks


def _collect_master_stage_names(
    manifest: ConversionManifest, fallback_master_name: str
) -> set[str]:
    names: set[str] = set()
    for master in manifest.masters.values():
        names.add(_force_usdc_filename(master.resolved_name()).lower())
    if manifest.defaults.master_name:
        names.add(_force_usdc_filename(manifest.defaults.master_name).lower())
    if manifest.defaults.master_id:
        master = manifest.masters.get(manifest.defaults.master_id)
        if master:
            names.add(_force_usdc_filename(master.resolved_name()).lower())
        else:
            names.add(_force_usdc_filename(manifest.defaults.master_id).lower())
    if fallback_master_name:
        names.add(_force_usdc_filename(fallback_master_name).lower())
    overall = _resolve_overall_master_name(manifest)
    if overall:
        names.add(str(Path(overall).name).lower())
    return names


def _apply_federation(
    tasks: Sequence[FederationTask],
    *,
    masters_root: Path,
    parent_prim: str,
    rebuild: bool,
    frame: Optional[str],
) -> list[str]:
    grouped: dict[str, list[FederationTask]] = {}
    for task in tasks:
        master_filename = _force_usdc_filename(task.plan.master_stage_filename)
        master_stage_path = join_path(masters_root, master_filename)
        grouped.setdefault(master_stage_path, []).append(task)

    built_master_paths: list[str] = []
    for master_stage_path, group in grouped.items():
        if not is_omniverse_path(master_stage_path):
            ensure_parent_directory(master_stage_path)
        payload_paths = [str(task.stage_path) for task in group]
        first_plan = group[0].plan
        federation_origin = _resolve_site_federation_origin(
            first_plan, DEFAULT_SHARED_BASE_POINT
        )
        federation_projected_crs = first_plan.projected_crs or "EPSG:7855"
        geodetic_crs = first_plan.geodetic_crs or DEFAULT_GEODETIC_CRS

        for task in group[1:]:
            task_origin = _resolve_site_federation_origin(
                task.plan, DEFAULT_SHARED_BASE_POINT
            )
            if task_origin != federation_origin:
                LOG.warning(
                    "Federation origin differs for %s; using the first plan's federation base point.",
                    path_name(task.stage_path),
                )
            if task.plan.projected_crs != federation_projected_crs:
                LOG.warning(
                    "Projected CRS differs for %s (%s vs %s); reprojecting anchors.",
                    path_name(task.stage_path),
                    task.plan.projected_crs,
                    federation_projected_crs,
                )

        LOG.info(
            "Building federated stage %s with %d payloads.",
            path_name(master_stage_path),
            len(payload_paths),
        )
        if _normalize_frame(frame) == "geodetic":
            _build_geodetic_federated_stage(
                out_stage_path=str(master_stage_path),
                payload_paths=payload_paths,
                origin_wgs84=_resolve_wgs84_origin(first_plan),
                projected_crs=federation_projected_crs,
                geodetic_crs=geodetic_crs,
                rebuild=rebuild,
                use_payloads=True,
                anchor_mode=group[0].anchor_mode,
            )
        else:
            build_federated_stage(
                payload_paths=payload_paths,
                out_stage_path=str(master_stage_path),
                federation_origin=federation_origin,
                federation_projected_crs=federation_projected_crs,
                geodetic_crs=geodetic_crs,
                parent_prim_path=parent_prim or "/World",
                use_payloads=True,
                rebuild=rebuild,
                frame=_normalize_frame(frame),
                anchor_mode=group[0].anchor_mode,
            )
        built_master_paths.append(str(master_stage_path))
    return built_master_paths


def _resolve_overall_master_name(manifest: ConversionManifest) -> Optional[str]:
    name = manifest.defaults.overall_master_name
    if not name:
        return None
    master = MasterConfig(id="__overall__", name=name)
    return _force_usdc_filename(master.resolved_name())


def _resolve_site_federation_origin(
    plan: ResolvedFilePlan, fallback_shared_site_base_point: BasePointConfig
) -> BasePointConfig:
    if plan.master.base_point is not None:
        return plan.master.base_point
    if plan.shared_site_base_point is not None:
        return plan.shared_site_base_point
    return fallback_shared_site_base_point


def _apply_overall_master(
    master_stage_paths: Sequence[str],
    *,
    manifest: ConversionManifest,
    masters_root: Path,
    parent_prim: str,
    fallback_projected_crs: str,
    fallback_geodetic_crs: str,
    fallback_shared_site_base_point: BasePointConfig,
    rebuild: bool,
    frame: Optional[str],
    anchor_mode: Optional[str],
) -> None:
    overall_name = _resolve_overall_master_name(manifest)
    if not overall_name:
        return
    overall_origin = (
        manifest.defaults.overall_base_point
        or manifest.defaults.overall_shared_site_base_point
        or manifest.defaults.shared_site_base_point
        or fallback_shared_site_base_point
    )
    if overall_origin is None:
        LOG.warning("Overall federation requested but no shared_site_base_point set.")
        return
    projected_crs = manifest.defaults.projected_crs or fallback_projected_crs
    geodetic_crs = manifest.defaults.geodetic_crs or fallback_geodetic_crs
    overall_stage_path = join_path(masters_root, overall_name)
    if any(str(path) == str(overall_stage_path) for path in master_stage_paths):
        LOG.warning(
            "Overall master %s conflicts with an existing master stage; skipping.",
            path_name(overall_stage_path),
        )
        return
    payload_paths: list[str] = []
    for path in sorted({str(path) for path in master_stage_paths}):
        if is_omniverse_path(path):
            payload_paths.append(path)
            continue
        if Path(path).exists():
            payload_paths.append(path)
        else:
            LOG.warning("Overall master skipped missing site master: %s", path)
    if not payload_paths:
        return
    if not is_omniverse_path(overall_stage_path):
        ensure_parent_directory(overall_stage_path)
    LOG.info(
        "Building overall federated stage %s with %d site masters.",
        path_name(overall_stage_path),
        len(payload_paths),
    )
    if _normalize_frame(frame) == "geodetic":
        _build_geodetic_federated_stage(
            out_stage_path=str(overall_stage_path),
            payload_paths=payload_paths,
            origin_wgs84=_resolve_overall_wgs84_origin(manifest),
            projected_crs=projected_crs,
            geodetic_crs=geodetic_crs,
            rebuild=rebuild,
            use_payloads=True,
            anchor_mode=anchor_mode,
        )
    else:
        build_federated_stage(
            payload_paths=payload_paths,
            out_stage_path=str(overall_stage_path),
            federation_origin=overall_origin,
            federation_projected_crs=projected_crs,
            geodetic_crs=geodetic_crs,
            parent_prim_path=parent_prim or "/World",
            use_payloads=True,
            rebuild=rebuild,
            frame=_normalize_frame(frame),
            anchor_mode=anchor_mode,
        )


def federate_into_stage(
    payload_paths: Sequence[PathLike],
    *,
    out_stage_path: PathLike,
    manifest: Optional[ConversionManifest] = None,
    parent_prim: str = "/World",
    map_coordinate_system: str = "EPSG:7855",
    fallback_shared_site_base_point: BasePointConfig = DEFAULT_SHARED_BASE_POINT,
    fallback_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    anchor_mode: Optional[str] = None,
    frame: Optional[str] = None,
    offline: bool = False,
    rebuild: bool = False,
) -> Optional[str]:
    """Append the provided payload stages into one target federated stage."""

    if not payload_paths:
        return None

    out_stage = _normalize_pathlike(out_stage_path)
    normalized_payloads = _dedupe_payload_paths(payload_paths, out_stage_path=out_stage)
    if not normalized_payloads:
        LOG.warning("No payload stages left to federate into %s.", out_stage)
        return None
    LOG.info("Resolved payload list for federate-into:")
    for payload in normalized_payloads:
        LOG.info("  - %s", payload)

    if not is_omniverse_path(out_stage):
        ensure_parent_directory(out_stage)

    resolved_anchor_mode = _normalize_anchor_mode(anchor_mode)
    federation_origin = fallback_shared_site_base_point
    projected_crs = map_coordinate_system
    geodetic_crs = fallback_geodetic_crs
    origin_wgs84 = None

    if manifest is not None:
        defaults = manifest.defaults
        federation_origin = (
            defaults.overall_base_point
            or defaults.overall_shared_site_base_point
            or defaults.shared_site_base_point
            or fallback_shared_site_base_point
        )
        projected_crs = defaults.projected_crs or projected_crs
        geodetic_crs = defaults.geodetic_crs or geodetic_crs
        origin_wgs84 = _resolve_overall_wgs84_origin(manifest)
    else:
        LOG.info(
            "No manifest supplied for federate-into; using existing target-stage metadata and built-in fallbacks."
        )

    frame_mode = _normalize_frame(frame)
    LOG.info(
        "Federate-into start: target=%s, payloads=%d, frame=%s, rebuild=%s, anchor_mode=%s",
        out_stage,
        len(normalized_payloads),
        frame_mode,
        rebuild,
        resolved_anchor_mode or "none",
    )

    initialize_usd(offline=offline)
    try:
        if not rebuild:
            (
                existing_origin,
                existing_projected_crs,
                existing_geodetic_crs,
                existing_origin_wgs84,
            ) = _resolve_existing_stage_hints(out_stage)
            if existing_origin is not None:
                LOG.info(
                    "Using existing target-stage projected origin for federate-into: %s",
                    out_stage,
                )
                federation_origin = existing_origin
            if existing_projected_crs:
                LOG.info(
                    "Using existing target-stage projected CRS for federate-into: %s",
                    existing_projected_crs,
                )
                projected_crs = existing_projected_crs
            if existing_geodetic_crs:
                geodetic_crs = existing_geodetic_crs
            if existing_origin_wgs84 is not None:
                LOG.info(
                    "Using existing target-stage WGS84 origin for federate-into: %s",
                    existing_origin_wgs84,
                )
                origin_wgs84 = existing_origin_wgs84
            if existing_origin is None and _normalize_frame(frame) == "projected":
                LOG.warning(
                    "Target stage %s has no projected origin metadata; using manifest/CLI hints.",
                    out_stage,
                )

        if frame_mode == "geodetic":
            _build_geodetic_federated_stage(
                out_stage_path=out_stage,
                payload_paths=normalized_payloads,
                origin_wgs84=origin_wgs84,
                projected_crs=projected_crs,
                geodetic_crs=geodetic_crs,
                rebuild=rebuild,
                use_payloads=True,
                anchor_mode=resolved_anchor_mode,
            )
        else:
            build_federated_stage(
                payload_paths=normalized_payloads,
                out_stage_path=out_stage,
                federation_origin=federation_origin,
                federation_projected_crs=projected_crs,
                geodetic_crs=geodetic_crs,
                parent_prim_path=parent_prim or "/World",
                use_payloads=True,
                rebuild=rebuild,
                frame=frame_mode,
                anchor_mode=resolved_anchor_mode,
            )
    finally:
        shutdown_usd_context()
    LOG.info("Federate-into complete: target=%s", out_stage)

    return out_stage


def federate_stages(
    stage_paths: Sequence[PathLike],
    *,
    manifest: ConversionManifest,
    masters_root: PathLike,
    parent_prim: str = "/World",
    map_coordinate_system: str = "EPSG:7855",
    fallback_base_point: BasePointConfig = DEFAULT_BASE_POINT,
    fallback_shared_site_base_point: BasePointConfig = DEFAULT_SHARED_BASE_POINT,
    fallback_master_stage: str = DEFAULT_MASTER_STAGE,
    fallback_geodetic_crs: str = DEFAULT_GEODETIC_CRS,
    anchor_mode: Optional[str] = None,
    frame: Optional[str] = None,
    offline: bool = False,
    rebuild: bool = False,
) -> Sequence[FederationTask]:
    """Federate the supplied stage files according to the provided manifest."""

    if not stage_paths:
        return []

    normalized_stage_paths: list[Path] = []
    for raw in stage_paths:
        if isinstance(raw, Path):
            normalized_stage_paths.append(raw)
            continue
        text = str(raw)
        if is_omniverse_path(text):
            normalized_stage_paths.append(Path(text))
        else:
            normalized_stage_paths.append(Path(text).resolve())

    if isinstance(masters_root, Path):
        masters_root_input = masters_root.as_posix()
    else:
        masters_root_input = str(masters_root)
    masters_root_path = _normalise_stage_root(masters_root_input)
    resolved_anchor_mode = _normalize_anchor_mode(anchor_mode)
    tasks = _plan_federation(
        manifest,
        normalized_stage_paths,
        fallback_projected_crs=map_coordinate_system,
        fallback_geodetic_crs=fallback_geodetic_crs,
        fallback_base_point=fallback_base_point,
        fallback_master_name=fallback_master_stage,
        fallback_shared_site_base_point=fallback_shared_site_base_point,
        anchor_mode=resolved_anchor_mode,
    )
    if not tasks:
        return []

    initialize_usd(offline=offline)
    try:
        master_stage_paths = _apply_federation(
            tasks,
            masters_root=masters_root_path,
            parent_prim=parent_prim,
            rebuild=rebuild,
            frame=frame,
        )
        _apply_overall_master(
            master_stage_paths,
            manifest=manifest,
            masters_root=masters_root_path,
            parent_prim=parent_prim,
            fallback_projected_crs=map_coordinate_system,
            fallback_geodetic_crs=fallback_geodetic_crs,
            fallback_shared_site_base_point=fallback_shared_site_base_point,
            rebuild=rebuild,
            frame=frame,
            anchor_mode=resolved_anchor_mode,
        )
    finally:
        shutdown_usd_context()
    return tasks


if __name__ == "__main__":
    from .federate_cli import main

    main()
