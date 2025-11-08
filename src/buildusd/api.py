from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

from .config.manifest import BasePointConfig, ConversionManifest
from .federate import (
    DEFAULT_BASE_POINT,
    DEFAULT_GEODETIC_CRS,
    DEFAULT_MASTER_STAGE,
    DEFAULT_SHARED_BASE_POINT,
    FederationTask,
    federate_stages as _federate_stages,
)
from .io_utils import is_omniverse_path
from .main import (
    ConversionOptions,
    ConversionResult,
    convert as _convert,
    set_stage_unit as _set_stage_unit,
    OPTIONS as DEFAULT_CONVERSION_OPTIONS,
)
from .process_usd import apply_stage_anchor_transform

PathLike = Union[str, Path]
AnchorMode = Literal["local", "site"]
AnchorModeSetting = Optional[AnchorMode]

__all__ = [
    "AnchorMode",
    "AnchorModeSetting",
    "ConversionDefaults",
    "ConversionSettings",
    "FederationDefaults",
    "FederationSettings",
    "FederationTask",
    "CONVERSION_DEFAULTS",
    "FEDERATION_DEFAULTS",
    "apply_stage_anchor_transform",
    "convert",
    "set_stage_unit",
    "federate_stages",
    "DEFAULT_BASE_POINT",
    "DEFAULT_SHARED_BASE_POINT",
    "DEFAULT_GEODETIC_CRS",
    "DEFAULT_MASTER_STAGE",
    "DEFAULT_CONVERSION_OPTIONS",
]


def _clone_base_point(value: BasePointConfig) -> BasePointConfig:
    return replace(value)


def _normalize_anchor_mode(value: Optional[str]) -> Optional[AnchorMode]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("local", "basepoint"):
        return "local"
    if normalized in ("site", "shared_site"):
        return "site"
    if normalized in ("none", ""):
        return None
    logging.getLogger(__name__).debug("Unknown anchor_mode '%s'; defaulting to None", value)
    return None


@dataclass(frozen=True)
class ConversionDefaults:
    map_coordinate_system: str = "EPSG:7855"
    usd_format: str = "usdc"
    usd_auto_binary_threshold_mb: Optional[float] = 50.0
    checkpoint: bool = False
    offline: bool = False
    anchor_mode: AnchorModeSetting = None

    def options(self) -> ConversionOptions:
        return replace(DEFAULT_CONVERSION_OPTIONS)


CONVERSION_DEFAULTS = ConversionDefaults()


@dataclass(slots=True)
class ConversionSettings:
    """Inputs that drive a conversion run via :func:`convert`."""

    input_path: PathLike
    output_dir: Optional[PathLike] = None
    map_coordinate_system: str = CONVERSION_DEFAULTS.map_coordinate_system
    manifest: Optional[ConversionManifest] = None
    manifest_path: Optional[PathLike] = None
    ifc_names: Optional[Sequence[str]] = None
    process_all: bool = False
    exclude_names: Optional[Sequence[str]] = None
    usd_format: str = CONVERSION_DEFAULTS.usd_format
    usd_auto_binary_threshold_mb: Optional[float] = CONVERSION_DEFAULTS.usd_auto_binary_threshold_mb
    checkpoint: bool = CONVERSION_DEFAULTS.checkpoint
    offline: bool = CONVERSION_DEFAULTS.offline
    anchor_mode: AnchorModeSetting = CONVERSION_DEFAULTS.anchor_mode
    logger: Optional[logging.Logger] = None


def convert(
    settings: ConversionSettings,
    *,
    options: Optional[ConversionOptions] = None,
    cancel_event: Any | None = None,
) -> list[ConversionResult]:
    """Convert IFC inputs described by ``settings``."""

    effective_options = replace(DEFAULT_CONVERSION_OPTIONS) if options is None else options
    normalized_anchor_mode = _normalize_anchor_mode(settings.anchor_mode)

    return _convert(
        settings.input_path,
        output_dir=settings.output_dir,
        map_coordinate_system=settings.map_coordinate_system,
        manifest=settings.manifest,
        manifest_path=settings.manifest_path,
        ifc_names=settings.ifc_names,
        process_all=settings.process_all,
        exclude_names=settings.exclude_names,
        options=effective_options,
        usd_format=settings.usd_format,
        usd_auto_binary_threshold_mb=settings.usd_auto_binary_threshold_mb,
        logger=settings.logger,
        checkpoint=settings.checkpoint,
        offline=settings.offline,
        anchor_mode=normalized_anchor_mode,
        cancel_event=cancel_event,
    )


def set_stage_unit(
    target_path: PathLike,
    meters_per_unit: float = 1.0,
    *,
    offline: bool = False,
) -> None:
    """Expose the CLI stage-unit helper as a reusable API call."""

    _set_stage_unit(target_path, meters_per_unit, offline=offline)


@dataclass(frozen=True)
class FederationDefaults:
    map_coordinate_system: str = "EPSG:7855"
    fallback_base_point: BasePointConfig = field(default_factory=lambda: _clone_base_point(DEFAULT_BASE_POINT))
    fallback_shared_site_base_point: BasePointConfig = field(default_factory=lambda: _clone_base_point(DEFAULT_SHARED_BASE_POINT))
    fallback_master_stage: str = DEFAULT_MASTER_STAGE
    fallback_geodetic_crs: str = DEFAULT_GEODETIC_CRS
    parent_prim: str = "/World"
    anchor_mode: AnchorModeSetting = None
    offline: bool = False


FEDERATION_DEFAULTS = FederationDefaults()


@dataclass(slots=True)
class FederationSettings:
    """Inputs controlling a federation pass over converted stage files."""

    stage_paths: Sequence[PathLike] = field(default_factory=list)
    manifest: Optional[ConversionManifest] = None
    manifest_path: Optional[PathLike] = None
    masters_root: Optional[PathLike] = None
    parent_prim: str = FEDERATION_DEFAULTS.parent_prim
    map_coordinate_system: str = FEDERATION_DEFAULTS.map_coordinate_system
    fallback_base_point: BasePointConfig = field(default_factory=lambda: _clone_base_point(DEFAULT_BASE_POINT))
    fallback_shared_site_base_point: BasePointConfig = field(default_factory=lambda: _clone_base_point(DEFAULT_SHARED_BASE_POINT))
    fallback_master_stage: str = DEFAULT_MASTER_STAGE
    fallback_geodetic_crs: str = DEFAULT_GEODETIC_CRS
    anchor_mode: AnchorModeSetting = FEDERATION_DEFAULTS.anchor_mode
    offline: bool = FEDERATION_DEFAULTS.offline

    def ensure_masters_root(self) -> PathLike:
        if self.masters_root is None:
            raise ValueError("FederationSettings.masters_root must be provided")
        return self.masters_root


def _resolve_manifest(manifest: Optional[ConversionManifest], manifest_path: Optional[PathLike]) -> ConversionManifest:
    if manifest is not None:
        return manifest
    if manifest_path is None:
        raise ValueError("FederationSettings requires either 'manifest' or 'manifest_path'.")
    path_obj = Path(manifest_path) if not isinstance(manifest_path, Path) else manifest_path
    return ConversionManifest.from_file(path_obj.resolve())


def federate_stages(settings: FederationSettings) -> Sequence[FederationTask]:
    """Federate the supplied stage files using manifest-driven routing."""

    if not settings.stage_paths:
        return []

    manifest_obj = _resolve_manifest(settings.manifest, settings.manifest_path)

    masters_root_input = settings.ensure_masters_root()
    if isinstance(masters_root_input, Path):
        masters_root_value: PathLike = masters_root_input.as_posix()
    else:
        masters_root_value = masters_root_input

    normalized_stage_paths: list[Path] = []
    for raw in settings.stage_paths:
        if isinstance(raw, Path):
            normalized_stage_paths.append(raw)
        else:
            text = str(raw)
            normalized_stage_paths.append(Path(text) if is_omniverse_path(text) else Path(text).resolve())

    return _federate_stages(
        stage_paths=normalized_stage_paths,
        manifest=manifest_obj,
        masters_root=masters_root_value,
        parent_prim=settings.parent_prim,
        map_coordinate_system=settings.map_coordinate_system,
        fallback_base_point=settings.fallback_base_point,
        fallback_shared_site_base_point=settings.fallback_shared_site_base_point,
        fallback_master_stage=settings.fallback_master_stage,
        fallback_geodetic_crs=settings.fallback_geodetic_crs,
        anchor_mode=_normalize_anchor_mode(settings.anchor_mode),
        offline=settings.offline,
    )
