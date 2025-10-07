from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

log = logging.getLogger(__name__)

_DEFAULT_MASTER_NAME = "Federated Model.usda"
_DEFAULT_GEODETIC_CRS = "EPSG:4326"


_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def _sanitize_filename(value: str) -> str:
    sanitized = ''.join(ch for ch in value if ch not in _INVALID_FILENAME_CHARS)
    sanitized = sanitized.rstrip(' .')
    sanitized = sanitized.strip()
    return sanitized


@dataclass
class BasePointConfig:
    """Projected base-point expressed as east/north/height in a given unit."""

    easting: float
    northing: float
    height: float = 0.0
    unit: str = "m"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]], fallback: Optional["BasePointConfig"] = None) -> Optional["BasePointConfig"]:
        if data is None:
            return fallback
        try:
            easting = float(data["easting"])
            northing = float(data["northing"])
        except KeyError as exc:
            raise ValueError(f"Base point mapping missing required key: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid base point values: {data}") from exc
        height = float(data.get("height", fallback.height if fallback else 0.0))
        unit = str(data.get("unit", fallback.unit if fallback else "m"))
        return cls(easting=easting, northing=northing, height=height, unit=unit)

    def with_fallback(self, fallback: Optional["BasePointConfig"]) -> "BasePointConfig":
        if fallback is None:
            return self
        return BasePointConfig(
            easting=self.easting if self.easting is not None else fallback.easting,
            northing=self.northing if self.northing is not None else fallback.northing,
            height=self.height if self.height is not None else fallback.height,
            unit=self.unit or fallback.unit,
        )


@dataclass
class GeodeticCoordinate:
    """Geodetic coordinate expressed as lon/lat (degrees) plus optional height."""

    longitude: float
    latitude: float
    height: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> Optional["GeodeticCoordinate"]:
        if not data:
            return None
        try:
            lon = float(data["longitude"])
            lat = float(data["latitude"])
        except KeyError as exc:
            raise ValueError(f"Geodetic mapping missing required key: {exc}") from exc
        height = data.get("height")
        h_val = float(height) if height is not None else None
        return cls(longitude=lon, latitude=lat, height=h_val)


@dataclass
class ManifestDefaults:
    master_id: Optional[str] = None
    master_name: Optional[str] = None
    projected_crs: Optional[str] = None
    geodetic_crs: Optional[str] = None
    base_point: Optional[BasePointConfig] = None


@dataclass
class MasterConfig:
    id: str
    name: str
    projected_crs: Optional[str] = None
    geodetic_crs: Optional[str] = None
    base_point: Optional[BasePointConfig] = None
    lonlat: Optional[GeodeticCoordinate] = None

    def resolved_name(self) -> str:
        raw = self.name.strip() if self.name else ""
        name = _sanitize_filename(raw)
        if not name:
            name = _sanitize_filename(_DEFAULT_MASTER_NAME)
        if not name.lower().endswith(".usda"):
            name = f"{name}.usda"
        name = _sanitize_filename(name)
        if not name:
            name = _DEFAULT_MASTER_NAME
        return name


@dataclass
class FileRule:
    name: Optional[str] = None
    pattern: Optional[str] = None
    master: Optional[str] = None
    projected_crs: Optional[str] = None
    geodetic_crs: Optional[str] = None
    base_point: Optional[BasePointConfig] = None
    lonlat: Optional[GeodeticCoordinate] = None

    def matches(self, path: Path) -> bool:
        target_name = path.name.lower()
        target_stem = path.stem.lower()
        if self.name:
            compare = self.name.lower()
            if compare == target_name or compare == target_stem:
                return True
        if self.pattern:
            pat = self.pattern
            if fnmatch.fnmatch(target_name, pat) or fnmatch.fnmatch(target_stem, pat):
                return True
        return False


@dataclass
class ResolvedFilePlan:
    ifc_path: Path
    master: MasterConfig
    projected_crs: str
    geodetic_crs: str
    base_point: BasePointConfig
    lonlat: Optional[GeodeticCoordinate] = None
    applied_rules: List[FileRule] = field(default_factory=list)

    @property
    def master_stage_filename(self) -> str:
        return self.master.resolved_name()


class ConversionManifest:
    """Manifest describing per-file anchoring and federated master routing."""

    def __init__(
        self,
        *,
        defaults: ManifestDefaults,
        masters: Dict[str, MasterConfig],
        file_rules: List[FileRule],
    ) -> None:
        self.defaults = defaults
        self.masters = masters
        self.file_rules = file_rules

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "ConversionManifest":
        defaults_data = data.get("defaults") or {}
        defaults = ManifestDefaults(
            master_id=defaults_data.get("master"),
            master_name=defaults_data.get("master_name"),
            projected_crs=defaults_data.get("projected_crs"),
            geodetic_crs=defaults_data.get("geodetic_crs"),
            base_point=BasePointConfig.from_mapping(defaults_data.get("base_point")),
        )

        masters: Dict[str, MasterConfig] = {}
        for entry in data.get("masters", []) or []:
            mid = str(entry.get("id") or "").strip()
            if not mid:
                log.warning("Manifest master entry missing id; skipping: %s", entry)
                continue
            name = str(entry.get("name") or "").strip() or _DEFAULT_MASTER_NAME
            projected = entry.get("projected_crs")
            geodetic = entry.get("geodetic_crs")
            base_point = BasePointConfig.from_mapping(entry.get("base_point"))
            lonlat = GeodeticCoordinate.from_mapping(entry.get("lonlat"))
            masters[mid] = MasterConfig(
                id=mid,
                name=name,
                projected_crs=projected,
                geodetic_crs=geodetic,
                base_point=base_point,
                lonlat=lonlat,
            )

        file_rules: List[FileRule] = []
        for entry in data.get("files", []) or []:
            base_point = BasePointConfig.from_mapping(entry.get("base_point"))
            lonlat = GeodeticCoordinate.from_mapping(entry.get("lonlat"))
            file_rules.append(
                FileRule(
                    name=entry.get("name"),
                    pattern=entry.get("pattern"),
                    master=entry.get("master"),
                    projected_crs=entry.get("projected_crs"),
                    geodetic_crs=entry.get("geodetic_crs"),
                    base_point=base_point,
                    lonlat=lonlat,
                )
        )

        return cls(defaults=defaults, masters=masters, file_rules=file_rules)

    @classmethod
    def from_file(cls, path: Path) -> "ConversionManifest":
        text = path.read_text(encoding="utf-8")
        data = cls._load_data_from_text(text, suffix=path.suffix)
        return cls.from_mapping(data)

    @classmethod
    def from_text(cls, text: str, *, suffix: str) -> "ConversionManifest":
        data = cls._load_data_from_text(text, suffix=suffix)
        return cls.from_mapping(data)

    def resolve_for_path(
        self,
        ifc_path: Path,
        *,
        fallback_master_name: str = _DEFAULT_MASTER_NAME,
        fallback_projected_crs: Optional[str] = None,
        fallback_geodetic_crs: str = _DEFAULT_GEODETIC_CRS,
        fallback_base_point: Optional[BasePointConfig] = None,
    ) -> ResolvedFilePlan:
        name = ifc_path.name

        projected_crs = self.defaults.projected_crs or fallback_projected_crs
        geodetic_crs = self.defaults.geodetic_crs or fallback_geodetic_crs
        base_point = self.defaults.base_point or fallback_base_point
        lonlat: Optional[GeodeticCoordinate] = None

        master_id = self.defaults.master_id
        master_name = self.defaults.master_name

        applied: List[FileRule] = []
        for rule in self._iter_matching_rules(ifc_path):
            applied.append(rule)
            if rule.projected_crs:
                projected_crs = rule.projected_crs
            if rule.geodetic_crs:
                geodetic_crs = rule.geodetic_crs
            if rule.base_point:
                base_point = rule.base_point
            if rule.master:
                master_id = rule.master
                master_name = None
            if rule.lonlat:
                lonlat = rule.lonlat

        master = self._resolve_master(master_id=master_id, master_name=master_name, fallback_name=fallback_master_name)

        if projected_crs is None:
            projected_crs = master.projected_crs or fallback_projected_crs
        if geodetic_crs is None:
            geodetic_crs = master.geodetic_crs or fallback_geodetic_crs

        if base_point is None:
            base_point = master.base_point or self.defaults.base_point or fallback_base_point
        if base_point is None:
            raise ValueError(f"No base point defined for IFC '{name}' in manifest or fallbacks")

        if projected_crs is None:
            raise ValueError(f"No projected CRS defined for IFC '{name}' in manifest or fallbacks")
        if geodetic_crs is None:
            geodetic_crs = _DEFAULT_GEODETIC_CRS

        if lonlat is None:
            lonlat = master.lonlat

        return ResolvedFilePlan(
            ifc_path=ifc_path,
            master=master,
            projected_crs=projected_crs,
            geodetic_crs=geodetic_crs,
            base_point=base_point,
            lonlat=lonlat,
            applied_rules=applied,
        )

    def _iter_matching_rules(self, ifc_path: Path) -> Iterable[FileRule]:
        for rule in self.file_rules:
            try:
                if rule.matches(ifc_path):
                    yield rule
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning("Manifest rule %s failed to evaluate for %s: %s", rule, ifc_path, exc)

    def _resolve_master(
        self,
        *,
        master_id: Optional[str],
        master_name: Optional[str],
        fallback_name: str,
    ) -> MasterConfig:
        master: Optional[MasterConfig] = None
        if master_id:
            master = self.masters.get(master_id)
            if master is None:
                log.warning("Manifest references undefined master '%s'; treating as ad-hoc file name", master_id)
                master = MasterConfig(id=master_id, name=master_id)
        if master is None:
            name = master_name or fallback_name
            master = MasterConfig(id="__default__", name=name)
        resolved_name = master.resolved_name()
        if master.name != resolved_name:
            master = replace(master, name=resolved_name)
        return master

    @staticmethod
    def _load_data_from_text(text: str, *, suffix: str) -> Dict[str, Any]:
        ext = (suffix or "").lower()
        if ext in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML manifests")
            loaded = yaml.safe_load(text)
            if not isinstance(loaded, dict):
                raise ValueError("YAML manifest must define a mapping at the top level")
            return loaded
        if ext == ".json":
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("JSON manifest must define a mapping at the top level")
            return loaded
        raise ValueError(f"Unsupported manifest type: {suffix}")


__all__ = [
    "BasePointConfig",
    "GeodeticCoordinate",
    "ManifestDefaults",
    "MasterConfig",
    "FileRule",
    "ResolvedFilePlan",
    "ConversionManifest",
]
