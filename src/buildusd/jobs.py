"""File-backed BuildUSD enrichment jobs.

The job layer is intentionally small and public-facing: it describes generic
conversion/enrichment work that can be submitted by any host application without
depending on that host's UI or scene graph.
"""

from __future__ import annotations

import json
import hashlib
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, TYPE_CHECKING
from uuid import uuid4

from .conversion import OPTIONS as DEFAULT_CONVERSION_OPTIONS
from .conversion import ConversionOptions, ConversionResult
from .io_utils import (
    PathLike,
    ensure_directory,
    is_omniverse_path,
    join_path,
    read_text,
    stat_entry,
    write_text,
)

if TYPE_CHECKING:
    from .api import ConversionSettings


JOB_SCHEMA = "buildusd.job.v1"
TARGETED_DETAIL_MANIFEST_SCHEMA = "buildusd.targeted_detail_manifest.v1"
JOB_RESULT_SCHEMA = "buildusd.job_result.v1"
TARGETED_DETAIL_CACHE_SCHEMA = "buildusd.targeted_detail_cache.v1"
SUPPORTED_JOB_TYPES = {"targeted_detail"}

JobStatus = Literal["completed", "failed"]
JobType = Literal["targeted_detail"]
Converter = Callable[..., Sequence[ConversionResult]]


@dataclass(frozen=True, slots=True)
class TargetedDetailTarget:
    """A single IFC product requested for object-scope detail conversion."""

    guid: Optional[str] = None
    step_id: Optional[int] = None
    source_usd_prim: Optional[str] = None
    reason: Optional[str] = None

    @property
    def detail_object(self) -> str | int:
        if self.guid:
            return self.guid
        if self.step_id is not None:
            return int(self.step_id)
        raise ValueError("Targeted detail target requires either 'guid' or 'step_id'.")

    @classmethod
    def from_payload(cls, payload: Any) -> "TargetedDetailTarget":
        if isinstance(payload, (str, int)):
            if isinstance(payload, int):
                return cls(step_id=payload)
            text = str(payload).strip()
            if not text:
                raise ValueError("Target identifier cannot be empty.")
            try:
                return cls(step_id=int(text))
            except ValueError:
                return cls(guid=text)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Target must be an object, string, or integer: {payload!r}"
            )
        step_value = payload.get("step_id")
        step_id = None if step_value in (None, "") else int(step_value)
        guid = payload.get("guid")
        source_prim = payload.get("source_usd_prim") or payload.get("current_usd_prim")
        return cls(
            guid=str(guid) if guid else None,
            step_id=step_id,
            source_usd_prim=str(source_prim) if source_prim else None,
            reason=str(payload.get("reason")) if payload.get("reason") else None,
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TargetedDetailJob:
    """Generic targeted detail job accepted by the file-backed worker."""

    source_ifc: PathLike
    output_dir: PathLike
    targets: tuple[TargetedDetailTarget, ...]
    job_id: str = field(default_factory=lambda: uuid4().hex)
    job_type: JobType = "targeted_detail"
    schema: str = JOB_SCHEMA
    source_stage: Optional[PathLike] = None
    semantic_graph: Optional[PathLike] = None
    detail_engine: str = "default"
    map_coordinate_system: str = "EPSG:7855"
    usd_format: str = "usdc"
    usd_auto_binary_threshold_mb: Optional[float] = 50.0
    offline: bool = False
    checkpoint: bool = False
    anchor_mode: Optional[str] = None
    geospatial_mode: str = "auto"
    include_2d: bool = False
    process_all: bool = False
    use_cache: bool = True
    cache_dir: Optional[PathLike] = None
    ifc_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()
    geom_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TargetedDetailJob":
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be a JSON object.")
        schema = str(payload.get("schema") or JOB_SCHEMA)
        if schema != JOB_SCHEMA:
            raise ValueError(f"Unsupported job schema: {schema}")
        job_type = str(payload.get("job_type") or "targeted_detail")
        if job_type not in SUPPORTED_JOB_TYPES:
            raise ValueError(f"Unsupported job type: {job_type}")
        targets = tuple(
            TargetedDetailTarget.from_payload(item)
            for item in payload.get("targets") or payload.get("detail_objects") or []
        )
        job = cls(
            schema=schema,
            job_type="targeted_detail",
            job_id=str(payload.get("job_id") or uuid4().hex),
            source_ifc=_required_path(payload, "source_ifc"),
            source_stage=_optional_path(payload, "source_stage"),
            semantic_graph=_optional_path(payload, "semantic_graph"),
            output_dir=_required_path(payload, "output_dir"),
            targets=targets,
            detail_engine=str(payload.get("detail_engine") or "default"),
            map_coordinate_system=str(
                payload.get("map_coordinate_system") or "EPSG:7855"
            ),
            usd_format=str(payload.get("usd_format") or "usdc"),
            usd_auto_binary_threshold_mb=_optional_float(
                payload.get("usd_auto_binary_threshold_mb"), default=50.0
            ),
            offline=bool(payload.get("offline", False)),
            checkpoint=bool(payload.get("checkpoint", False)),
            anchor_mode=_optional_str(payload.get("anchor_mode")),
            geospatial_mode=str(payload.get("geospatial_mode") or "auto"),
            include_2d=bool(payload.get("include_2d", False)),
            process_all=bool(payload.get("process_all", False)),
            use_cache=bool(payload.get("use_cache", True)),
            cache_dir=_optional_path(payload, "cache_dir"),
            ifc_names=tuple(str(value) for value in payload.get("ifc_names") or []),
            exclude_names=tuple(
                str(value) for value in payload.get("exclude_names") or []
            ),
            geom_overrides=dict(payload.get("geom_overrides") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )
        job.validate()
        return job

    @classmethod
    def from_json_file(cls, path: PathLike) -> "TargetedDetailJob":
        return cls.from_payload(json.loads(read_text(path)))

    @property
    def detail_objects(self) -> tuple[str | int, ...]:
        objects: list[str | int] = []
        seen: set[str] = set()
        for target in self.targets:
            value = target.detail_object
            key = f"{type(value).__name__}:{value}"
            if key in seen:
                continue
            seen.add(key)
            objects.append(value)
        return tuple(objects)

    @property
    def cache_key(self) -> str:
        payload = {
            "schema": TARGETED_DETAIL_CACHE_SCHEMA,
            "buildusd_version": _buildusd_version(),
            "source": _source_identity(self.source_ifc),
            "source_stage": str(self.source_stage) if self.source_stage else None,
            "semantic_graph": str(self.semantic_graph) if self.semantic_graph else None,
            "targets": _target_cache_payload(self.targets),
            "detail_engine": self.detail_engine,
            "map_coordinate_system": self.map_coordinate_system,
            "usd_format": self.usd_format,
            "usd_auto_binary_threshold_mb": self.usd_auto_binary_threshold_mb,
            "offline": self.offline,
            "checkpoint": self.checkpoint,
            "anchor_mode": self.anchor_mode,
            "geospatial_mode": self.geospatial_mode,
            "include_2d": self.include_2d,
            "process_all": self.process_all,
            "ifc_names": sorted(self.ifc_names),
            "exclude_names": sorted(self.exclude_names),
            "geom_overrides": _json_safe(self.geom_overrides),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not str(self.source_ifc).strip():
            raise ValueError("source_ifc is required.")
        if not str(self.output_dir).strip():
            raise ValueError("output_dir is required.")
        if not self.targets:
            raise ValueError("At least one targeted detail target is required.")
        for target in self.targets:
            target.detail_object

    def to_conversion_options(
        self, base_options: ConversionOptions | None = None
    ) -> ConversionOptions:
        options = (
            replace(base_options)
            if base_options is not None
            else replace(DEFAULT_CONVERSION_OPTIONS)
        )
        geom_overrides = dict(getattr(options, "geom_overrides", {}) or {})
        geom_overrides.update(self.geom_overrides or {})
        return replace(
            options,
            detail_mode=True,
            detail_scope="object",
            detail_objects=self.detail_objects,
            detail_engine=self.detail_engine,
            include_2d=bool(self.include_2d or getattr(options, "include_2d", False)),
            geom_overrides=geom_overrides,
        )

    def to_conversion_settings(self) -> ConversionSettings:
        from .api import ConversionSettings

        return ConversionSettings(
            input_path=self.source_ifc,
            output_dir=self.output_dir,
            map_coordinate_system=self.map_coordinate_system,
            ifc_names=self.ifc_names or None,
            process_all=self.process_all,
            exclude_names=self.exclude_names or None,
            usd_format=self.usd_format,
            usd_auto_binary_threshold_mb=self.usd_auto_binary_threshold_mb,
            checkpoint=self.checkpoint,
            offline=self.offline,
            anchor_mode=self.anchor_mode,
            geospatial_mode=self.geospatial_mode,
            include_2d=self.include_2d,
            geom_overrides=dict(self.geom_overrides or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_ifc"] = str(self.source_ifc)
        payload["source_stage"] = str(self.source_stage) if self.source_stage else None
        payload["semantic_graph"] = (
            str(self.semantic_graph) if self.semantic_graph else None
        )
        payload["output_dir"] = str(self.output_dir)
        payload["cache_dir"] = str(self.cache_dir) if self.cache_dir else None
        payload["targets"] = [target.to_payload() for target in self.targets]
        return payload


@dataclass(frozen=True, slots=True)
class BuildUSDJobResult:
    """Result JSON written by the file-backed worker."""

    job_id: str
    job_type: JobType
    status: JobStatus
    schema: str = JOB_RESULT_SCHEMA
    created_utc: str = field(default_factory=lambda: _utc_now())
    manifest_path: Optional[PathLike] = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    conversion_results: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    cache_key: Optional[str] = None
    cache_hit: bool = False
    error: Optional[str] = None
    traceback: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_path"] = (
            str(self.manifest_path) if self.manifest_path else None
        )
        return _json_safe(payload)

    def write_json(self, path: PathLike) -> None:
        write_text(path, json.dumps(self.to_payload(), indent=2, sort_keys=True))


def read_job(path: PathLike) -> TargetedDetailJob:
    return TargetedDetailJob.from_json_file(path)


def write_job(path: PathLike, job: TargetedDetailJob) -> None:
    write_text(path, json.dumps(job.to_payload(), indent=2, sort_keys=True))


def run_job(
    job: TargetedDetailJob,
    *,
    converter: Callable[..., Sequence[ConversionResult]] | None = None,
) -> BuildUSDJobResult:
    """Run a single targeted detail job and write its replacement manifest."""

    try:
        job.validate()
        cache_key = job.cache_key
        if job.use_cache:
            cached = _read_cached_job_result(job, cache_key)
            if cached is not None:
                return cached
        conversion_settings = job.to_conversion_settings()
        conversion_options = job.to_conversion_options()
        convert_fn = converter or _default_converter
        results = list(convert_fn(conversion_settings, options=conversion_options))
        manifest_path = _write_targeted_detail_manifest(job, results)
        result = BuildUSDJobResult(
            job_id=job.job_id,
            job_type=job.job_type,
            status="completed",
            manifest_path=manifest_path,
            cache_key=cache_key,
            cache_hit=False,
            artifacts={
                "manifest": str(manifest_path),
                "stages": [str(result.stage_path) for result in results],
                "semantic_graphs": [
                    str(result.layers.get("semantic_graph"))
                    for result in results
                    if result.layers.get("semantic_graph")
                ],
            },
            conversion_results=tuple(
                _conversion_result_payload(result) for result in results
            ),
        )
        if job.use_cache:
            _write_cached_job_result(job, result)
        return result
    except Exception as exc:
        return BuildUSDJobResult(
            job_id=job.job_id,
            job_type=job.job_type,
            status="failed",
            cache_key=_safe_cache_key(job),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _default_converter(
    settings: ConversionSettings, *, options: ConversionOptions
) -> Sequence[ConversionResult]:
    from .api import convert

    return convert(settings, options=options)


def _write_targeted_detail_manifest(
    job: TargetedDetailJob,
    results: Sequence[ConversionResult],
) -> PathLike:
    ensure_directory(job.output_dir)
    manifest_path = join_path(
        job.output_dir, f"{_safe_token(job.job_id)}.targeted_detail.manifest.json"
    )
    payload = {
        "schema": TARGETED_DETAIL_MANIFEST_SCHEMA,
        "created_utc": _utc_now(),
        "cache_key": job.cache_key,
        "job": job.to_payload(),
        "contract": {
            "source_layers_are_not_mutated": True,
            "detail_scope": "object",
            "downstream_tool_controls_composition": True,
        },
        "targets": [target.to_payload() for target in job.targets],
        "artifacts": {
            "stages": [str(result.stage_path) for result in results],
            "layers": [result.layers for result in results],
        },
        "conversion_results": [
            _conversion_result_payload(result) for result in results
        ],
    }
    write_text(manifest_path, json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return manifest_path


def _read_cached_job_result(
    job: TargetedDetailJob, cache_key: str
) -> Optional[BuildUSDJobResult]:
    cache_record_path = _cache_record_path(job, cache_key)
    if stat_entry(cache_record_path) is None:
        return None
    payload = json.loads(read_text(cache_record_path))
    if payload.get("schema") != TARGETED_DETAIL_CACHE_SCHEMA:
        return None
    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        return None
    manifest_path = result_payload.get("manifest_path")
    if manifest_path and stat_entry(str(manifest_path)) is None:
        return None
    artifacts = dict(result_payload.get("artifacts") or {})
    artifacts["cache_record"] = str(cache_record_path)
    return BuildUSDJobResult(
        job_id=job.job_id,
        job_type=job.job_type,
        status="completed",
        manifest_path=manifest_path,
        artifacts=artifacts,
        conversion_results=tuple(result_payload.get("conversion_results") or ()),
        warnings=tuple(result_payload.get("warnings") or ()),
        cache_key=cache_key,
        cache_hit=True,
    )


def _write_cached_job_result(job: TargetedDetailJob, result: BuildUSDJobResult) -> None:
    if result.status != "completed" or not result.manifest_path:
        return
    cache_key = result.cache_key or job.cache_key
    cache_record_path = _cache_record_path(job, cache_key)
    ensure_directory(_cache_entry_dir(job, cache_key))
    payload = {
        "schema": TARGETED_DETAIL_CACHE_SCHEMA,
        "created_utc": _utc_now(),
        "cache_key": cache_key,
        "source": _source_identity(job.source_ifc),
        "targets": _target_cache_payload(job.targets),
        "detail_engine": job.detail_engine,
        "job": job.to_payload(),
        "result": result.to_payload(),
    }
    write_text(
        cache_record_path, json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    )


def _cache_root(job: TargetedDetailJob) -> PathLike:
    if job.cache_dir:
        return job.cache_dir
    return join_path(job.output_dir, ".buildusd_cache", "targeted_detail")


def _cache_entry_dir(job: TargetedDetailJob, cache_key: str) -> PathLike:
    return join_path(_cache_root(job), cache_key)


def _cache_record_path(job: TargetedDetailJob, cache_key: str) -> PathLike:
    return join_path(_cache_entry_dir(job, cache_key), "cache.json")


def _conversion_result_payload(result: ConversionResult) -> dict[str, Any]:
    return {
        "ifc_path": str(result.ifc_path),
        "stage_path": str(result.stage_path),
        "master_stage_path": str(result.master_stage_path),
        "layers": dict(result.layers or {}),
        "projected_crs": result.projected_crs,
        "geodetic_crs": result.geodetic_crs,
        "geodetic_coordinates": result.geodetic_coordinates,
        "geospatial_mode": result.geospatial_mode,
        "counts": dict(result.counts or {}),
        "revision": result.revision,
    }


def _source_identity(source_ifc: PathLike) -> dict[str, Any]:
    text = str(source_ifc)
    identity: dict[str, Any] = {"path": text}
    if is_omniverse_path(text):
        identity["kind"] = "omniverse"
        return identity
    path = Path(text)
    if path.exists() and path.is_file():
        identity["kind"] = "local_file"
        identity["sha256"] = _file_sha256(path)
        try:
            stat = path.stat()
            identity["size"] = int(stat.st_size)
        except Exception:
            pass
    else:
        identity["kind"] = "path_only"
    return identity


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_cache_payload(
    targets: Sequence[TargetedDetailTarget],
) -> list[dict[str, Any]]:
    payload = [target.to_payload() for target in targets]
    return sorted(
        payload,
        key=lambda item: (
            str(item.get("guid") or ""),
            "" if item.get("step_id") is None else str(item.get("step_id")),
            str(item.get("source_usd_prim") or ""),
        ),
    )


def _buildusd_version() -> str:
    try:
        return importlib_metadata.version("buildusd")
    except Exception:
        return "0+local"


def _safe_cache_key(job: TargetedDetailJob) -> Optional[str]:
    try:
        return job.cache_key
    except Exception:
        return None


def _required_path(payload: dict[str, Any], key: str) -> PathLike:
    value = payload.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} is required.")
    return str(value)


def _optional_path(payload: dict[str, Any], key: str) -> Optional[PathLike]:
    value = payload.get(key)
    return str(value) if value is not None and str(value).strip() else None


def _optional_float(value: Any, *, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    return float(value)


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value is not None and str(value).strip() else None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _safe_token(value: str) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value)
    )
    return token.strip("_") or uuid4().hex


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value
