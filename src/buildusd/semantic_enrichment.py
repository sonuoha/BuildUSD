"""Targeted enrichment planning from BuildUSD semantic graph sidecars."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence, Union

from .io_utils import PathLike, read_text


DetailObject = Union[int, str]


@dataclass(frozen=True)
class EnrichmentTarget:
    """A single graph request that can be fed back into object-scope detail."""

    guid: Optional[str]
    step_id: Optional[int]
    ifc_class: Optional[str]
    current_usd_prim: Optional[str]
    requested_parts: tuple[str, ...]

    @property
    def detail_object(self) -> Optional[DetailObject]:
        """Prefer stable IFC GUIDs, falling back to local STEP ids."""

        if self.guid:
            return self.guid
        return self.step_id


@dataclass(frozen=True)
class TargetedEnrichmentPlan:
    """Object-scope detail request derived from graph enrichment entries."""

    targets: tuple[EnrichmentTarget, ...]
    detail_engine: str = "default"

    @property
    def detail_objects(self) -> tuple[DetailObject, ...]:
        objects: list[DetailObject] = []
        seen: set[str] = set()
        for target in self.targets:
            value = target.detail_object
            if value is None:
                continue
            key = f"{type(value).__name__}:{value}"
            if key in seen:
                continue
            seen.add(key)
            objects.append(value)
        return tuple(objects)

    def to_conversion_options(self, base_options: Any | None = None) -> Any:
        """Return ``ConversionOptions`` configured for targeted decomposition."""

        from .process_ifc import ConversionOptions

        options = replace(base_options) if base_options is not None else ConversionOptions()
        return replace(
            options,
            detail_mode=True,
            detail_scope="object",
            detail_objects=self.detail_objects,
            detail_engine=self.detail_engine,
        )


def load_semantic_graph(path: PathLike) -> dict[str, Any]:
    """Load a semantic graph sidecar from a local path or supported URI."""

    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Semantic graph at {path} must contain a JSON object.")
    return payload


def build_targeted_enrichment_plan(
    graph_or_path: dict[str, Any] | PathLike,
    *,
    selected_objects: Optional[Sequence[DetailObject]] = None,
    detail_engine: str = "default",
    max_targets: Optional[int] = None,
) -> TargetedEnrichmentPlan:
    """Create an object-scope detail plan from graph enrichment requests.

    ``selected_objects`` may contain GUIDs or STEP ids. When omitted, all graph
    enrichment requests are included.
    """

    graph = (
        graph_or_path
        if isinstance(graph_or_path, dict)
        else load_semantic_graph(graph_or_path)
    )
    selected = _selected_tokens(selected_objects)
    targets: list[EnrichmentTarget] = []

    for request in graph.get("enrichment_requests", []) or []:
        if not isinstance(request, dict):
            continue
        target = _target_from_request(request)
        if target.detail_object is None:
            continue
        if selected and not _matches_selected(target, selected):
            continue
        targets.append(target)
        if max_targets is not None and len(targets) >= max_targets:
            break

    return TargetedEnrichmentPlan(tuple(targets), detail_engine=detail_engine)


def _target_from_request(request: dict[str, Any]) -> EnrichmentTarget:
    step_id = _optional_int(request.get("step_id"))
    requested_parts = tuple(
        str(part)
        for part in request.get("requested_parts", []) or []
        if str(part).strip()
    )
    guid = request.get("guid")
    return EnrichmentTarget(
        guid=str(guid) if guid else None,
        step_id=step_id,
        ifc_class=str(request.get("ifc_class")) if request.get("ifc_class") else None,
        current_usd_prim=str(request.get("current_usd_prim"))
        if request.get("current_usd_prim")
        else None,
        requested_parts=requested_parts,
    )


def _selected_tokens(values: Optional[Sequence[DetailObject]]) -> set[str]:
    if not values:
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _matches_selected(target: EnrichmentTarget, selected: set[str]) -> bool:
    candidates: Iterable[Any] = (
        target.guid,
        target.step_id,
        target.current_usd_prim,
    )
    return any(
        str(candidate).strip().lower() in selected
        for candidate in candidates
        if candidate is not None
    )


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None
