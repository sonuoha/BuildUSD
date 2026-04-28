"""2D annotation extraction helpers for IFC processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AnnotationCurve:
    step_id: int
    name: str
    points: List[Tuple[float, float, float]]
    hierarchy: Tuple[Tuple[str, Optional[int]], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnnotationHooks:
    entity_on_hidden_layer: Callable[[Any], bool]
    collect_spatial_hierarchy: Callable[[Any], Tuple[Tuple[str, Optional[int]], ...]]
    entity_label: Callable[[Any], str]
    object_placement_to_np: Callable[[Any], np.ndarray]
    context_to_np: Callable[[Any], np.ndarray]
    mapping_item_transform: Callable[[Any, Any], np.ndarray]
    extract_curve_points: Callable[[Any], List[Tuple[float, float, float]]]
    transform_points: Callable[
        [List[Tuple[float, float, float]], np.ndarray], List[Tuple[float, float, float]]
    ]


def extract_annotation_curves(
    ifc_file,
    hierarchy_cache: Dict[int, Tuple[Tuple[str, Optional[int]], ...]],
    hooks: AnnotationHooks,
) -> Dict[int, AnnotationCurve]:
    """Harvest annotation curves (alignment strings, polylines, etc.)."""

    if hooks is None:
        raise ValueError("AnnotationHooks must be provided for annotation extraction.")

    annotations: Dict[int, AnnotationCurve] = {}
    try:
        contexts = [
            ctx
            for ctx in (ifc_file.by_type("IfcGeometricRepresentationContext") or [])
            if str(getattr(ctx, "ContextType", "") or "").strip().lower()
            == "annotation"
            or str(getattr(ctx, "ContextIdentifier", "") or "").strip().lower()
            == "annotation"
        ]
    except Exception:
        contexts = []
    context_ids = set()
    for ctx in contexts:
        try:
            context_ids.add(ctx.id())
        except Exception:
            continue

    annotation_rep_types = {
        "annotation",
        "annotation2d",
        "curve",
        "curve2d",
        "curve3d",
        "geometriccurveset",
        "geometricset",
    }
    annotation_ident_tokens = {"annotation", "alignment"}

    def _hierarchy_for(ent) -> Tuple[Tuple[str, Optional[int]], ...]:
        try:
            key = ent.id()
        except Exception:
            key = id(ent)
        cached = hierarchy_cache.get(key)
        if cached is None:
            cached = hooks.collect_spatial_hierarchy(ent)
            hierarchy_cache[key] = cached
        return cached

    for product in ifc_file.by_type("IfcProduct") or []:
        if hooks.entity_on_hidden_layer(product):
            continue
        rep = getattr(product, "Representation", None)
        if not rep:
            continue
        hierarchy = _hierarchy_for(product)
        name = hooks.entity_label(product)
        try:
            placement_np = hooks.object_placement_to_np(
                getattr(product, "ObjectPlacement", None)
            )
        except Exception:
            placement_np = np.eye(4, dtype=float)

        for rep_ctx in getattr(rep, "Representations", []) or []:
            ctx = getattr(rep_ctx, "ContextOfItems", None)
            rep_type = (
                str(getattr(rep_ctx, "RepresentationType", "") or "").strip().lower()
            )
            rep_ident = (
                str(getattr(rep_ctx, "RepresentationIdentifier", "") or "")
                .strip()
                .lower()
            )
            ctx_id = None
            if ctx is not None:
                try:
                    ctx_id = ctx.id()
                except Exception:
                    ctx_id = None
            ctx_is_ann = ctx_id in context_ids if ctx_id is not None else False
            rep_is_ann = (rep_type in annotation_rep_types) or any(
                t in rep_ident for t in annotation_ident_tokens
            )

            for item in getattr(rep_ctx, "Items", []) or []:
                if not hasattr(item, "is_a"):
                    continue

                item_ann = ctx_is_ann or rep_is_ann
                context_np = (
                    hooks.context_to_np(ctx)
                    if ctx is not None
                    else np.eye(4, dtype=float)
                )
                transform = context_np @ placement_np
                item_type = str(item.is_a() if hasattr(item, "is_a") else "").lower()

                if item.is_a("IfcMappedItem"):
                    src = getattr(item, "MappingSource", None)
                    mapped = getattr(src, "MappedRepresentation", None) if src else None
                    mapped_type = (
                        str(getattr(mapped, "RepresentationType", "") or "")
                        .strip()
                        .lower()
                        if mapped
                        else ""
                    )
                    mapped_ident = (
                        str(getattr(mapped, "RepresentationIdentifier", "") or "")
                        .strip()
                        .lower()
                        if mapped
                        else ""
                    )
                    mapped_ctx = (
                        getattr(mapped, "ContextOfItems", None) if mapped else None
                    )
                    mapped_ctx_np = (
                        hooks.context_to_np(mapped_ctx)
                        if mapped_ctx is not None
                        else np.eye(4, dtype=float)
                    )
                    if mapped_ctx is not None:
                        try:
                            ctx_id = mapped_ctx.id()
                        except Exception:
                            ctx_id = None
                        if ctx_id is not None and ctx_id in context_ids:
                            item_ann = True
                    if mapped_type in annotation_rep_types or any(
                        t in mapped_ident for t in annotation_ident_tokens
                    ):
                        item_ann = True
                    try:
                        transform = context_np @ (
                            mapped_ctx_np @ hooks.mapping_item_transform(product, item)
                        )
                    except Exception:
                        transform = context_np @ (mapped_ctx_np @ placement_np)
                else:
                    if (
                        item.is_a("IfcGeometricSet")
                        or item.is_a("IfcGeometricCurveSet")
                        or item.is_a("IfcPolyline")
                    ):
                        item_ann = True
                    elif "curve" in item_type and "surface" not in item_type:
                        item_ann = True

                if not item_ann:
                    continue

                pts = hooks.extract_curve_points(item)
                if len(pts) < 2:
                    continue
                try:
                    pts_world = hooks.transform_points(pts, transform)
                except Exception:
                    pts_world = pts
                if len(pts_world) < 2:
                    continue

                try:
                    step_id = item.id()
                except Exception:
                    step_id = id(item)
                annotations[step_id] = AnnotationCurve(
                    step_id=step_id, name=name, points=pts_world, hierarchy=hierarchy
                )

    return annotations
