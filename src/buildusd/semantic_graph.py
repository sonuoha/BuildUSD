"""Semantic graph sidecar generation for converted BuildUSD models.

The graph intentionally stays outside the USD layers. It records what the
pipeline knows about IFC elements, their source materials, decomposition state,
and targeted enrichment work that downstream systems can request later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from .io_utils import PathLike, join_path, write_text


GRAPH_SCHEMA_VERSION = 1

DECOMPOSITION_LOGICAL_ONLY = "logical_only"
DECOMPOSITION_MATERIAL_SUBSET = "material_subset"
DECOMPOSITION_SEMANTIC_MESH_PARTS = "semantic_mesh_parts"
DECOMPOSITION_OCC_DETAIL_PARTS = "occ_detail_parts"
DECOMPOSITION_WHOLE_OBJECT = "whole_object"


@dataclass(frozen=True)
class CompositeTemplate:
    parts: tuple[str, ...]
    default_materials: Dict[str, str]


COMPOSITE_TEMPLATES: Dict[str, CompositeTemplate] = {
    "IFCWINDOW": CompositeTemplate(
        parts=("Frame", "Glazing", "Hardware"),
        default_materials={
            "Frame": "semantic:painted_or_powder_coated_frame",
            "Glazing": "semantic:clear_glazing",
            "Hardware": "semantic:brushed_metal_hardware",
        },
    ),
    "IFCWINDOWSTANDARDCASE": CompositeTemplate(
        parts=("Frame", "Glazing", "Hardware"),
        default_materials={
            "Frame": "semantic:painted_or_powder_coated_frame",
            "Glazing": "semantic:clear_glazing",
            "Hardware": "semantic:brushed_metal_hardware",
        },
    ),
    "IFCDOOR": CompositeTemplate(
        parts=("Panel", "Frame", "Hardware", "Glazing"),
        default_materials={
            "Panel": "semantic:door_panel",
            "Frame": "semantic:painted_or_powder_coated_frame",
            "Hardware": "semantic:brushed_metal_hardware",
            "Glazing": "semantic:clear_glazing",
        },
    ),
    "IFCDOORSTANDARDCASE": CompositeTemplate(
        parts=("Panel", "Frame", "Hardware", "Glazing"),
        default_materials={
            "Panel": "semantic:door_panel",
            "Frame": "semantic:painted_or_powder_coated_frame",
            "Hardware": "semantic:brushed_metal_hardware",
            "Glazing": "semantic:clear_glazing",
        },
    ),
    "IFCCURTAINWALL": CompositeTemplate(
        parts=("Panel", "Mullion", "Transom", "Glazing"),
        default_materials={
            "Panel": "semantic:curtain_panel",
            "Mullion": "semantic:aluminium_mullion",
            "Transom": "semantic:aluminium_transom",
            "Glazing": "semantic:clear_glazing",
        },
    ),
    "IFCRAILING": CompositeTemplate(
        parts=("Frame", "Hardware"),
        default_materials={
            "Frame": "semantic:painted_or_galvanized_metal",
            "Hardware": "semantic:brushed_metal_hardware",
        },
    ),
    "IFCSTAIR": CompositeTemplate(
        parts=("Flight", "Landing", "Railing"),
        default_materials={
            "Flight": "semantic:stair_flight",
            "Landing": "semantic:stair_landing",
            "Railing": "semantic:painted_or_galvanized_metal",
        },
    ),
}


def persist_semantic_graph(
    graph_dir: PathLike,
    base_name: str,
    caches: Any,
    *,
    ifc_path: Optional[PathLike] = None,
    stage_path: Optional[PathLike] = None,
    layers: Optional[Dict[str, Any]] = None,
    projected_crs: Optional[str] = None,
    geodetic_crs: Optional[str] = None,
) -> PathLike:
    """Build and persist a semantic graph sidecar for a converted model."""

    graph_path = join_path(graph_dir, f"{base_name}.semantic_graph.json")
    graph = build_semantic_graph(
        caches,
        base_name=base_name,
        ifc_path=ifc_path,
        stage_path=stage_path,
        layers=layers,
        projected_crs=projected_crs,
        geodetic_crs=geodetic_crs,
    )
    write_text(graph_path, json.dumps(graph, indent=2, sort_keys=True))
    return graph_path


def build_semantic_graph(
    caches: Any,
    *,
    base_name: str,
    ifc_path: Optional[PathLike] = None,
    stage_path: Optional[PathLike] = None,
    layers: Optional[Dict[str, Any]] = None,
    projected_crs: Optional[str] = None,
    geodetic_crs: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable graph for IFC/USD semantic consumers."""

    model_id = f"model:{_slug(base_name)}"
    nodes: list[Dict[str, Any]] = [
        {
            "id": model_id,
            "type": "Model",
            "label": base_name,
            "properties": _drop_none(
                {
                    "ifc_path": str(ifc_path) if ifc_path is not None else None,
                    "stage_path": str(stage_path) if stage_path is not None else None,
                    "layers": _json_safe(layers or {}),
                    "projected_crs": projected_crs,
                    "geodetic_crs": geodetic_crs,
                }
            ),
        }
    ]
    edges: list[Dict[str, Any]] = []
    enrichment_requests: list[Dict[str, Any]] = []
    spatial_nodes: dict[str, str] = {}
    material_nodes: dict[str, str] = {}

    for record in _iter_records(caches):
        element_id = _element_id(record)
        usd_path = _record_usd_path(record)
        ifc_class = _norm_class(getattr(record, "ifc_class", None))
        decomposition_quality = _decomposition_quality(record, ifc_class)
        source_materials = _source_materials(record)
        semantic_parts = _semantic_part_labels(record)
        material_evidence = _material_evidence(record, source_materials)

        nodes.append(
            {
                "id": element_id,
                "type": "Element",
                "label": getattr(record, "name", None) or element_id,
                "properties": _drop_none(
                    {
                        "ifc_class": ifc_class or None,
                        "step_id": getattr(record, "step_id", None),
                        "product_id": getattr(record, "product_id", None),
                        "guid": getattr(record, "guid", None),
                        "usd_path": usd_path,
                        "prototype_path": getattr(record, "prototype_path", None),
                        "decomposition_quality": decomposition_quality,
                        "source_materials": source_materials,
                        "semantic_parts": semantic_parts,
                        "attributes": _json_safe(
                            getattr(record, "attributes", {}) or {}
                        ),
                        "material_evidence": material_evidence,
                    }
                ),
            }
        )
        edges.append({"source": model_id, "target": element_id, "type": "contains"})

        previous_spatial_id = model_id
        for label, step_id in getattr(record, "hierarchy", ()) or ():
            spatial_id = (
                f"spatial:{step_id}"
                if step_id is not None
                else f"spatial:{_slug(label)}"
            )
            if spatial_id not in spatial_nodes:
                spatial_nodes[spatial_id] = label
                nodes.append(
                    {
                        "id": spatial_id,
                        "type": "SpatialNode",
                        "label": label,
                        "properties": _drop_none({"step_id": step_id}),
                    }
                )
            edges.append(
                {
                    "source": previous_spatial_id,
                    "target": spatial_id,
                    "type": "contains",
                }
            )
            previous_spatial_id = spatial_id
        if previous_spatial_id != model_id:
            edges.append(
                {
                    "source": previous_spatial_id,
                    "target": element_id,
                    "type": "contains",
                }
            )

        for material_name in source_materials:
            material_id = f"source_material:{_slug(material_name)}"
            if material_id not in material_nodes:
                material_nodes[material_id] = material_name
                nodes.append(
                    {
                        "id": material_id,
                        "type": "SourceMaterial",
                        "label": material_name,
                        "properties": {},
                    }
                )
            edges.append(
                {
                    "source": element_id,
                    "target": material_id,
                    "type": "hasSourceMaterial",
                }
            )

        part_nodes, request = _part_nodes_and_request(
            record,
            element_id=element_id,
            ifc_class=ifc_class,
            usd_path=usd_path,
            decomposition_quality=decomposition_quality,
        )
        nodes.extend(part_nodes)
        edges.extend(
            {
                "source": element_id,
                "target": part_node["id"],
                "type": "hasLogicalPart",
            }
            for part_node in part_nodes
        )
        if request is not None:
            enrichment_requests.append(request)

    return {
        "schema": GRAPH_SCHEMA_VERSION,
        "graph_type": "buildusd.semantic_graph",
        "model": model_id,
        "nodes": nodes,
        "edges": _dedupe_edges(edges),
        "enrichment_requests": enrichment_requests,
    }


def _iter_records(caches: Any) -> Iterable[Any]:
    instances = getattr(caches, "instances", {}) or {}
    return instances.values()


def _element_id(record: Any) -> str:
    guid = getattr(record, "guid", None)
    if guid:
        return f"element:guid:{guid}"
    return f"element:step:{getattr(record, 'step_id', 'unknown')}"


def _record_usd_path(record: Any) -> Optional[str]:
    value = getattr(record, "usd_path", None)
    return str(value) if value else None


def _norm_class(value: Any) -> str:
    return str(value or "").strip().upper()


def _decomposition_quality(record: Any, ifc_class: str) -> str:
    hint = getattr(record, "decomposition_quality_hint", None)
    if hint in {
        DECOMPOSITION_LOGICAL_ONLY,
        DECOMPOSITION_MATERIAL_SUBSET,
        DECOMPOSITION_SEMANTIC_MESH_PARTS,
        DECOMPOSITION_OCC_DETAIL_PARTS,
        DECOMPOSITION_WHOLE_OBJECT,
    }:
        return str(hint)
    if getattr(record, "semantic_parts", None):
        return DECOMPOSITION_SEMANTIC_MESH_PARTS
    detail_mesh = getattr(record, "detail_mesh", None)
    if detail_mesh is not None and getattr(detail_mesh, "faces", None):
        return DECOMPOSITION_OCC_DETAIL_PARTS
    if _has_material_subsets(record):
        return DECOMPOSITION_MATERIAL_SUBSET
    if ifc_class in COMPOSITE_TEMPLATES:
        return DECOMPOSITION_LOGICAL_ONLY
    return DECOMPOSITION_WHOLE_OBJECT


def _has_material_subsets(record: Any) -> bool:
    style_groups = getattr(record, "style_face_groups", None) or {}
    if len(style_groups) > 1:
        return True
    material_ids = list(getattr(record, "material_ids", None) or [])
    return len(set(material_ids)) > 1


def _semantic_part_labels(record: Any) -> list[str]:
    return sorted(str(label) for label in (getattr(record, "semantic_parts", {}) or {}))


def _source_materials(record: Any) -> list[str]:
    names: list[str] = []
    for entry in _material_entries(getattr(record, "materials", None)):
        name = _material_name(entry)
        if name:
            names.append(name)
    for entry in (getattr(record, "style_face_groups", None) or {}).values():
        name = _material_name(
            entry.get("material") if isinstance(entry, dict) else entry
        )
        if name:
            names.append(name)
    style_material = getattr(record, "style_material", None)
    name = _material_name(style_material)
    if name:
        names.append(name)
    return sorted(set(names))


def _material_entries(materials: Any) -> Iterable[Any]:
    if materials is None:
        return ()
    if isinstance(materials, dict):
        return materials.values()
    if isinstance(materials, (str, bytes)):
        return (materials,)
    try:
        return tuple(materials)
    except TypeError:
        return (materials,)


def _material_name(entry: Any) -> Optional[str]:
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry.strip() or None
    if isinstance(entry, dict):
        for key in ("name", "Name", "label", "Label", "material", "Material"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("name", "Name", "label", "Label", "ElementName", "Description"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _material_evidence(record: Any, source_materials: Sequence[str]) -> Dict[str, Any]:
    return _drop_none(
        {
            "source_materials": list(source_materials),
            "material_ids": list(getattr(record, "material_ids", None) or []),
            "style_group_count": len(getattr(record, "style_face_groups", None) or {}),
        }
    )


def _part_nodes_and_request(
    record: Any,
    *,
    element_id: str,
    ifc_class: str,
    usd_path: Optional[str],
    decomposition_quality: str,
) -> tuple[list[Dict[str, Any]], Optional[Dict[str, Any]]]:
    template = COMPOSITE_TEMPLATES.get(ifc_class)
    semantic_parts = getattr(record, "semantic_parts", {}) or {}
    semantic_part_paths = getattr(record, "semantic_part_prim_paths", {}) or {}
    labels = (
        tuple(semantic_parts.keys())
        if semantic_parts
        else (template.parts if template else ())
    )
    nodes: list[Dict[str, Any]] = []

    for label in labels:
        part_id = f"{element_id}:part:{_slug(label)}"
        represented_by = semantic_part_paths.get(label) or usd_path
        if semantic_parts:
            representation_status = DECOMPOSITION_SEMANTIC_MESH_PARTS
        elif decomposition_quality == DECOMPOSITION_MATERIAL_SUBSET:
            representation_status = DECOMPOSITION_MATERIAL_SUBSET
        else:
            representation_status = "inferred_only"
        desired_material = (
            template.default_materials.get(str(label)) if template else None
        )
        nodes.append(
            {
                "id": part_id,
                "type": "LogicalPart",
                "label": str(label),
                "properties": _drop_none(
                    {
                        "owner": element_id,
                        "represented_by": represented_by,
                        "representation_status": representation_status,
                        "desired_semantic_material": desired_material,
                    }
                ),
            }
        )

    if template is None or decomposition_quality != DECOMPOSITION_LOGICAL_ONLY:
        return nodes, None
    return nodes, {
        "kind": "targeted_decomposition",
        "reason": "Composite element has inferred logical parts but no part-level geometry.",
        "guid": getattr(record, "guid", None),
        "step_id": getattr(record, "step_id", None),
        "ifc_class": ifc_class,
        "current_usd_prim": usd_path,
        "requested_parts": list(template.parts),
        "suggested_detail_objects": [
            value
            for value in (
                getattr(record, "guid", None),
                getattr(record, "step_id", None),
            )
            if value is not None
        ],
    }


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _dedupe_edges(edges: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Dict[str, Any]] = []
    for edge in edges:
        key = (str(edge.get("source")), str(edge.get("target")), str(edge.get("type")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9_.$-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"
