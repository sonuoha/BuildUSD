"""
semantic_subcomponents.py
-------------------------

Best-effort semantic splitting of a single triangulated IFC product mesh into
logical sub-components (Panel / Frame / Glazing / Body / Hardware, etc.).

This module is designed to be **semantics-first**:

* It inspects IFC semantics (class, representation identifiers, shape aspects)
  to decide which *roles* we expect (panel, frame, glazing).
* It then uses per-face style groups + material ids to assign faces to roles.
* It does **not** rely purely on colour names, although name tokens are used
  as a last-resort fallback when exporters don't expose rich semantics.

It deliberately does **not** depend on OCC / BRep topology; it works entirely
on the triangulated iterator mesh and the existing style grouping machinery
used by ``process_ifc``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SemanticParts = Dict[str, Dict[str, Any]]  # label -> {vertices, faces, material_ids, style_groups}


_PANEL_TOKENS = ("PANEL", "LEAF", "SASH", "CASEMENT", "DOOR PANEL", "DOOR_LEAF", "SHUTTER")
_FRAME_TOKENS = ("FRAME", "LINING", "JAMB", "MULLION", "TRANSOM", "REVEAL")
_GLAZING_TOKENS = ("GLAZ", "GLASS", "IGU", "PANE")
_HARDWARE_TOKENS = ("HANDLE", "HINGE", "LOCK", "LATCH", "KEEPER", "STRIKE")


def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).strip().upper()


def _ifc_class(product: Any) -> str:
    try:
        if hasattr(product, "is_a"):
            return _norm(product.is_a())
    except Exception:
        pass
    return ""


def _material_name(entry: Any) -> str:
    if entry is None:
        return ""
    for attr in ("name", "Name", "ElementName", "Description", "label", "Label"):
        if hasattr(entry, attr):
            v = getattr(entry, attr)
            if isinstance(v, str) and v.strip():
                return _norm(v)
    if isinstance(entry, dict):
        for key in ("name", "Name", "ElementName", "Description", "label", "Label"):
            v = entry.get(key)
            if isinstance(v, str) and v.strip():
                return _norm(v)
    return ""


def _group_name(entry: Mapping[str, Any]) -> str:
    for key in ("name", "label", "id", "guid", "shapeAspect"):
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return _norm(v)
    mat = entry.get("material")
    if mat is not None:
        return _material_name(mat)
    return ""


def _aspect_name(aspect: Any) -> str:
    for attr in ("Name", "Description", "Identification"):
        try:
            v = getattr(aspect, attr, None)
        except Exception:
            v = None
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _classify_aspect_role(aspect: Any) -> Optional[str]:
    name = _norm(_aspect_name(aspect))
    if not name:
        return None
    if any(tok in name for tok in _PANEL_TOKENS):
        return "Panel"
    if any(tok in name for tok in _FRAME_TOKENS):
        return "Frame"
    if any(tok in name for tok in _GLAZING_TOKENS):
        return "Glazing"
    if any(tok in name for tok in _HARDWARE_TOKENS):
        return "Hardware"
    return None


def _classify_rep_identifier(rep: Any, ifc_class: str) -> Optional[str]:
    ident = _norm(getattr(rep, "RepresentationIdentifier", None))
    if not ident:
        return None
    if ident in {"PANEL", "LEAF"}:
        return "Panel"
    if ident in {"FRAME", "LINING"}:
        return "Frame"
    if ident in {"GLAZING", "GLASS"}:
        return "Glazing"
    if ident in {"BODY", "TESSellation", "FACET"}:
        return None
    if ifc_class in {"IFCDOOR", "IFCDOORSTANDARDCASE", "IFCWINDOW", "IFCWINDOWSTANDARDCASE"}:
        if "PANEL" in ident or "LEAF" in ident:
            return "Panel"
        if "FRAME" in ident or "LINING" in ident or "JAMB" in ident:
            return "Frame"
        if "GLAZ" in ident or "GLASS" in ident:
            return "Glazing"
    return None


def _compact_subset_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: Sequence[int],
    material_ids: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[int]]]:
    """Create a compacted (verts, faces, material_ids) for the given face indices."""
    if not face_indices:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int), None

    face_indices = np.asarray(face_indices, dtype=int)
    subset_faces = faces[face_indices]
    used_vertex_ids = np.unique(subset_faces.reshape(-1))
    remap = -np.ones(vertices.shape[0], dtype=int)
    remap[used_vertex_ids] = np.arange(used_vertex_ids.size, dtype=int)
    new_vertices = vertices[used_vertex_ids]
    new_faces = remap[subset_faces]

    subset_material_ids: Optional[List[int]] = None
    if material_ids is not None and len(material_ids) == faces.shape[0]:
        material_ids_arr = np.asarray(material_ids, dtype=int)
        subset_material_ids = material_ids_arr[face_indices].tolist()

    return new_vertices, new_faces, subset_material_ids


def split_product_by_semantic_roles(
    ifc_file: Any,
    product: Any,
    mesh_dict: Mapping[str, Any],
    material_ids: Optional[Sequence[int]],
    face_style_groups: Mapping[str, Mapping[str, Any]],
    *,
    materials: Optional[Sequence[Any]] = None,
    min_faces_per_part: int = 16,
) -> SemanticParts:
    """
    Best-effort split of a product's triangulated mesh into semantic parts.
    """
    verts = np.asarray(mesh_dict.get("vertices"), dtype=float)
    faces = np.asarray(mesh_dict.get("faces"), dtype=int)

    if verts.size == 0 or faces.size == 0:
        return {}

    face_count = faces.shape[0]
    if face_count < min_faces_per_part * 2:
        return {}

    ifc_cls = _ifc_class(product)

    expected_roles: List[str] = []
    if ifc_cls in {
        "IFCDOOR",
        "IFCDOORSTANDARDCASE",
        "IFCWINDOW",
        "IFCWINDOWSTANDARDCASE",
        "IFCCOLUMN",
        "IFCBEAM",
        "IFCMEMBER",
        "IFCPLATE",
        "IFCWALL",
        "IFCWALLSTANDARDCASE",
        "IFCSLAB",
    }:
        expected_roles = ["Panel", "Frame", "Glazing", "Core", "Cladding", "Insulation", "Lining", "Hardware"]
    elif ifc_cls in {"IFCCURTAINWALL"}:
        expected_roles = ["Panel", "Frame", "Mullion", "Transom"]
    elif ifc_cls in {"IFCRAILING"}:
        expected_roles = ["Frame", "Hardware"]

    rep_role_hints: Dict[Any, str] = {}
    rep_container = getattr(product, "Representation", None)
    reps = getattr(rep_container, "Representations", []) or []
    for rep in reps:
        role = _classify_rep_identifier(rep, ifc_cls)
        if role:
            rep_role_hints[rep] = role

    aspect_roles: Dict[Any, str] = {}
    for aspect in getattr(product, "HasShapeAspects", []) or []:
        role = _classify_aspect_role(aspect)
        if role:
            aspect_roles[aspect] = role

    if not rep_role_hints and not aspect_roles and not expected_roles:
        return {}

    group_role: Dict[str, str] = {}

    for group_key, entry in (face_style_groups or {}).items():
        gname = _group_name(entry)
        if not gname:
            continue
        for aspect, role in aspect_roles.items():
            aname = _norm(_aspect_name(aspect))
            if aname and aname in gname:
                group_role[group_key] = role
                break
        if group_role.get(group_key):
            continue
        for rep, role in rep_role_hints.items():
            ident = _norm(getattr(rep, "RepresentationIdentifier", None))
            if ident and ident in gname:
                group_role[group_key] = role
                break

    for group_key, entry in (face_style_groups or {}).items():
        if group_key in group_role:
            continue
        gname = _group_name(entry)
        if not gname:
            continue
        if "Panel" in expected_roles and any(tok in gname for tok in _PANEL_TOKENS):
            group_role[group_key] = "Panel"
        elif "Frame" in expected_roles and any(tok in gname for tok in _FRAME_TOKENS):
            group_role[group_key] = "Frame"
        elif "Glazing" in expected_roles and any(tok in gname for tok in _GLAZING_TOKENS):
            group_role[group_key] = "Glazing"
        elif "Hardware" in expected_roles and any(tok in gname for tok in _HARDWARE_TOKENS):
            group_role[group_key] = "Hardware"

    face_role: List[Optional[str]] = [None] * face_count
    for group_key, entry in (face_style_groups or {}).items():
        role = group_role.get(group_key)
        if not role:
            continue
        for idx in entry.get("faces", []) or []:
            try:
                fi = int(idx)
            except Exception:
                continue
            if 0 <= fi < face_count:
                face_role[fi] = role

    if not any(face_role):
        return {}

    default_label = "Body"
    if len(expected_roles) == 1:
        default_label = expected_roles[0]

    for i in range(face_count):
        if face_role[i] is None:
            face_role[i] = default_label

    label_to_faces: Dict[str, List[int]] = {}
    for idx, label in enumerate(face_role):
        if not label:
            continue
        label_to_faces.setdefault(label, []).append(idx)

    for label in list(label_to_faces.keys()):
        if len(label_to_faces[label]) < min_faces_per_part:
            if label != default_label:
                label_to_faces.setdefault(default_label, []).extend(label_to_faces[label])
            del label_to_faces[label]

    effective_labels = [lbl for lbl in label_to_faces.keys()]
    if len(effective_labels) <= 1:
        return {}

    parts: SemanticParts = {}
    for label, face_idx_list in label_to_faces.items():
        if not face_idx_list:
            continue
        sub_verts, sub_faces, sub_mat_ids = _compact_subset_mesh(
            verts, faces, face_idx_list, material_ids=material_ids
        )
        if sub_faces.size == 0:
            continue
        parts[label] = {
            "vertices": sub_verts,
            "faces": sub_faces,
            "material_ids": sub_mat_ids,
            "style_groups": {},
        }

    return parts
