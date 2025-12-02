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


def _get_tokens(tokens_dict: Optional[Dict[str, Sequence[str]]], key: str, defaults: Sequence[str]) -> Sequence[str]:
    if not tokens_dict:
        return defaults
    return tokens_dict.get(key) or defaults


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


def _classify_aspect_role(aspect: Any, tokens: Dict[str, Sequence[str]]) -> Optional[str]:
    name = _norm(_aspect_name(aspect))
    if not name:
        return None
    if any(tok in name for tok in tokens["Panel"]):
        return "Panel"
    if any(tok in name for tok in tokens["Frame"]):
        return "Frame"
    if any(tok in name for tok in tokens["Glazing"]):
        return "Glazing"
    if any(tok in name for tok in tokens["Hardware"]):
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
    semantic_tokens: Optional[Dict[str, Sequence[str]]] = None,
) -> SemanticParts:
    """
    Best-effort split of a product's triangulated mesh into semantic parts.
    """
    verts = np.asarray(mesh_dict.get("vertices"), dtype=float)
    faces = np.asarray(mesh_dict.get("faces"), dtype=int)

    if verts.size == 0 or faces.size == 0:
        return {}

    face_count = faces.shape[0]
    if material_ids is not None and len(material_ids) != face_count:
        # Safety check: mismatched material IDs can cause crashes or corruption.
        material_ids = None

    if face_count < min_faces_per_part * 2:
        return {}

    # Resolve tokens
    tokens = {
        "Panel": _get_tokens(semantic_tokens, "Panel", _PANEL_TOKENS),
        "Frame": _get_tokens(semantic_tokens, "Frame", _FRAME_TOKENS),
        "Glazing": _get_tokens(semantic_tokens, "Glazing", _GLAZING_TOKENS),
        "Hardware": _get_tokens(semantic_tokens, "Hardware", _HARDWARE_TOKENS),
    }

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
    aspect_by_guid: Dict[str, Any] = {}
    for aspect in getattr(product, "HasShapeAspects", []) or []:
        guid = getattr(aspect, "GlobalId", None)
        if guid:
            aspect_by_guid[guid] = aspect
        role = _classify_aspect_role(aspect, tokens)
        if role:
            aspect_roles[aspect] = role

    if not rep_role_hints and not aspect_roles and not expected_roles:
        return {}

    # Fallback: If we have no explicit style groups (e.g. from IfcStyledItem),
    # but we DO have material IDs on faces, synthesize groups so we can check material names.
    effective_style_groups = face_style_groups
    if not effective_style_groups and material_ids is not None and materials:
        from collections import defaultdict
        mat_to_faces: Dict[int, List[int]] = defaultdict(list)
        for f_idx, m_id in enumerate(material_ids):
            if m_id >= 0:
                mat_to_faces[m_id].append(f_idx)
        
        if mat_to_faces:
            effective_style_groups = {}
            for m_id, f_indices in mat_to_faces.items():
                if 0 <= m_id < len(materials):
                    mat_obj = materials[m_id]
                    # Create a synthetic group entry compatible with _group_name inspection
                    effective_style_groups[f"Material_{m_id}"] = {
                        "material": mat_obj,
                        "faces": f_indices,
                        "name": _material_name(mat_obj) # Helper to ensure _group_name finds it easily
                    }

    group_role: Dict[str, str] = {}

    for group_key, entry in (effective_style_groups or {}).items():
        # Robust linking: check for aspect GUIDs first
        aspect_ids = entry.get("aspect_ids")
        if aspect_ids:
            found_role = None
            for guid in aspect_ids:
                aspect = aspect_by_guid.get(guid)
                if aspect and aspect in aspect_roles:
                    found_role = aspect_roles[aspect]
                    break
            if found_role:
                group_role[group_key] = found_role
                continue

        gname = _group_name(entry)
        if not gname:
            # Fallback: construct granular label for unclassified parts
            item_type = entry.get("item_type")
            step_id = entry.get("step_id")
            mat_name = _material_name(entry.get("material"))
            
            # Prioritize item info if available for granular splitting
            if item_type and step_id:
                # e.g. Unclassified_IfcExtrudedAreaSolid_123
                # Clean up item type (remove 'Ifc' prefix if desired, but full name is safer)
                clean_type = item_type[3:] if item_type.startswith("Ifc") else item_type
                label = f"Unclassified_{clean_type}_{step_id}"
                # Optionally append material for clarity? 
                # User asked for "isolate the post, panel...". 
                # Item type + ID is unique. Material is helpful context.
                # Let's stick to unique ID for now.
                group_role[group_key] = label
                continue
            
            # Fallback to material name if no item info
            if mat_name:
                group_role[group_key] = f"Unclassified_{mat_name}"
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

    for group_key, entry in (effective_style_groups or {}).items():
        if group_key in group_role:
            continue
        gname = _group_name(entry)
        
        # If we have a granular label already generated above (when gname was empty), use it?
        # Wait, the logic above only runs if `not gname`. 
        # If `gname` exists (e.g. material name), we still want to potentially use item info if no role matched.
        
        if "Panel" in expected_roles and any(tok in gname for tok in tokens["Panel"]):
            group_role[group_key] = "Panel"
        elif "Frame" in expected_roles and any(tok in gname for tok in tokens["Frame"]):
            group_role[group_key] = "Frame"
        elif "Glazing" in expected_roles and any(tok in gname for tok in tokens["Glazing"]):
            group_role[group_key] = "Glazing"
        elif "Hardware" in expected_roles and any(tok in gname for tok in tokens["Hardware"]):
            group_role[group_key] = "Hardware"
        else:
            # No semantic role matched. Use granular item label.
            item_type = entry.get("item_type")
            step_id = entry.get("step_id")
            if item_type and step_id:
                clean_type = item_type[3:] if item_type.startswith("Ifc") else item_type
                group_role[group_key] = f"Unclassified_{clean_type}_{step_id}"
            elif gname:
                # Fallback to whatever gname is (likely material name)
                group_role[group_key] = f"Unclassified_{gname}"

    face_role: List[Optional[str]] = [None] * face_count
    for group_key, entry in (effective_style_groups or {}).items():
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
            # If we have other parts, mark this as Unclassified so it doesn't get merged into Body
            # unless Body is the only thing.
            face_role[i] = "Unclassified"

    label_to_faces: Dict[str, List[int]] = {}
    for idx, label in enumerate(face_role):
        if not label:
            continue
        label_to_faces.setdefault(label, []).append(idx)

    # Merge Unclassified into default_label if it's the only other part or if default_label exists
    # Actually, simpler logic: if we have "Unclassified" and "Body", keep both?
    # Or should "Unclassified" fallback to "Body"?
    # Let's say: if we have explicit roles, Unclassified stays Unclassified.
    # If we ONLY have Unclassified, it becomes Body.
    
    unique_labels = set(label_to_faces.keys())
    if "Unclassified" in unique_labels:
        if len(unique_labels) == 1:
             # Only Unclassified -> rename to Body (or default_label)
             label_to_faces[default_label] = label_to_faces.pop("Unclassified")
        elif default_label in unique_labels:
             # We have Body and Unclassified. Merge Unclassified into Body?
             # User requested "Only merge Unclassified into Body if Body is the only other part"
             # But here we have Body. So we merge?
             # Let's keep them separate as requested: "Introduce Unclassified... Only merge if Body is the only other part"
             # Wait, if Body is the only other part, then we have [Body, Unclassified]. 
             # Merging them makes sense to avoid fragmentation.
             # If we have [Frame, Panel, Unclassified], we keep Unclassified separate.
             if len(unique_labels) == 2:
                 label_to_faces[default_label].extend(label_to_faces.pop("Unclassified"))

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
