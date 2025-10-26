"""Helpers for extracting IFC presentation styles and PBR-friendly materials.

The functions provided here favour IfcSurfaceStyle-based visual metadata and
fallback to IfcMaterial associations when no explicit rendering information is
available.  They return lightweight dataclasses that can be consumed by the USD
authoring layer without dragging Omniverse/pxr dependencies into the IFC pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import re

import ifcopenshell
import ifcopenshell.util.element

__all__ = [
    "PBRMaterial",
    "build_material_for_product",
    "extract_uvs_for_product",
    "extract_face_style_groups",
]


@dataclass
class PBRMaterial:
    name: str
    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    opacity: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_color_tex: Optional[str] = None
    normal_tex: Optional[str] = None
    orm_tex: Optional[str] = None


@dataclass
class MeshUV:
    uvs: List[Tuple[float, float]]


def extract_uvs_for_product(product: ifcopenshell.entity_instance) -> Optional[MeshUV]:
    for item in _iter_representation_items(product):
        if not item or not item.is_a():
            continue
        if item.is_a("IfcTessellatedFaceSet") or item.is_a("IfcTriangulatedFaceSet"):
            uv = _resolve_uv_ifc4_polygonal_texture_map(item)
            if uv is not None:
                return uv
    return None


def build_material_for_product(
    product: ifcopenshell.entity_instance,
    prefer_surface_style: bool = True,
) -> Optional[PBRMaterial]:
    style_mat = _material_from_surface_style(product) if prefer_surface_style else None
    phys_name = _material_name_from_associations(product)
    if style_mat and phys_name and style_mat.name != phys_name:
        style_mat.name = f"{phys_name} ({style_mat.name})"
        return style_mat
    if style_mat:
        return style_mat
    if phys_name:
        return PBRMaterial(name=phys_name)
    default_name = _default_name_for_product(product)
    return PBRMaterial(name=default_name)


# --- internal helpers -----------------------------------------------------

def _iter_representation_items(product: ifcopenshell.entity_instance) -> Iterable[ifcopenshell.entity_instance]:
    rep = getattr(product, "Representation", None)
    if not rep or not getattr(rep, "Representations", None):
        return []
    items: List[ifcopenshell.entity_instance] = []
    for shape_rep in rep.Representations:
        entries = getattr(shape_rep, "Items", None)
        if entries:
            items.extend(entries)
    return items


def _resolve_uv_ifc4_polygonal_texture_map(
    tess: ifcopenshell.entity_instance,
) -> Optional[MeshUV]:
    model = getattr(tess, "_ifc", None) or getattr(tess, "wrapped_data", None)
    if model is None:
        return None
    try:
        maps = model.by_type("IfcIndexedPolygonalTextureMap")
    except Exception:
        return None
    for mapping in maps or []:
        if mapping.MappedTo != tess:
            continue
        tvl = getattr(mapping, "TexCoords", None)
        coords = getattr(tvl, "TexCoordsList", None) if tvl else None
        if not coords:
            continue
        uv_pool = [(float(u), float(v)) for (u, v) in coords]
        indices = mapping.TexCoordIndex or []
        flattened: List[Tuple[float, float]] = []
        for face in indices:
            for idx in face:
                flattened.append(uv_pool[idx - 1])
        if flattened:
            return MeshUV(uvs=flattened)
    return None


def _material_from_surface_style(product: ifcopenshell.entity_instance) -> Optional[PBRMaterial]:
    for item in _iter_representation_items(product):
        styled_items = getattr(item, "StyledByItem", None) or []
        for styled in styled_items:
            for style in getattr(styled, "Styles", []) or []:
                if style.is_a("IfcPresentationStyleAssignment"):
                    for sub in getattr(style, "Styles", []) or []:
                        mat = _pbr_from_surface_style(sub)
                        if mat:
                            return mat
                else:
                    mat = _pbr_from_surface_style(style)
                    if mat:
                        return mat
    return None


def _pbr_from_surface_style(style: ifcopenshell.entity_instance) -> Optional[PBRMaterial]:
    if not style or not style.is_a():
        return None
    if style.is_a("IfcSurfaceStyle"):
        elements = getattr(style, "Styles", []) or []
        rendering = next((elem for elem in elements if elem.is_a("IfcSurfaceStyleRendering")), None)
        if rendering is not None:
            base_color = _as_rgb(rendering.SurfaceColour)
            opacity = 1.0 - float(getattr(rendering, "Transparency", 0.0) or 0.0)
            spec = _as_rgb(getattr(rendering, "SpecularColour", None))
            spec_level = max(spec) if spec else 0.0
            roughness = max(0.05, 1.0 - min(0.95, spec_level))
            return PBRMaterial(
                name=style.Name or "SurfaceStyle",
                base_color=base_color,
                opacity=opacity,
                roughness=roughness,
            )
        shading = next((elem for elem in elements if elem.is_a("IfcSurfaceStyleShading")), None)
        if shading is not None:
            base_color = _as_rgb(shading.SurfaceColour)
            return PBRMaterial(name=style.Name or "SurfaceStyle", base_color=base_color)
    if style.is_a("IfcSurfaceStyleRendering"):
        base_color = _as_rgb(style.SurfaceColour)
        opacity = 1.0 - float(getattr(style, "Transparency", 0.0) or 0.0)
        return PBRMaterial(name=style.Name or "Rendering", base_color=base_color, opacity=opacity)
    return None


def _as_rgb(col) -> Tuple[float, float, float]:
    if not col:
        return (0.8, 0.8, 0.8)
    r = float(getattr(col, "Red", getattr(col, "R", 0.8)) or 0.8)
    g = float(getattr(col, "Green", getattr(col, "G", 0.8)) or 0.8)
    b = float(getattr(col, "Blue", getattr(col, "B", 0.8)) or 0.8)
    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
    )


def _material_name_from_associations(product: ifcopenshell.entity_instance) -> Optional[str]:
    try:
        mats = ifcopenshell.util.element.get_material(product)
    except Exception:
        mats = None
    if not mats:
        return None

    names: List[str] = []

    def collect(entry):
        if entry is None:
            return
        if isinstance(entry, list):
            for child in entry:
                collect(child)
            return
        if isinstance(entry, dict):
            if "materials" in entry and isinstance(entry["materials"], list):
                for child in entry["materials"]:
                    collect(child)
            elif entry.get("name"):
                names.append(str(entry["name"]))
            return
        candidate = getattr(entry, "Name", None)
        if candidate:
            names.append(str(candidate))

    collect(mats)
    unique: List[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    if not unique:
        return None
    return " | ".join(unique)


def _default_name_for_product(product: ifcopenshell.entity_instance) -> str:
    label = getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "IfcProduct"
    type_name = product.is_a() if hasattr(product, "is_a") else "IfcProduct"
    return f"{type_name}_{label}"


def extract_face_style_groups(product: ifcopenshell.entity_instance) -> Dict[str, Dict[str, Any]]:
    combined: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    face_offset = 0
    for item in _iter_representation_items(product):
        item_groups, face_count = _face_style_groups_from_item(item)
        if face_count == 0:
            continue
        for key, entry in item_groups.items():
            faces = [face_offset + int(idx) for idx in entry.get("faces", [])]
            if not faces:
                continue
            grouped = combined.setdefault(key, {"material": entry.get("material"), "faces": []})
            grouped["faces"].extend(faces)
        face_offset += face_count
    result: Dict[str, Dict[str, Any]] = {}
    for index, (key, entry) in enumerate(combined.items()):
        material = entry.get("material")
        name_hint = getattr(material, "name", None) or "Style"
        token = _sanitize_style_token(f"{name_hint}_{index}")
        result[token] = {"material": material, "faces": entry.get("faces", [])}
    return result


def _face_style_groups_from_item(item: ifcopenshell.entity_instance) -> Tuple[Dict[Tuple[Any, ...], Dict[str, Any]], int]:
    face_count = _face_count_from_item(item)
    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    maps = getattr(item, "HasColours", None) or []
    for colour_map in maps:
        colours_entity = getattr(colour_map, "Colours", None)
        colour_list = getattr(colours_entity, "ColourList", None) if colours_entity else None
        indices = getattr(colour_map, "ColourIndex", None) or []
        if not colour_list or not indices:
            continue
        for face_idx, colour_indices in enumerate(indices):
            if colour_indices in (None, []):
                continue
            if isinstance(colour_indices, int):
                colour_index = int(colour_indices) - 1
            else:
                try:
                    first_index = (colour_indices or [0])[0]
                    colour_index = int(first_index) - 1
                except Exception:
                    continue
            if colour_index < 0 or colour_index >= len(colour_list):
                continue
            raw_colour = colour_list[colour_index]
            try:
                r, g, b = (float(raw_colour[0]), float(raw_colour[1]), float(raw_colour[2]))
            except Exception:
                continue
            base_color = (
                max(0.0, min(1.0, r)),
                max(0.0, min(1.0, g)),
                max(0.0, min(1.0, b)),
            )
            name_hint = getattr(colour_map, "Name", None) or f"Colour_{colour_index}"
            pbr = PBRMaterial(name=name_hint, base_color=base_color)
            key = ("colour", round(base_color[0], 5), round(base_color[1], 5), round(base_color[2], 5))
            entry = groups.setdefault(key, {"material": pbr, "faces": []})
            entry["faces"].append(face_idx)
    if groups:
        return groups, face_count
    # Fallback to styles applied to the entire item
    styles = _styles_for_item(item)
    if styles and face_count:
        for idx, mat in enumerate(styles):
            key = ("style", idx, mat.name or "Style")
            groups[key] = {"material": mat, "faces": list(range(face_count))}
            break
    return groups, face_count


def _styles_for_item(item: ifcopenshell.entity_instance) -> List[PBRMaterial]:
    styled_items = getattr(item, "StyledByItem", None) or []
    materials: List[PBRMaterial] = []
    for styled in styled_items:
        for style in getattr(styled, "Styles", []) or []:
            if style.is_a("IfcPresentationStyleAssignment"):
                for sub_style in getattr(style, "Styles", []) or []:
                    mat = _pbr_from_surface_style(sub_style)
                    if mat:
                        materials.append(mat)
            else:
                mat = _pbr_from_surface_style(style)
                if mat:
                    materials.append(mat)
    return materials


def _face_count_from_item(item: ifcopenshell.entity_instance) -> int:
    coord_index = getattr(item, "CoordIndex", None)
    if coord_index:
        return len(coord_index)
    faces = getattr(item, "Faces", None)
    if faces:
        return len(faces)
    return 0


def _sanitize_style_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", value)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = "Style"
    if token[0].isdigit():
        token = f"Style_{token}"
    return token[:64]
