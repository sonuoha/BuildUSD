"""Helpers for extracting IFC presentation styles and PBR-friendly materials.

The functions provided here favour IfcSurfaceStyle-based visual metadata and
fallback to IfcMaterial associations when no explicit rendering information is
available. They return lightweight dataclasses that can be consumed by the USD
authoring layer without dragging Omniverse/pxr dependencies into the IFC pass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
    if not style_mat:
        assoc_styles = _styles_from_material_associations(product)
        if assoc_styles:
            style_mat = assoc_styles[0]
    phys_name = _material_name_from_associations(product)
    if style_mat and phys_name and style_mat.name != phys_name:
        style_mat = replace(style_mat, name=f"{phys_name} ({style_mat.name})")
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

    collected: List[ifcopenshell.entity_instance] = []
    seen: Set[int] = set()

    def _entity_id(entity: Any) -> Optional[int]:
        if entity is None:
            return None
        try:
            return int(entity.id())
        except Exception:
            try:
                return int(getattr(entity, "id", None))
            except Exception:
                return None

    def visit(item: Any) -> None:
        if not item or not getattr(item, "is_a", None):
            return
        ent_id = _entity_id(item)
        if ent_id is not None and ent_id in seen:
            return
        if ent_id is not None:
            seen.add(ent_id)
        collected.append(item)

        type_name = item.is_a()
        if type_name == "IfcMappedItem":
            mapping_source = getattr(item, "MappingSource", None)
            mapped_rep = getattr(mapping_source, "MappedRepresentation", None) if mapping_source else None
            if mapped_rep is not None:
                for sub_item in getattr(mapped_rep, "Items", None) or []:
                    visit(sub_item)
        else:
            nested_items = getattr(item, "Items", None)
            if nested_items:
                for sub_item in nested_items:
                    visit(sub_item)

    for shape_rep in rep.Representations:
        for entry in getattr(shape_rep, "Items", None) or []:
            visit(entry)
    return collected


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


def _styles_from_layer_assignments(entity: Any) -> List[PBRMaterial]:
    assignments = getattr(entity, "LayerAssignments", None) or []
    if not assignments:
        return []
    materials: List[PBRMaterial] = []
    seen: Set[Tuple[float, float, float, str]] = set()
    for assignment in assignments:
        if not assignment or not getattr(assignment, "is_a", None):
            continue
        if not assignment.is_a("IfcPresentationLayerWithStyle"):
            continue
        layer_name = getattr(assignment, "Name", None)
        styles = getattr(assignment, "LayerStyles", None) or []
        for style in styles:
            mat = _pbr_from_surface_style(style)
            if not mat:
                continue
            if layer_name:
                pretty_name = layer_name
                if mat.name and mat.name not in (layer_name, f"{layer_name} ({mat.name})"):
                    pretty_name = f"{layer_name} ({mat.name})"
                mat = replace(mat, name=pretty_name)
            key = (round(mat.base_color[0], 6), round(mat.base_color[1], 6), round(mat.base_color[2], 6), mat.name)
            if key in seen:
                continue
            seen.add(key)
            materials.append(mat)
    return materials


def _styles_for_item(item: Any) -> List[PBRMaterial]:
    if not item:
        return []
    materials: List[PBRMaterial] = []
    styled_items = getattr(item, "StyledByItem", None) or []
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
    layer_materials = _styles_from_layer_assignments(item)
    if layer_materials:
        materials.extend(layer_materials)
    return materials


def _material_from_surface_style(product: ifcopenshell.entity_instance) -> Optional[PBRMaterial]:
    # Highest priority: explicit styles on representation items
    for item in _iter_representation_items(product):
        styles = _styles_for_item(item)
        if styles:
            return styles[0]

    # Fallback to styles applied directly to the product
    direct_styles = _styles_for_item(product)
    if direct_styles:
        return direct_styles[0]

    # Next, check presentation layers attached to the representation containers
    rep = getattr(product, "Representation", None)
    if rep:
        rep_styles = _styles_for_item(rep)
        if rep_styles:
            return rep_styles[0]
        representations = getattr(rep, "Representations", None) or []
        for shape_rep in representations:
            styles = _styles_for_item(shape_rep)
            if styles:
                return styles[0]

    return None


def _pbr_from_surface_style(style: ifcopenshell.entity_instance) -> Optional[PBRMaterial]:
    if not style or not style.is_a():
        return None
    if style.is_a("IfcSurfaceStyleWithTextures"):
        textures = getattr(style, "Textures", None) or []
        for texture in textures:
            colour = _colour_from_texture(texture)
            if colour:
                name = style.Name or getattr(texture, "Name", None) or "SurfaceStyleTexture"
                return PBRMaterial(name=name, base_color=colour)
        return PBRMaterial(name=style.Name or "SurfaceStyleTexture")
    if style.is_a("IfcSurfaceStyle"):
        elements = getattr(style, "Styles", []) or []
        rendering = next((elem for elem in elements if elem.is_a("IfcSurfaceStyleRendering")), None)
        if rendering is not None:
            surface_colour = getattr(rendering, "SurfaceColour", None)
            if surface_colour is not None:
                base_color = _as_rgb(surface_colour)
            else:
                diffuse = getattr(rendering, "DiffuseColour", None)
                base_color = _colour_or_factor(diffuse) or (0.8, 0.8, 0.8)
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


def _colour_from_texture(texture: Any) -> Optional[Tuple[float, float, float]]:
    if not texture:
        return None
    if getattr(texture, "is_a", None) and texture.is_a("IfcSurfaceTexture"):
        for attr in ("DiffuseColour", "SpecularColour", "ReflectionColour", "TransmissionColour", "Colour"):
            value = getattr(texture, attr, None)
            colour = _colour_or_factor(value)
            if colour:
                return colour
    return None


def _colour_or_factor(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if getattr(value, "is_a", None):
        if value.is_a("IfcColourRgb"):
            return _as_rgb(value)
        wrapped = getattr(value, "wrappedValue", None)
        if wrapped is not None:
            try:
                scalar = float(wrapped)
            except (TypeError, ValueError):
                return None
            return (scalar, scalar, scalar)
    if isinstance(value, (tuple, list)):
        if len(value) >= 3:
            try:
                return (
                    max(0.0, min(1.0, float(value[0]))),
                    max(0.0, min(1.0, float(value[1]))),
                    max(0.0, min(1.0, float(value[2]))),
                )
            except (TypeError, ValueError):
                return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return (scalar, scalar, scalar)


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
    base_styles = _styles_for_item(item)
    base_style = base_styles[0] if base_styles else None
    base_style_name = base_style.name if base_style else None

    if face_count <= 0:
        return groups, 0

    assigned: List[bool] = [False] * face_count
    colour_entries: Dict[Tuple[float, float, float], Dict[str, Any]] = {}

    for colour_map in maps:
        colours_entity = getattr(colour_map, "Colours", None)
        colour_list = getattr(colours_entity, "ColourList", None) if colours_entity else None
        indices = getattr(colour_map, "ColourIndex", None) or []
        if not colour_list or not indices:
            continue
        for face_idx, colour_indices in enumerate(indices):
            if colour_indices in (None, []):
                continue
            if face_idx < 0 or face_idx >= face_count:
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
            key = (round(base_color[0], 5), round(base_color[1], 5), round(base_color[2], 5))
            entry = colour_entries.setdefault(key, {"faces": []})
            entry["faces"].append(face_idx)
            assigned[face_idx] = True

    if colour_entries:
        name_counts: Dict[str, int] = {}
        for key, entry in colour_entries.items():
            base_color = (
                float(key[0]),
                float(key[1]),
                float(key[2]),
            )
            if base_style is not None:
                candidate = replace(base_style, base_color=base_color)
            else:
                candidate_name = base_style_name or "Colour"
                candidate = PBRMaterial(name=candidate_name, base_color=base_color)
            base_name = candidate.name or "Colour"
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            if count:
                candidate = replace(candidate, name=f"{base_name}_{count}")
            entry["material"] = candidate
            material_key = ("colour", candidate.name or base_name, key)
            groups[material_key] = {"material": candidate, "faces": entry["faces"]}

        if base_style is not None:
            unassigned = [index for index, flag in enumerate(assigned) if not flag]
            if unassigned:
                groups[("styleFallback", base_style.name or "Style")] = {
                    "material": base_style,
                    "faces": unassigned,
                }
        return groups, face_count

    # Fallback to styles applied to the entire item (styled or via layer)
    if base_style is not None and face_count:
        key = ("style", 0, base_style.name or "Style")
        groups[key] = {"material": base_style, "faces": list(range(face_count))}
    return groups, face_count


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


def _styles_from_material_associations(product: ifcopenshell.entity_instance) -> List[PBRMaterial]:
    associations = getattr(product, "HasAssociations", None) or []
    if not associations:
        return []
    materials: List[PBRMaterial] = []
    visited_entities: Set[int] = set()
    for rel in associations:
        if not rel or not getattr(rel, "is_a", None) or not rel.is_a("IfcRelAssociatesMaterial"):
            continue
        relating = getattr(rel, "RelatingMaterial", None)
        if relating is None:
            continue
        materials.extend(_collect_styles_from_material_definition(relating, visited_entities))
    return _dedupe_materials(materials)


def _collect_styles_from_material_definition(definition: Any, visited: Set[int]) -> List[PBRMaterial]:
    stack: List[Any] = [definition]
    collected: List[PBRMaterial] = []
    while stack:
        entity = stack.pop()
        if not entity or not getattr(entity, "is_a", None):
            continue
        entity_id = None
        try:
            entity_id = entity.id()
        except Exception:
            try:
                entity_id = int(getattr(entity, "id", None))
            except Exception:
                entity_id = None
        if entity_id is not None:
            if entity_id in visited:
                continue
            visited.add(entity_id)
        if entity.is_a("IfcMaterialDefinitionRepresentation"):
            collected.extend(_styles_from_material_representation(entity))
            continue
        representations = []
        has_representation = getattr(entity, "HasRepresentation", None)
        if has_representation:
            representations.extend(has_representation)
        has_representations = getattr(entity, "HasRepresentations", None)
        if has_representations:
            representations.extend(has_representations)
        for rep in representations:
            stack.append(rep)
        stack.extend(_child_material_definitions(entity))
    return collected


def _child_material_definitions(entity: Any) -> List[Any]:
    try:
        type_name = entity.is_a()
    except Exception:
        return []
    children: List[Any] = []
    if type_name == "IfcMaterialLayerSetUsage":
        children.append(getattr(entity, "ForLayerSet", None))
    elif type_name == "IfcMaterialLayerSet":
        children.extend(getattr(entity, "MaterialLayers", None) or [])
    elif type_name == "IfcMaterialLayer":
        children.append(getattr(entity, "Material", None))
    elif type_name == "IfcMaterialProfileSetUsage":
        children.append(getattr(entity, "ForProfileSet", None))
    elif type_name == "IfcMaterialProfileSet":
        children.extend(getattr(entity, "MaterialProfiles", None) or [])
    elif type_name == "IfcMaterialProfile":
        children.append(getattr(entity, "Material", None))
    elif type_name == "IfcMaterialConstituentSet":
        children.extend(getattr(entity, "Constituents", None) or [])
    elif type_name == "IfcMaterialConstituent":
        children.append(getattr(entity, "Material", None))
    elif type_name == "IfcMaterialList":
        children.extend(getattr(entity, "Materials", None) or [])
    elif type_name == "IfcMaterial":
        reps = getattr(entity, "HasRepresentation", None) or []
        children.extend(reps)
    return [child for child in children if child is not None]


def _styles_from_material_representation(material_rep: Any) -> List[PBRMaterial]:
    materials: List[PBRMaterial] = []
    representations = getattr(material_rep, "Representations", None) or []
    for representation in representations:
        items = getattr(representation, "Items", None) or []
        for item in items:
            if not item:
                continue
            if item.is_a("IfcStyledItem"):
                materials.extend(_materials_from_styled_item(item))
            else:
                materials.extend(_styles_for_item(item))
    return materials


def _materials_from_styled_item(styled_item: Any) -> List[PBRMaterial]:
    materials: List[PBRMaterial] = []
    styles = getattr(styled_item, "Styles", None) or []
    for style in styles:
        if style.is_a("IfcPresentationStyleAssignment"):
            for sub_style in getattr(style, "Styles", None) or []:
                mat = _pbr_from_surface_style(sub_style)
                if mat:
                    materials.append(mat)
        else:
            mat = _pbr_from_surface_style(style)
            if mat:
                materials.append(mat)
    return materials


def _dedupe_materials(materials: List[PBRMaterial]) -> List[PBRMaterial]:
    deduped: List[PBRMaterial] = []
    seen: Set[Tuple[float, float, float, str]] = set()
    for mat in materials:
        if not mat:
            continue
        key = (
            round(mat.base_color[0], 6),
            round(mat.base_color[1], 6),
            round(mat.base_color[2], 6),
            mat.name or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mat)
    return deduped
