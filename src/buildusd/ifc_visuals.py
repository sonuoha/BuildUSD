"""IFC → PBR Material Extraction (IFC 8.9.3.10 Compliant + Backward Compatible)"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
import re

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.representation

__all__ = [
    "PBRMaterial",
    "build_material_for_product",
    "extract_face_style_groups",
    "extract_uvs_for_product",
]


# === PBR DATACLASS ===
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


# === UV DATACLASS ===
@dataclass
class MeshUV:
    uvs: List[Tuple[float, float]]


# === PUBLIC API: BACKWARD COMPATIBLE ===

def build_material_for_product(
    product: ifcopenshell.entity_instance,
    prefer_surface_style: bool = True,  # ← kept for compatibility (ignored)
) -> Optional[PBRMaterial]:
    """
    Backward-compatible wrapper.
    Always follows IFC 8.9.3.10 precedence.
    Returns single PBRMaterial (highest priority).
    """
    model = product.file
    result = _build_material_for_product_internal(product, model)

    # Priority: per-face > object > fallback
    if result.get("per_face"):
        return next(iter(result["per_face"].values()))
    if result.get("object"):
        return result["object"]
    return result["fallback"]


def extract_face_style_groups(
    product: ifcopenshell.entity_instance
) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible wrapper."""
    model = product.file
    return _extract_face_style_groups_internal(product, model)


def extract_uvs_for_product(product: ifcopenshell.entity_instance) -> Optional[MeshUV]:
    """Extract UVs from IfcIndexedPolygonalTextureMap (IFC4)."""
    for item in _iter_representation_items(product):
        if item.is_a(("IfcTessellatedFaceSet", "IfcTriangulatedFaceSet")):
            uv = _resolve_uv_ifc4_polygonal_texture_map(item)
            if uv:
                return uv
    return None


# === INTERNAL: HARDENED LOGIC ===

def _build_material_for_product_internal(product, model) -> Dict[str, Any]:
    result = {}

    # 1. Per-face (highest priority)
    face_styles = get_face_styles(model, product)
    if face_styles:
        result["per_face"] = {item_id: _to_pbr(styles[0]) for item_id, styles in face_styles.items()}

    # 2. Object-level (A→E precedence)
    obj_styles = get_surface_styles(model, product)
    if obj_styles:
        result["object"] = _to_pbr(obj_styles[0])

    # 3. Fallback: material name
    name = _material_name_from_associations(product) or _default_name_for_product(product)
    result["fallback"] = PBRMaterial(name=name)

    return result


# === IFC 8.9.3.10 PRECEDENCE: A → B → C → D → E ===

def get_surface_styles(model: ifcopenshell.file, product: ifcopenshell.entity_instance) -> List[ifcopenshell.entity_instance]:
    styles = []

    # A: IfcStyledItem on geometry
    for rep in (product.Representation.Representations if product.Representation else []):
        for item in rep.Items:
            for styled in getattr(item, "StyledByItem", []):
                for psa in styled.Styles:
                    styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])
    if styles:
        return styles

    # B/C: Material → IfcMaterialDefinitionRepresentation (layers, profiles, constituents)
    for rel in getattr(product, "HasAssociations", []):
        if rel.is_a("IfcRelAssociatesMaterial"):
            mat = rel.RelatingMaterial
            styles.extend(_styles_from_material_definition(mat))
            if styles:
                return styles

    # D: IfcShapeAspect
    for aspect in getattr(product, "HasShapeAspects", []):
        for mdr in getattr(aspect, "Representation", []):
            for rep in getattr(mdr, "Representations", []):
                for item in getattr(rep, "Items", []):
                    for styled in getattr(item, "StyledByItem", []):
                        for psa in styled.Styles:
                            styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])
        if styles:
            return styles

    # E: Type-level material
    if typ := ifcopenshell.util.element.get_type(product):
        for rel in getattr(typ, "HasAssociations", []):
            if rel.is_a("IfcRelAssociatesMaterial"):
                mat = rel.RelatingMaterial
                styles.extend(_styles_from_material_definition(mat))
                if styles:
                    return styles

    return []


def _styles_from_material_definition(mat: ifcopenshell.entity_instance) -> List[ifcopenshell.entity_instance]:
    styles = []
    visited = set()

    def collect(entity):
        if not entity or entity.id() in visited:
            return
        visited.add(entity.id())

        if entity.is_a("IfcMaterialDefinitionRepresentation"):
            for rep in getattr(entity, "Representations", []):
                for item in getattr(rep, "Items", []):
                    if item.is_a("IfcStyledItem"):
                        for psa in item.Styles:
                            styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])
                    else:
                        for psa in getattr(item, "Styles", []):
                            styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])

        # Recurse into sets
        children = []
        if entity.is_a("IfcMaterialLayerSetUsage"):
            children.append(entity.ForLayerSet)
        elif entity.is_a("IfcMaterialLayerSet"):
            children.extend(entity.MaterialLayers or [])
        elif entity.is_a("IfcMaterialLayer"):
            children.append(entity.Material)
        elif entity.is_a("IfcMaterialProfileSetUsage"):
            children.append(entity.ForProfileSet)
        elif entity.is_a("IfcMaterialProfileSet"):
            children.extend(entity.MaterialProfiles or [])
        elif entity.is_a("IfcMaterialProfile"):
            children.append(entity.Material)
        elif entity.is_a("IfcMaterialConstituentSet"):
            children.extend(entity.MaterialConstituents or [])
        elif entity.is_a("IfcMaterialConstituent"):
            children.append(entity.Material)
        elif entity.is_a("IfcMaterialList"):
            children.extend(entity.Materials or [])
        elif entity.is_a("IfcMaterial"):
            children.extend(entity.HasRepresentation or [])

        for child in children:
            collect(child)

    collect(mat)
    return styles


# === PER-FACE MAPPING ===

def get_face_styles(model: ifcopenshell.file, product: ifcopenshell.entity_instance) -> Dict[int, List[ifcopenshell.entity_instance]]:
    mapping = {}
    for rep in (product.Representation.Representations if product.Representation else []):
        for item in rep.Items:
            _collect_styled_items(item, mapping)
    return mapping


def _collect_styled_items(item, mapping):
    for styled in getattr(item, "StyledByItem", []):
        styles = []
        for psa in styled.Styles:
            styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])
        if styles:
            mapping[id(item)] = styles

    if item.is_a("IfcBooleanResult"):
        _collect_styled_items(item.FirstOperand, mapping)
        _collect_styled_items(item.SecondOperand, mapping)
    elif item.is_a("IfcMappedItem"):
        _collect_styled_items(item.MappingSource.MappedRepresentation.Items[0], mapping)
    elif hasattr(item, "Operands"):
        for op in item.Operands:
            _collect_styled_items(op, mapping)


# === FACE GROUPING ===

def _extract_face_style_groups_internal(product, model) -> Dict[str, Dict[str, Any]]:
    face_styles = get_face_styles(model, product)
    if not face_styles:
        return {}

    combined = {}
    face_offset = 0
    for item in _iter_representation_items(product):
        item_groups, face_count = _face_style_groups_from_item(item, face_styles)
        for key, entry in item_groups.items():
            faces = [face_offset + idx for idx in entry.get("faces", [])]
            if not faces:
                continue
            grouped = combined.setdefault(key, {"material": entry["material"], "faces": []})
            grouped["faces"].extend(faces)
        face_offset += face_count

    result = {}
    for i, (key, entry) in enumerate(combined.items()):
        mat = entry["material"]
        token = _sanitize_style_token(f"{mat.name}_{i}")
        result[token] = {"material": mat, "faces": entry["faces"]}
    return result


def _face_style_groups_from_item(item, face_styles) -> Tuple[Dict, int]:
    face_count = _face_count_from_item(item)
    if face_count == 0:
        return {}, 0

    item_id = id(item)
    if item_id in face_styles:
        mat = _to_pbr(face_styles[item_id][0])
        return {("styled", mat.name): {"material": mat, "faces": list(range(face_count))}}, face_count

    # Fallback: IfcIndexedColourMap
    groups = {}
    maps = getattr(item, "HasColours", []) or []
    assigned = [False] * face_count
    color_entries = {}

    for cmap in maps:
        colors = getattr(cmap.Colours, "ColourList", []) if cmap.Colours else []
        indices = getattr(cmap, "ColourIndex", []) or []
        for face_idx, idx_list in enumerate(indices):
            if not idx_list or face_idx >= face_count:
                continue
            idx = idx_list[0] - 1 if isinstance(idx_list, (list, tuple)) else idx_list - 1
            if idx < 0 or idx >= len(colors):
                continue
            r, g, b = colors[idx]
            key = (round(r, 5), round(g, 5), round(b, 5))
            entry = color_entries.setdefault(key, {"faces": []})
            entry["faces"].append(face_idx)
            assigned[face_idx] = True

    if color_entries:
        for key, entry in color_entries.items():
            mat = PBRMaterial(name="Color", base_color=(key[0], key[1], key[2]))
            groups[("color", key)] = {"material": mat, "faces": entry["faces"]}

    unassigned = [i for i, f in enumerate(assigned) if not f]
    if unassigned:
        mat = PBRMaterial(name="Default")
        groups[("default",)] = {"material": mat, "faces": unassigned}

    return groups, face_count


def _face_count_from_item(item) -> int:
    if ci := getattr(item, "CoordIndex", None):
        return len(ci)
    if faces := getattr(item, "Faces", None):
        return len(faces)
    return 0


# === REPRESENTATION ITERATOR ===
def _iter_representation_items(product):
    rep = getattr(product, "Representation", None)
    if not rep or not getattr(rep, "Representations", None):
        return
    seen = set()
    def visit(item):
        if not item or id(item) in seen:
            return
        seen.add(id(item))
        yield item
        if item.is_a("IfcMappedItem"):
            for sub in getattr(item.MappingSource.MappedRepresentation, "Items", []):
                yield from visit(sub)
        else:
            for sub in getattr(item, "Items", []):
                yield from visit(sub)
    for shape_rep in rep.Representations:
        for entry in getattr(shape_rep, "Items", []):
            yield from visit(entry)


# === UV EXTRACTION ===
def _resolve_uv_ifc4_polygonal_texture_map(tess) -> Optional[MeshUV]:
    model = tess.file
    for mapping in model.by_type("IfcIndexedPolygonalTextureMap"):
        if mapping.MappedTo != tess:
            continue
        tvl = mapping.TexCoords
        coords = tvl.TexCoordsList if tvl else None
        if not coords:
            continue
        uv_pool = [(float(u), float(v)) for u, v in coords]
        indices = mapping.TexCoordIndex or []
        flattened = []
        for face in indices:
            for idx in face:
                flattened.append(uv_pool[idx - 1])
        if flattened:
            return MeshUV(uvs=flattened)
    return None


# === PBR CONVERSION ===
def _to_pbr(style: ifcopenshell.entity_instance) -> PBRMaterial:
    if style.is_a("IfcSurfaceStyleWithTextures"):
        for tex in getattr(style, "Textures", []) or []:
            if color := _color_from_texture(tex):
                return PBRMaterial(name=style.Name or "Texture", base_color=color)
        return PBRMaterial(name=style.Name or "Texture")

    if style.is_a("IfcSurfaceStyle"):
        elements = getattr(style, "Styles", []) or []
        rendering = next((e for e in elements if e.is_a("IfcSurfaceStyleRendering")), None)
        if rendering:
            base_color = _get_base_color(rendering)
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
        shading = next((e for e in elements if e.is_a("IfcSurfaceStyleShading")), None)
        if shading:
            return PBRMaterial(name=style.Name or "SurfaceStyle", base_color=_as_rgb(shading.SurfaceColour))

    if style.is_a("IfcSurfaceStyleRendering"):
        return PBRMaterial(
            name=style.Name or "Rendering",
            base_color=_get_base_color(style),
            opacity=1.0 - float(getattr(style, "Transparency", 0.0) or 0.0)
        )

    return PBRMaterial(name=style.Name or "Style")


def _get_base_color(rendering) -> Tuple[float, float, float]:
    if sc := getattr(rendering, "SurfaceColour", None):
        return _as_rgb(sc)
    if dc := getattr(rendering, "DiffuseColour", None):
        return _color_or_factor(dc) or (0.8, 0.8, 0.8)
    return (0.8, 0.8, 0.8)


def _as_rgb(col) -> Tuple[float, float, float]:
    if not col:
        return (0.8, 0.8, 0.8)
    r = float(getattr(col, "Red", 0.8))
    g = float(getattr(col, "Green", 0.8))
    b = float(getattr(col, "Blue", 0.8))
    return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))


def _color_from_texture(tex) -> Optional[Tuple[float, float, float]]:
    for attr in ("DiffuseColour", "SpecularColour", "ReflectionColour", "TransmissionColour", "Colour"):
        val = getattr(tex, attr, None)
        if color := _color_or_factor(val):
            return color
    return None


def _color_or_factor(val) -> Optional[Tuple[float, float, float]]:
    if not val:
        return None
    if val.is_a("IfcColourRgb"):
        return _as_rgb(val)
    if hasattr(val, "wrappedValue"):
        try:
            s = float(val.wrappedValue)
            return (s, s, s)
        except:
            pass
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            return tuple(max(0.0, min(1.0, float(v))) for v in val[:3])
        except:
            pass
    return None


# === FALLBACK NAMES ===
def _material_name_from_associations(product) -> Optional[str]:
    mat = ifcopenshell.util.element.get_material(product, should_inherit=True)
    if not mat:
        return None
    names = []
    def collect(e):
        if hasattr(e, "Name") and e.Name:
            names.append(str(e.Name))
        for attr in ("MaterialLayers", "MaterialProfiles", "MaterialConstituents", "Materials"):
            if children := getattr(e, attr, None):
                for c in children:
                    collect(c.Material if hasattr(c, "Material") else c)
    collect(mat)
    return " | ".join(dict.fromkeys(names)) if names else None


def _default_name_for_product(product) -> str:
    label = getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "IfcProduct"
    return f"{product.is_a()}_{label}"


def _sanitize_style_token(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "Style_" + s
    return s[:64]