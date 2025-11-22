"""IFC → PBR Material Extraction (IFC 8.9.3.10 Compliant + Backward Compatible)"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
import re
import logging

import ifcopenshell
import ifcopenshell.util.element
import ifcopenshell.util.representation

__all__ = [
    "PBRMaterial",
    "build_material_for_product",
    "extract_face_style_groups",
    "extract_uvs_for_product",
    "pbr_from_surface_style",
]

LOG = logging.getLogger(__name__)


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
        result["per_face"] = {item_id: _to_pbr(styles[0], model) for item_id, styles in face_styles.items()}

    # 2. Object-level (A→E precedence)
    obj_styles = get_surface_styles(model, product)
    if obj_styles:
        result["object"] = _to_pbr(obj_styles[0], model)

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

    # D: IfcShapeAspect (walk standard ShapeRepresentations first; keep older Representation attr as fallback)
    for aspect in getattr(product, "HasShapeAspects", []):
        aspect_reps = list(getattr(aspect, "ShapeRepresentations", []) or [])
        legacy_repr = getattr(aspect, "Representation", None)
        if legacy_repr is not None:
            aspect_reps.extend(getattr(legacy_repr, "Representations", []) or [])
        for rep in aspect_reps:
            for item in getattr(rep, "Items", []):
                for styled in getattr(item, "StyledByItem", []):
                    for psa in getattr(styled, "Styles", []) or []:
                        styles.extend([s for s in getattr(psa, "Styles", []) or [] if s.is_a("IfcSurfaceStyle")])
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


# Map material constituent names to styles (used to correlate with shape aspect names).
def _constituent_styles_by_name(product: ifcopenshell.entity_instance) -> Dict[str, List[ifcopenshell.entity_instance]]:
    mapping: Dict[str, List[ifcopenshell.entity_instance]] = {}
    visited: Set[int] = set()

    def record(name: Optional[str], source: Optional[ifcopenshell.entity_instance]):
        if not name or source is None:
            return
        key = _sanitize_style_token(name)
        styles = _styles_from_material_definition(source)
        if styles:
            mapping.setdefault(key, []).extend(styles)

    def collect(entity):
        if entity is None:
            return
        try:
            ent_id = int(entity.id())
        except Exception:
            ent_id = id(entity)
        if ent_id in visited:
            return
        visited.add(ent_id)

        if entity.is_a("IfcMaterialConstituentSet"):
            for c in getattr(entity, "MaterialConstituents", []) or []:
                cname = getattr(c, "Name", None) or getattr(getattr(c, "Material", None), "Name", None)
                record(cname, getattr(c, "Material", None) or c)
                collect(c)
        elif entity.is_a("IfcMaterialConstituent"):
            cname = getattr(entity, "Name", None) or getattr(getattr(entity, "Material", None), "Name", None)
            record(cname, getattr(entity, "Material", None) or entity)
        elif entity.is_a("IfcMaterial"):
            record(getattr(entity, "Name", None), entity)
        elif entity.is_a("IfcMaterialLayerSetUsage"):
            collect(entity.ForLayerSet)
        elif entity.is_a("IfcMaterialProfileSetUsage"):
            collect(entity.ForProfileSet)

        # Follow associated material definitions.
        for child_attr in ("HasAssociations", "ForLayerSet", "ForProfileSet", "MaterialConstituents"):
            child = getattr(entity, child_attr, None)
            if isinstance(child, (list, tuple)):
                for sub in child:
                    collect(sub)
            elif child is not None:
                collect(child)

    for rel in getattr(product, "HasAssociations", []) or []:
        if rel.is_a("IfcRelAssociatesMaterial"):
            collect(rel.RelatingMaterial)

    # Type-level fallback
    typ = ifcopenshell.util.element.get_type(product)
    if typ:
        for rel in getattr(typ, "HasAssociations", []) or []:
            if rel.is_a("IfcRelAssociatesMaterial"):
                collect(rel.RelatingMaterial)

    return mapping


# === PER-FACE MAPPING ===

def get_face_styles(model: ifcopenshell.file, product: ifcopenshell.entity_instance) -> Dict[int, List[ifcopenshell.entity_instance]]:
    if not product or not hasattr(product, "Representation"):
        return {}

    shape_styles: Dict[int, List[ifcopenshell.entity_instance]] = {}
    aspect_name_by_item = _shape_aspect_name_map(product)
    constituent_styles = _constituent_styles_by_name(product) if aspect_name_by_item else {}

    # Collect styles from the explicit representation tree.
    rep = product.Representation
    for shape_rep in getattr(rep, "Representations", []) or []:
        for item in getattr(shape_rep, "Items", []) or []:
            _collect_styled_items(item, shape_styles)
            # If no explicit style, try matching shape aspect name to material constituent.
            if id(item) not in shape_styles and aspect_name_by_item and constituent_styles:
                aspect_name = aspect_name_by_item.get(id(item))
                if aspect_name:
                    styles = constituent_styles.get(aspect_name)
                    if styles:
                        shape_styles[id(item)] = styles

    # Merge in any styles provided via IfcStyledItem associations on the product.
    for styled in getattr(product, "StyledByItem", []) or []:
        styles = []
        for psa in getattr(styled, "Styles", []) or []:
            if hasattr(psa, "Styles"):
                styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])
            elif psa.is_a("IfcSurfaceStyle"):
                styles.append(psa)
        if not styles:
            continue
        # Associate with each representation item referenced by the styled item.
        targets = [
            getattr(styled, "Item", None),
            getattr(styled, "SecondaryItem", None),
            getattr(styled, "ThirdItem", None),
        ]
        for target in targets:
            if target is None:
                continue
            for rep in getattr(target, "Representations", []) or []:
                for item in getattr(rep, "Items", []) or []:
                    shape_styles.setdefault(id(item), []).extend(styles)

    return shape_styles


def _shape_aspect_name_map(product: ifcopenshell.entity_instance) -> Dict[int, str]:
    """Return mapping from representation item id() to a sanitized shape-aspect label."""
    mapping: Dict[int, str] = {}
    aspects = getattr(product, "HasShapeAspects", None) or []
    for aspect in aspects:
        raw_name = getattr(aspect, "Name", None) or getattr(aspect, "Description", None)
        if not raw_name:
            continue
        aspect_name = _sanitize_style_token(raw_name)
        reps = getattr(aspect, "ShapeRepresentations", None) or []
        for rep in reps:
            for item in getattr(rep, "Items", None) or []:
                mapping[id(item)] = aspect_name
    return mapping


def _collect_styled_items(item, mapping):
    for styled in getattr(item, "StyledByItem", []):
        styles = []
        for psa in styled.Styles:
            if hasattr(psa, "Styles"):
                styles.extend([s for s in psa.Styles if s.is_a("IfcSurfaceStyle")])
            elif psa.is_a("IfcSurfaceStyle"):
                styles.append(psa)
            else:
                styles.append(psa)
        if styles:
            mapping[id(item)] = styles

    if item.is_a("IfcBooleanResult"):
        _collect_styled_items(item.FirstOperand, mapping)
        _collect_styled_items(item.SecondOperand, mapping)
    elif item.is_a("IfcMappedItem"):
        source = getattr(item, "MappingSource", None)
        mapped = getattr(source, "MappedRepresentation", None) if source is not None else None
        if mapped is not None:
            for sub in getattr(mapped, "Items", []) or []:
                _collect_styled_items(sub, mapping)
    elif hasattr(item, "Operands"):
        for op in item.Operands:
            _collect_styled_items(op, mapping)


# === FACE GROUPING ===

def _extract_face_style_groups_internal(product, model) -> Dict[str, Dict[str, Any]]:
    face_styles = get_face_styles(model, product)
    if not face_styles:
        return {}

    shape_name_by_item = _shape_aspect_name_map(product)
    combined = {}
    face_offset = 0
    for item in _iter_representation_items(product):
        item_groups, face_count = _face_style_groups_from_item(item, face_styles, model, shape_name_by_item)
        for key, entry in item_groups.items():
            faces = [face_offset + idx for idx in entry.get("faces", [])]
            if not faces:
                continue
            grouped = combined.setdefault(
                key,
                {"material": entry["material"], "faces": [], "style_id": entry.get("style_id")},
            )
            grouped["faces"].extend(faces)
        face_offset += face_count

    result = {}
    for i, (key, entry) in enumerate(combined.items()):
        mat = entry["material"]
        token = _sanitize_style_token(f"{mat.name}_{i}")
        payload = {"material": mat, "faces": entry["faces"]}
        if entry.get("style_id") is not None:
            payload["style_id"] = entry["style_id"]
        result[token] = payload
    return result


def _face_style_groups_from_item(item, face_styles, model, shape_name_by_item) -> Tuple[Dict, int]:
    face_count = _face_count_from_item(item)
    if face_count == 0:
        return {}, 0

    item_id = id(item)
    if item_id in face_styles:
        style = face_styles[item_id][0]
        resolved_style = _resolve_surface_style_entity(style) or style
        friendly_name = _friendly_style_name(resolved_style, shape_name_by_item.get(item_id))
        mat = _to_pbr(resolved_style, model=model, preferred_name=friendly_name)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                "Face style item=%s style=%s friendly=%s base=%s opacity=%.4f metallic=%.4f roughness=%.4f",
                getattr(item, "GlobalId", None) or getattr(item, "id", None) or hex(item_id),
                getattr(resolved_style, "id", lambda: None)(),
                friendly_name,
                tuple(round(c, 4) for c in mat.base_color),
                mat.opacity,
                getattr(mat, "metallic", 0.0),
                getattr(mat, "roughness", 0.5),
            )
        try:
            style_id = int(resolved_style.id())
        except Exception:
            style_id = None
        return {("styled", mat.name): {"material": mat, "faces": list(range(face_count)), "style_id": style_id}}, face_count

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
            groups[("color", key)] = {"material": mat, "faces": entry["faces"], "style_id": None}

    unassigned = [i for i, f in enumerate(assigned) if not f]
    if unassigned:
        mat = PBRMaterial(name="Default")
        groups[("default",)] = {"material": mat, "faces": unassigned, "style_id": None}

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
def _get_material_for_style(model, style) -> Optional[ifcopenshell.entity_instance]:
    """Resolve the IfcMaterial that owns ``style`` via MDR chains."""
    try:
        mdrs = model.by_type("IfcMaterialDefinitionRepresentation")
    except Exception:
        mdrs = []
    for mdr in mdrs:
        represented = getattr(mdr, "RepresentedMaterial", None)
        for rep in getattr(mdr, "Representations", []) or []:
            for item in getattr(rep, "Items", []) or []:
                if item is None:
                    continue
                if item.is_a("IfcStyledItem"):
                    styles = getattr(item, "Styles", []) or []
                    if any(s == style for s in styles):
                        return represented
                elif hasattr(item, "Styles"):
                    for psa in item.Styles or []:
                        if style in (psa.Styles or []):
                            return represented
    return None

def _resolve_surface_style_entity(style: Optional[ifcopenshell.entity_instance]) -> Optional[ifcopenshell.entity_instance]:
    """Chase through presentation assignments/styled items to find an IfcSurfaceStyle."""
    if style is None:
        return None
    seen: Set[int] = set()
    stack: List[ifcopenshell.entity_instance] = [style]
    while stack:
        entity = stack.pop()
        if entity is None:
            continue
        try:
            key = int(entity.id())
        except Exception:
            key = id(entity)
        if key in seen:
            continue
        seen.add(key)
        if hasattr(entity, "is_a") and entity.is_a("IfcSurfaceStyle"):
            return entity
        for attr_name in ("Styles", "styles", "Items", "items"):
            values = getattr(entity, attr_name, None)
            if not values:
                continue
            if isinstance(values, (list, tuple)):
                stack.extend(v for v in values if v is not None)
            else:
                stack.append(values)
    return None


def _friendly_style_name(style: ifcopenshell.entity_instance, shape_name: Optional[str] = None) -> str:
    """Readable token combining shape aspect name + material/style name."""
    model = getattr(style, "file", None)
    material = _get_material_for_style(model, style)
    mat_name = getattr(material, "Name", None)
    style_name = getattr(style, "Name", None)

    if mat_name and style_name and style_name != mat_name:
        core = f"{mat_name} ({style_name})"
    else:
        core = mat_name or style_name or "Material"

    parts = [shape_name, core]
    raw = "_".join([p for p in parts if p])
    return _sanitize_style_token(raw or "Material")


def _to_pbr(
    style: ifcopenshell.entity_instance,
    model: Optional[ifcopenshell.file] = None,
    *,
    preferred_name: Optional[str] = None,
) -> PBRMaterial:
    surface_style = _resolve_surface_style_entity(style) or style
    material = _get_material_for_style(model or getattr(surface_style, "file", None), surface_style)
    mat_name = getattr(material, "Name", None)
    style_name = getattr(surface_style, "Name", None)
    material_name = preferred_name or mat_name or style_name or "Unknown"
    pbr = PBRMaterial(name=material_name)

    if surface_style is not None and surface_style.is_a("IfcSurfaceStyle"):
        elements = list(getattr(surface_style, "Styles", []) or [])
        rendering = next((e for e in elements if e.is_a("IfcSurfaceStyleRendering")), None)
        shading = next((e for e in elements if e.is_a("IfcSurfaceStyleShading")), None)
        if rendering:
            base_color = _rendering_base_color(rendering)
            pbr.base_color = base_color
            pbr.opacity = _rendering_opacity(rendering)
            pbr.roughness = _rendering_roughness(rendering)
            pbr.metallic = _rendering_metallic(rendering)
            emissive = _rendering_emissive(rendering, base_color)
            if emissive:
                pbr.emissive = emissive
        elif shading:
            pbr.base_color = _as_rgb(getattr(shading, "SurfaceColour", None))
        _log_style_resolution(surface_style, rendering, shading, pbr)
    else:
        _log_style_resolution(surface_style or style, None, None, pbr)

    if preferred_name is None:
        if mat_name and style_name and style_name != mat_name:
            pbr.name = f"{mat_name} ({style_name})"
        elif mat_name:
            pbr.name = mat_name

    return pbr


def pbr_from_surface_style(
    style: ifcopenshell.entity_instance,
    model: Optional[ifcopenshell.file] = None,
    *,
    preferred_name: Optional[str] = None,
) -> PBRMaterial:
    """Public wrapper for converting an IfcSurfaceStyle (or related) into PBRMaterial."""
    return _to_pbr(style, model=model, preferred_name=preferred_name)


def _log_style_resolution(style, rendering, shading, pbr) -> None:
    root_logger = logging.getLogger()
    logger: logging.Logger
    if LOG.isEnabledFor(logging.DEBUG):
        logger = LOG
        level = logging.DEBUG
    elif root_logger.isEnabledFor(logging.DEBUG):
        logger = root_logger
        level = logging.DEBUG
    else:
        # Fall back to INFO so the message still lands in stdout when DEBUG isn't enabled.
        logger = LOG if LOG.isEnabledFor(logging.INFO) else root_logger
        level = logging.INFO

    def _entity_id(entity: Optional[ifcopenshell.entity_instance]) -> Optional[int]:
        if entity is None:
            return None
        try:
            return int(entity.id())
        except Exception:
            return None

    render_snapshot = _rendering_debug_snapshot(rendering)
    shading_snapshot = _shading_debug_snapshot(shading)

    logger.log(
        level,
        "IfcSurfaceStyle '%s' (style_id=%s render_id=%s shade_id=%s) raw_render=%s raw_shading=%s -> material '%s' base=%s opacity=%.4f roughness=%.4f metallic=%.4f emissive=%s",
        getattr(style, "Name", None) or "<Unnamed>",
        _entity_id(style),
        _entity_id(rendering),
        _entity_id(shading),
        render_snapshot,
        shading_snapshot,
        pbr.name,
        tuple(round(c, 4) for c in pbr.base_color),
        pbr.opacity,
        pbr.roughness,
        pbr.metallic,
        tuple(round(c, 4) for c in pbr.emissive) if pbr.emissive else None,
    )


def _rendering_debug_snapshot(rendering: Optional[ifcopenshell.entity_instance]) -> Optional[Dict[str, Any]]:
    if rendering is None:
        return None

    def _rounded_color(value):
        color = _color_or_factor(value)
        if not color:
            return None
        return tuple(round(c, 4) for c in color)

    snapshot = {
        "SurfaceColour": _rounded_color(getattr(rendering, "SurfaceColour", None)),
        "DiffuseColour": _rounded_color(getattr(rendering, "DiffuseColour", None)),
        "SpecularColour": _rounded_color(getattr(rendering, "SpecularColour", None)),
        "SpecularHighlight": _rounded_color(getattr(rendering, "SpecularHighlight", None)),
        "EmissiveColour": _rounded_color(getattr(rendering, "EmissiveColour", None)),
        "Transparency": _float_value(getattr(rendering, "Transparency", None)),
        "SpecularRoughness": _float_value(getattr(rendering, "SpecularRoughness", None)),
        "SelfLuminous": _float_value(getattr(rendering, "SelfLuminous", None)),
        "ReflectanceMethod": getattr(rendering, "ReflectanceMethod", None),
    }
    return snapshot


def _shading_debug_snapshot(shading: Optional[ifcopenshell.entity_instance]) -> Optional[Dict[str, Any]]:
    if shading is None:
        return None
    surface = getattr(shading, "SurfaceColour", None)
    color = _color_or_factor(surface) if surface is not None else None
    return {
        "SurfaceColour": tuple(round(c, 4) for c in color) if color else None,
    }


def _rendering_base_color(rendering) -> Tuple[float, float, float]:
    base = _color_or_factor(getattr(rendering, "SurfaceColour", None)) or (0.8, 0.8, 0.8)
    diffuse = _color_or_factor(getattr(rendering, "DiffuseColour", None))
    if diffuse:
        base = tuple(_clamp01(base[i] * diffuse[i]) for i in range(3))
    return base


def _rendering_opacity(rendering) -> float:
    transparency = _float_value(getattr(rendering, "Transparency", None)) or 0.0
    return _clamp01(1.0 - transparency)


def _rendering_roughness(rendering) -> float:
    rough = _float_value(getattr(rendering, "SpecularRoughness", None))
    if rough is not None:
        return _clamp01(rough)
    spec = _color_or_factor(getattr(rendering, "SpecularColour", None))
    highlight = _color_or_factor(getattr(rendering, "SpecularHighlight", None))
    level = None
    if spec:
        level = max(spec)
    elif highlight:
        level = max(highlight)
    if level is None:
        return 0.5
    return max(0.03, min(1.0, 1.0 - min(level, 0.97)))


def _rendering_metallic(rendering) -> float:
    method = getattr(rendering, "ReflectanceMethod", None)
    if not method:
        return 0.0
    token = str(method).strip().strip(".").lower()
    if token in {"metal", "mirror"}:
        return 1.0
    return 0.0


def _rendering_emissive(rendering, base_color: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
    emissive = _color_or_factor(getattr(rendering, "EmissiveColour", None))
    if emissive:
        return tuple(_clamp01(c) for c in emissive)
    luminous = _float_value(getattr(rendering, "SelfLuminous", None))
    if luminous and luminous > 0.0:
        return tuple(_clamp01(base_color[i] * luminous) for i in range(3))
    return None


def _as_rgb(col) -> Tuple[float, float, float]:
    if col is None:
        return (0.8, 0.8, 0.8)
    r = float(getattr(col, "Red", 0.8))
    g = float(getattr(col, "Green", 0.8))
    b = float(getattr(col, "Blue", 0.8))
    return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b)))


def _color_or_factor(val) -> Optional[Tuple[float, float, float]]:
    if val is None:
        return None
    if hasattr(val, "is_a") and val.is_a("IfcColourRgb"):
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


def _float_value(val: Any) -> Optional[float]:
    if val is None:
        return None
    if hasattr(val, "wrappedValue"):
        val = val.wrappedValue
    try:
        return float(val)
    except Exception:
        return None


def _clamp01(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


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
    if not names:
        return None
    # Prefer the first declared material name instead of concatenating the entire stack.
    return next(iter(dict.fromkeys(names)))


def _default_name_for_product(product) -> str:
    label = getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "IfcProduct"
    return f"{product.is_a()}_{label}"


def _sanitize_style_token(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s or s[0].isdigit():
        s = "Style_" + s
    return s[:64]
