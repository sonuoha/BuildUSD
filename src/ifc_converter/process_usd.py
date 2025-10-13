"""USD authoring helpers for the ifc_converter pipeline.

The functions below consume the prototype/instance caches produced by
``process_ifc`` and emit the various USD layers (prototypes, materials,
instances, 2D geometry) expected by ``main.py``.  The code mirrors the behaviour
of the original project but has been pared back to the essentials required by
the new iterator-driven workflow.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from textwrap import dedent
import fnmatch
import json
import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .pxr_utils import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from .config.manifest import BasePointConfig, GeodeticCoordinate
from .io_utils import join_path, path_stem, write_text, is_omniverse_path, stat_entry
from .process_ifc import ConversionOptions, CurveWidthRule, InstanceRecord, PrototypeCaches, PrototypeKey
from .utils.matrix_utils import np_to_gf_matrix, scale_matrix_translation_only

# ---------------- Constants / Units ----------------

_UNIT_FACTORS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "dm": 0.1,
    "decimeter": 0.1,
    "decimeters": 0.1,
}

PathLike = Union[str, Path]

# ---------------- Name helpers ----------------

def sanitize_name(raw_name, fallback=None):
    """Make a USD-legal prim name (deterministic; no time-based suffix)."""
    base = str(raw_name or fallback or "Unnamed")
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    return name[:63]

def _sanitize_identifier(raw_name: Optional[str], fallback: Optional[str] = None) -> str:
    """Return a USD-safe token useful for attribute or namespace names."""
    base = str(raw_name or fallback or "Unnamed")
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    return name[:63]


def _unique_name(base: str, used: Dict[str, int]) -> str:
    """Generate a unique name by appending an incrementing suffix when needed."""
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count}"


def _author_namespaced_dictionary_attributes(prim: Usd.Prim, namespace: str, entries: Dict[str, Any]) -> None:
    """Write IFC property dictionaries into BIMData:* USD attributes."""
    if not entries:
        return

    root_ns = "BIMData"
    ns_map = {"pset": "Psets", "qtos": "QTO", "qto": "QTO"}
    ns1 = ns_map.get(namespace.lower(), _sanitize_identifier(namespace, fallback="Meta"))

    def _set_attr(full_name: str, value: Any) -> None:
        if isinstance(value, bool):
            attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.Bool)
            attr.Set(bool(value))
            return
        if isinstance(value, int) and not isinstance(value, bool):
            attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.Int)
            attr.Set(int(value))
            return
        if isinstance(value, float):
            attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.Double)
            attr.Set(float(value))
            return
        if isinstance(value, (list, tuple)):
            seq = list(value)
            if all(isinstance(v, bool) for v in seq):
                attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.BoolArray)
                attr.Set(Vt.BoolArray([bool(v) for v in seq]))
                return
            if all(isinstance(v, int) and not isinstance(v, bool) for v in seq):
                attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.IntArray)
                attr.Set(Vt.IntArray([int(v) for v in seq]))
                return
            if all(isinstance(v, (int, float)) for v in seq):
                attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.DoubleArray)
                attr.Set(Vt.DoubleArray([float(v) for v in seq]))
                return
            if all(isinstance(v, str) for v in seq):
                attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.StringArray)
                attr.Set(Vt.StringArray([str(v) for v in seq]))
                return
            attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.String)
            attr.Set(str(value))
            return
        attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.String)
        attr.Set(str(value))

    for set_name, payload in entries.items():
        if payload is None:
            continue
        set_token = _sanitize_identifier(set_name, fallback="Unnamed")
        if isinstance(payload, dict):
            for prop_name, prop_value in payload.items():
                if prop_value is None:
                    continue
                prop_token = _sanitize_identifier(prop_name, fallback="Value")
                full_attr = f"{root_ns}:{ns1}:{set_token}:{prop_token}"
                _set_attr(full_attr, prop_value)
        else:
            full_attr = f"{root_ns}:{ns1}:{set_token}:Value"
            _set_attr(full_attr, payload)


def _author_instance_attributes(prim: Usd.Prim, attributes: Optional[Dict[str, Any]]) -> None:
    if not attributes or not isinstance(attributes, dict):
        return
    psets = attributes.get("psets", {})
    qtos = attributes.get("qtos", {})
    _author_namespaced_dictionary_attributes(prim, "pset", psets)
    _author_namespaced_dictionary_attributes(prim, "qto", qtos)


# ---------------- PreparedInstance (used by grouping/variants) ----------------

@dataclass
class PreparedInstance:
    """Lightweight representation of an IFC instance ready for regrouping."""

    step_id: Any
    name: str
    transform: Optional[Tuple[float, ...]]
    attributes: Dict[str, Any]
    hierarchy: Tuple[Tuple[str, Optional[int]], ...]
    prototype_path: Optional[Sdf.Path]
    prototype_delta: Optional[Tuple[float, ...]]
    material_ids: List[int]
    materials: List[Any]
    mesh: Optional[Dict[str, Any]]
    source_path: Optional[Sdf.Path] = None
    guid: Optional[str] = None


# ---------------- Stage helpers ----------------

def create_usd_stage(usd_path, meters_per_unit=1.0):
    """Create a new USD stage with Z-up and a `/World` default prim."""
    if isinstance(usd_path, Path):
        identifier = usd_path.resolve().as_posix()
    else:
        identifier = str(usd_path)
    stage = Usd.Stage.CreateNew(identifier)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", float(meters_per_unit))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


# ---------------- Mesh helpers ----------------

def write_usd_mesh(stage, parent_path, mesh_name, verts, faces, abs_mat=None,
                   material_ids=None, stage_meters_per_unit=1.0,
                   scale_matrix_translation=False):
    """Author a Mesh prim beneath ``parent_path`` with the supplied data."""
    mesh_path = Sdf.Path(parent_path).AppendChild(mesh_name)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    if verts:
        n = len(verts) // 3
        pts = Vt.Vec3fArray(n)
        for i in range(n):
            x, y, z = verts[3*i:3*i+3]
            pts[i] = Gf.Vec3f(float(x), float(y), float(z))
        mesh.CreatePointsAttr(pts)

    if faces:
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(faces)//3)))

    if abs_mat is not None:
        gf = np_to_gf_matrix(abs_mat)
        if scale_matrix_translation and stage_meters_per_unit != 1.0:
            gf = scale_matrix_translation_only(gf, 1.0/float(stage_meters_per_unit))
        xf = UsdGeom.Xformable(mesh)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(gf)

    if material_ids:
        mesh.GetPrim().CreateAttribute("ifc:materialIds", Sdf.ValueTypeNames.IntArray)\
            .Set(Vt.IntArray([int(i) for i in material_ids]))
    return mesh


def _flatten_mesh_arrays(mesh: Dict[str, Any]) -> Optional[Tuple[List[float], List[int]]]:
    """Return flattened vertex/face arrays from a mesh dict, or None if missing."""
    if not mesh:
        return None
    verts = mesh.get("vertices")
    faces = mesh.get("faces")
    if verts is None or faces is None:
        return None
    verts_list = verts.flatten().tolist() if hasattr(verts, "flatten") else list(verts)
    faces_list = faces.flatten().tolist() if hasattr(faces, "flatten") else list(faces)
    if not verts_list or not faces_list:
        return None
    return verts_list, faces_list


# ---------------- Prototype authoring ----------------

def _name_for_repmap(proto: Any) -> str:
    """Return the prototype prim name for a repmap-backed prototype."""
    # Use Defining Type name EXACTLY (as requested)
    if getattr(proto, "type_name", None):
        return _sanitize_identifier(proto.type_name)
    return _sanitize_identifier(f"RepMap_{proto.repmap_id}")


def _name_for_hash(proto: Any) -> str:
    """Return the prototype name for a hash-backed (occurrence) mesh."""
    base = (getattr(proto, "name", None) or getattr(proto, "type_name", None)
            or "Fallback")
    digest = getattr(proto, "digest", "unknown")
    return _sanitize_identifier(f"{base}_Fallback_{str(digest)[:12]}")


def _layer_identifier(path: PathLike) -> str:
    """Normalise layer paths to a USD identifier string."""
    if isinstance(path, Path):
        return path.resolve().as_posix()
    return str(path).replace("\\", "/")


def _sublayer_identifier(parent_layer: Sdf.Layer, child_path: PathLike) -> str:
    """
    Compute a stable asset path to author into `parent_layer.subLayerPaths`.

    Preference order:
        1. Omniverse identifiers stay absolute (already portable).
        2. Relative path from the parent layer directory when possible.
        3. Raw POSIX string fallback.
    """
    raw_child = str(child_path)
    if is_omniverse_path(raw_child):
        return raw_child.replace("\\", "/")

    parent_real = getattr(parent_layer, "realPath", None)
    if parent_real:
        try:
            parent_dir = Path(parent_real).resolve().parent
            child_path_obj = Path(child_path) if isinstance(child_path, Path) else Path(raw_child)
            child_abs = child_path_obj.resolve()
            rel = child_abs.relative_to(parent_dir)
            return PurePosixPath(rel.as_posix()).as_posix()
        except Exception:
            pass

    if isinstance(child_path, Path):
        return child_path.as_posix()

    try:
        return PurePosixPath(raw_child).as_posix()
    except Exception:
        return raw_child.replace("\\", "/")


def _iter_prototypes(caches: PrototypeCaches) -> Iterable[Tuple[PrototypeKey, Any]]:
    """Yield prototypes in a deterministic order for layer authoring."""
    for rep_id, proto in sorted(caches.repmaps.items()):
        yield PrototypeKey(kind="repmap", identifier=rep_id), proto
    for digest, proto in sorted(caches.hashes.items()):
        yield PrototypeKey(kind="hash", identifier=digest), proto


def author_prototype_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    layer_path: PathLike,
    base_name: str,
    options: ConversionOptions,
) -> Tuple[Sdf.Layer, Dict[PrototypeKey, Sdf.Path]]:
    """Author a prototype layer and fill /World/__Prototypes with meshes.

    - Prototype Xform is named after the Defining Type (or a stable fallback).
    - Mesh is authored as a child "Geom".
    - Prototype Xform is marked purpose=guide (so it won't render by itself).
    - Prototype Xform is NOT instanceable (instances are instanceable instead).
    """
    proto_layer = Sdf.Layer.CreateNew(_layer_identifier(layer_path))

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    proto_root = Sdf.Path("/World/__Prototypes")
    proto_paths: Dict[PrototypeKey, Sdf.Path] = {}
    used_names: Dict[str, int] = {}

    def _unique_name(base: str) -> str:
        c = used_names.get(base, 0)
        used_names[base] = c + 1
        return base if c == 0 else f"{base}_{c}"

    with Usd.EditContext(stage, proto_layer):
        # container scope
        UsdGeom.Scope.Define(stage, proto_root)

        for key, proto in _iter_prototypes(caches):
            mesh_data = _flatten_mesh_arrays(getattr(proto, "mesh", None))
            if not mesh_data:
                continue
            verts, faces = mesh_data

            # Name the prototype Xform
            if key.kind == "repmap":
                base = _name_for_repmap(proto)  # Defining Type name (sanitized)
            else:
                base = _name_for_hash(proto)
            prim_name = _unique_name(base)

            proto_path = proto_root.AppendChild(prim_name)
            proto_paths[key] = proto_path

            # Define prototype Xform and mark as 'guide' (no direct render)
            x = UsdGeom.Xform.Define(stage, proto_path)
            x.ClearXformOpOrder()
            proto_prim = x.GetPrim()
            # DO NOT set instanceable on the prototype
            UsdGeom.Imageable(proto_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

            # Author the mesh as a child "Geom"
            write_usd_mesh(
                stage,
                proto_path,
                "Geom",
                verts,
                faces,
                abs_mat=None,
                material_ids=list(getattr(proto, "material_ids", []) or []),
                stage_meters_per_unit=float(stage.GetMetadata("metersPerUnit") or 1.0),
            )

    return proto_layer, proto_paths



# ---------------- Materials (kept minimal; wiring points preserved) ----------------

def _prototype_materials(caches: PrototypeCaches, key: PrototypeKey) -> List[Any]:
    if key.kind == "repmap":
        proto = caches.repmaps.get(int(key.identifier))
    else:
        proto = caches.hashes.get(str(key.identifier))
    if not proto:
        return []
    return list(getattr(proto, "materials", []) or [])


@dataclass(frozen=True)
class _MaterialProperties:
    name: str
    display_color: Tuple[float, float, float]
    opacity: float
    emissive_color: Optional[Tuple[float, float, float]] = None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_color_value(obj: Any) -> Optional[Tuple[float, float, float]]:
    """Extract an RGB tuple from various ifcopenshell colour representations."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        vals = [float(v) for v in obj[:3]]
        if len(vals) == 3:
            return tuple(vals)
        return None
    if isinstance(obj, dict):
        for keys in (("Red", "Green", "Blue"), ("R", "G", "B"), ("red", "green", "blue"), ("x", "y", "z")):
            if all(k in obj for k in keys):
                try:
                    vals = [float(obj[k]) for k in keys]
                    return tuple(vals[:3])
                except Exception:
                    continue
        return None
    for attr_set in (("Red", "Green", "Blue"), ("R", "G", "B"), ("red", "green", "blue")):
        values = []
        success = True
        for attr in attr_set:
            if hasattr(obj, attr):
                component = getattr(obj, attr)
                component_val = _as_float(component)
                if component_val is None:
                    success = False
                    break
                values.append(component_val)
            else:
                success = False
                break
        if success and len(values) == 3:
            return tuple(values)
    if hasattr(obj, "rgb"):
        try:
            values = obj.rgb
            if isinstance(values, (list, tuple)) and len(values) >= 3:
                return tuple(float(v) for v in values[:3])
        except Exception:
            pass
    return None


def _extract_material_properties(
    material: Any,
    *,
    prototype_label: str,
    index: int,
) -> Optional[_MaterialProperties]:
    """Derive a usable material definition from an ifcopenshell geometry material."""
    def _get_value(obj: Any, *names: str) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                value = obj[name]
                if value is not None:
                    return value
            elif hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    name_value = _get_value(material, "name", "Name", "label", "Label", "description", "Description")
    if isinstance(name_value, str):
        name_value = name_value.strip()
    if not name_value:
        name_value = f"{prototype_label}_Material_{index + 1}"

    color_source = _get_value(
        material,
        "diffuse_color",
        "DiffuseColor",
        "DiffuseColour",
        "diffuseColour",
        "albedo",
        "AlbedoColor",
        "Color",
        "colour",
    )
    color = _resolve_color_value(color_source)
    if color is None:
        color = _resolve_color_value(material)
    if color is None:
        color = (0.8, 0.8, 0.8)
    max_component = max(color)
    if max_component > 1.0:
        # Assume 0-255 range; normalise to 0-1
        color = tuple(float(c) / 255.0 for c in color)
    display_color = _normalize_color(tuple(float(c) for c in color))

    transparency = _get_value(
        material,
        "transparency",
        "Transparency",
        "TransparencyFactor",
        "transparencyFactor",
    )
    opacity_value: Optional[float] = None
    if transparency is not None:
        transparency_float = _as_float(transparency)
        if transparency_float is not None:
            opacity_value = 1.0 - transparency_float
    opacity_attr = _get_value(material, "opacity", "Opacity", "Alpha")
    if opacity_attr is not None:
        opacity_candidate = _as_float(opacity_attr)
        if opacity_candidate is not None:
            opacity_value = opacity_candidate
    if opacity_value is None:
        opacity_value = 1.0
    opacity = max(0.0, min(1.0, float(opacity_value)))

    emissive_source = _get_value(
        material,
        "emissive_color",
        "EmissiveColor",
        "EmissionColor",
        "Emission",
    )
    emissive_color = _resolve_color_value(emissive_source)
    if emissive_color is not None:
        max_emissive = max(emissive_color)
        if max_emissive > 1.0:
            emissive_color = tuple(float(c) / 255.0 for c in emissive_color)
        emissive_color = _normalize_color(tuple(float(c) for c in emissive_color))

    return _MaterialProperties(
        name=name_value,
        display_color=display_color,
        opacity=opacity,
        emissive_color=emissive_color,
    )


def _material_signature(props: _MaterialProperties) -> Tuple[Any, ...]:
    emissive = tuple(round(c, 5) for c in props.emissive_color) if props.emissive_color else None
    return (
        tuple(round(c, 5) for c in props.display_color),
        round(props.opacity, 5),
        emissive,
        sanitize_name(props.name, fallback="Material").lower(),
    )


def author_material_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    layer_path: PathLike,
    base_name: str,
    proto_layer: Sdf.Layer,
    options: ConversionOptions,
) -> Tuple[Sdf.Layer, Dict[PrototypeKey, List[Sdf.Path]]]:
    """Author a material layer and return mapping from prototype key to materials.

    This is a minimal binding scaffold; you can restore richer bindings without
    changing any public signatures.
    """
    material_layer = Sdf.Layer.CreateNew(_layer_identifier(layer_path))
    log = logging.getLogger(__name__)

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    proto_sub_path = _sublayer_identifier(proto_layer, layer_path)
    if proto_sub_path not in proto_layer.subLayerPaths:
        proto_layer.subLayerPaths.append(proto_sub_path)

    material_root = Sdf.Path("/World/__Materials")
    material_paths: Dict[PrototypeKey, List[Sdf.Path]] = {}
    used_names: Dict[str, int] = {}
    authored_signatures: Dict[Tuple[Any, ...], Sdf.Path] = {}

    with Usd.EditContext(stage, material_layer):
        UsdGeom.Scope.Define(stage, material_root)

        for key, proto in _iter_prototypes(caches):
            proto_path = proto_paths.get(key)
            if proto_path is None:
                continue
            materials = _prototype_materials(caches, key)
            if not materials:
                continue

            proto_label = sanitize_name(proto_path.name, fallback="Prototype")
            material_prims: List[Sdf.Path] = []

            for idx, material in enumerate(materials):
                props = _extract_material_properties(material, prototype_label=proto_label, index=idx)
                if props is None:
                    continue
                signature = _material_signature(props)
                material_prim_path = authored_signatures.get(signature)
                if material_prim_path is None:
                    base_name = sanitize_name(props.name, fallback=f"{proto_label}_Material")
                    count = used_names.get(base_name, 0)
                    used_names[base_name] = count + 1
                    if count:
                        token = f"{base_name}_{count}"
                    else:
                        token = base_name
                    material_prim_path = material_root.AppendChild(token)
                    authored_signatures[signature] = material_prim_path

                    material_prim = UsdShade.Material.Define(stage, material_prim_path)
                    shader_path = material_prim_path.AppendChild("PreviewSurface")
                    shader = UsdShade.Shader.Define(stage, shader_path)
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*props.display_color))
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(props.opacity))
                    if props.emissive_color:
                        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*props.emissive_color))
                    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                    material_prim.CreateSurfaceOutput().ConnectToSource(surface_output)
                    material_prim.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f, custom=True).Set(
                        Gf.Vec3f(*props.display_color)
                    )
                    material_prim.CreateAttribute("displayOpacity", Sdf.ValueTypeNames.Float, custom=True).Set(float(props.opacity))
                    log.debug(
                        "Authored material %s (color=%.3f,%.3f,%.3f opacity=%.3f)",
                        material_prim_path,
                        props.display_color[0],
                        props.display_color[1],
                        props.display_color[2],
                        props.opacity,
                    )

                material_prims.append(material_prim_path)

            if material_prims:
                material_paths[key] = material_prims
                log.debug(
                    "Prototype %s uses %d material(s)",
                    proto_path,
                    len(material_prims),
                )

    return material_layer, material_paths


def bind_materials_to_prototypes(
    stage: Usd.Stage,
    proto_layer: Sdf.Layer,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    material_paths: Dict[PrototypeKey, List[Sdf.Path]],
) -> None:
    """Bind first available material to each prototype mesh if present."""
    if not material_paths:
        return

    with Usd.EditContext(stage, proto_layer):
        for key, proto_path in proto_paths.items():
            mesh_path = proto_path.AppendChild("Geom")
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if not mesh_prim:
                continue
            materials = material_paths.get(key)
            if not materials:
                continue
            mat_prim = stage.GetPrimAtPath(materials[0])
            if not mat_prim:
                continue
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(UsdShade.Material(mat_prim))
            display_attr = mat_prim.GetAttribute("displayColor")
            if display_attr and display_attr.HasAuthoredValue():
                try:
                    color_value = display_attr.Get()
                except Exception:
                    color_value = None
                if color_value is not None:
                    if isinstance(color_value, Gf.Vec3f):
                        color_vec = color_value
                    else:
                        try:
                            color_vec = Gf.Vec3f(float(color_value[0]), float(color_value[1]), float(color_value[2]))
                        except Exception:
                            color_vec = None
                    if color_vec is not None:
                        mesh_geom = UsdGeom.Mesh(mesh_prim)
                        mesh_display_attr = mesh_geom.GetDisplayColorAttr()
                        mesh_display_attr.Set(Vt.Vec3fArray([color_vec]))


# ---------------- Instance authoring ----------------

def author_instance_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    layer_path: PathLike,
    base_name: str,
    options: ConversionOptions,
) -> Sdf.Layer:
    """
    Author a layer containing instanceable prototype references.

    This function is responsible for creating a new layer containing all
    instanceable prototype references, including any transforms or metadata
    associated with the instances.

    :param stage: The USD stage to author the layer in.
    :param caches: The cached prototype meshes.
    :param proto_paths: A mapping from prototype key to USD path.
    :param layer_path: The path to the layer to be created.
    :param base_name: The base name of the layer.
    :param options: Optional conversion options.
    :return: The authored layer.
    """
    inst_layer = Sdf.Layer.CreateNew(_layer_identifier(layer_path))

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    inst_root_name = _sanitize_identifier(f"{base_name}_Instances", fallback="Instances")
    inst_root = Sdf.Path(f"/World/{inst_root_name}")

    from collections import defaultdict
    name_counters: Dict[Sdf.Path, Dict[str, int]] = defaultdict(dict)
    hierarchy_nodes_by_step: Dict[int, Sdf.Path] = {}
    hierarchy_nodes_by_label: Dict[Tuple[Sdf.Path, str], Sdf.Path] = {}

    def _unique_child_name(parent_path: Sdf.Path, base: str) -> str:
        used = name_counters[parent_path]
        count = used.get(base, 0)
        used[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def _ensure_group(parent_path: Sdf.Path, label: str, step_id: Optional[int]) -> Sdf.Path:
        if step_id is not None and step_id in hierarchy_nodes_by_step:
            return hierarchy_nodes_by_step[step_id]
        sanitized = _sanitize_identifier(label, fallback=f"Group_{step_id or 'Node'}") or "Group"
        if step_id is None:
            existing = hierarchy_nodes_by_label.get((parent_path, sanitized))
            if existing is not None:
                return existing
        token = _unique_child_name(parent_path, sanitized)
        child_path = parent_path.AppendChild(token)
        prim = stage.GetPrimAtPath(child_path)
        if not prim:
            xf = UsdGeom.Xform.Define(stage, child_path)
            xf.ClearXformOpOrder()
            prim = xf.GetPrim()
            Usd.ModelAPI(prim).SetKind('group')
            prim.SetCustomDataByKey("ifc:label", label)
            if step_id is not None:
                try:
                    prim.SetCustomDataByKey("ifc:stepId", int(step_id))
                except Exception:
                    prim.SetCustomDataByKey("ifc:stepId", step_id)
        if step_id is not None:
            hierarchy_nodes_by_step[step_id] = child_path
        else:
            hierarchy_nodes_by_label[(parent_path, sanitized)] = child_path
        return child_path

    with Usd.EditContext(stage, inst_layer):
        inst_root_xf = UsdGeom.Xform.Define(stage, inst_root)
        inst_root_xf.ClearXformOpOrder()
        inst_root_xf.GetPrim().SetCustomDataByKey("ifc:sourceFile", base_name)

        for record in caches.instances.values():
            parent_path = inst_root
            for label, step_id in record.hierarchy:
                parent_path = _ensure_group(parent_path, label, step_id)

            base_name_candidate = _sanitize_identifier(record.name, fallback=f"Ifc_{record.step_id}")
            inst_name = _unique_child_name(parent_path, base_name_candidate)
            inst_path = parent_path.AppendChild(inst_name)

            xform = UsdGeom.Xform.Define(stage, inst_path)
            xform.ClearXformOpOrder()
            inst_prim = xform.GetPrim()
            Usd.ModelAPI(inst_prim).SetKind('component')

            # ✅ Make the INSTANCE prim instanceable (per your requirement)
            inst_prim.SetInstanceable(bool(options.enable_instancing))

            try:
                inst_prim.SetCustomDataByKey("ifc:instanceStepId", int(record.step_id))
            except Exception:
                inst_prim.SetCustomDataByKey("ifc:instanceStepId", record.step_id)
            if getattr(record, "guid", None):
                inst_prim.SetCustomDataByKey("ifc:guid", str(record.guid))

            if record.transform is not None:
                xform.AddTransformOp().Set(np_to_gf_matrix(record.transform))

            if getattr(options, "convert_metadata", True) and getattr(record, "attributes", None):
                _author_instance_attributes(inst_prim, record.attributes)

            # Per-instance mesh fallback (non-instanced geometry)
            if record.mesh:
                mesh_data = _flatten_mesh_arrays(record.mesh)
                if mesh_data:
                    verts, faces = mesh_data
                    write_usd_mesh(
                        stage,
                        inst_path,
                        "Geom",
                        verts,
                        faces,
                        abs_mat=None,
                        material_ids=list(record.material_ids or []),
                        stage_meters_per_unit=float(stage.GetMetadata("metersPerUnit") or 1.0),
                    )
                continue

            # Prototype reference
            if record.prototype is None:
                continue
            proto_path = proto_paths.get(record.prototype)
            if proto_path is None:
                continue

            ref_path = inst_path.AppendChild("Prototype")
            ref_prim = stage.DefinePrim(ref_path, "Xform")
            refs = ref_prim.GetReferences()
            refs.ClearReferences()
            refs.AddReference("", proto_path)

            ref_xf = UsdGeom.Xform(ref_prim)
            ref_xf.ClearXformOpOrder()
            if record.prototype_delta:
                ref_xf.AddTransformOp().Set(np_to_gf_matrix(record.prototype_delta))

            # ❌ Do NOT make the reference holder instanceable
            # ref_prim.SetInstanceable(...)

            # Ensure the referenced payload renders even if prototype purpose is 'guide'
            UsdGeom.Imageable(ref_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)

    return inst_layer



# ---------------- 2D annotation geometry ----------------

@dataclass(frozen=True)
class CurveWidthRuleParsed:
    width: float
    unit: Optional[str] = None
    layer: Optional[str] = None
    curve: Optional[str] = None
    hierarchy: Optional[str] = None
    step_id: Optional[int] = None


def _normalize_color(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, float(c))) for c in color)


def _rule_from_mapping(mapping: dict[str, Any], *, context: str) -> CurveWidthRule:
    if not isinstance(mapping, dict):
        raise ValueError(f"{context}: rule must be a mapping, got {type(mapping).__name__}.")
    lowered = {str(k).lower(): v for k, v in mapping.items()}
    width_spec = lowered.pop("width", None)
    if width_spec is None:
        raise ValueError(f"{context}: missing required 'width' entry.")
    unit_hint = lowered.pop("unit", None)

    def _parse_width_value(value: Any, unit_hint: Optional[str], *, context: str) -> tuple[float, Optional[str]]:
        if isinstance(value, dict):
            lowered = {str(k).lower(): v for k, v in value.items()}
            nested_value = lowered.get("value", lowered.get("amount"))
            if nested_value is None:
                raise ValueError(f"{context}: width mapping requires 'value' or 'amount'.")
            nested_unit = lowered.get("unit")
            return _parse_width_value(nested_value, nested_unit or unit_hint, context=context)
        unit = unit_hint
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            m = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z]+)?\s*$", value)
            if not m:
                raise ValueError(f"{context}: could not parse width value {value!r}.")
            numeric = float(m.group(1)); unit = m.group(2) or unit
        else:
            raise ValueError(f"{context}: width must be numeric or string, got {type(value).__name__}.")
        unit_normalized = unit.strip().lower() if isinstance(unit, str) and unit.strip() else None
        return numeric, unit_normalized

    width_value, width_unit = _parse_width_value(width_spec, unit_hint, context=context)
    layer = lowered.pop("layer", None)
    curve = lowered.pop("curve", None)
    hierarchy = lowered.pop("hierarchy", None)
    step_id = lowered.pop("step_id", None)
    if lowered:
        unknown = ", ".join(sorted(lowered))
        raise ValueError(f"{context}: unrecognised keys: {unknown}.")
    if step_id is not None:
        try:
            step_id = int(step_id)
        except Exception as exc:
            raise ValueError(f"{context}: step_id must be an integer, got {step_id!r}.") from exc
    return CurveWidthRule(
        width=float(width_value),
        unit=width_unit,
        layer=layer,
        curve=curve,
        hierarchy=hierarchy,
        step_id=step_id,
    )


def _rules_from_config_payload(payload: Any, *, source: str) -> list[CurveWidthRule]:
    if payload is None:
        return []
    rules: list[CurveWidthRule] = []
    if isinstance(payload, list):
        for idx, entry in enumerate(payload):
            if not isinstance(entry, dict):
                raise ValueError(f"{source}: rule #{idx + 1} must be a mapping.")
            rules.append(_rule_from_mapping(entry, context=f"{source} rule #{idx + 1}"))
        return rules
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: configuration must be a mapping or a list.")
    if "default" in payload:
        rules.append(_rule_from_mapping({"width": payload["default"]}, context=f"{source} default"))
    for idx, entry in enumerate(payload.get("rules", []) or []):
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: entry #{idx + 1} under 'rules' must be a mapping.")
        rules.append(_rule_from_mapping(entry, context=f"{source} rules[{idx}]"))
    for layer_pattern, value in (payload.get("layers") or {}).items():
        rule_mapping = {"width": value, "layer": layer_pattern}
        rules.append(_rule_from_mapping(rule_mapping, context=f"{source} layers[{layer_pattern}]"))
    for curve_pattern, value in (payload.get("curves") or {}).items():
        rule_mapping = {"width": value, "curve": curve_pattern}
        rules.append(_rule_from_mapping(rule_mapping, context=f"{source} curves[{curve_pattern}]"))
    for layer_pattern, entries in (payload.get("layer_curves") or {}).items():
        if not isinstance(entries, dict):
            raise ValueError(f"{source}: layer_curves[{layer_pattern}] must be a mapping.")
        for curve_pattern, value in entries.items():
            if isinstance(value, dict):
                rule_mapping = dict(value)
                rule_mapping.setdefault("layer", layer_pattern)
                rule_mapping.setdefault("curve", curve_pattern)
            else:
                rule_mapping = {"width": value, "layer": layer_pattern, "curve": curve_pattern}
            rules.append(_rule_from_mapping(rule_mapping, context=f"{source} layer_curves[{layer_pattern}][{curve_pattern}]"))
    for hierarchy_pattern, value in (payload.get("hierarchies") or {}).items():
        rule_mapping = {"width": value, "hierarchy": hierarchy_pattern}
        rules.append(_rule_from_mapping(rule_mapping, context=f"{source} hierarchies[{hierarchy_pattern}]"))
    for layer_pattern, entries in (payload.get("layer_hierarchies") or {}).items():
        if not isinstance(entries, dict):
            raise ValueError(f"{source}: layer_hierarchies[{layer_pattern}] must be a mapping.")
        for hierarchy_pattern, value in entries.items():
            if isinstance(value, dict):
                rule_mapping = dict(value)
                rule_mapping.setdefault("layer", layer_pattern)
                rule_mapping.setdefault("hierarchy", hierarchy_pattern)
            else:
                rule_mapping = {"width": value, "layer": layer_pattern, "hierarchy": hierarchy_pattern}
            rules.append(_rule_from_mapping(rule_mapping, context=f"{source} layer_hierarchies[{layer_pattern}][{hierarchy_pattern}]"))
    return rules


def author_geometry2d_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    layer_path: PathLike,
    base_name: str,
    options: ConversionOptions,
) -> Optional[Sdf.Layer]:
    """Author 2D annotation curves as BasisCurves under the instance root hierarchy."""
    if not getattr(caches, "annotations", None):
        return None

    geom_layer = Sdf.Layer.CreateNew(_layer_identifier(layer_path))

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    name_counters: Dict[Sdf.Path, Dict[str, int]] = defaultdict(dict)
    hierarchy_nodes: Dict[Tuple[Sdf.Path, str, Optional[int]], Sdf.Path] = {}
    curves_root_cache: Dict[Sdf.Path, Sdf.Path] = {}

    def _unique_child_name(parent_path: Sdf.Path, base: str) -> str:
        used = name_counters[parent_path]
        count = used.get(base, 0)
        used[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def _ensure_group(parent_path: Sdf.Path, label: str, step_id: Optional[int]) -> Sdf.Path:
        key = (parent_path, label, step_id)
        cached = hierarchy_nodes.get(key)
        if cached is not None:
            return cached
        token = _unique_child_name(parent_path, _sanitize_identifier(label, fallback=f"Group_{step_id or 'Node'}") or "Group")
        group_path = parent_path.AppendChild(token)
        xf = UsdGeom.Xform.Define(stage, group_path)
        xf.ClearXformOpOrder()
        prim = xf.GetPrim()
        Usd.ModelAPI(prim).SetKind('group')
        prim.SetCustomDataByKey("ifc:label", label)
        if step_id is not None:
            try:
                prim.SetCustomDataByKey("ifc:stepId", int(step_id))
            except Exception:
                prim.SetCustomDataByKey("ifc:stepId", step_id)
        hierarchy_nodes[key] = group_path
        return group_path

    inst_root = Sdf.Path(f"/World/{sanitize_name(base_name)}_Instances")

    def _ensure_curves_container(parent_path: Sdf.Path) -> Sdf.Path:
        existing = curves_root_cache.get(parent_path)
        if existing:
            return existing
        token = sanitize_name("Curves2D")
        container_path = parent_path.AppendChild(token)
        if not stage.GetPrimAtPath(container_path):
            curves_xf = UsdGeom.Xform.Define(stage, container_path)
            curves_xf.ClearXformOpOrder()
            Usd.ModelAPI(curves_xf.GetPrim()).SetKind('group')
        curves_root_cache[parent_path] = container_path
        return container_path

    with Usd.EditContext(stage, geom_layer):
        if not stage.GetPrimAtPath(inst_root):
            UsdGeom.Xform.Define(stage, inst_root)

        # Resolve a default width helper (from rules)
        rules: Tuple[CurveWidthRule, ...] = tuple(getattr(options, "curve_width_rules", ()) or ())
        stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)

        def _pattern_match(pattern: str, candidates: Iterable[str]) -> bool:
            pattern_norm = pattern.strip().lower()
            if not pattern_norm:
                return False
            for candidate in candidates:
                if candidate is None:
                    continue
                value = candidate.strip()
                if not value:
                    continue
                if fnmatch.fnmatchcase(value.lower(), pattern_norm):
                    return True
            return False

        def _candidate_layer_names() -> Tuple[str, ...]:
            sanitized = _sanitize_identifier(base_name, fallback=base_name) if base_name else None
            values = [base_name] if base_name else []
            if sanitized and sanitized not in values:
                values.append(sanitized)
            return tuple(values)

        def _curve_name_candidates(curve) -> Tuple[str, ...]:
            values: List[str] = []
            raw = getattr(curve, "name", None)
            if raw:
                values.append(raw)
            sanitized = _sanitize_identifier(raw, fallback=None) if raw else None
            if sanitized:
                values.append(sanitized)
            fallback = f"Geometry2D_{curve.step_id}"
            if fallback not in values:
                values.append(fallback)
            return tuple(values)

        def _hierarchy_candidates(curve) -> Tuple[str, ...]:
            labels = [label for label, _ in getattr(curve, "hierarchy", ()) if label]
            candidates: List[str] = []
            if labels:
                raw_path = "/".join(labels)
                if raw_path:
                    candidates.append(raw_path)
                sanitized_labels = [
                    _sanitize_identifier(label, fallback=label) or label for label in labels
                ]
                san_path = "/".join(sanitized_labels)
                if san_path and san_path not in candidates:
                    candidates.append(san_path)
                for label, sanitized in zip(labels, sanitized_labels):
                    if label and label not in candidates:
                        candidates.append(label)
                    if sanitized and sanitized not in candidates:
                        candidates.append(sanitized)
            return tuple(candidates)

        def _rule_applies(rule: CurveWidthRule, curve) -> bool:
            if rule.layer and not _pattern_match(rule.layer, _candidate_layer_names()):
                return False
            if rule.curve and not _pattern_match(rule.curve, _curve_name_candidates(curve)):
                return False
            if rule.hierarchy and not _pattern_match(rule.hierarchy, _hierarchy_candidates(curve)):
                return False
            if rule.step_id is not None and rule.step_id != getattr(curve, "step_id", None):
                return False
            return True

        def _convert_rule_width_to_stage_units(width: float, unit: Optional[str]) -> Optional[float]:
            if width <= 0.0:
                return None
            unit_norm = (unit or "stage").strip().lower()
            if unit_norm in {"stage", "stage_unit", "stage-units", "stageunits"}:
                return float(width)
            factor = _UNIT_FACTORS.get(unit_norm)
            if factor is None:
                return None
            if stage_meters_per_unit <= 0:
                scale = 1.0
            else:
                scale = 1.0 / stage_meters_per_unit
            return float(width) * factor * scale

        def _resolved_curve_width(curve) -> Optional[float]:
            if not rules:
                return None
            resolved: Optional[float] = None
            for rule in rules:
                if not _rule_applies(rule, curve):
                    continue
                converted = _convert_rule_width_to_stage_units(float(rule.width), rule.unit)
                if converted is None:
                    continue
                resolved = converted
            return resolved

        for curve in caches.annotations.values():
            parent_path = inst_root
            for label, step_id in curve.hierarchy:
                parent_path = _ensure_group(parent_path, label, step_id)

            curves_parent = _ensure_curves_container(parent_path)

            curve_base = sanitize_name(curve.name, fallback=f"Geometry2D_{curve.step_id}") or "Geometry2D"
            curve_token = _unique_child_name(curves_parent, curve_base)
            curve_path = curves_parent.AppendChild(curve_token)

            if len(curve.points) < 2:
                continue

            basis = UsdGeom.BasisCurves.Define(stage, curve_path)
            basis.CreateTypeAttr(UsdGeom.Tokens.linear)
            basis.CreateBasisAttr(UsdGeom.Tokens.linear)
            basis.CreateWrapAttr(UsdGeom.Tokens.open)

            counts = Vt.IntArray([len(curve.points)])
            basis.CreateCurveVertexCountsAttr(counts)

            pt_array = Vt.Vec3fArray(len(curve.points))
            for idx, (x, y, z) in enumerate(curve.points):
                pt_array[idx] = (float(x), float(y), float(z))
            basis.CreatePointsAttr(pt_array)

            width_value = _resolved_curve_width(curve)
            if width_value is not None:
                basis.CreateWidthsAttr(Vt.FloatArray([float(width_value)]))

            prim = basis.GetPrim()
            try:
                prim.SetCustomDataByKey("ifc:annotationStepId", int(curve.step_id))
            except Exception:
                prim.SetCustomDataByKey("ifc:annotationStepId", curve.step_id)
            prim.SetCustomDataByKey("ifc:label", curve.name)

    return geom_layer


# ---------------- Persist instance cache (for debugging) ----------------

def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        if hasattr(value, "tolist"):
            return _json_safe(value.tolist())
        return str(value)


def persist_instance_cache(
    cache_dir: PathLike,
    base_name: str,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
) -> PathLike:
    cache_path = join_path(cache_dir, f"{base_name}.json")
    payload = {
        "schema": 1,
        "base_name": base_name,
        "instances": [
            {
                "step_id": record.step_id,
                "name": record.name,
                "guid": getattr(record, "guid", None),
                "transform": list(record.transform) if record.transform is not None else None,
                "prototype": {
                    "kind": record.prototype.kind,
                    "identifier": record.prototype.identifier,
                } if record.prototype is not None else None,
                "prototype_path": proto_paths.get(record.prototype).pathString if (record.prototype and proto_paths.get(record.prototype)) else None,
                "material_ids": list(record.material_ids or []),
                "attributes": _json_safe(record.attributes),
            }
            for record in caches.instances.values()
        ],
    }
    write_text(cache_path, json.dumps(payload, indent=2))
    return cache_path


# ---------------- Georef + Anchor — stubs (retain signatures) ----------------
# Plug back your full implementations here if you use pyproj & Cesium metadata.

def assign_world_geolocation(
    stage: Usd.Stage,
    base_point: BasePointConfig,
    projected_crs: str,
    geodetic_crs: str = "EPSG:4326",
    unit_hint: Optional[str] = None,
    lonlat_override: Optional[GeodeticCoordinate] = None,
):
    return None


def apply_stage_anchor_transform(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    base_point: Optional[BasePointConfig] = None,
    *,
    align_axes_to_map: bool = False,
    enable_geo_anchor: bool = True,  # kept for signature compatibility (unused)
    projected_crs: Optional[str] = None,
    lonlat: Optional[GeodeticCoordinate] = None,
    **_unused,
) -> None:
    """Anchor the USD stage to the survey/base-point reference frame.

    - Translates /World so the requested base-point (map coordinates) sits at the origin
      in stage space. Units are respected (base-point unit → meters → stage units).
    - When `align_axes_to_map` and an IfcMapConversion are present, rotates /World so the
      local XY axes honour the published map azimuth.
    - Persists the chosen anchor (translation/rotation) as prim custom data for debugging.
    - If geodetic lon/lat is supplied, writes them as authored attributes on /World so any
      downstream consumer can recover survey metadata.
    """

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        world_xf = UsdGeom.Xform.Define(stage, "/World")
        world_prim = world_xf.GetPrim()
    world_xf = UsdGeom.Xformable(world_prim)
    world_xf.ClearXformOpOrder()

    stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    if stage_meters_per_unit <= 0.0:
        stage_meters_per_unit = 1.0
    def _unit_to_meters(value: float, unit: Optional[str]) -> float:
        if unit is None:
            return float(value)
        factor = _UNIT_FACTORS.get(unit.strip().lower(), None)
        if factor is None:
            return float(value)
        return float(value) * factor

    bp_unit = None
    bp_easting_m = bp_northing_m = bp_height_m = 0.0
    if base_point is not None:
        bp_unit = base_point.unit or "m"
        bp_easting_m = _unit_to_meters(base_point.easting, bp_unit)
        bp_northing_m = _unit_to_meters(base_point.northing, bp_unit)
        bp_height_m = _unit_to_meters(base_point.height, bp_unit)

    map_conv = getattr(caches, "map_conversion", None)
    translation_stage = Gf.Vec3d(0.0, 0.0, 0.0)
    rotation_deg = 0.0

    if map_conv is not None:
        scale = float(getattr(map_conv, "scale", 1.0) or 1.0)
        eastings_m = float(getattr(map_conv, "eastings", 0.0) or 0.0) * scale
        northings_m = float(getattr(map_conv, "northings", 0.0) or 0.0) * scale
        ortho_height_m = float(getattr(map_conv, "orthogonal_height", 0.0) or 0.0) * scale
        if base_point is not None:
            d_e = bp_easting_m - eastings_m
            d_n = bp_northing_m - northings_m
            ax, ay = map_conv.normalized_axes()
            local_x = ax * d_e + ay * d_n
            local_y = -ay * d_e + ax * d_n
            local_z = bp_height_m - ortho_height_m
            translation_stage = Gf.Vec3d(
                -local_x / stage_meters_per_unit,
                -local_y / stage_meters_per_unit,
                -local_z / stage_meters_per_unit,
            )
        if align_axes_to_map:
            try:
                rotation_deg = float(map_conv.rotation_degrees())
            except Exception:
                rotation_deg = 0.0
    elif base_point is not None:
        translation_stage = Gf.Vec3d(
            -bp_easting_m / stage_meters_per_unit,
            -bp_northing_m / stage_meters_per_unit,
            -bp_height_m / stage_meters_per_unit,
        )

    anchor_matrix = Gf.Matrix4d(1.0)
    if rotation_deg:
        anchor_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_deg))
    anchor_matrix.SetTranslateOnly(translation_stage)
    world_xf.AddTransformOp().Set(anchor_matrix)

    translation_attr = world_prim.CreateAttribute("ifc:stageAnchorTranslation", Sdf.ValueTypeNames.Double3)
    translation_attr.Set(tuple(float(translation_stage[i]) for i in range(3)))
    rotation_attr = world_prim.CreateAttribute("ifc:stageAnchorRotationDegrees", Sdf.ValueTypeNames.Double)
    rotation_attr.Set(float(rotation_deg))
    bp_attr = world_prim.CreateAttribute("ifc:stageAnchorBasePointMeters", Sdf.ValueTypeNames.Double3)
    bp_attr.Set((float(bp_easting_m), float(bp_northing_m), float(bp_height_m)))
    unit_attr = world_prim.CreateAttribute("ifc:stageAnchorBasePointUnit", Sdf.ValueTypeNames.String)
    unit_attr.Set(bp_unit or "")
    crs_attr = world_prim.CreateAttribute("ifc:stageAnchorProjectedCRS", Sdf.ValueTypeNames.String)
    crs_attr.Set(projected_crs or "")
    if map_conv is not None:
        scale_attr = world_prim.CreateAttribute("ifc:stageAnchorMapScale", Sdf.ValueTypeNames.Double)
        scale_attr.Set(float(getattr(map_conv, "scale", 1.0) or 1.0))

    if lonlat is not None:
        longitude = float(lonlat.longitude)
        latitude = float(lonlat.latitude)
        height_val = float(lonlat.height) if lonlat.height is not None else None
        for attr_name, value in (
            ("ifc:longitude", longitude),
            ("ifc:latitude", latitude),
        ):
            attr = world_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Double)
            attr.Set(value)
        if height_val is not None:
            attr = world_prim.CreateAttribute("ifc:ellipsoidalHeight", Sdf.ValueTypeNames.Double)
            attr.Set(height_val)
        cesium_lon = world_prim.CreateAttribute("CesiumLongitude", Sdf.ValueTypeNames.Double)
        cesium_lon.Set(longitude)
        cesium_lat = world_prim.CreateAttribute("CesiumLatitude", Sdf.ValueTypeNames.Double)
        cesium_lat.Set(latitude)
        if height_val is not None:
            cesium_h = world_prim.CreateAttribute("CesiumHeight", Sdf.ValueTypeNames.Double)
            cesium_h.Set(height_val)


# ---------------- Federated master view — stub (retain signature) ----------------

def update_federated_view(
    master_stage_path: PathLike,
    payload_stage_path: PathLike,
    payload_prim_name: Optional[str] = None,
    parent_prim_path: str = "/World",
) -> Sdf.Layer:
    identifier = _layer_identifier(master_stage_path)
    stage = Usd.Stage.Open(identifier)
    created = False
    if stage is None:
        stage = Usd.Stage.CreateNew(identifier)
        created = True
    parent_path = Sdf.Path(parent_prim_path or "/World")
    if not parent_path:
        parent_path = Sdf.Path("/World")
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim:
        parent_prim = UsdGeom.Xform.Define(stage, parent_path).GetPrim()
    if not stage.HasDefaultPrim() or stage.GetDefaultPrim() != parent_prim:
        stage.SetDefaultPrim(parent_prim)
    if created:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    payload_name = payload_prim_name or sanitize_name(path_stem(payload_stage_path))
    target_path = parent_path.AppendChild(payload_name)
    payload_prim = stage.GetPrimAtPath(target_path)
    if not payload_prim:
        payload_prim = stage.DefinePrim(target_path, "Xform")
    refs = payload_prim.GetReferences()
    refs.ClearReferences()
    refs.AddReference(_layer_identifier(payload_stage_path))
    UsdGeom.Imageable(payload_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
    stage.GetRootLayer().Save()
    return stage.GetRootLayer()

