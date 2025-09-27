import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from src.config.manifest import BasePointConfig, GeodeticCoordinate
from src.process_ifc import ConversionOptions, InstanceRecord, PrototypeCaches, PrototypeKey
from src.utils.matrix_utils import np_to_gf_matrix, scale_matrix_translation_only

try:
    from pyproj import CRS, Transformer
    _HAVE_PYPROJ = True
except Exception:
    _HAVE_PYPROJ = False


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


# 1. ---------------------------- Name helpers ----------------------------

def sanitize_name(raw_name, fallback=None):
    """Make a USD-legal prim name with a short uniq suffix."""
    base = str(raw_name or fallback or "Unnamed")
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    unique_suffix = f"_{int(time.time()*1000)%10000}"
    return (name + unique_suffix)[:63]


def _sanitize_identifier(raw_name: Optional[str], fallback: Optional[str] = None) -> str:
    base = str(raw_name or fallback or "Unnamed")
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    return name[:63]


def _unique_name(base: str, used: Dict[str, int]) -> str:
    count = used.get(base, 0)
    used[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count}"


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


# 2. ---------------------------- Mesh helpers ----------------------------

def create_usd_stage(usd_path, meters_per_unit=1.0):
    """Create a new USD stage with Z-up and a /World default prim.

    Args:
        usd_path: Filesystem path where the stage will be written.
        meters_per_unit: Stage linear units (USD metadata metersPerUnit).

    Returns:
        The newly created Usd.Stage instance.
    """
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", float(meters_per_unit))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


def write_usd_mesh(stage, parent_path, mesh_name, verts, faces, abs_mat=None,
                   material_ids=None, stage_meters_per_unit=1.0,
                   scale_matrix_translation=False):
    """
    Author a mesh with local points + single absolute xformOp:transform.
    """
    mesh_path = Sdf.Path(parent_path).AppendChild(mesh_name)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    if verts:
        n = len(verts) // 3
        pts = Vt.Vec3fArray(n)
        for i in range(n):
            x, y, z = verts[3*i:3*i+3]
            pts[i] = Gf.Vec3f(float(x), float(y), float(z))
        mesh.CreatePointsAttr(pts)
    else:
        print(f"?? No vertices for {mesh_name}; creating empty mesh.")

    if faces:
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(faces)//3)))
    else:
        print(f"?? No faces for {mesh_name}; mesh will be empty.")

    if abs_mat is not None:
        try:
            gf = np_to_gf_matrix(abs_mat)
            if scale_matrix_translation and stage_meters_per_unit != 1.0:
                gf = scale_matrix_translation_only(gf, 1.0/float(stage_meters_per_unit))
            xf = UsdGeom.Xformable(mesh)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(gf)
        except Exception as e:
            print(f"?? Invalid matrix for {mesh_name}: {e}")

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


def _iter_prototypes(caches: PrototypeCaches) -> Iterable[Tuple[PrototypeKey, Any]]:
    for rep_id, proto in sorted(caches.repmaps.items()):
        yield PrototypeKey(kind="repmap", identifier=rep_id), proto
    for digest, proto in sorted(caches.hashes.items()):
        yield PrototypeKey(kind="hash", identifier=digest), proto



# 3. ---------------------------- Material helpers ----------------------------

_MATERIAL_PRESET_LIBRARY = [
    {
        "keywords": {"stainless", "steel"},
        "color": (0.58, 0.58, 0.6),
        "metallic": 1.0,
        "roughness": 0.28,
        "specular": 0.55,
    },
    {
        "keywords": {"steel"},
        "color": (0.45, 0.45, 0.45),
        "metallic": 1.0,
        "roughness": 0.35,
        "specular": 0.5,
    },
    {
        "keywords": {"aluminium", "aluminum"},
        "color": (0.77, 0.77, 0.78),
        "metallic": 1.0,
        "roughness": 0.22,
        "specular": 0.55,
    },
    {
        "keywords": {"concrete"},
        "color": (0.55, 0.55, 0.52),
        "metallic": 0.0,
        "roughness": 0.9,
    },
    {
        "keywords": {"asphalt"},
        "color": (0.08, 0.08, 0.08),
        "metallic": 0.0,
        "roughness": 0.95,
    },
    {
        "keywords": {"glass"},
        "color": (0.92, 0.95, 0.98),
        "metallic": 0.0,
        "roughness": 0.05,
        "opacity": 0.12,
        "ior": 1.45,
        "specular": 0.1,
    },
    {
        "keywords": {"timber"},
        "color": (0.63, 0.45, 0.32),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    {
        "keywords": {"wood"},
        "color": (0.6, 0.43, 0.3),
        "metallic": 0.0,
        "roughness": 0.6,
    },
    {
        "keywords": {"brick"},
        "color": (0.65, 0.28, 0.23),
        "metallic": 0.0,
        "roughness": 0.75,
    },
    {
        "keywords": {"plaster"},
        "color": (0.83, 0.83, 0.8),
        "metallic": 0.0,
        "roughness": 0.7,
    },
    {
        "keywords": {"paint"},
        "metallic": 0.0,
        "roughness": 0.4,
    },
    {
        "keywords": {"gypsum"},
        "color": (0.85, 0.84, 0.81),
        "metallic": 0.0,
        "roughness": 0.7,
    },
    {
        "keywords": {"plastic"},
        "metallic": 0.0,
        "roughness": 0.35,
        "specular": 0.6,
    },
    {
        "keywords": {"ceramic"},
        "metallic": 0.0,
        "roughness": 0.15,
        "specular": 0.5,
    },
]

def _normalize_color(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, float(c))) for c in color)


def _material_color(material: Any) -> Tuple[float, float, float]:
    candidates = [
        "Diffuse", "diffuse", "DiffuseColor", "diffuseColor",
        "SurfaceColour", "surfaceColour", "surface_colour", "Color",
    ]
    for attr in candidates:
        value = getattr(material, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return _normalize_color((value[0], value[1], value[2]))
            except Exception:
                continue
    return (0.8, 0.8, 0.8)


def _material_opacity(material: Any) -> float:
    value = None
    for attr in ("Transparency", "transparency", "Opacity", "opacity"):
        data = getattr(material, attr, None)
        if data is None:
            continue
        value = data() if callable(data) else data
        break
    if value is None:
        return 1.0
    try:
        opacity = 1.0 - float(value)
    except Exception:
        opacity = 1.0
    return max(0.0, min(1.0, opacity))


def _material_display_name(material: Any, fallback: str) -> str:
    name = None
    for attr in ("Name", "name", "Label", "label"):
        value = getattr(material, attr, None)
        if isinstance(value, str) and value.strip():
            name = value
            break
    return _sanitize_identifier(name, fallback=fallback)


def _material_strings(material: Any) -> List[str]:
    tokens: List[str] = []
    def _consume(text: Optional[str]) -> None:
        if not text:
            return
        for piece in re.split(r"[^A-Za-z0-9]+", text.lower()):
            if piece:
                tokens.append(piece)
    if isinstance(material, str):
        _consume(material)
    else:
        for attr in ("Name", "name", "Category", "CategoryName", "Description", "MaterialCategory"):
            value = getattr(material, attr, None)
            if isinstance(value, str):
                _consume(value)
        if hasattr(material, "is_a"):
            try:
                isa = material.is_a()
                if isinstance(isa, str):
                    _consume(isa)
            except Exception:
                pass
    return tokens


def _match_material_preset(material: Any) -> Optional[Dict[str, Any]]:
    tokens = set(_material_strings(material))
    for preset in _MATERIAL_PRESET_LIBRARY:
        if preset["keywords"].issubset(tokens):
            return preset
    return None


def _material_pbr_parameters(material: Any) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "color": _material_color(material),
        "metallic": 0.0,
        "roughness": 0.6,
        "opacity": _material_opacity(material),
        "specular": 0.5,
    }
    preset = _match_material_preset(material)
    if preset:
        if "color" in preset:
            params["color"] = preset["color"]
        for key in ("metallic", "roughness", "opacity", "specular", "ior", "emissiveColor"):
            if key in preset:
                params[key] = preset[key]
    params["color"] = _normalize_color(params["color"])
    if "emissiveColor" in params:
        params["emissiveColor"] = _normalize_color(params["emissiveColor"])
    params["metallic"] = max(0.0, min(1.0, float(params.get("metallic", 0.0))))
    params["roughness"] = max(0.0, min(1.0, float(params.get("roughness", 0.6))))
    params["opacity"] = max(0.0, min(1.0, float(params.get("opacity", 1.0))))
    params["specular"] = max(0.0, min(1.0, float(params.get("specular", 0.5))))
    return params

# 4. ---------------------------- Prototype authoring ----------------------------

def _name_for_repmap(proto: Any) -> str:
    parts: List[str] = []
    if getattr(proto, "type_name", None):
        parts.append(_sanitize_identifier(proto.type_name))
    else:
        parts.append(f"RepMap_{proto.repmap_id}")
    if getattr(proto, "type_class", None):
        parts.append(_sanitize_identifier(proto.type_class))
    if getattr(proto, "type_guid", None):
        parts.append(_sanitize_identifier(proto.type_guid))
    if getattr(proto, "repmap_index", None) is not None:
        parts.append(f"i{proto.repmap_index}")
    return _sanitize_identifier("_".join(parts), fallback=f"RepMap_{proto.repmap_id}")


def _name_for_hash(proto: Any) -> str:
    base = (getattr(proto, "name", None) or getattr(proto, "type_name", None)
            or "Fallback")
    digest = getattr(proto, "digest", "unknown")
    return _sanitize_identifier(f"{base}_Fallback_{str(digest)[:12]}")


def author_prototype_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    output_dir: Path,
    base_name: str,
    options: ConversionOptions,
) -> Tuple[Sdf.Layer, Dict[PrototypeKey, Sdf.Path]]:
    """Author a prototype layer and fill /World/__Prototypes with meshes.

    Returns the authored Sdf.Layer and mapping from PrototypeKey to prim path.
    """
    proto_file = (output_dir / f"{base_name}_prototypes.usda").resolve()
    proto_layer = Sdf.Layer.CreateNew(proto_file.as_posix())

    root_layer = stage.GetRootLayer()
    if proto_layer.identifier not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(proto_layer.identifier)

    proto_root = Sdf.Path("/World/__Prototypes")
    proto_paths: Dict[PrototypeKey, Sdf.Path] = {}
    used_names: Dict[str, int] = {}

    with Usd.EditContext(stage, proto_layer):
        UsdGeom.Scope.Define(stage, proto_root)

        for key, proto in _iter_prototypes(caches):
            mesh_data = _flatten_mesh_arrays(getattr(proto, "mesh", None))
            if not mesh_data:
                continue
            verts, faces = mesh_data

            if key.kind == "repmap":
                base_name_candidate = _name_for_repmap(proto)
            else:
                base_name_candidate = _name_for_hash(proto)
            prim_name = _unique_name(base_name_candidate, used_names)
            proto_path = proto_root.AppendChild(prim_name)
            proto_paths[key] = proto_path

            xform = UsdGeom.Xform.Define(stage, proto_path)
            xform.ClearXformOpOrder()

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


# 5. ---------------------------- Material authoring ----------------------------

def _prototype_materials(caches: PrototypeCaches, key: PrototypeKey) -> List[Any]:
    if key.kind == "repmap":
        proto = caches.repmaps.get(int(key.identifier))
    else:
        proto = caches.hashes.get(str(key.identifier))
    if not proto:
        return []
    return list(getattr(proto, "materials", []) or [])


def author_material_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    output_dir: Path,
    base_name: str,
    proto_layer: Sdf.Layer,
    options: ConversionOptions,
) -> Tuple[Sdf.Layer, Dict[PrototypeKey, List[Sdf.Path]]]:
    """Author a material layer and return mapping from prototype key to materials."""
    material_file = (output_dir / f"{base_name}_materials.usda").resolve()
    material_layer = Sdf.Layer.CreateNew(material_file.as_posix())

    root_layer = stage.GetRootLayer()
    if material_layer.identifier not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(material_layer.identifier)
    if material_layer.identifier not in proto_layer.subLayerPaths:
        proto_layer.subLayerPaths.append(material_layer.identifier)

    material_root = Sdf.Path("/World/__Materials")
    material_paths: Dict[PrototypeKey, List[Sdf.Path]] = {}
    used_names: Dict[str, int] = {}

    with Usd.EditContext(stage, material_layer):
        UsdGeom.Scope.Define(stage, material_root)

        for key, proto_path in proto_paths.items():
            materials = _prototype_materials(caches, key)
            if not materials:
                continue
            material_entries: List[Sdf.Path] = []
            base = proto_path.name
            for idx, material in enumerate(materials):
                display_name = _material_display_name(material, fallback=f"{base}_Mat{idx}")
                mat_name = _unique_name(display_name, used_names)
                mat_path = material_root.AppendChild(mat_name)
                shader_path = mat_path.AppendChild("Shader")

                material_prim = UsdShade.Material.Define(stage, mat_path)
                shader = UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                params = _material_pbr_parameters(material)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*params["color"]))
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(params["opacity"]))
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(params["metallic"]))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(params["roughness"]))
                shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(float(params.get("specular", 0.5)))
                if "ior" in params:
                    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(params["ior"]))
                if "emissiveColor" in params:
                    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*params["emissiveColor"]))
                surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                material_prim.CreateSurfaceOutput().ConnectToSource(surface_output)

                material_entries.append(mat_path)
            if material_entries:
                material_paths[key] = material_entries

    return material_layer, material_paths


# 6. ---------------------------- Material binding ----------------------------

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




def _author_namespaced_dictionary_attributes(prim: Usd.Prim, namespace: str, entries: Dict[str, Any]) -> None:
    """Author IFC psets/qtos as USD attributes under a BIMData namespace.

    The attribute naming convention is:
      BIMData:Psets:<PsetName>:<PropName>
      BIMData:QTO:<QtoName>:<PropName>

    Names are sanitized to be USD-identifier-safe. Values are typed based on Python types.
    """
    if not entries:
        return

    root_ns = "BIMData"
    ns_map = {"pset": "Psets", "qtos": "QTO", "qto": "QTO"}
    ns1 = ns_map.get(namespace.lower(), _sanitize_identifier(namespace, fallback="Meta"))

    def _set_attr(full_name: str, value: Any) -> None:
        # Decide type
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
            # Fallback: serialize to string
            attr = prim.CreateAttribute(full_name, Sdf.ValueTypeNames.String)
            attr.Set(str(value))
            return
        # Default to string for other types (tokens/enums can be strings)
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
            # If entries is not a dict-of-dicts, write the single value under the set name
            full_attr = f"{root_ns}:{ns1}:{set_token}:Value"
            _set_attr(full_attr, payload)


def _author_instance_attributes(prim: Usd.Prim, attributes: Optional[Dict[str, Any]]) -> None:
    """Author IFC property sets and quantities as BIMData attributes on a prim."""
    if not attributes:
        return
    if not isinstance(attributes, dict):
        return
    psets = attributes.get("psets", {})
    qtos = attributes.get("qtos", {})
    _author_namespaced_dictionary_attributes(prim, "pset", psets)
    _author_namespaced_dictionary_attributes(prim, "qto", qtos)


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


def _serialize_instance_record(record: InstanceRecord, proto_paths: Dict[PrototypeKey, Sdf.Path]) -> Dict[str, Any]:
    proto_path_str = None
    if record.prototype is not None and proto_paths:
        proto_path = proto_paths.get(record.prototype)
        if proto_path is not None:
            proto_path_str = proto_path.pathString
    def _as_list(val):
        return list(val) if val is not None else None
    mesh_payload = None
    if record.mesh:
        flattened = _flatten_mesh_arrays(record.mesh)
        if flattened:
            verts_list, faces_list = flattened
            mesh_payload = {
                "vertices": verts_list,
                "faces": faces_list,
            }
    return {
        "step_id": record.step_id,
        "product_id": record.product_id,
        "name": record.name,
        "transform": _as_list(record.transform),
        "material_ids": list(record.material_ids or []),
        "materials": _json_safe(record.materials),
        "attributes": _json_safe(record.attributes),
        "prototype_delta": _as_list(record.prototype_delta),
        "hierarchy": [[label, sid] for label, sid in (record.hierarchy or tuple())],
        "mesh": mesh_payload,
        "prototype": {
            "kind": record.prototype.kind,
            "identifier": record.prototype.identifier,
        } if record.prototype is not None else None,
        "prototype_path": proto_path_str,
    }


def persist_instance_cache(cache_dir: Path, base_name: str, caches: PrototypeCaches, proto_paths: Dict[PrototypeKey, Sdf.Path]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{base_name}.json"
    payload = {
        "schema": 1,
        "base_name": base_name,
        "instances": [
            _serialize_instance_record(record, proto_paths) for record in caches.instances.values()
        ],
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cache_path


def _prepared_instances_from_serialized(data: Dict[str, Any]) -> List[PreparedInstance]:
    prepared: List[PreparedInstance] = []
    for entry in data.get("instances", []):
        proto_path_str = entry.get("prototype_path")
        proto_path = Sdf.Path(proto_path_str) if proto_path_str else None
        hierarchy = tuple(
            (str(label), int(step) if step is not None else None)
            for label, step in entry.get("hierarchy", [])
        )
        transform_list = entry.get("transform")
        transform = tuple(transform_list) if transform_list else None
        proto_delta_list = entry.get("prototype_delta")
        prototype_delta = tuple(proto_delta_list) if proto_delta_list else None
        mesh_entry = entry.get("mesh")
        mesh_payload = None
        if mesh_entry:
            mesh_payload = {
                "vertices": mesh_entry.get("vertices"),
                "faces": mesh_entry.get("faces"),
            }
        prepared.append(
            PreparedInstance(
                step_id=entry.get("step_id"),
                name=entry.get("name", ""),
                transform=transform,
                attributes=entry.get("attributes") or {},
                hierarchy=hierarchy,
                prototype_path=proto_path,
                prototype_delta=prototype_delta,
                material_ids=list(entry.get("material_ids") or []),
                materials=list(entry.get("materials") or []),
                mesh=mesh_payload,
                source_path=None,
            )
        )
    return prepared


def load_instance_cache(cache_path: Path) -> List[PreparedInstance]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return _prepared_instances_from_serialized(data)


def _prepare_instances_from_cache(caches: PrototypeCaches, proto_paths: Dict[PrototypeKey, Sdf.Path]) -> List[PreparedInstance]:
    prepared: List[PreparedInstance] = []
    for record in caches.instances.values():
        proto_path = None
        if record.prototype is not None and proto_paths:
            proto_path = proto_paths.get(record.prototype)
        prepared.append(
            PreparedInstance(
                step_id=record.step_id,
                name=record.name,
                transform=tuple(record.transform) if record.transform is not None else None,
                attributes=record.attributes or {},
                hierarchy=tuple(record.hierarchy or tuple()),
                prototype_path=proto_path,
                prototype_delta=tuple(record.prototype_delta) if record.prototype_delta else None,
                material_ids=list(record.material_ids or []),
                materials=list(record.materials or []),
                mesh=record.mesh,
                source_path=None,
            )
        )
    return prepared


def _collect_bimdata_attributes_from_prim(prim: Usd.Prim) -> Dict[str, Any]:
    result: Dict[str, Any] = {"psets": {}, "qtos": {}}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith("BIMData:"):
            continue
        tokens = name.split(":")
        if len(tokens) < 4:
            continue
        _, ns, set_name, prop_name, *rest = tokens
        if rest:
            prop_name = ":".join([prop_name] + rest)
        try:
            value = attr.Get()
        except Exception:
            continue
        if ns.lower().startswith("pset"):
            bucket = result.setdefault("psets", {})
        elif ns.lower().startswith("qto"):
            bucket = result.setdefault("qtos", {})
        else:
            bucket = result.setdefault(ns, {})
        group = bucket.setdefault(set_name, {})
        group[prop_name] = value
    if not result.get("psets") and not result.get("qtos"):
        return {}
    return result


def _collect_hierarchy_for_prim(prim: Usd.Prim, inst_root: Sdf.Path) -> Tuple[Tuple[str, Optional[int]], ...]:
    nodes: List[Tuple[str, Optional[int]]] = []
    parent = prim.GetParent()
    while parent and parent.GetPath().HasPrefix(inst_root) and parent.GetPath() != inst_root:
        label = parent.GetCustomDataByKey("ifc:label")
        step_id = parent.GetCustomDataByKey("ifc:stepId")
        if label:
            try:
                step_val = int(step_id) if step_id is not None else None
            except Exception:
                step_val = step_id
            nodes.append((str(label), step_val))
        parent = parent.GetParent()
    nodes.reverse()
    return tuple(nodes)


def _extract_mesh_from_prim(prim: Usd.Prim) -> Optional[Dict[str, Any]]:
    if not prim or not prim.IsValid():
        return None
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None
    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or indices is None:
        return None
    verts: List[float] = []
    for pt in points:
        verts.extend([float(pt[0]), float(pt[1]), float(pt[2])])
    faces = [int(i) for i in indices]
    if not verts or not faces:
        return None
    return {"vertices": verts, "faces": faces}


def _prepare_instances_from_stage(stage: Usd.Stage, inst_root: Sdf.Path) -> List[PreparedInstance]:
    prepared: List[PreparedInstance] = []
    inst_root_prim = stage.GetPrimAtPath(inst_root)
    if not inst_root_prim:
        return prepared
    grouping_set = inst_root_prim.GetVariantSets().GetVariantSet("Grouping") if inst_root_prim.HasVariantSets() else None
    previous_selection = None
    if grouping_set:
        previous_selection = grouping_set.GetVariantSelection()
        if "Original" in grouping_set.GetVariantNames():
            grouping_set.SetVariantSelection("Original")
    try:
        for prim in Usd.PrimRange(inst_root_prim):
            if prim == inst_root_prim:
                continue
            if prim.GetTypeName() != "Xform":
                continue
            step_id = prim.GetCustomDataByKey("ifc:instanceStepId")
            if step_id is None:
                continue
            try:
                step_val = int(step_id)
            except Exception:
                step_val = step_id
            xformable = UsdGeom.Xformable(prim)
            matrix, _ = xformable.GetLocalTransformation()
            transform = tuple(matrix[i][j] for i in range(4) for j in range(4)) if matrix else None
            hierarchy = _collect_hierarchy_for_prim(prim, inst_root)
            attributes = _collect_bimdata_attributes_from_prim(prim)
            prototype_path = None
            prototype_delta = None
            prototype_child = next((child for child in prim.GetChildren() if child.GetName() == "Prototype"), None)
            if prototype_child:
                refs = prototype_child.GetReferences()
                ref_items = refs.GetAddedOrExplicitItems()
                if ref_items:
                    prototype_path = ref_items[0].primPath
                proto_xform = UsdGeom.Xformable(prototype_child)
                proto_matrix, _ = proto_xform.GetLocalTransformation()
                if proto_matrix and proto_matrix != Gf.Matrix4d(1):
                    prototype_delta = tuple(proto_matrix[i][j] for i in range(4) for j in range(4))
            geom_child = prim.GetChild("Geom")
            mesh_payload = _extract_mesh_from_prim(geom_child) if geom_child and prototype_path is None else None
            material_ids_attr = prim.GetAttribute("ifc:materialIds")
            material_ids = list(material_ids_attr.Get() or []) if material_ids_attr and material_ids_attr.HasAuthoredValueOpinion() else []
            bound_materials: List[Any] = []
            target_for_binding = geom_child if geom_child else prim
            try:
                binding = UsdShade.MaterialBindingAPI(target_for_binding).ComputeBoundMaterial()[0]
                if binding:
                    bound_materials.append(binding.GetPath().pathString)
            except Exception:
                pass
            prepared.append(
                PreparedInstance(
                    step_id=step_val,
                    name=prim.GetName(),
                    transform=transform,
                    attributes=attributes,
                    hierarchy=hierarchy,
                    prototype_path=prototype_path,
                    prototype_delta=prototype_delta,
                    material_ids=material_ids,
                    materials=bound_materials,
                    mesh=mesh_payload,
                    source_path=prim.GetPath(),
                )
            )
    finally:
        if grouping_set and previous_selection:
            grouping_set.SetVariantSelection(previous_selection)
    return prepared

# 7. ---------------------------- Instance authoring ----------------------------

def author_instance_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    output_dir: Path,
    base_name: str,
    options: ConversionOptions,
) -> Sdf.Layer:
    inst_file = (output_dir / f"{base_name}_instances.usda").resolve()
    inst_layer = Sdf.Layer.CreateNew(inst_file.as_posix())

    root_layer = stage.GetRootLayer()
    if inst_layer.identifier not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(inst_layer.identifier)

    inst_root_name = _sanitize_identifier(f"{base_name}_Instances", fallback="Instances")
    inst_root = Sdf.Path(f"/World/{inst_root_name}")
    name_counters: Dict[Sdf.Path, Dict[str, int]] = defaultdict(dict)
    hierarchy_nodes_by_step: Dict[int, Sdf.Path] = {}
    hierarchy_nodes_by_label: Dict[Tuple[Sdf.Path, str], Sdf.Path] = {}
    stage_meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)

    def _unique_child_name(parent_path: Sdf.Path, base: str) -> str:
        return _unique_name(base, name_counters[parent_path])

    def _ensure_group(parent_path: Sdf.Path, label: str, step_id: Optional[int]) -> Sdf.Path:
        if step_id is not None and step_id in hierarchy_nodes_by_step:
            return hierarchy_nodes_by_step[step_id]
        sanitized = _sanitize_identifier(label, fallback=f"Group_{step_id or 'Node'}")
        if not sanitized:
            sanitized = "Group"
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
            try:
                inst_prim.SetCustomDataByKey("ifc:instanceStepId", int(record.step_id))
            except Exception:
                inst_prim.SetCustomDataByKey("ifc:instanceStepId", record.step_id)
            if record.transform:
                try:
                    xf = np_to_gf_matrix(record.transform)
                    xform.AddTransformOp().Set(xf)
                except Exception as exc:
                    print(f"?? Failed to author transform for instance {record.step_id}: {exc}")

            if options.convert_metadata and record.attributes:
                _author_instance_attributes(inst_prim, record.attributes)

            if record.mesh:
                mesh_data = _flatten_mesh_arrays(record.mesh)
                if not mesh_data:
                    print(f"?? Missing mesh payload for instance {record.step_id}; skipping")
                    continue
                verts, faces = mesh_data
                write_usd_mesh(
                    stage,
                    inst_path,
                    "Geom",
                    verts,
                    faces,
                    abs_mat=None,
                    material_ids=list(record.material_ids or []),
                    stage_meters_per_unit=stage_meters_per_unit,
                )
                continue

            proto_path = proto_paths.get(record.prototype)
            if proto_path is None:
                print(f"?? Missing prototype for instance {record.step_id}; skipping")
                continue

            ref_path = inst_path.AppendChild("Prototype")
            ref_prim = stage.DefinePrim(ref_path, "Xform")
            refs = ref_prim.GetReferences()
            refs.ClearReferences()
            refs.AddReference("", proto_path)

            ref_xform = UsdGeom.Xform(ref_prim)
            ref_xform.ClearXformOpOrder()
            if record.prototype_delta:
                try:
                    ref_xform.AddTransformOp().Set(np_to_gf_matrix(record.prototype_delta))
                except Exception as exc:
                    print(f"?? Failed to apply prototype delta for instance {record.step_id}: {exc}")

            ref_prim.SetInstanceable(bool(options.enable_instancing))

    return inst_layer


class GroupingSpecError(ValueError):
    """Raised when a grouping specification cannot be resolved."""


def _ensure_layer_on_stage(stage: Usd.Stage, layer: Sdf.Layer, *, persist: bool) -> None:
    """Attach a layer to the stage's session or root stack if not already present."""
    target = stage.GetRootLayer() if persist else stage.GetSessionLayer()
    if layer.identifier not in target.subLayerPaths:
        target.subLayerPaths.append(layer.identifier)


def _map_instance_paths(stage: Usd.Stage, inst_root: Sdf.Path) -> Dict[int, Sdf.Path]:
    """Build a map from IFC instance step id to its authored prim path."""
    result: Dict[int, Sdf.Path] = {}
    root_prim = stage.GetPrimAtPath(inst_root)
    if not root_prim:
        return result
    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim:
            continue
        if prim.GetTypeName() != "Xform":
            continue
        step_id = prim.GetCustomDataByKey("ifc:instanceStepId")
        if step_id is None:
            continue
        try:
            key = int(step_id)
        except Exception:
            key = step_id
        result[key] = prim.GetPath()
    return result


def _resolve_group_value(record: InstanceRecord, accessor: Union[str, Callable[[InstanceRecord], Any]]) -> Any:
    if callable(accessor):
        return accessor(record)
    if not accessor:
        return None
    current: Any = record.attributes
    for token in str(accessor).split('.'):
        if isinstance(current, dict):
            current = current.get(token)
        else:
            return None
    return current


def _format_group_token(value: Any, *, prefix: str) -> str:
    display = "Unspecified" if value in (None, "", []) else str(value)
    base = _sanitize_identifier(f"{prefix}_{display}", fallback="Group")
    return base or "Group"


def author_instance_grouping_variant(
    stage: Usd.Stage,
    caches: Optional[PrototypeCaches],
    proto_paths: Optional[Dict[PrototypeKey, Sdf.Path]],
    base_name: str,
    options: ConversionOptions,
    grouping_accessor: Union[str, Callable[[Any], Any]],
    variant_name: str,
    *,
    cache_path: Optional[Path] = None,
    drop_leading_levels: int = 0,
    keep_class_anchor: bool = True,
    persist_path: Optional[Path] = None,
    grouping_label_prefix: Optional[str] = None,
) -> Sdf.Layer:
    """Author a dynamic grouping variant under the instance root.

    The variant is created beneath /World/<base>_Instances with variant set "Grouping".
    Existing spatial hierarchy prims remain in the default "Original" variant. The
    new variant deactivates the original instance prims and instantiates clones under
    new grouping nodes derived from `grouping_accessor`.
    """
    inst_root_name = _sanitize_identifier(f"{base_name}_Instances", fallback="Instances")
    inst_root = Sdf.Path(f"/World/{inst_root_name}")
    inst_root_prim = stage.GetPrimAtPath(inst_root)
    if not inst_root_prim:
        raise GroupingSpecError(f"Instance root prim not found at {inst_root}")

    instances_data: List[PreparedInstance] = []
    if caches is not None:
        if proto_paths is None:
            raise GroupingSpecError("proto_paths is required when caches are provided")
        instances_data = _prepare_instances_from_cache(caches, proto_paths)
    elif cache_path is not None:
        if not cache_path.exists():
            raise GroupingSpecError(f"Cache file not found: {cache_path}")
        instances_data = load_instance_cache(cache_path)
    else:
        instances_data = _prepare_instances_from_stage(stage, inst_root)

    if not instances_data:
        raise GroupingSpecError("No instances available to author grouping variant")

    if persist_path is not None:
        persist = True
        variant_layer = Sdf.Layer.CreateNew(Path(persist_path).as_posix())
    else:
        persist = False
        variant_layer = Sdf.Layer.CreateAnonymous(f"group_{variant_name}")

    _ensure_layer_on_stage(stage, variant_layer, persist=persist)

    grouping_label_prefix = grouping_label_prefix or _sanitize_identifier(variant_name, fallback="Group") or "Group"

    with Usd.EditContext(stage, variant_layer):
        variant_sets = inst_root_prim.GetVariantSets()
        grouping_set = variant_sets.AddVariantSet("Grouping")
        variant_names = set(grouping_set.GetVariantNames())
        if "Original" not in variant_names:
            grouping_set.AddVariant("Original")
        if variant_name in variant_names:
            grouping_set.RemoveVariant(variant_name)
        grouping_set.AddVariant(variant_name)
        grouping_set.SetVariantSelection(variant_name)

        with grouping_set.GetVariantEditContext(variant_name):
            instance_path_map = _map_instance_paths(stage, inst_root)
            for path in instance_path_map.values():
                stage.OverridePrim(path).SetActive(False)

            name_counters: Dict[Sdf.Path, Dict[str, int]] = defaultdict(dict)
            group_cache: Dict[Tuple[Sdf.Path, str, Optional[int]], Sdf.Path] = {}

            def _unique_child_name(parent: Sdf.Path, base: str) -> str:
                used = name_counters[parent]
                count = used.get(base, 0)
                used[base] = count + 1
                return base if count == 0 else f"{base}_{count}"

            def _ensure_group(parent_path: Sdf.Path, label: str, step_id: Optional[int]) -> Sdf.Path:
                key = (parent_path, label, step_id)
                cached = group_cache.get(key)
                if cached is not None:
                    return cached
                base_token = _sanitize_identifier(label, fallback=f"Group_{step_id or 'Node'}") or "Group"
                token = _unique_child_name(parent_path, base_token)
                group_path = parent_path.AppendChild(token)
                group_cache[key] = group_path
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
                return group_path

            stage_meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
            drop_count = max(0, int(drop_leading_levels))

            for instance in instances_data:
                group_value = _resolve_group_value(instance, grouping_accessor)
                parent_path = inst_root
                hierarchy = list(instance.hierarchy or ())
                trimmed = hierarchy[drop_count:] if drop_count else hierarchy
                if keep_class_anchor and not trimmed and hierarchy:
                    trimmed = [hierarchy[-1]]
                for label, step_id in trimmed:
                    parent_path = _ensure_group(parent_path, label, step_id)

                group_label = str(group_value if group_value not in (None, "", []) else "Unspecified")
                group_token = _format_group_token(group_value, prefix=grouping_label_prefix)
                group_path = _ensure_group(parent_path, group_token, None)
                group_prim = stage.GetPrimAtPath(group_path)
                group_prim.SetCustomDataByKey("ifc:groupingKey", str(grouping_accessor))
                group_prim.SetCustomDataByKey("ifc:groupingValue", group_label)

                inst_base = _sanitize_identifier(instance.name, fallback=f"Ifc_{instance.step_id}") or "Instance"
                inst_token = _unique_child_name(group_path, inst_base)
                inst_path = group_path.AppendChild(inst_token)

                xform = UsdGeom.Xform.Define(stage, inst_path)
                xform.ClearXformOpOrder()
                inst_prim = xform.GetPrim()
                try:
                    inst_prim.SetCustomDataByKey("ifc:instanceStepId", int(instance.step_id))
                except Exception:
                    inst_prim.SetCustomDataByKey("ifc:instanceStepId", instance.step_id)

                if instance.transform:
                    try:
                        xform.AddTransformOp().Set(np_to_gf_matrix(instance.transform))
                    except Exception as exc:
                        print(f"?? Failed to author transform for grouping variant instance {instance.step_id}: {exc}")

                if options.convert_metadata and instance.attributes:
                    _author_instance_attributes(inst_prim, instance.attributes)

                if instance.mesh:
                    mesh_payload = _flatten_mesh_arrays(instance.mesh)
                    if mesh_payload:
                        verts, faces = mesh_payload
                        write_usd_mesh(
                            stage,
                            inst_path,
                            "Geom",
                            verts,
                            faces,
                            abs_mat=None,
                            material_ids=list(instance.material_ids or []),
                            stage_meters_per_unit=stage_meters_per_unit,
                        )
                        continue

                proto_path = instance.prototype_path
                if proto_path is None and instance.source_path is not None:
                    ref_prim = stage.DefinePrim(inst_path.AppendChild("Source"), "Xform")
                    refs = ref_prim.GetReferences()
                    refs.ClearReferences()
                    refs.AddReference("", instance.source_path)
                    continue

                if proto_path is None:
                    print(f"?? Missing prototype for grouping variant instance {instance.step_id}; skipping")
                    continue

                ref_path = inst_path.AppendChild("Prototype")
                ref_prim = stage.DefinePrim(ref_path, "Xform")
                refs = ref_prim.GetReferences()
                refs.ClearReferences()
                refs.AddReference("", proto_path)

                ref_xform = UsdGeom.Xform(ref_prim)
                ref_xform.ClearXformOpOrder()
                if instance.prototype_delta:
                    try:
                        ref_xform.AddTransformOp().Set(np_to_gf_matrix(instance.prototype_delta))
                    except Exception as exc:
                        print(f"?? Failed to apply prototype delta for grouping variant instance {instance.step_id}: {exc}")

                ref_prim.SetInstanceable(bool(options.enable_instancing))

    return variant_layer

def reset_instance_grouping(stage: Usd.Stage, base_name: str, variant_set: str = "Grouping") -> None:
    """Reset the grouping variant selection back to the canonical hierarchy."""
    inst_root_name = _sanitize_identifier(f"{base_name}_Instances", fallback="Instances")
    inst_root = Sdf.Path(f"/World/{inst_root_name}")
    prim = stage.GetPrimAtPath(inst_root)
    if not prim:
        return
    vs = prim.GetVariantSets().GetVariantSet(variant_set)
    if vs:
        vs.SetVariantSelection("Original")


# 8. ---------------------------- Stage alignment ----------------------------

def _length_to_meters(value: float, unit_hint: str) -> float:
    factor = _UNIT_FACTORS.get((unit_hint or "m").lower(), 1.0)
    return float(value) * factor



def _projected_to_lonlat(
    base_point: BasePointConfig,
    projected_crs: str,
    geodetic_crs: str,
) -> Optional[Tuple[float, float, Optional[float]]]:
    """Convert a base point in a projected CRS to geodetic coordinates."""
    if not _HAVE_PYPROJ:
        print("?? assign_world_geolocation: pyproj unavailable; skipping lon/lat computation")
        return None

    unit = (base_point.unit or "m").lower()
    scale = _UNIT_FACTORS.get(unit, 1.0)
    try:
        e_m = float(base_point.easting) * scale
        n_m = float(base_point.northing) * scale
        h_m = float(base_point.height or 0.0) * scale
    except Exception as exc:
        print(f"?? assign_world_geolocation: invalid base point values: {exc}")
        return None

    try:
        src = CRS.from_user_input(projected_crs)
        dest = CRS.from_user_input(geodetic_crs or "EPSG:4326")
        transformer = Transformer.from_crs(src, dest, always_xy=True)
    except Exception as exc:
        print(f"?? assign_world_geolocation: failed to build transformer for '{projected_crs}' -> '{geodetic_crs}': {exc}")
        return None

    try:
        lon, lat, alt = transformer.transform(e_m, n_m, h_m, errcheck=False)
    except Exception:
        try:
            lon, lat = transformer.transform(e_m, n_m, errcheck=False)
            alt = None
        except Exception as exc:
            print(f"?? assign_world_geolocation: failed to transform coordinates: {exc}")
            return None

    return float(lon), float(lat), (float(alt) if alt is not None else None)


def assign_world_geolocation(
    stage: Usd.Stage,
    base_point: BasePointConfig,
    projected_crs: str,
    geodetic_crs: str = "EPSG:4326",
    unit_hint: Optional[str] = None,
    lonlat_override: Optional[GeodeticCoordinate] = None,
) -> Optional[Tuple[float, float, Optional[float]]]:
    """Author geodetic metadata (Cesium) for the stage based on a base point."""

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        raise RuntimeError("/World prim not found on stage")

    if lonlat_override is not None:
        lon, lat = float(lonlat_override.longitude), float(lonlat_override.latitude)
        alt = float(lonlat_override.height) if lonlat_override.height is not None else None
    else:
        result = _projected_to_lonlat(base_point, projected_crs, geodetic_crs)
        if result is None:
            return None
        lon, lat, alt = result

    unit = unit_hint or base_point.unit or "m"
    alt_fallback = _length_to_meters(base_point.height or 0.0, unit)
    alt_value = float(alt) if alt is not None else float(alt_fallback)

    target_prim = stage.GetPrimAtPath("/World/CesiumGeoreferencing")
    if not target_prim or not target_prim.IsValid():
        target_prim = world_prim

    geo_attrs = {
        "cesium:georeferenceOrigin:latitude": float(lat),
        "cesium:georeferenceOrigin:longitude": float(lon),
        "cesium:georeferenceOrigin:height": alt_value,
    }
    for name, val in geo_attrs.items():
        attr = target_prim.CreateAttribute(name, Sdf.ValueTypeNames.Double)
        attr.Set(float(val))

    return float(lon), float(lat), alt_value


def apply_stage_anchor_transform(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    base_point: Optional[BasePointConfig] = None,
    *,
    align_axes_to_map: bool = False,
    enable_geo_anchor: bool = True,
    projected_crs: Optional[str] = None,
    lonlat: Optional[GeodeticCoordinate] = None,
    map_easting: Optional[float] = None,
    map_northing: Optional[float] = None,
    map_height: Optional[float] = 0.0,
    unit_hint: str = "m",
    coordinate_system: Optional[str] = None,
    longitude: Optional[float] = None,
    latitude: Optional[float] = None,
    altitude: float = 0.0,
) -> None:
    """Anchor the /World prim so the supplied base point sits at the origin."""

    if not enable_geo_anchor:
        return

    map_conv = caches.map_conversion
    meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
    meters_to_stage = 1.0 / meters_per_unit if meters_per_unit else 1.0

    unit = unit_hint
    anchor_e = map_easting
    anchor_n = map_northing
    anchor_h = map_height if map_height is not None else 0.0

    if base_point is not None:
        unit = base_point.unit or unit_hint
        anchor_e = base_point.easting
        anchor_n = base_point.northing
        anchor_h = base_point.height

    geo_override = lonlat
    if geo_override is None and longitude is not None and latitude is not None:
        geo_override = GeodeticCoordinate(longitude=float(longitude), latitude=float(latitude), height=float(altitude))

    crs_for_geo = projected_crs or coordinate_system

    if (anchor_e is None or anchor_n is None) and geo_override is not None:
        if not crs_for_geo:
            raise ValueError("projected_crs or coordinate_system is required when supplying longitude/latitude")
        if not _HAVE_PYPROJ:
            raise RuntimeError("pyproj is required to convert geographic coordinates; install pyproj or provide map easting/northing")
        transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs_for_geo), always_xy=True)
        try:
            result = transformer.transform(
                float(geo_override.longitude),
                float(geo_override.latitude),
                float(geo_override.height or 0.0),
                errcheck=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to transform lon/lat using {crs_for_geo}: {exc}") from exc
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            anchor_e = float(result[0])
            anchor_n = float(result[1])
            if len(result) >= 3:
                anchor_h = float(result[2])
        unit = "m"

    if anchor_e is None or anchor_n is None:
        print("?? apply_stage_anchor_transform: missing anchoring coordinates; skipping geo alignment")
        return

    use_unit = unit or "m"
    e_m = _length_to_meters(anchor_e, use_unit)
    n_m = _length_to_meters(anchor_n, use_unit)
    h_m = _length_to_meters(anchor_h or 0.0, use_unit)

    if map_conv:
        x_local, y_local = map_conv.map_to_local_xy(e_m, n_m)
    else:
        x_local, y_local = e_m, n_m
        if align_axes_to_map:
            print("?? align_axes_to_map=True but no map conversion available; skipping axis alignment")

    translation = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(-x_local * meters_to_stage, -y_local * meters_to_stage, -h_m * meters_to_stage)
    )

    matrix = translation
    if align_axes_to_map and map_conv:
        rotation_deg = map_conv.rotation_degrees()
        rotation = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_deg))
        matrix = rotation * translation

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        raise RuntimeError("/World prim not found on stage")

    world_xform = UsdGeom.Xform(world_prim)
    world_xform.ClearXformOpOrder()
    world_xform.AddTransformOp().Set(matrix)


# 9. ---------------------------- Federated master view ----------------------------

def _sanitize_identifier(raw_name: Optional[str], fallback: Optional[str] = None) -> str:
    """Local-lightweight sanitizer for prim names (dup of earlier helper)."""
    import re as _re
    base = str(raw_name or fallback or "Unnamed")
    name = _re.sub(r"[^A-Za-z0-9_]", "_", base)
    name = _re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    return name[:63]


def update_federated_view(
    master_stage_path: Path,
    payload_stage_path: Path,
    payload_prim_name: Optional[str] = None,
    parent_prim_path: str = "/World/Federated",
) -> Sdf.Layer:
    """Create or update a federated master stage and add an unloaded payload.

    - Creates the master stage if it does not exist, with /World as default prim.
    - Adds/ensures an Xform at /World/Federated.
    - Adds a child Xform for the payload (named by `payload_prim_name` or file stem),
      authors a payload arc to the payload stage's /World, and sets the child inactive
      on first creation (so it's unloaded by default). Existing prims are left as-is.
    - Avoids duplicate payload entries.
    """

    # Open payload stage to read its units
    payload_stage = Usd.Stage.Open(Path(payload_stage_path).resolve().as_posix())
    payload_mpu = float(payload_stage.GetMetadata("metersPerUnit") or 1.0)

    # Open or create master stage
    master_stage_path = Path(master_stage_path).resolve()
    if master_stage_path.exists():
        stage = Usd.Stage.Open(master_stage_path.as_posix())
    else:
        stage = Usd.Stage.CreateNew(master_stage_path.as_posix())
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        # Set federated master to meters; do not scale payloads
        stage.SetMetadata("metersPerUnit", 1.0)

    # Ensure /World exists and is default prim
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    if stage.GetDefaultPrim() != world_prim:
        stage.SetDefaultPrim(world_prim)

    # Validate master stage metersPerUnit matches payload's
    master_mpu = float(stage.GetMetadata("metersPerUnit") or 1.0)
    if abs(master_mpu - payload_mpu) > 1e-9:
        print(f"!! federated_view: units mismatch (master={master_mpu} m, payload={payload_mpu}); payload will not be auto-scaled: {payload_stage_path}")
    else:
        print(f"federated_view: units aligned at {master_mpu} m for {payload_stage_path}")

    # Ensure parent exists (e.g., "/World" or "/World/Federated")
    fed_root_path = Sdf.Path(parent_prim_path)
    fed_root = UsdGeom.Xform.Define(stage, fed_root_path)
    fed_root.ClearXformOpOrder()

    # Choose child prim name
    child_name = _sanitize_identifier(payload_prim_name or Path(payload_stage_path).stem)
    child_path = fed_root_path.AppendChild(child_name)

    # Define or get child prim
    created = False
    if not stage.GetPrimAtPath(child_path):
        child_x = UsdGeom.Xform.Define(stage, child_path)
        child_prim = child_x.GetPrim()
        created = True
    else:
        child_prim = stage.GetPrimAtPath(child_path)

    # Author payload (avoid duplicates)
    payloads = child_prim.GetPayloads()
    existing = []
    try:
        existing = payloads.GetAddedOrExplicitItems()
    except Exception:
        existing = []
    asset = payload_stage_path.as_posix()
    # Use the payload stage's default prim if available to avoid grafting an extra '/World'
    default_prim = payload_stage.GetDefaultPrim()
    target_prim = (default_prim.GetPath() if default_prim else Sdf.Path("/World"))
    already = any((p.assetPath == asset and p.primPath == target_prim) for p in (existing or []))
    if not already:
        payloads.AddPayload(asset, target_prim)

    # Set inactive on first creation so payload is unloaded by default
    if created:
        child_prim.SetActive(False)

    stage.GetRootLayer().Save()
    return stage.GetRootLayer()

