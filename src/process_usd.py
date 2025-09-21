import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from src.process_ifc import PrototypeCaches, PrototypeKey
from src.utils.matrix_utils import np_to_gf_matrix, scale_matrix_translation_only


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


# 2. ---------------------------- Mesh helpers ----------------------------

def create_usd_stage(usd_path, meters_per_unit=1.0):
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

def _material_color(material: Any) -> Tuple[float, float, float]:
    candidates = [
        "Diffuse", "diffuse", "DiffuseColor", "diffuseColor",
        "SurfaceColour", "surfaceColour", "surface_colour", "Color"
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
                return (float(value[0]), float(value[1]), float(value[2]))
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
) -> Tuple[Sdf.Layer, Dict[PrototypeKey, Sdf.Path]]:
    proto_file = (output_dir / f"{base_name}_prototypes.usda").resolve()
    proto_layer = Sdf.Layer.CreateNew(proto_file.as_posix())

    root_layer = stage.GetRootLayer()
    if proto_layer.identifier not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(proto_layer.identifier)

    proto_root = Sdf.Path("/__Prototypes")
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
) -> Tuple[Sdf.Layer, Dict[PrototypeKey, List[Sdf.Path]]]:
    material_file = (output_dir / f"{base_name}_materials.usda").resolve()
    material_layer = Sdf.Layer.CreateNew(material_file.as_posix())

    root_layer = stage.GetRootLayer()
    if material_layer.identifier not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(material_layer.identifier)
    if material_layer.identifier not in proto_layer.subLayerPaths:
        proto_layer.subLayerPaths.append(material_layer.identifier)

    material_root = Sdf.Path("/__Materials")
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
                color = _material_color(material)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(_material_opacity(material))
                material_prim.CreateSurfaceOutput().ConnectToSource(shader, "surface")

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
    if not entries:
        return
    for raw_name, value in entries.items():
        if value is None:
            continue
        token = _sanitize_identifier(raw_name, fallback="Unnamed")
        attr_name = f"ifc:{namespace}:{token}"
        if isinstance(value, dict):
            payload = {k: v for k, v in value.items() if v is not None}
            if not payload:
                continue
        else:
            payload = {"value": value}
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Dictionary)
        attr.Set(payload)


def _author_instance_attributes(prim: Usd.Prim, attributes: Optional[Dict[str, Any]]) -> None:
    if not attributes:
        return
    if not isinstance(attributes, dict):
        return
    psets = attributes.get("psets", {})
    qtos = attributes.get("qtos", {})
    _author_namespaced_dictionary_attributes(prim, "pset", psets)
    _author_namespaced_dictionary_attributes(prim, "qto", qtos)

# 7. ---------------------------- Instance authoring ----------------------------

def author_instance_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    output_dir: Path,
    base_name: str,
) -> Sdf.Layer:
    inst_file = (output_dir / f"{base_name}_instances.usda").resolve()
    inst_layer = Sdf.Layer.CreateNew(inst_file.as_posix())

    root_layer = stage.GetRootLayer()
    if inst_layer.identifier not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(inst_layer.identifier)

    inst_root = Sdf.Path("/World/Instances")
    used_names: Dict[str, int] = {}

    with Usd.EditContext(stage, inst_layer):
        UsdGeom.Xform.Define(stage, inst_root)

        for record in caches.instances.values():
            base_name_candidate = _sanitize_identifier(record.name, fallback=f"Ifc_{record.step_id}")
            inst_name = _unique_name(base_name_candidate, used_names)
            inst_path = inst_root.AppendChild(inst_name)

            proto_path = proto_paths.get(record.prototype)
            if proto_path is None:
                print(f"?? Missing prototype for instance {record.step_id}; skipping")
                continue

            xform = UsdGeom.Xform.Define(stage, inst_path)
            xform.ClearXformOpOrder()
            inst_prim = xform.GetPrim()
            if record.transform:
                try:
                    xf = np_to_gf_matrix(record.transform)
                    xform.AddTransformOp().Set(xf)
                except Exception as exc:
                    print(f"?? Failed to author transform for instance {record.step_id}: {exc}")

            _author_instance_attributes(inst_prim, record.attributes)

            ref_path = inst_path.AppendChild("Prototype")
            ref_prim = stage.DefinePrim(ref_path, "Xform")
            refs = ref_prim.GetReferences()
            refs.ClearReferences()
            refs.AddReference("", proto_path)
            ref_prim.SetInstanceable(True)

    return inst_layer


# 8. ---------------------------- Stage alignment ----------------------------

def _length_to_meters(value: float, unit_hint: str) -> float:
    factor = _UNIT_FACTORS.get((unit_hint or "m").lower(), 1.0)
    return float(value) * factor


def apply_stage_anchor_transform(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    map_easting: float,
    map_northing: float,
    map_height: float,
    unit_hint: str = "m",
    align_axes_to_map: bool = False,
) -> None:
    map_conv = caches.map_conversion
    meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
    meters_to_stage = 1.0 / meters_per_unit if meters_per_unit else 1.0

    e_m = _length_to_meters(map_easting, unit_hint)
    n_m = _length_to_meters(map_northing, unit_hint)
    h_m = _length_to_meters(map_height, unit_hint)

    if map_conv:
        x_local, y_local = map_conv.map_to_local_xy(e_m, n_m)
    else:
        x_local, y_local = e_m, n_m

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
