import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from src.process_ifc import PrototypeCaches, PrototypeKey
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
    map_easting: Optional[float] = None,
    map_northing: Optional[float] = None,
    map_height: Optional[float] = 0.0,
    unit_hint: str = "m",
    align_axes_to_map: bool = False,
    enable_geo_anchor: bool = True,
    longitude: Optional[float] = None,
    latitude: Optional[float] = None,
    altitude: float = 0.0,
    coordinate_system: Optional[str] = None,
) -> None:
    """Anchor the /World prim so a supplied geospatial point sits at the origin."""
    if not enable_geo_anchor:
        return

    map_conv = caches.map_conversion
    meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
    meters_to_stage = 1.0 / meters_per_unit if meters_per_unit else 1.0

    anchor_e = map_easting
    anchor_n = map_northing
    anchor_h = map_height if map_height is not None else 0.0

    if (anchor_e is None or anchor_n is None) and longitude is not None and latitude is not None:
        if coordinate_system is None:
            raise ValueError("coordinate_system is required when supplying longitude/latitude")
        if not _HAVE_PYPROJ:
            raise RuntimeError("pyproj is required to convert geographic coordinates; install pyproj or provide map_easting/map_northing")
        try:
            dest = CRS.from_user_input(coordinate_system)
        except Exception as exc:
            raise ValueError(f"Unable to interpret coordinate_system '{coordinate_system}': {exc}") from exc
        if dest.is_geographic and not dest.is_projected:
            raise ValueError(f"Coordinate system '{dest.name}' is geographic; provide a projected system for east/north conversion")
        transformer = Transformer.from_crs(CRS.from_epsg(4326), dest, always_xy=True)
        try:
            result = transformer.transform(float(longitude), float(latitude), float(altitude), errcheck=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to transform lon/lat using {dest}: {exc}") from exc
        if isinstance(result, (list, tuple)):
            if len(result) >= 2:
                anchor_e = float(result[0])
                anchor_n = float(result[1])
            if len(result) >= 3:
                anchor_h = float(result[2])
        else:
            raise RuntimeError("Unexpected transform result when converting geographic coordinates")
        unit_hint = "m"

    if anchor_e is None or anchor_n is None:
        print("?? apply_stage_anchor_transform: missing anchoring coordinates; skipping geo alignment")
        return

    e_m = _length_to_meters(anchor_e, unit_hint)
    n_m = _length_to_meters(anchor_n, unit_hint)
    h_m = _length_to_meters(anchor_h or 0.0, unit_hint)

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
