# 0 . ---------------------------- Imports ----------------------------
from collections import Counter
import multiprocessing
import numpy as np
import ifcopenshell
from pxr import Gf

import math
import re
from typing import Dict, Optional, Union, Literal, List, Tuple, Any
from dataclasses import dataclass, field
import hashlib


# ifcopenshell util for robust matrices (mapped/type geometry)
try:
    from ifcopenshell.util import shape as ifc_shape_util
    _HAVE_IFC_UTIL_SHAPE = True
except Exception:
    _HAVE_IFC_UTIL_SHAPE = False

def _is_identity16(mat16, atol=1e-10):
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
        return np.allclose(arr, np.eye(4), atol=atol)
    except Exception:
        return False


# 1. ---------------------------- Object Prototyping ----------------------------

    # 1.1 ---------------------------- Prototype Data Model ----------------------

@dataclass
class MeshProto:
    repmap_id: int
    type_name: Optional[str] = None
    type_class: Optional[str] = None
    type_guid: Optional[str] = None
    repmap_index: Optional[int] = None
    mesh: Optional[dict] = None
    materials: List[Any] = field(default_factory=list)
    material_ids: List[int] = field(default_factory=list)
    count: int = 0

@dataclass
class HashProto:
    digest: str
    name: Optional[str] = None
    type_name: Optional[str] = None
    type_guid: Optional[str] = None
    mesh: Optional[dict] = None
    materials: List[Any] = field(default_factory=list)
    material_ids: List[int] = field(default_factory=list)
    signature: Optional[Tuple[Any, ...]] = None
    count: int = 0

@dataclass(frozen=True)
class PrototypeKey:
    kind: Literal["repmap", "hash"]
    identifier: Union[int, str]

@dataclass
class MapConversionData:
    eastings: float = 0.0
    northings: float = 0.0
    orthogonal_height: float = 0.0
    x_axis_abscissa: float = 1.0
    x_axis_ordinate: float = 0.0
    scale: float = 1.0

@dataclass
class InstanceRecord:
    step_id: int
    product_id: Optional[int]
    prototype: PrototypeKey
    name: str
    transform: Optional[Tuple[float, ...]]
    material_ids: List[int] = field(default_factory=list)
    materials: List[Any] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PrototypeCaches:
    repmaps: Dict[int, MeshProto]
    repmap_counts: Counter
    hashes: Dict[str, HashProto]
    step_keys: Dict[int, PrototypeKey]
    instances: Dict[int, "InstanceRecord"]
    map_conversion: Optional[MapConversionData] = None


    def normalized_axes(self) -> Tuple[float, float]:
        ax = float(self.x_axis_abscissa) or 0.0
        ay = float(self.x_axis_ordinate) or 0.0
        length = math.hypot(ax, ay)
        if length <= 1e-12:
            return 1.0, 0.0
        return ax / length, ay / length

    def rotation_degrees(self) -> float:
        ax, ay = self.normalized_axes()
        return math.degrees(math.atan2(ay, ax))

    def map_to_local_xy(self, easting: float, northing: float) -> Tuple[float, float]:
        ax, ay = self.normalized_axes()
        scale = self.scale or 1.0
        d_e = float(easting) - float(self.eastings)
        d_n = float(northing) - float(self.northings)
        x = scale * (ax * d_e + ay * d_n)
        y = scale * (-ay * d_e + ax * d_n)
        return x, y



def _select_map_conversion_entity(ifc_file) -> Optional[Any]:
    best = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not op.is_a("IfcMapConversion"):
                continue
            if getattr(ctx, "ContextType", None) == "Model" and getattr(ctx, "CoordinateSpaceDimension", None) == 3:
                return op
            if best is None:
                best = op
    return best


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    entity = _select_map_conversion_entity(ifc_file)
    if entity is None:
        return None
    return MapConversionData(
        eastings=_as_float(getattr(entity, "Eastings", 0.0), 0.0),
        northings=_as_float(getattr(entity, "Northings", 0.0), 0.0),
        orthogonal_height=_as_float(getattr(entity, "OrthogonalHeight", 0.0), 0.0),
        x_axis_abscissa=_as_float(getattr(entity, "XAxisAbscissa", 1.0), 1.0),
        x_axis_ordinate=_as_float(getattr(entity, "XAxisOrdinate", 0.0), 0.0),
        scale=_as_float(getattr(entity, "Scale", 1.0), 1.0) or 1.0,
    )



def sanitize_name(raw_name: Optional[str], fallback: Optional[str] = None) -> str:
    base = str(raw_name or fallback or "Unnamed")
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "Unnamed"
    if name[0].isdigit():
        name = "_" + name
    return name[:63]

# 2. ---------------------------- IFC Geometry Utilities ----------------------------

    # 2.1 ---------- helpers ----------
def safe_set(s, key, val):
    try: s.set(key, val); return True
    except: 
        try: s.set(key.replace("-", "_").upper(), val); return True
        except: return False

def triangulated_to_dict(geom):
    """
    Convert IfcOpenShell Triangulation or TriangulationElement to
    a dict with vertices and faces arrays.
    """
    # unwrap .geometry if present
    g = getattr(geom, "geometry", geom)

    # vertices
    if hasattr(g, "verts"):
        v = np.array(g.verts, dtype=float).reshape(-1, 3)
    elif hasattr(g, "coordinates"):
        v = np.array(g.coordinates, dtype=float).reshape(-1, 3)
    else:
        raise AttributeError(f"No verts/coordinates found on {type(g)}")

    # faces
    if hasattr(g, "faces"):
        f = np.array(g.faces, dtype=int).reshape(-1, 3)
    elif hasattr(g, "triangles"):
        f = np.array(g.triangles, dtype=int).reshape(-1, 3)
    else:
        raise AttributeError(f"No faces/triangles found on {type(g)}")

    return {"vertices": v, "faces": f}


def mesh_hash(mesh, precision=6):
    """
    Stable-ish hash ignoring vertex/face order differences.
    """
    v = np.round(mesh["vertices"], precision)
    uniq, inv = np.unique(v, axis=0, return_inverse=True)
    f = inv[mesh["faces"]]

    # sort face indices and then sort faces to ignore order differences
    f = np.sort(np.sort(f, axis=1), axis=0)
    h = hashlib.sha256(); h.update(uniq.tobytes()); h.update(f.tobytes())
    return h.hexdigest()

def get_type_name(prod):
    for rel in getattr(prod, "IsTypedBy", []) or []:
        t = rel.RelatingType
        nm = getattr(t, "Name", None) or getattr(t, "ElementType", None)
        if nm: return nm
    return getattr(prod, "Name", None) or prod.is_a()


def _ifc_value_to_python(value):
    if value is None:
        return None
    if hasattr(value, 'wrappedValue'):
        return _ifc_value_to_python(value.wrappedValue)
    if hasattr(value, 'ValueComponent'):
        return _ifc_value_to_python(value.ValueComponent)
    if isinstance(value, (list, tuple)):
        return [_ifc_value_to_python(v) for v in value]
    if isinstance(value, (int, float, bool, str)):
        return value
    try:
        return str(value)
    except Exception:
        return None


def _extract_property_value(prop):
    if prop is None:
        return None
    try:
        if prop.is_a('IfcPropertySingleValue'):
            return _ifc_value_to_python(getattr(prop, 'NominalValue', None))
        if prop.is_a('IfcPropertyEnumeratedValue'):
            vals = getattr(prop, 'EnumerationValues', None) or []
            return [_ifc_value_to_python(v) for v in vals]
        if prop.is_a('IfcPropertyListValue'):
            vals = getattr(prop, 'ListValues', None) or []
            return [_ifc_value_to_python(v) for v in vals]
        if prop.is_a('IfcPropertyReferenceValue'):
            ref = getattr(prop, 'PropertyReference', None)
            if ref is not None:
                return getattr(ref, 'Name', None) or getattr(ref, 'Description', None) or str(ref)
        if prop.is_a('IfcComplexProperty'):
            nested = {}
            for sub in getattr(prop, 'HasProperties', None) or []:
                val = _extract_property_value(sub)
                nested[sub.Name] = val
            return nested
    except Exception:
        pass
    for attr in ('NominalValue', 'LengthValue', 'AreaValue', 'VolumeValue', 'WeightValue', 'TimeValue', 'IntegerValue', 'RealValue', 'BooleanValue', 'LogicalValue', 'TextValue'):
        if hasattr(prop, attr):
            return _ifc_value_to_python(getattr(prop, attr))
    return None


def _extract_quantity_value(quantity):
    if quantity is None:
        return None
    for attr in ('LengthValue', 'AreaValue', 'VolumeValue', 'CountValue', 'WeightValue', 'TimeValue', 'Value', 'NominalValue'):
        if hasattr(quantity, attr):
            return _ifc_value_to_python(getattr(quantity, attr))
    for attr in ('Formula', 'Description'):
        if hasattr(quantity, attr) and getattr(quantity, attr):
            return getattr(quantity, attr)
    return None


def collect_instance_attributes(product) -> Dict[str, Dict[str, Dict[str, Any]]]:
    attrs: Dict[str, Dict[str, Dict[str, Any]]] = {'psets': {}, 'qtos': {}}

    def merge(container: Dict[str, Dict[str, Any]], name: Optional[str], data: Dict[str, Any]):
        if not data:
            return
        key = name or 'Unnamed'
        entry = container.setdefault(key, {})
        entry.update({k: v for k, v in data.items() if v is not None})

    for rel in getattr(product, 'IsDefinedBy', []) or []:
        definition = getattr(rel, 'RelatingPropertyDefinition', None)
        if definition is None:
            continue
        if definition.is_a('IfcPropertySet'):
            props = {}
            for prop in getattr(definition, 'HasProperties', None) or []:
                props[prop.Name] = _extract_property_value(prop)
            merge(attrs['psets'], getattr(definition, 'Name', None), props)
        elif definition.is_a('IfcElementQuantity'):
            quants = {}
            for qty in getattr(definition, 'Quantities', None) or []:
                quants[qty.Name] = _extract_quantity_value(qty)
            merge(attrs['qtos'], getattr(definition, 'Name', None), quants)

    for rel in getattr(product, 'IsTypedBy', []) or []:
        type_obj = getattr(rel, 'RelatingType', None)
        if type_obj is None:
            continue
        for pset in getattr(type_obj, 'HasPropertySets', None) or []:
            if pset.is_a('IfcPropertySet'):
                props = {}
                for prop in getattr(pset, 'HasProperties', None) or []:
                    props[prop.Name] = _extract_property_value(prop)
                merge(attrs['psets'], getattr(pset, 'Name', None), props)
        for qset in getattr(type_obj, 'HasQuantities', None) or []:
            if qset.is_a('IfcElementQuantity'):
                quants = {}
                for qty in getattr(qset, 'Quantities', None) or []:
                    quants[qty.Name] = _extract_quantity_value(qty)
                merge(attrs['qtos'], getattr(qset, 'Name', None), quants)

    return attrs

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Resolve an absolute transform for the iterator step as a 16-tuple."""
    mat = None
    tr = getattr(shape, "transformation", None)
    if tr is not None and hasattr(tr, "matrix"):
        mat = tuple(tr.matrix)
        if _is_identity16(mat):
            mat = None

    if mat is None and _HAVE_IFC_UTIL_SHAPE:
        try:
            gm = ifc_shape_util.get_shape_matrix(shape)
            gm = np.array(gm, dtype=float).reshape(4, 4)
            mat = tuple(gm.flatten().tolist())
            if _is_identity16(mat):
                mat = None
        except Exception:
            mat = None

    if mat is None and element is not None:
        try:
            place = getattr(element, "ObjectPlacement", None)
            gf = compose_object_placement(place, length_to_m=1.0)
            mat = gf_to_tuple16(gf)
        except Exception:
            mat = None

    return mat

# Fallback when rep-map direct tessellation fails:
def derive_mesh_from_occurrence(ifc, repmap, settings_local):
    """Fallback mesh derivation when a representation map cannot be tessellated directly."""

    def _gf_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
        return np.array(gf_to_tuple16(matrix), dtype=float).reshape(4, 4)

    def _axis_placement_to_np(axis_placement) -> np.ndarray:
        if axis_placement is None:
            return np.eye(4)
        return _gf_to_np(axis2placement_to_matrix(axis_placement, length_to_m=1.0))

    def _local_placement_to_np(local_placement) -> np.ndarray:
        if local_placement is None:
            return np.eye(4)
        return _gf_to_np(compose_object_placement(local_placement, length_to_m=1.0))

    def _cartesian_transform_to_np(op) -> np.ndarray:
        if op is None:
            return np.eye(4)

        def _vec_from(direction, fallback):
            data = direction if direction is not None else fallback
            return np.array([_as_float(c) for c in data], dtype=float)

        origin = np.array([
            _as_float(c) for c in (getattr(getattr(op, "LocalOrigin", None), "Coordinates", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
        ], dtype=float)

        x_axis = _vec_from(getattr(getattr(op, "Axis1", None), "DirectionRatios", None), (1.0, 0.0, 0.0))
        y_axis = _vec_from(getattr(getattr(op, "Axis2", None), "DirectionRatios", None), (0.0, 1.0, 0.0))
        z_axis = _vec_from(getattr(getattr(op, "Axis3", None), "DirectionRatios", None), (0.0, 0.0, 1.0))

        def _norm(vec: np.ndarray) -> np.ndarray:
            length = np.linalg.norm(vec)
            return vec if length == 0.0 else vec / length

        x_axis = _norm(x_axis)
        y_axis = _norm(y_axis)
        z_axis = _norm(z_axis if np.linalg.norm(z_axis) > 0 else np.cross(x_axis, y_axis))
        if np.linalg.norm(z_axis) == 0.0:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(x_axis, y_axis)) > 0.9999:
            y_axis = _norm(np.cross(z_axis, x_axis))
        x_axis = _norm(np.cross(y_axis, z_axis))

        scale = _as_float(getattr(op, "Scale", 1.0) or 1.0, 1.0)
        sx = scale
        sy = _as_float(getattr(op, "Scale2", scale) or scale, scale)
        sz = _as_float(getattr(op, "Scale3", scale) or scale, scale)

        transform = np.eye(4, dtype=float)
        transform[:3, 0] = x_axis * sx
        transform[:3, 1] = y_axis * sy
        transform[:3, 2] = z_axis * sz
        transform[:3, 3] = origin
        return transform

    for prod in ifc.by_type("IfcProduct"):
        repC = getattr(prod, "Representation", None)
        if not repC:
            continue
        for rep in repC.Representations or []:
            for it in (rep.Items or []):
                if not it.is_a("IfcMappedItem") or it.MappingSource != repmap:
                    continue

                sh = ifcopenshell.geom.create_shape(settings_local, rep)
                geom = sh.geometry
                v = np.array(geom.verts, float).reshape(-1, 3)
                f = np.array(geom.faces, int).reshape(-1, 3)

                placement_np = _local_placement_to_np(getattr(prod, "ObjectPlacement", None))
                target_np = _cartesian_transform_to_np(getattr(it, "MappingTarget", None))
                origin_np = _axis_placement_to_np(getattr(repmap, "MappingOrigin", None))

                M = placement_np @ target_np @ origin_np
                Minv = np.linalg.inv(M)
                v_loc = (np.c_[v, np.ones((len(v), 1))] @ Minv.T)[:, :3]
                return {"vertices": v_loc, "faces": f}
    return None


def build_prototypes(ifc_file) -> PrototypeCaches:
    """Discover prototypes and instances in a single iterator pass."""
    s_local = ifcopenshell.geom.settings()
    safe_set(s_local, "use-world-coords", False)
    safe_set(s_local, "weld-vertices", True)
    safe_set(s_local, "disable-opening-subtractions", True)
    safe_set(s_local, "apply-default-materials", False)

    repmaps: Dict[int, MeshProto] = {}
    repmap_counts: Counter = Counter()
    hashes: Dict[str, HashProto] = {}
    step_keys: Dict[int, PrototypeKey] = {}
    instances: Dict[int, InstanceRecord] = {}

    repmap_to_type: Dict[int, Tuple[Optional[Any], Optional[int]]] = {}
    for t in ifc_file.by_type("IfcTypeProduct"):
        for idx, rm in enumerate(getattr(t, "RepresentationMaps", []) or []):
            repmap_to_type[rm.id()] = (t, idx)

    signature_to_digest: Dict[Tuple[Any, ...], str] = {}
    type_to_digest: Dict[Tuple[str, str], str] = {}

    def ensure_repmap_entry(rmid: int) -> MeshProto:
        info = repmaps.get(rmid)
        if info is None:
            type_ref, idx = repmap_to_type.get(rmid, (None, None))
            info = MeshProto(
                repmap_id=rmid,
                type_name=getattr(type_ref, "Name", None) if type_ref else None,
                type_class=type_ref.is_a() if type_ref else None,
                type_guid=getattr(type_ref, "GlobalId", None) if type_ref else None,
                repmap_index=idx,
            )
            rm_entity = ifc_file.by_id(rmid)
            mesh_payload = None
            materials_payload: List[Any] = []
            material_ids_payload: List[int] = []
            if rm_entity is not None:
                repmap_rep = getattr(rm_entity, "MappedRepresentation", None)
                if repmap_rep is not None:
                    try:
                        sh = ifcopenshell.geom.create_shape(s_local, repmap_rep)
                        mesh_payload = triangulated_to_dict(sh)
                        geom_payload = getattr(sh, "geometry", None)
                        if geom_payload is not None:
                            materials_payload = list(getattr(geom_payload, "materials", []) or [])
                            material_ids_payload = list(getattr(geom_payload, "material_ids", []) or [])
                    except Exception:
                        mesh_payload = derive_mesh_from_occurrence(ifc_file, rm_entity, s_local)
            info.mesh = mesh_payload
            if materials_payload:
                info.materials = materials_payload
            if material_ids_payload:
                info.material_ids = material_ids_payload
            repmaps[rmid] = info
        return info

    it = ifcopenshell.geom.iterator(s_local, ifc_file, multiprocessing.cpu_count())
    if not it.initialize():
        raise RuntimeError("iterator init failed")

    while it.next():
        shape = it.get()
        if shape is None:
            continue

        product = ifc_file.by_id(shape.id)
        if product is None:
            continue

        geom = getattr(shape, "geometry", None)
        if geom is None:
            continue

        verts = list(getattr(geom, "verts", []) or [])
        faces = list(getattr(geom, "faces", []) or [])
        if not verts or not faces:
            continue

        materials = list(getattr(geom, "materials", []) or [])
        material_ids = list(getattr(geom, "material_ids", []) or [])
        abs_mat = resolve_absolute_matrix(shape, product)

        rep = getattr(product, "Representation", None)
        reps = [r for r in (rep.Representations or []) if r.is_a("IfcShapeRepresentation")] if rep else []
        body = next((r for r in reps if (r.RepresentationIdentifier or "").lower() == "body"), reps[0] if reps else None)
        mapped_items = [mi for mi in (body.Items or []) if mi.is_a("IfcMappedItem")] if body else []

        type_ref = None
        type_guid = None
        type_name = None
        for rel in (getattr(product, "IsTypedBy", []) or []):
            rel_type = getattr(rel, "RelatingType", None)
            if rel_type is not None:
                type_ref = rel_type
                type_guid = getattr(rel_type, "GlobalId", None)
                type_name = getattr(rel_type, "Name", None)
                break

        primary_key: Optional[PrototypeKey] = None

        if mapped_items:
            for mi in mapped_items:
                rmid = mi.MappingSource.id()
                repmap_counts[rmid] += 1
                info = ensure_repmap_entry(rmid)
                info.count += 1
                if not info.materials and materials:
                    info.materials = list(materials)
                if not info.material_ids and material_ids:
                    info.material_ids = list(material_ids)
            primary_key = PrototypeKey(kind="repmap", identifier=mapped_items[0].MappingSource.id())
        else:
            repmap_from_type = None
            if type_ref is not None:
                for rm in getattr(type_ref, "RepresentationMaps", None) or []:
                    repmap_from_type = rm
                    break
            if repmap_from_type is not None:
                rmid = repmap_from_type.id()
                repmap_counts[rmid] += 1
                info = ensure_repmap_entry(rmid)
                info.count += 1
                if not info.materials and materials:
                    info.materials = list(materials)
                if not info.material_ids and material_ids:
                    info.material_ids = list(material_ids)
                primary_key = PrototypeKey(kind="repmap", identifier=rmid)
            else:
                mesh = triangulated_to_dict(geom)
                signature = compute_mesh_signature(mesh)
                type_key = None
                if type_guid or type_name:
                    type_key = (type_guid or "", type_name or "")
                canonical_digest = None
                if type_key and type_key in type_to_digest:
                    canonical_digest = type_to_digest[type_key]
                else:
                    canonical_digest = signature_to_digest.get(signature)
                if canonical_digest is None:
                    digest = mesh_hash(mesh, precision=6)
                    signature_to_digest[signature] = digest
                    if type_key:
                        type_to_digest[type_key] = digest
                    bucket = HashProto(
                        digest=digest,
                        name=get_type_name(product),
                        type_name=type_name,
                        type_guid=type_guid,
                        mesh=mesh,
                        materials=list(materials),
                        material_ids=list(material_ids),
                        signature=signature,
                        count=1,
                    )
                    hashes[digest] = bucket
                    canonical_digest = digest
                else:
                    bucket = hashes[canonical_digest]
                    bucket.count += 1
                    if bucket.mesh is None:
                        bucket.mesh = mesh
                    if not bucket.materials and materials:
                        bucket.materials = list(materials)
                    if not bucket.material_ids and material_ids:
                        bucket.material_ids = list(material_ids)
                primary_key = PrototypeKey(kind="hash", identifier=canonical_digest)

        if primary_key is None:
            continue

        step_id = shape.id
        step_keys[step_id] = primary_key

        fallback = getattr(product, "GlobalId", None)
        if not fallback:
            try:
                fallback = str(product.id())
            except Exception:
                fallback = str(step_id)
        name = sanitize_name(getattr(product, "Name", None), fallback=fallback)
        try:
            product_id = product.id()
        except Exception:
            product_id = None

        instances[step_id] = InstanceRecord(
            step_id=step_id,
            product_id=product_id,
            prototype=primary_key,
            name=name,
            transform=abs_mat,
            material_ids=list(material_ids),
            materials=list(materials),
            attributes=collect_instance_attributes(product),
        )

    map_conversion = extract_map_conversion(ifc_file)

    return PrototypeCaches(
        repmaps=repmaps,
        repmap_counts=repmap_counts,
        hashes=hashes,
        step_keys=step_keys,
        instances=instances,
        map_conversion=map_conversion,
    )






def _as_float(v, default=0.0):
    """
    Attempt to convert the input value into a float. If the value has a "wrappedValue" attribute, use that instead.
    Function converts placement coordinates and direction ratios before building USD Gf.Vec3d vectors; this lets the code accept plain floats, 
     IFC measure wrappers, or even strings without crashing.
    If the conversion fails, return the default value.
    :param v: The value to convert into a float.
    :param default: The default value to return if the conversion fails.
    :return: The converted float value, or the default value if the conversion fails.
    """
    try:
        if hasattr(v, "wrappedValue"):
            return float(v.wrappedValue)
        return float(v)
    except Exception:
        return float(default)
     
def axis2placement_to_matrix(place, length_to_m=1.0):
    """Convert an IfcAxis2Placement2D/3D into a USD 4×4 transform.

    Normalises the placement’s axes, reconstructs an orthonormal frame even when
    IFC data is missing or degenerate, and scales coordinates by `length_to_m`.
    Returns an identity matrix if `place` is None.
    Args:
        place (IfcAxis2Placement2D/3D): The placement to convert.
        length_to_m (float): The conversion factor from IFC length units to meters.
    Returns:
        Gf.Matrix4d: The 4x4 matrix representing the placement in world coordinates.
    """

    if place is None:
        return Gf.Matrix4d(1)
    coords = getattr(getattr(place, "Location", None), "Coordinates", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    loc = Gf.Vec3d(*((_as_float(c) * length_to_m) for c in (coords + (0.0, 0.0, 0.0))[:3]))

    z_axis = getattr(getattr(place, "Axis", None), "DirectionRatios", (0.0, 0.0, 1.0)) or (0.0, 0.0, 1.0)
    x_axis = getattr(getattr(place, "RefDirection", None), "DirectionRatios", (1.0, 0.0, 0.0)) or (1.0, 0.0, 0.0)

    z = Gf.Vec3d(*map(_as_float, z_axis)); z = z if z.GetLength() > 1e-12 else Gf.Vec3d(0.0,0.0,1.0)
    z = z.GetNormalized()
    x = Gf.Vec3d(*map(_as_float, x_axis))
    if abs(Gf.Dot(z, x)) > 0.9999 or x.GetLength() < 1e-12:
        x = Gf.Vec3d(1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else Gf.Vec3d(0.0, 1.0, 0.0)
    else:
        x = x.GetNormalized()
    y = Gf.Cross(z, x).GetNormalized() # orthogonalize/derive y via cross(z, x)
    x = Gf.Cross(y, z).GetNormalized() # re-orthogonalize x  via cross(y, z) to correct accumulated drift.

    rot = Gf.Matrix4d(
        x[0], x[1], x[2], 0.0,
        y[0], y[1], y[2], 0.0,
        z[0], z[1], z[2], 0.0,
        0.0,  0.0,  0.0,  1.0
    )
    trans = Gf.Matrix4d().SetTranslate(loc)
    return trans * rot

def compose_object_placement(obj_placement, length_to_m=1.0):
    """
    Compose a 4x4 matrix from an IfcObjectPlacement by recursively combining
    relative placements and their parents.

    Args:
        obj_placement (IfcObjectPlacement): The object placement to compose.
        length_to_m (float): The conversion factor from IFC length units to meters.

    Returns:
        Gf.Matrix4d: The composed 4x4 matrix representing the object placement in world coordinates.
    """
    if obj_placement is None:
        return Gf.Matrix4d(1)
    local = axis2placement_to_matrix(getattr(obj_placement, "RelativePlacement", None), length_to_m)
    parent = compose_object_placement(getattr(obj_placement, "PlacementRelTo", None), length_to_m)
    return parent * local

def gf_to_tuple16(gf: Gf.Matrix4d):
    """Row-major 16-tuple from a Gf.Matrix4d."""
    return tuple(gf[i, j] for i in range(4) for j in range(4))


def generate_mesh_data(ifc_file, settings):
    """
    Yield (element, faces, verts, abs_mat, materials, material_ids).
    Robust absolute matrix resolution order:
      1) shape.transformation.matrix (if present & not identity)
      2) ifcopenshell.util.shape.get_shape_matrix(shape)  (if available)
      3) compose_object_placement(element.ObjectPlacement)
    """
    it = ifcopenshell.geom.iterator(settings, ifc_file, multiprocessing.cpu_count())
    if it.initialize():
        while True:
            element = None
            try:
                shape = it.get()
                element = ifc_file.by_id(shape.id)

                geom = shape.geometry
                faces = list(getattr(geom, "faces", []) or [])
                verts = list(getattr(geom, "verts", []) or [])
                materials    = list(getattr(geom, "materials", []) or [])
                material_ids = list(getattr(geom, "material_ids", []) or [])

                mat = resolve_absolute_matrix(shape, element)

                yield element, faces, verts, mat, materials, material_ids

            except Exception as e:
                if element is not None:
                    print(f"⚠️ Skipping element {getattr(element,'id',lambda: '?')()} due to error: {e}")
                else:
                    print(f"⚠️ Iterator error: {e}")
            if not it.next():
                break





