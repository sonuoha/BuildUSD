# ===============================
# process_ifc.py (FULL REPLACEMENT)
# ===============================
from __future__ import annotations
import logging
from collections import Counter
import multiprocessing
import numpy as np
import ifcopenshell
import os
import math
import re
from typing import Dict, Optional, Union, Literal, List, Tuple, Any, TYPE_CHECKING, Iterable
from dataclasses import dataclass, field
import hashlib

# USD-lite vector/matrix shim for placement helpers
try:
    from .pxr_utils import Gf  # same local helper you already ship
except Exception:  # pragma: no cover
    class _GfShim:
        class Matrix4d(list):
            def __init__(self, v=1):
                if v == 1:
                    super().__init__([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
                else:
                    super().__init__(v)
            def __mul__(self, other):
                a = np.array(self, dtype=float); b = np.array(other, dtype=float)
                return _GfShim.Matrix4d((a @ b).tolist())
        class Vec3d(tuple):
            def __new__(cls, x=0.0, y=0.0, z=0.0):
                return super().__new__(cls, (float(x), float(y), float(z)))
        class Rotation:
            def __init__(self, axis, angle):
                self.axis = axis; self.angle = angle
        @staticmethod
        def Dot(a,b):
            return float(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
        @staticmethod
        def Cross(a,b):
            ax,ay,az=a; bx,by,bz=b
            return _GfShim.Vec3d(ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx)
        Matrix4d.SetTranslate = lambda self, v: _GfShim.Matrix4d([[1,0,0,v[0]],[0,1,0,v[1]],[0,0,1,v[2]],[0,0,0,1]])
        Matrix4d.SetRotate = lambda self, r: _GfShim.Matrix4d(1)
    Gf = _GfShim()

if TYPE_CHECKING:
    from .config.manifest import ConversionManifest

log = logging.getLogger(__name__)

# ---------------------------------
# Thread count helper
# ---------------------------------

def _resolve_threads(env_var="IFC_GEOM_THREADS", minimum=1):
    val = os.getenv(env_var)
    if val and val.strip():
        try: return max(minimum, int(val))
        except ValueError: pass
    try: return max(minimum, multiprocessing.cpu_count())
    except NotImplementedError: return minimum

threads = _resolve_threads()

# ---------------------------------
# Small utilities
# ---------------------------------

def round_tuple_list(xs: Iterable[Iterable[float]], tol: int = 9) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(round(float(v), tol) for v in x) for x in xs)

def stable_mesh_hash(verts: Any, faces: Any) -> str:
    if hasattr(verts, "tolist"): verts_iter = verts.tolist()
    else: verts_iter = list(verts)
    if hasattr(faces, "tolist"): faces_iter = faces.tolist()
    else: faces_iter = list(faces)
    rverts = round_tuple_list(verts_iter, tol=9)
    rfaces = tuple(tuple(int(i) for i in f) for f in faces_iter)
    h = hashlib.sha256(); h.update(str(rverts).encode("utf-8")); h.update(b"|"); h.update(str(rfaces).encode("utf-8"))
    return h.hexdigest()

def sanitize_name(raw_name: Optional[str], fallback: Optional[str] = None) -> str:
    base = str(raw_name or fallback or "Unnamed")
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name: name = "Unnamed"
    if name[0].isdigit(): name = "_" + name
    return name[:63]

# ---------------------------------
# Data model (unchanged signatures)
# ---------------------------------

@dataclass
class MeshProto:
    repmap_id: int  # we use IfcTypeObject id here as the canonical repmap anchor
    type_name: Optional[str] = None
    type_class: Optional[str] = None
    type_guid: Optional[str] = None
    repmap_index: Optional[int] = None
    mesh: Optional[dict] = None
    mesh_hash: Optional[str] = None
    settings_fp: Optional[str] = None
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
    canonical_frame: Optional[Tuple[float, ...]] = None
    settings_fp: Optional[str] = None
    count: int = 0

@dataclass(frozen=True)
class CurveWidthRule:
    width: float
    unit: Optional[str] = None
    layer: Optional[str] = None
    curve: Optional[str] = None
    hierarchy: Optional[str] = None
    step_id: Optional[int] = None

@dataclass
class ConversionOptions:
    enable_instancing: bool = True
    enable_hash_dedup: bool = True
    convert_metadata: bool = True
    manifest: Optional['ConversionManifest'] = None
    curve_width_rules: Tuple[CurveWidthRule, ...] = tuple()

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
    def normalized_axes(self) -> Tuple[float, float]:
        ax = float(self.x_axis_abscissa) or 0.0
        ay = float(self.x_axis_ordinate) or 0.0
        length = math.hypot(ax, ay)
        if length <= 1e-12: return 1.0, 0.0
        return ax/length, ay/length
    def rotation_degrees(self) -> float:
        ax, ay = self.normalized_axes(); return math.degrees(math.atan2(ay, ax))
    def map_to_local_xy(self, easting: float, northing: float) -> Tuple[float, float]:
        ax, ay = self.normalized_axes(); scale = self.scale or 1.0
        d_e = float(easting) - float(self.eastings)
        d_n = float(northing) - float(self.northings)
        x = scale * (ax * d_e + ay * d_n)
        y = scale * (-ay * d_e + ax * d_n)
        return x, y

@dataclass
class InstanceRecord:
    step_id: int
    product_id: Optional[int]
    prototype: Optional[PrototypeKey]
    name: str
    transform: Optional[Tuple[float, ...]]
    material_ids: List[int] = field(default_factory=list)
    materials: List[Any] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    prototype_delta: Optional[Tuple[float, ...]] = None
    hierarchy: Tuple[Tuple[str, Optional[int]], ...] = field(default_factory=tuple)
    mesh: Optional[Dict[str, Any]] = None
    guid: Optional[str] = None

@dataclass
class AnnotationCurve:
    step_id: int
    name: str
    points: List[Tuple[float, float, float]]
    hierarchy: Tuple[Tuple[str, Optional[int]], ...] = field(default_factory=tuple)

@dataclass
class PrototypeCaches:
    repmaps: Dict[int, MeshProto]
    repmap_counts: Counter
    hashes: Dict[str, HashProto]
    step_keys: Dict[int, PrototypeKey]
    instances: Dict[int, InstanceRecord]
    annotations: Dict[int, AnnotationCurve] = field(default_factory=dict)
    map_conversion: Optional[MapConversionData] = None

# ---------------------------------
# IFC helpers used by the pipeline
# ---------------------------------

def _as_float(v, default=0.0):
    try:
        if hasattr(v, "wrappedValue"): return float(v.wrappedValue)
        return float(v)
    except Exception:
        try: return float(default)
        except Exception: return 0.0

# Minimal placement composition (matches your prior signatures)

def axis2placement_to_matrix(place, length_to_m=1.0):
    if place is None: return Gf.Matrix4d(1)
    coords = getattr(getattr(place, "Location", None), "Coordinates", (0.0,0.0,0.0)) or (0.0,0.0,0.0)
    loc = Gf.Vec3d(*((_as_float(c) * length_to_m) for c in (coords + (0.0,0.0,0.0))[:3]))
    z_axis = getattr(getattr(place, "Axis", None), "DirectionRatios", (0.0,0.0,1.0)) or (0.0,0.0,1.0)
    x_axis = getattr(getattr(place, "RefDirection", None), "DirectionRatios", (1.0,0.0,0.0)) or (1.0,0.0,0.0)
    z = Gf.Vec3d(*map(_as_float, z_axis)); x = Gf.Vec3d(*map(_as_float, x_axis))
    # fallback orthonormal frame
    if abs(Gf.Dot(z, x)) > 0.9999: x = Gf.Vec3d(1.0,0.0,0.0)
    y = Gf.Cross(z, x)
    # build matrix (row-major)
    rot = Gf.Matrix4d([
        [x[0], y[0], z[0], 0.0],
        [x[1], y[1], z[1], 0.0],
        [x[2], y[2], z[2], 0.0],
        [0.0,  0.0,  0.0,  1.0],
    ])
    trans = Gf.Matrix4d(1).SetTranslate(loc)
    return trans * rot

def compose_object_placement(obj_placement, length_to_m=1.0):
    if obj_placement is None: return Gf.Matrix4d(1)
    local = axis2placement_to_matrix(getattr(obj_placement, "RelativePlacement", None), length_to_m)
    parent = compose_object_placement(getattr(obj_placement, "PlacementRelTo", None), length_to_m)
    return parent * local

def gf_to_tuple16(gf: Gf.Matrix4d):
    return tuple(gf[i][j] for i in range(4) for j in range(4))

# Properties/QTOs collector (same signature)

def _ifc_value_to_python(value):
    if value is None: return None
    if hasattr(value, 'wrappedValue'): return _ifc_value_to_python(value.wrappedValue)
    if hasattr(value, 'ValueComponent'): return _ifc_value_to_python(value.ValueComponent)
    if isinstance(value, (list, tuple)): return [_ifc_value_to_python(v) for v in value]
    if isinstance(value, (int, float, bool, str)): return value
    try: return str(value)
    except Exception: return None

def _extract_property_value(prop):
    if prop is None: return None
    try:
        if prop.is_a('IfcPropertySingleValue'): return _ifc_value_to_python(getattr(prop, 'NominalValue', None))
        if prop.is_a('IfcPropertyEnumeratedValue'): return [_ifc_value_to_python(v) for v in (getattr(prop,'EnumerationValues',None) or [])]
        if prop.is_a('IfcPropertyListValue'): return [_ifc_value_to_python(v) for v in (getattr(prop,'ListValues',None) or [])]
        if prop.is_a('IfcPropertyReferenceValue'):
            ref = getattr(prop, 'PropertyReference', None)
            if ref is not None: return getattr(ref, 'Name', None) or getattr(ref, 'Description', None) or str(ref)
        if prop.is_a('IfcComplexProperty'):
            nested = {}
            for sub in getattr(prop, 'HasProperties', None) or []:
                nested[sub.Name] = _extract_property_value(sub)
            return nested
    except Exception:
        pass
    for attr in ('NominalValue','LengthValue','AreaValue','VolumeValue','WeightValue','TimeValue','IntegerValue','RealValue','BooleanValue','LogicalValue','TextValue'):
        if hasattr(prop, attr): return _ifc_value_to_python(getattr(prop, attr))
    return None

def _extract_quantity_value(quantity):
    if quantity is None: return None
    for attr in ('LengthValue','AreaValue','VolumeValue','CountValue','WeightValue','TimeValue','Value','NominalValue'):
        if hasattr(quantity, attr): return _ifc_value_to_python(getattr(quantity, attr))
    for attr in ('Formula','Description'):
        if hasattr(quantity, attr) and getattr(quantity, attr): return getattr(quantity, attr)
    return None

def collect_instance_attributes(product) -> Dict[str, Dict[str, Dict[str, Any]]]:
    attrs: Dict[str, Dict[str, Dict[str, Any]]] = {'psets': {}, 'qtos': {}}
    def merge(container, name, data):
        if not data: return
        key = name or 'Unnamed'; entry = container.setdefault(key, {})
        entry.update({k:v for k,v in data.items() if v is not None})
    for rel in getattr(product, 'IsDefinedBy', []) or []:
        definition = getattr(rel, 'RelatingPropertyDefinition', None)
        if definition is None: continue
        if definition.is_a('IfcPropertySet'):
            props = {prop.Name: _extract_property_value(prop) for prop in (getattr(definition,'HasProperties',None) or [])}
            merge(attrs['psets'], getattr(definition,'Name',None), props)
        elif definition.is_a('IfcElementQuantity'):
            quants = {qty.Name: _extract_quantity_value(qty) for qty in (getattr(definition,'Quantities',None) or [])}
            merge(attrs['qtos'], getattr(definition,'Name',None), quants)
    for rel in getattr(product, 'IsTypedBy', []) or []:
        type_obj = getattr(rel, 'RelatingType', None)
        if type_obj is None: continue
        for pset in getattr(type_obj, 'HasPropertySets', None) or []:
            if pset.is_a('IfcPropertySet'):
                props = {prop.Name: _extract_property_value(prop) for prop in (getattr(pset,'HasProperties',None) or [])}
                merge(attrs['psets'], getattr(pset,'Name',None), props)
        for qset in getattr(type_obj, 'HasQuantities', None) or []:
            if qset.is_a('IfcElementQuantity'):
                quants = {qty.Name: _extract_quantity_value(qty) for qty in (getattr(qset,'Quantities',None) or [])}
                merge(attrs['qtos'], getattr(qset,'Name',None), quants)
    return attrs

# Mesh extraction from iterator triangulation

def triangulated_to_dict(geom) -> Dict[str, Any]:
    g = getattr(geom, "geometry", geom)
    if hasattr(g, "verts"): v = np.array(g.verts, dtype=float).reshape(-1,3)
    elif hasattr(g, "coordinates"): v = np.array(g.coordinates, dtype=float).reshape(-1,3)
    else: raise AttributeError("No verts/coordinates found")
    if hasattr(g, "faces"): f = np.array(g.faces, dtype=int).reshape(-1,3)
    elif hasattr(g, "triangles"): f = np.array(g.triangles, dtype=int).reshape(-1,3)
    else: raise AttributeError("No faces/triangles found")
    return {"vertices": v, "faces": f}

# Spatial hierarchy (unchanged)

def _entity_label(entity) -> str:
    for attr in ("Name","LongName","Description"):
        value = getattr(entity, attr, None)
        if isinstance(value, str) and value.strip(): return value.strip()
    label = entity.is_a() if hasattr(entity, "is_a") else "IfcEntity"
    try: step_id = entity.id()
    except Exception: step_id = None
    return f"{label}_{step_id}" if step_id is not None else label

def _resolve_spatial_parent(element):
    for rel in (getattr(element, "ContainedInStructure", None) or []):
        parent = getattr(rel, "RelatingStructure", None)
        if parent is not None: return parent
    for rel in (getattr(element, "Decomposes", None) or []):
        parent = getattr(rel, "RelatingObject", None)
        if parent is not None: return parent
    return None

def _collect_spatial_hierarchy(product) -> Tuple[Tuple[str, Optional[int]], ...]:
    if product is None or not hasattr(product, "is_a"): return tuple()
    parents: List[Any] = []; seen: set[int] = set(); current = product
    while True:
        parent = _resolve_spatial_parent(current)
        if parent is None: break
        try: parent_id = parent.id()
        except Exception: parent_id = None
        if parent_id is not None:
            if parent_id in seen: break
            seen.add(parent_id)
        if hasattr(parent, "is_a") and (parent.is_a("IfcSpatialStructureElement") or parent.is_a("IfcProject")):
            parents.append(parent)
        current = parent
    hierarchy: List[Tuple[str, Optional[int]]] = []
    for ancestor in reversed(parents):
        label = _entity_label(ancestor)
        try: step_id = ancestor.id()
        except Exception: step_id = None
        hierarchy.append((label, step_id))
    class_label = product.is_a() if hasattr(product, "is_a") else "IfcProduct"
    hierarchy.append((class_label, None))
    return tuple(hierarchy)

# Minimal map conversion (same signature as before)

def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    best = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not op.is_a("IfcMapConversion"): continue
            if getattr(ctx, "ContextType", None) == "Model" and getattr(ctx, "CoordinateSpaceDimension", None) == 3:
                return MapConversionData(
                    eastings=_as_float(getattr(op,"Eastings",0.0),0.0),
                    northings=_as_float(getattr(op,"Northings",0.0),0.0),
                    orthogonal_height=_as_float(getattr(op,"OrthogonalHeight",0.0),0.0),
                    x_axis_abscissa=_as_float(getattr(op,"XAxisAbscissa",1.0),1.0),
                    x_axis_ordinate=_as_float(getattr(op,"XAxisOrdinate",0.0),0.0),
                    scale=_as_float(getattr(op,"Scale",1.0),1.0) or 1.0,
                )
            best = best or op
    if best is None: return None
    return MapConversionData(
        eastings=_as_float(getattr(best,"Eastings",0.0),0.0),
        northings=_as_float(getattr(best,"Northings",0.0),0.0),
        orthogonal_height=_as_float(getattr(best,"OrthogonalHeight",0.0),0.0),
        x_axis_abscissa=_as_float(getattr(best,"XAxisAbscissa",1.0),1.0),
        x_axis_ordinate=_as_float(getattr(best,"XAxisOrdinate",0.0),0.0),
        scale=_as_float(getattr(best,"Scale",1.0),1.0) or 1.0,
    )

# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4x4 for THIS iterator piece.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    NOTE: Never elide identity — author explicit transforms on instances.
    """
    # 1) iterator-provided per-piece matrix
    tr = getattr(shape, "transformation", None)
    if tr is not None and hasattr(tr, "matrix"):
        return tuple(tr.matrix)  # keep even if identity

    # 2) robust util fallback
    try:
        from ifcopenshell.util import shape as ifc_shape_util
        gm = ifc_shape_util.get_shape_matrix(shape)
        gm = np.array(gm, dtype=float).reshape(4, 4)
        return tuple(gm.flatten().tolist())   # keep even if identity
    except Exception:
        pass

    # 3) composed local placement
    if element is not None:
        try:
            place = getattr(element, "ObjectPlacement", None)
            gf = compose_object_placement(place, length_to_m=1.0)
            return gf_to_tuple16(gf)          # keep even if identity
        except Exception:
            pass

    return None


def _gf_matrix_to_np(matrix) -> np.ndarray:
    """Convert a pxr.Gf matrix or sequence into a 4x4 numpy array."""
    if matrix is None:
        return np.eye(4, dtype=float)
    if isinstance(matrix, np.ndarray):
        arr = np.array(matrix, dtype=float)
        return arr.reshape(4, 4)
    if isinstance(matrix, (list, tuple)):
        arr = np.array(matrix, dtype=float)
        return arr.reshape(4, 4)
    try:
        return np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=float)
    except Exception:
        return np.eye(4, dtype=float)

def _axis_placement_to_np(axis_placement) -> np.ndarray:
    """Convert an IfcAxis2Placement into a numpy 4x4 matrix."""
    if axis_placement is None:
        return np.eye(4, dtype=float)
    try:
        return _gf_matrix_to_np(axis2placement_to_matrix(axis_placement, length_to_m=1.0))
    except Exception:
        return np.eye(4, dtype=float)


def _is_identity16(mat16, atol=1e-10):
    try:
        arr = np.array(mat16, dtype=float).reshape(4,4)
        return np.allclose(arr, np.eye(4), atol=atol)
    except Exception:
        return False

# Dummy 2D curves extraction kept (no change to signature)


def _cartesian_point_to_tuple(point) -> Tuple[float, float, float]:
    coords = list(getattr(point, "Coordinates", []) or [])
    x = _as_float(coords[0] if len(coords) > 0 else 0.0)
    y = _as_float(coords[1] if len(coords) > 1 else 0.0)
    z = _as_float(coords[2] if len(coords) > 2 else 0.0)
    return (x, y, z)


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a numpy 4x4 matrix."""
    if op is None:
        return np.eye(4, dtype=float)

    try:
        origin_src = getattr(getattr(op, "LocalOrigin", None), "Coordinates", None) or (0.0, 0.0, 0.0)
        origin_tuple = tuple(origin_src) + (0.0, 0.0, 0.0)
        origin = np.array([_as_float(c) for c in origin_tuple[:3]], dtype=float)
    except Exception:
        origin = np.zeros(3, dtype=float)

    def _vec(data, fallback):
        if not data:
            return np.array(fallback, dtype=float)
        return np.array([_as_float(c) for c in data], dtype=float)

    x_axis = _vec(getattr(getattr(op, "Axis1", None), "DirectionRatios", None), (1.0, 0.0, 0.0))
    y_axis = _vec(getattr(getattr(op, "Axis2", None), "DirectionRatios", None), (0.0, 1.0, 0.0))
    z_data = getattr(getattr(op, "Axis3", None), "DirectionRatios", None)
    z_axis = _vec(z_data, (0.0, 0.0, 1.0)) if z_data else np.cross(x_axis, y_axis)

    def _norm(vec: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(vec)
        return vec if length <= 1e-12 else vec / length

    x_axis = _norm(x_axis)
    y_axis = _norm(y_axis)
    if np.linalg.norm(z_axis) <= 1e-12:
        z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) <= 1e-12:
        z_axis = np.array((0.0, 0.0, 1.0), dtype=float)
    z_axis = _norm(z_axis)

    if abs(float(np.dot(x_axis, y_axis))) > 0.9999:
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

def _repmap_rt_matrix(mapped_item) -> np.ndarray:
    """Return MappingTarget ∘ MappingOrigin (RepresentationMap frame to product frame)."""
    source = getattr(mapped_item, "MappingSource", None)
    origin_np = _axis_placement_to_np(getattr(source, "MappingOrigin", None)) if source is not None else np.eye(4, dtype=float)
    target_np = _cartesian_transform_to_np(getattr(mapped_item, "MappingTarget", None))
    return target_np @ origin_np


def _mapping_item_transform(product, mapped_item) -> np.ndarray:
    """R->W for a mapped item: placement (P->W) composed with map (R->P)."""
    placement_np = _object_placement_to_np(getattr(product, "ObjectPlacement", None))
    map_np = _repmap_rt_matrix(mapped_item)
    return placement_np @ map_np


def _map_conversion_to_np(conv) -> np.ndarray:
    """Build a 4x4 from IfcMapConversion parameters."""
    scale = _as_float(getattr(conv, "Scale", 1.0) or 1.0, 1.0)
    east = _as_float(getattr(conv, "Eastings", 0.0), 0.0)
    north = _as_float(getattr(conv, "Northings", 0.0), 0.0)
    height = _as_float(getattr(conv, "OrthogonalHeight", 0.0), 0.0)
    ax = _as_float(getattr(conv, "XAxisAbscissa", 1.0), 1.0)
    ay = _as_float(getattr(conv, "XAxisOrdinate", 0.0), 0.0)
    norm = math.hypot(ax, ay) or 1.0
    cos = ax / norm
    sin = ay / norm
    mat = np.eye(4, dtype=float)
    mat[0, 0] = cos * scale
    mat[0, 1] = -sin * scale
    mat[1, 0] = sin * scale
    mat[1, 1] = cos * scale
    mat[0, 3] = east
    mat[1, 3] = north
    mat[2, 3] = height
    return mat

def _context_to_np(ctx) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    visited = set()
    current = ctx
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        wcs = getattr(current, "WorldCoordinateSystem", None)
        if wcs is not None:
            transform = transform @ _axis_placement_to_np(wcs)
        for op in getattr(current, "HasCoordinateOperation", None) or []:
            if op is None:
                continue
            if op.is_a("IfcMapConversion"):
                try:
                    transform = transform @ _map_conversion_to_np(op)
                except Exception:
                    continue
        current = getattr(current, "ParentContext", None)
    return transform

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    if item is None:
        return []
    if hasattr(item, "is_a") and item.is_a("IfcPolyline"):
        pts = []
        for p in getattr(item, "Points", []) or []:
            try:
                pts.append(_cartesian_point_to_tuple(p))
            except Exception:
                continue
        return pts
    if hasattr(item, "is_a") and item.is_a("IfcCompositeCurve"):
        pts = []
        for seg in getattr(item, "Segments", []) or []:
            parent = getattr(seg, "ParentCurve", None)
            pts.extend(_extract_curve_points(parent))
        return pts
    if hasattr(item, "is_a") and item.is_a("IfcTrimmedCurve"):
        return _extract_curve_points(getattr(item, "BasisCurve", None))
    if hasattr(item, "is_a") and item.is_a("IfcGeometricSet"):
        pts = []
        for el in getattr(item, "Elements", []) or []:
            pts.extend(_extract_curve_points(el))
        return pts
    if hasattr(item, "is_a") and item.is_a("IfcMappedItem"):
        src = getattr(item, "MappingSource", None)
        mapped = getattr(src, "MappedRepresentation", None) if src else None
        if mapped is not None:
            pts = []
            for sub in getattr(mapped, "Items", []) or []:
                pts.extend(_extract_curve_points(sub))
            return pts
    return []

def extract_annotation_curves(
    ifc_file,
    hierarchy_cache: Dict[int, Tuple[Tuple[str, Optional[int]], ...]],
) -> Dict[int, AnnotationCurve]:
    annotations: Dict[int, AnnotationCurve] = {}
    try:
        contexts = [
            ctx for ctx in (ifc_file.by_type("IfcGeometricRepresentationContext") or [])
            if str(getattr(ctx, "ContextType", "") or "").strip().lower() == "annotation"
            or str(getattr(ctx, "ContextIdentifier", "") or "").strip().lower() == "annotation"
        ]
    except Exception:
        contexts = []
    context_ids = {ctx.id() for ctx in contexts}

    annotation_rep_types = {"annotation","annotation2d","curve","curve2d","curve3d","geometriccurveset","geometricset"}
    annotation_ident_tokens = {"annotation","alignment"}

    def _hierarchy_for(ent) -> Tuple[Tuple[str, Optional[int]], ...]:
        try: key = ent.id()
        except Exception: key = id(ent)
        cached = hierarchy_cache.get(key)
        if cached is None:
            cached = _collect_spatial_hierarchy(ent)
            hierarchy_cache[key] = cached
        return cached

    for product in ifc_file.by_type("IfcProduct") or []:
        rep = getattr(product, "Representation", None)
        if not rep:
            continue
        hierarchy = _hierarchy_for(product)
        name = _entity_label(product)
        try:
            placement_np = _object_placement_to_np(getattr(product, "ObjectPlacement", None))
        except Exception:
            placement_np = np.eye(4, dtype=float)

        for rep_ctx in getattr(rep, "Representations", []) or []:
            ctx = getattr(rep_ctx, "ContextOfItems", None)
            rep_type = str(getattr(rep_ctx, "RepresentationType", "") or "").strip().lower()
            rep_ident = str(getattr(rep_ctx, "RepresentationIdentifier", "") or "").strip().lower()
            ctx_is_ann = ctx is not None and ctx.id() in context_ids
            rep_is_ann = (rep_type in annotation_rep_types) or any(t in rep_ident for t in annotation_ident_tokens)

            for item in getattr(rep_ctx, "Items", []) or []:
                if not hasattr(item, "is_a"):
                    continue

                item_ann = ctx_is_ann or rep_is_ann
                context_np = _context_to_np(ctx) if ctx is not None else np.eye(4, dtype=float)
                transform = context_np @ placement_np
                item_type = str(item.is_a() if hasattr(item, "is_a") else "").lower()

                if item.is_a("IfcMappedItem"):
                    src = getattr(item, "MappingSource", None)
                    mapped = getattr(src, "MappedRepresentation", None) if src else None
                    mapped_type = str(getattr(mapped, "RepresentationType", "") or "").strip().lower() if mapped else ""
                    mapped_ident = str(getattr(mapped, "RepresentationIdentifier", "") or "").strip().lower() if mapped else ""
                    mapped_ctx = getattr(mapped, "ContextOfItems", None) if mapped else None
                    mapped_ctx_np = _context_to_np(mapped_ctx) if mapped_ctx is not None else np.eye(4, dtype=float)
                    if mapped_ctx is not None and mapped_ctx.id() in context_ids:
                        item_ann = True
                    if mapped_type in annotation_rep_types or any(t in mapped_ident for t in annotation_ident_tokens):
                        item_ann = True
                    try:
                        transform = context_np @ (mapped_ctx_np @ _mapping_item_transform(product, item))
                    except Exception:
                        transform = context_np @ (mapped_ctx_np @ placement_np)
                else:
                    if item.is_a("IfcGeometricSet") or item.is_a("IfcGeometricCurveSet") or item.is_a("IfcPolyline"):
                        item_ann = True
                    elif "curve" in item_type and "surface" not in item_type:
                        item_ann = True

                if not item_ann:
                    continue

                pts = _extract_curve_points(item)
                if len(pts) < 2:
                    continue
                try:
                    pts_world = _transform_points(pts, transform)
                except Exception:
                    pts_world = pts
                if len(pts_world) < 2:
                    continue

                try: step_id = item.id()
                except Exception: step_id = id(item)
                annotations[step_id] = AnnotationCurve(
                    step_id=step_id, name=name, points=pts_world, hierarchy=hierarchy
                )
    return annotations



# Helper

def get_type_name(prod):
    for rel in getattr(prod, "IsTypedBy", []) or []:
        t = rel.RelatingType
        nm = getattr(t, "Name", None) or getattr(t, "ElementType", None)
        if nm: return nm
    return getattr(prod, "Name", None) or prod.is_a()

# ---------------------------------
# SINGLE-PASS iterator: build_prototypes()
# ---------------------------------

def build_prototypes(ifc_file, options: ConversionOptions) -> PrototypeCaches:
    """Single-pass iterator: builds prototype caches and per-instance records.
    Prototype identity = (Defining Type id, base geometry hash, settings fp)
    Instance transform = absolute/world 4×4 (resolve_absolute_matrix)
    """
    settings = ifcopenshell.geom.settings()
    # keep geometry in local coords; do not bake placement
    for k, v in {
        "use-world-coords": False,
        "weld-vertices": True,
        "disable-opening-subtractions": False,
        "apply-default-materials": False,
    }.items():
        try: settings.set(k, v)
        except Exception:
            try: settings.set(str(k).replace("-","_").upper(), v)
            except Exception: pass

    repmaps: Dict[int, MeshProto] = {}
    repmap_counts: Counter = Counter()
    hashes: Dict[str, HashProto] = {}
    step_keys: Dict[int, PrototypeKey] = {}
    instances: Dict[int, InstanceRecord] = {}
    hierarchy_cache: Dict[int, Tuple[Tuple[str, Optional[int]], ...]] = {}

    fingerprint_keys = [
        "use-world-coords",
        "weld-vertices",
        "disable-opening-subtractions",
        "apply-default-materials",
        "deflection-tolerance",
        "angle-tolerance",
        "max-facet",
    ]
    fp_vals: List[str] = []
    for key in fingerprint_keys:
        try:
            if hasattr(settings, "get"):
                val = settings.get(key)
            else:
                val = getattr(settings, key, None)
        except Exception:
            val = None
        if val is not None:
            fp_vals.append(str(val))
    if not fp_vals:
        fp_vals.append(str(id(settings)))
    settings_fp = hashlib.md5("|".join(fp_vals).encode("utf-8")).hexdigest()

    iterator = ifcopenshell.geom.iterator(settings, ifc_file, threads)
    if not iterator.initialize():
        # Fallback: still return caches for downstream authoring (empty)
        return PrototypeCaches(
            repmaps=repmaps, repmap_counts=repmap_counts, hashes=hashes,
            step_keys=step_keys, instances=instances,
            annotations=extract_annotation_curves(ifc_file, hierarchy_cache),
            map_conversion=extract_map_conversion(ifc_file),
        )

    while True:
        shape = iterator.get()
        if shape is None: break

        product = ifc_file.by_id(shape.id)
        if product is None:
            if not iterator.next(): break
            continue

        geom = getattr(shape, "geometry", None)
        if geom is None:
            if not iterator.next(): break
            continue

        # Mesh + hash
        try:
            mesh_dict = triangulated_to_dict(geom)
            mesh_hash = stable_mesh_hash(mesh_dict["vertices"], mesh_dict["faces"])
        except Exception:
            if not iterator.next(): break
            continue

        materials = list(getattr(geom, "materials", []) or [])
        material_ids = list(getattr(geom, "material_ids", []) or [])

        # World transform
        xf_tuple = resolve_absolute_matrix(shape, product)

        # Defining Type anchor
        type_ref = None; type_guid=None; type_name=None; type_id=None
        for rel in (getattr(product, "IsTypedBy", []) or []):
            rel_type = getattr(rel, "RelatingType", None)
            if rel_type is not None:
                type_ref = rel_type
                type_guid = getattr(rel_type, "GlobalId", None)
                type_name = getattr(rel_type, "Name", None)
                break
        if type_ref is not None and mesh_hash is not None:
            try: type_id = int(type_ref.id())
            except Exception: type_id = None

        primary_key: Optional[PrototypeKey] = None
        instance_mesh: Optional[Dict[str, Any]] = None

        if type_id is not None:
            info = repmaps.get(type_id)
            if info is None:
                info = MeshProto(
                    repmap_id=type_id,
                    type_name=type_name,
                    type_class=type_ref.is_a() if hasattr(type_ref, "is_a") else None,
                    type_guid=type_guid,
                    repmap_index=None,
                )
                repmaps[type_id] = info
            else:
                if info.type_name is None and type_name is not None: info.type_name = type_name
                if info.type_class is None and hasattr(type_ref, "is_a"): info.type_class = type_ref.is_a()
                if info.type_guid is None and type_guid is not None: info.type_guid = type_guid

            if info.mesh is None and mesh_hash is not None:
                info.mesh = mesh_dict; info.mesh_hash = mesh_hash; info.settings_fp = settings_fp
                if not info.materials and materials: info.materials = list(materials)
                if not info.material_ids and material_ids: info.material_ids = list(material_ids)
                info.count = 0

            if info.mesh is not None and info.mesh_hash == mesh_hash and info.settings_fp == settings_fp:
                info.count += 1; repmap_counts[type_id] += 1
                primary_key = PrototypeKey(kind="repmap", identifier=type_id)

        if primary_key is None:
            if options.enable_hash_dedup and mesh_hash is not None:
                digest = f"{mesh_hash}|{settings_fp}"
                bucket = hashes.get(digest)
                if bucket is None:
                    bucket = HashProto(
                        digest=digest,
                        name=sanitize_name(getattr(product, "Name", None), fallback=product.is_a() if hasattr(product, "is_a") else None),
                        type_name=get_type_name(product),
                        type_guid=getattr(product, "GlobalId", None),
                        mesh=mesh_dict,
                        materials=list(materials),
                        material_ids=list(material_ids),
                        canonical_frame=xf_tuple,
                        settings_fp=settings_fp,
                        count=0,
                    )
                    hashes[digest] = bucket
                bucket.count += 1
                if not bucket.materials and materials: bucket.materials = list(materials)
                if not bucket.material_ids and material_ids: bucket.material_ids = list(material_ids)
                if bucket.mesh is None: bucket.mesh = mesh_dict
                if bucket.canonical_frame is None: bucket.canonical_frame = xf_tuple
                primary_key = PrototypeKey(kind="hash", identifier=digest)
            else:
                instance_mesh = mesh_dict

        if primary_key is None and instance_mesh is None:
            if not iterator.next(): break
            continue

        # Build InstanceRecord
        step_id = shape.id
        if primary_key is not None:
            step_keys[step_id] = primary_key

        fallback = getattr(product, "GlobalId", None) or str(getattr(product, "id", lambda: step_id)())
        name = sanitize_name(getattr(product, "Name", None), fallback=fallback)
        try: product_id = product.id()
        except Exception: product_id = None

        proto_delta_tuple: Optional[Tuple[float, ...]] = None
        if options.enable_instancing and primary_key is not None and primary_key.kind == "hash":
            bucket = hashes.get(primary_key.identifier)
            if bucket and bucket.canonical_frame is not None and xf_tuple is not None:
                try:
                    canonical_np = np.array(bucket.canonical_frame, dtype=float).reshape(4,4)
                    xf_np = np.array(xf_tuple, dtype=float).reshape(4,4)
                    delta_np = np.matmul(np.linalg.inv(canonical_np), xf_np)
                    if not np.allclose(delta_np, np.eye(4), atol=1e-8):
                        proto_delta_tuple = tuple(float(delta_np[i, j]) for i in range(4) for j in range(4))
                except Exception:
                    proto_delta_tuple = None

        attributes = collect_instance_attributes(product) if options.convert_metadata else {}

        key = product_id if product_id is not None else id(product)
        hierarchy_nodes = hierarchy_cache.get(key) or _collect_spatial_hierarchy(product)
        hierarchy_cache[key] = hierarchy_nodes

        " Debug Logging"
        watch = {"Railing_3000mm_Hoarding_Retaining_Wall_7071173",
                "Strip_Footing_SF1_2000W_x_1200_D_Acoustic_Shed_1736226"}
        nm = getattr(product, "Name", None) or ""
        if any(nm.startswith(tok) for tok in watch):
            import math
            mat = np.array(xf_tuple or np.eye(4)).reshape(4,4)
            log.info("DBG %s step=%s T=(%.3f, %.3f, %.3f) | row3=(%.3f, %.3f, %.3f, %.3f)",
                    nm, step_id, mat[0,3], mat[1,3], mat[2,3],
                    mat[3,0], mat[3,1], mat[3,2], mat[3,3])



        guid = getattr(product, "GlobalId", None)
        instances[step_id] = InstanceRecord(
            step_id=step_id,
            product_id=product_id,
            prototype=primary_key,
            name=name,
            transform=xf_tuple,
            material_ids=list(material_ids),
            materials=list(materials),
            attributes=attributes,
            prototype_delta=proto_delta_tuple,
            hierarchy=hierarchy_nodes,
            mesh=instance_mesh,
            guid=guid,
        )

        if not iterator.next(): break

    return PrototypeCaches(
        repmaps=repmaps,
        repmap_counts=repmap_counts,
        hashes=hashes,
        step_keys=step_keys,
        instances=instances,
        annotations=extract_annotation_curves(ifc_file, hierarchy_cache),
        map_conversion=extract_map_conversion(ifc_file),
    )
