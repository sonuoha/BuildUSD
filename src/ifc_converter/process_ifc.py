# 0 . ---------------------------- Imports ----------------------------
from __future__ import annotations
import logging
from collections import Counter
import multiprocessing
import numpy as np
import ifcopenshell
import os

import math
import re
from typing import Dict, Optional, Union, Literal, List, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import hashlib

from .pxr_utils import Gf

if TYPE_CHECKING:
    from .config.manifest import ConversionManifest


# ifcopenshell util for robust matrices (mapped/type geometry)
try:
    from ifcopenshell.util import shape as ifc_shape_util
    _HAVE_IFC_UTIL_SHAPE = True
except Exception:
    _HAVE_IFC_UTIL_SHAPE = False

log = logging.getLogger(__name__)


def resolve_threads(env_var="IFC_GEOM_THREADS", minimum=1):
    val = os.getenv(env_var)
    if val is not None and val.strip() != "":
        try:
            n = int(val)
        except ValueError:
            log.warning("Invalid %s=%r; using CPU count fallback.", env_var, val)
            n = None
    else:
        n = None

    if n is None:
        try:
            n = multiprocessing.cpu_count()  # returns int, may raise NotImplementedError
        except NotImplementedError:
            n = minimum

    return max(minimum, n)

threads = resolve_threads()

def _localize_mesh(mesh: dict, matrix_inv: np.ndarray) -> dict:
    """Return a copy of mesh with vertices transformed by matrix_inv (4x4)."""
    if mesh is None:
        return None
    verts = mesh.get("vertices")
    if verts is None:
        return mesh
    verts_np = np.asarray(verts, dtype=float).reshape(-1, 3)
    ones = np.ones((verts_np.shape[0], 1), dtype=float)
    homo = np.hstack((verts_np, ones))
    localized = (homo @ matrix_inv.T)[:, :3]
    mesh_copy = dict(mesh)
    mesh_copy["vertices"] = localized
    return mesh_copy

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
    """Prototype representing a single IFC representation map (type-based mesh)."""
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
    """Prototype representing a hashed fallback mesh aggregated from instances."""
    digest: str
    name: Optional[str] = None
    type_name: Optional[str] = None
    type_guid: Optional[str] = None
    mesh: Optional[dict] = None
    materials: List[Any] = field(default_factory=list)
    material_ids: List[int] = field(default_factory=list)
    signature: Optional[Tuple[Any, ...]] = None
    canonical_frame: Optional[Tuple[float, ...]] = None
    count: int = 0

@dataclass(frozen=True)
class CurveWidthRule:
    """User-defined width override for annotation curves authored as BasisCurves."""
    width: float
    unit: Optional[str] = None
    layer: Optional[str] = None
    curve: Optional[str] = None
    hierarchy: Optional[str] = None
    step_id: Optional[int] = None

@dataclass
class ConversionOptions:
    """Controls conversion behavior: instancing, hash de-dup, and metadata."""
    enable_instancing: bool = True
    enable_hash_dedup: bool = True
    convert_metadata: bool = True
    manifest: Optional['ConversionManifest'] = None
    curve_width_rules: Tuple[CurveWidthRule, ...] = tuple()

@dataclass(frozen=True)
class PrototypeKey:
    """Stable key for prototypes: either by repmap id or fallback hash."""
    kind: Literal["repmap", "hash"]
    identifier: Union[int, str]

@dataclass
class MapConversionData:
    """IFC IfcMapConversion parameters used to align stage to map coordinates."""
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

@dataclass
class InstanceRecord:
    """A single placed IFC product occurrence and its authored payload."""
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
    """Polyline-style annotation extracted from an IFC annotation context."""

    step_id: int
    name: str
    points: List[Tuple[float, float, float]]
    hierarchy: Tuple[Tuple[str, Optional[int]], ...] = field(default_factory=tuple)


@dataclass
class PrototypeCaches:
    """All prototypes and instances discovered plus optional map conversion."""
    repmaps: Dict[int, MeshProto]
    repmap_counts: Counter
    hashes: Dict[str, HashProto]
    step_keys: Dict[int, PrototypeKey]
    instances: Dict[int, "InstanceRecord"]
    annotations: Dict[int, AnnotationCurve] = field(default_factory=dict)
    map_conversion: Optional[MapConversionData] = None



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
    """Extract IfcMapConversion parameters from an IFC file if present."""
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

def _entity_label(entity) -> str:
    for attr in ("Name", "LongName", "Description"):
        value = getattr(entity, attr, None)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    label = entity.is_a() if hasattr(entity, "is_a") else "IfcEntity"
    try:
        step_id = entity.id()
    except Exception:
        step_id = None
    if step_id is not None:
        return f"{label}_{step_id}"
    return label


def _resolve_spatial_parent(element):
    for rel in (getattr(element, "ContainedInStructure", None) or []):
        parent = getattr(rel, "RelatingStructure", None)
        if parent is not None:
            return parent
    for rel in (getattr(element, "Decomposes", None) or []):
        parent = getattr(rel, "RelatingObject", None)
        if parent is not None:
            return parent
    return None


def _collect_spatial_hierarchy(product) -> Tuple[Tuple[str, Optional[int]], ...]:
    if product is None or not hasattr(product, "is_a"):
        return tuple()
    parents: List[Any] = []
    seen: set[int] = set()
    current = product
    while True:
        parent = _resolve_spatial_parent(current)
        if parent is None:
            break
        try:
            parent_id = parent.id()
        except Exception:
            parent_id = None
        if parent_id is not None:
            if parent_id in seen:
                break
            seen.add(parent_id)
        if hasattr(parent, "is_a") and (parent.is_a("IfcSpatialStructureElement") or parent.is_a("IfcProject")):
            parents.append(parent)
        current = parent
    hierarchy: List[Tuple[str, Optional[int]]] = []
    for ancestor in reversed(parents):
        label = _entity_label(ancestor)
        try:
            step_id = ancestor.id()
        except Exception:
            step_id = None
        hierarchy.append((label, step_id))
    class_label = product.is_a() if hasattr(product, "is_a") else "IfcProduct"
    hierarchy.append((class_label, None))
    return tuple(hierarchy)


def _cartesian_point_to_tuple(point) -> Tuple[float, float, float]:
    coords = list(getattr(point, "Coordinates", []) or [])
    x = _as_float(coords[0] if len(coords) > 0 else 0.0)
    y = _as_float(coords[1] if len(coords) > 1 else 0.0)
    z = _as_float(coords[2] if len(coords) > 2 else 0.0)
    return (x, y, z)







def _as_float(v, default=0.0):
    """Safely coerce IFC numeric wrappers or Python scalars to float."""
    try:
        if hasattr(v, "wrappedValue"):
            return float(v.wrappedValue)
        return float(v)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0

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


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)


def _axis_placement_to_np(axis_placement) -> np.ndarray:
    """Convert an IfcAxis2Placement into a numpy 4x4 matrix."""
    if axis_placement is None:
        return np.eye(4, dtype=float)
    try:
        return _gf_matrix_to_np(axis2placement_to_matrix(axis_placement, length_to_m=1.0))
    except Exception:
        return np.eye(4, dtype=float)


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


def _transform_points(points, matrix: np.ndarray) -> List[Tuple[float, float, float]]:
    """Apply a homogeneous transform to a sequence of (x, y, z) points."""
    if not points:
        return []
    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    ones = np.ones((arr.shape[0], 1), dtype=float)
    homo = np.hstack((arr, ones))
    mat = np.asarray(matrix, dtype=float).reshape(4, 4)
    transformed = homo @ mat
    return [
        (float(x), float(y), float(z))
        for x, y, z in transformed[:, :3]
    ]


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



def _context_to_np(ctx) -> np.ndarray:
    """Compose world transforms defined by an IFC representation context chain."""

    transform = np.eye(4, dtype=float)
    visited = set()
    current = ctx
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        wcs = getattr(current, "WorldCoordinateSystem", None)
        if wcs is not None:
            transform = transform @ _axis_placement_to_np(wcs)
        current = getattr(current, "ParentContext", None)
    return transform


def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    if item is None:
        return []
    if hasattr(item, "is_a") and item.is_a("IfcPolyline"):
        pts: List[Tuple[float, float, float]] = []
        for p in getattr(item, "Points", []) or []:
            try:
                pts.append(_cartesian_point_to_tuple(p))
            except Exception:
                continue
        return pts
    if hasattr(item, "is_a") and item.is_a("IfcCompositeCurve"):
        pts: List[Tuple[float, float, float]] = []
        for segment in getattr(item, "Segments", []) or []:
            parent = getattr(segment, "ParentCurve", None)
            pts.extend(_extract_curve_points(parent))
        return pts
    if hasattr(item, "is_a") and item.is_a("IfcTrimmedCurve"):
        return _extract_curve_points(getattr(item, "BasisCurve", None))
    if hasattr(item, "is_a") and item.is_a("IfcGeometricSet"):
        pts: List[Tuple[float, float, float]] = []
        for element in getattr(item, "Elements", []) or []:
            pts.extend(_extract_curve_points(element))
        return pts
    if hasattr(item, "is_a") and item.is_a("IfcMappedItem"):
        source = getattr(item, "MappingSource", None)
        if source is not None:
            mapped = getattr(source, "MappedRepresentation", None)
            if mapped is not None:
                pts: List[Tuple[float, float, float]] = []
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
            ctx
            for ctx in (ifc_file.by_type("IfcGeometricRepresentationContext") or [])
            if str(getattr(ctx, "ContextType", "") or "").strip().lower() == "annotation"
            or str(getattr(ctx, "ContextIdentifier", "") or "").strip().lower() == "annotation"
        ]
    except Exception:
        contexts = []
    context_ids = {ctx.id() for ctx in contexts}

    annotation_rep_types = {
        "annotation",
        "annotation2d",
        "curve",
        "curve2d",
        "curve3d",
        "geometriccurveset",
        "geometricset",
    }
    annotation_ident_tokens = {"annotation", "alignment"}

    def _hierarchy_for(entity) -> Tuple[Tuple[str, Optional[int]], ...]:
        try:
            key = entity.id()
        except Exception:
            key = id(entity)
        cached = hierarchy_cache.get(key)
        if cached is None:
            cached = _collect_spatial_hierarchy(entity)
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
            ctx_is_annotation = ctx is not None and ctx.id() in context_ids
            rep_is_annotation = (
                rep_type in annotation_rep_types
                or any(token in rep_ident for token in annotation_ident_tokens)
            )

            for item in getattr(rep_ctx, "Items", []) or []:
                if not hasattr(item, "is_a"):
                    continue

                item_annotation = ctx_is_annotation or rep_is_annotation
                context_np = _context_to_np(ctx) if ctx is not None else np.eye(4, dtype=float)
                transform = context_np @ placement_np
                item_type = str(item.is_a() if hasattr(item, "is_a") else "").lower()

                if item.is_a("IfcMappedItem"):
                    mapped_source = getattr(item, "MappingSource", None)
                    mapped = getattr(mapped_source, "MappedRepresentation", None) if mapped_source else None
                    mapped_type = str(getattr(mapped, "RepresentationType", "") or "").strip().lower() if mapped else ""
                    mapped_ident = str(getattr(mapped, "RepresentationIdentifier", "") or "").strip().lower() if mapped else ""
                    mapped_ctx = getattr(mapped, "ContextOfItems", None) if mapped else None
                    mapped_context_np = _context_to_np(mapped_ctx) if mapped_ctx is not None else np.eye(4, dtype=float)
                    if mapped_ctx is not None and mapped_ctx.id() in context_ids:
                        item_annotation = True
                    if mapped_type in annotation_rep_types or any(token in mapped_ident for token in annotation_ident_tokens):
                        item_annotation = True
                    try:
                        transform = context_np @ (mapped_context_np @ _mapping_item_transform(product, item))
                    except Exception:
                        transform = context_np @ (mapped_context_np @ placement_np)
                else:
                    if item.is_a("IfcGeometricSet") or item.is_a("IfcGeometricCurveSet") or item.is_a("IfcPolyline"):
                        item_annotation = True
                    elif "curve" in item_type and "surface" not in item_type:
                        item_annotation = True

                if not item_annotation:
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

                try:
                    step_id = item.id()
                except Exception:
                    step_id = id(item)
                annotations[step_id] = AnnotationCurve(
                    step_id=step_id,
                    name=name,
                    points=pts_world,
                    hierarchy=hierarchy,
                )
    return annotations



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
    """Collect property sets (psets) and quantities (qtos) for an IFC product.

    Returns a nested dict: {'psets': {Name: {k:v}}, 'qtos': {Name: {k:v}}}.
    """
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


def build_prototypes(ifc_file, options: ConversionOptions) -> PrototypeCaches:
    """Discover prototypes and instances respecting conversion options."""
    s_local = ifcopenshell.geom.settings()
    safe_set(s_local, "use-world-coords", False)
    safe_set(s_local, "weld-vertices", True)
    safe_set(s_local, "disable-opening-subtractions", True)
    safe_set(s_local, "apply-default-materials", False)

    if options.enable_hash_dedup:
        log.warning("Hash-based deduplication is experimental and may produce inaccurate results; use with caution.")

    repmaps: Dict[int, MeshProto] = {}
    repmap_counts: Counter = Counter()
    hashes: Dict[str, HashProto] = {}
    step_keys: Dict[int, PrototypeKey] = {}
    instances: Dict[int, InstanceRecord] = {}
    hierarchy_cache: Dict[int, Tuple[Tuple[str, Optional[int]], ...]] = {}

    repmap_to_type: Dict[int, Tuple[Optional[Any], Optional[int]]] = {}
    for t in ifc_file.by_type("IfcTypeProduct"):
        for idx, rm in enumerate(getattr(t, "RepresentationMaps", []) or []):
            repmap_to_type[rm.id()] = (t, idx)

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

    it = ifcopenshell.geom.iterator(s_local, ifc_file, threads)
    if not it.initialize():
        geom_log = (ifcopenshell.get_log() or "").strip()
        if "ContextType 'Annotation' not allowed" in geom_log:
            log.warning("ifcopenshell geom iterator refused annotation context; switching to manual 2D curve extraction. Details: %s", geom_log)
        else:
            log.error("Geometry iterator initialisation failed: %s", geom_log or "<no log output>")
        annotations = extract_annotation_curves(ifc_file, hierarchy_cache)
        return PrototypeCaches(
            repmaps=repmaps,
            repmap_counts=repmap_counts,
            hashes=hashes,
            step_keys=step_keys,
            instances=instances,
            annotations=annotations,
            map_conversion=extract_map_conversion(ifc_file),
        )

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
        abs_np = np.eye(4, dtype=float)
        if abs_mat is not None:
            try:
                abs_np = np.array(abs_mat, dtype=float).reshape(4, 4)
            except Exception:
                abs_np = np.eye(4, dtype=float)

        instance_mesh: Optional[Dict[str, Any]] = None

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
                for rm in getattr(type_ref, "RepresentationMaps", []) or []:
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
                instance_mesh = triangulated_to_dict(geom)
                primary_key = None

        if primary_key is None and instance_mesh is None:
            continue

        step_id = shape.id
        if primary_key is not None:
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

        proto_delta_tuple: Optional[Tuple[float, ...]] = None
        if (
            options.enable_instancing
            and abs_mat is not None
            and primary_key is not None
            and primary_key.kind == "hash"
        ):
            bucket = hashes.get(primary_key.identifier)
            if bucket and bucket.canonical_frame is not None:
                try:
                    canonical_np = np.array(bucket.canonical_frame, dtype=float).reshape(4, 4)
                    delta_np = np.matmul(np.linalg.inv(canonical_np), abs_np)
                    if not np.allclose(delta_np, np.eye(4), atol=1e-8):
                        proto_delta_tuple = tuple(delta_np.reshape(-1).tolist())
                except Exception:
                    proto_delta_tuple = None

        attributes = collect_instance_attributes(product) if options.convert_metadata else {}

        hierarchy_key = product_id if product_id is not None else id(product)
        hierarchy_nodes = hierarchy_cache.get(hierarchy_key)
        if hierarchy_nodes is None:
            hierarchy_nodes = _collect_spatial_hierarchy(product)
            hierarchy_cache[hierarchy_key] = hierarchy_nodes

        if mapped_items:
            try:
                inst_np = _mapping_item_transform(product, mapped_items[0])
            except Exception:
                inst_np = abs_np if abs_mat is not None else _object_placement_to_np(getattr(product, "ObjectPlacement", None))
        else:
            inst_np = abs_np if abs_mat is not None else _object_placement_to_np(getattr(product, "ObjectPlacement", None))
        inst_np = np.array(inst_np, dtype=float)
        inst_transform: Optional[Tuple[float, ...]] = tuple(inst_np.reshape(-1).tolist())

        guid = getattr(product, "GlobalId", None)
        instances[step_id] = InstanceRecord(
            step_id=step_id,
            product_id=product_id,
            prototype=primary_key,
            name=name,
            transform=inst_transform,
            material_ids=list(material_ids),
            materials=list(materials),
            attributes=attributes,
            prototype_delta=proto_delta_tuple,
            hierarchy=hierarchy_nodes,
            mesh=instance_mesh,
            guid=guid,
        )
        if inst_np is not None and getattr(product, "is_a", lambda *_: False)("IfcFooting"):
            parent_label = hierarchy_nodes[-2][0] if len(hierarchy_nodes) > 1 else "?"
            log.info(
                "IfcFooting instance step=%s name=%s parent=%s translation=(%.3f, %.3f, %.3f)",
                step_id,
                name,
                parent_label,
                float(inst_np[0, 3]),
                float(inst_np[1, 3]),
                float(inst_np[2, 3]),
            )

    annotations = extract_annotation_curves(ifc_file, hierarchy_cache)
    map_conversion = extract_map_conversion(ifc_file)

    return PrototypeCaches(
        repmaps=repmaps,
        repmap_counts=repmap_counts,
        hashes=hashes,
        step_keys=step_keys,
        instances=instances,
        annotations=annotations,
        map_conversion=map_conversion,
    )

def axis2placement_to_matrix(place, length_to_m=1.0):
    """Convert an IfcAxis2Placement2D/3D into a USD 4x4 transform.

    Normalises the placement's axes, reconstructs an orthonormal frame even when
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
    it = ifcopenshell.geom.iterator(settings, ifc_file, threads)
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







def compute_mesh_signature(mesh: Dict[str, Any], precision: int = 5) -> Tuple[Any, ...]:
    """Return a rotation/translation-robust signature for a triangulated mesh."""
    if mesh is None:
        return tuple()
    verts = mesh.get("vertices")
    faces = mesh.get("faces")
    if verts is None or faces is None:
        return tuple()
    try:
        v = np.asarray(verts, dtype=float).reshape(-1, 3)
    except Exception:
        return tuple()
    if v.size == 0:
        return (0,)
    try:
        f = np.asarray(faces, dtype=int).reshape(-1, 3)
    except Exception:
        f = np.zeros((0, 3), dtype=int)
    centroid = v.mean(axis=0) if len(v) else np.zeros(3, dtype=float)
    centered = v - centroid
    bbox = np.max(v, axis=0) - np.min(v, axis=0) if len(v) else np.zeros(3, dtype=float)
    bbox_sig = tuple(np.round(np.sort(np.abs(bbox)), precision).tolist())
    cov = centered.T @ centered
    if len(v):
        cov /= float(len(v))
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except Exception:
        eigvals = np.zeros(3, dtype=float)
    eig_sig = tuple(np.round(np.sort(np.abs(eigvals)), precision).tolist())
    try:
        tris = v[f.astype(int)] if len(f) else np.zeros((0, 3, 3), dtype=float)
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]) if len(tris) else np.zeros((0, 3), dtype=float)
        areas = 0.5 * np.linalg.norm(cross, axis=1) if len(cross) else np.zeros(0, dtype=float)
        total_area = float(np.sum(areas))
        area_sq = float(np.sum(areas ** 2))
    except Exception:
        total_area = 0.0
        area_sq = 0.0
    try:
        radii = np.linalg.norm(centered, axis=1) if len(centered) else np.zeros(0, dtype=float)
        if radii.size:
            q = np.quantile(radii, [0.0, 0.25, 0.5, 0.75, 1.0])
            radial_sig = tuple(np.round(q, precision).tolist())
        else:
            radial_sig = (0.0, 0.0, 0.0, 0.0, 0.0)
    except Exception:
        radial_sig = (0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        int(v.shape[0]),
        int(f.shape[0]),
        bbox_sig,
        eig_sig,
        radial_sig,
        round(float(total_area), precision),
        round(float(area_sq), precision),
    )
