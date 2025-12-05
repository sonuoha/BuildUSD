"""OpenCASCADE helpers for detail-mode tessellation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Literal, Set, Callable

import numpy as np
import ifcopenshell

from .occ_detail_bootstrap import (
    bootstrap_occ,
    is_available as _occ_bootstrap_available,
    last_failure_reason as _occ_last_failure,
    sym,
    sym_optional,
)

log = logging.getLogger(__name__)

_OCC_READY = False

# NOTE: This module only produces meshes in the local IFC coordinate frame.
# World/local normalization is handled upstream (process_ifc) via pre/post
# transforms so USD authors can decide how anchors/geolocation are applied.

__all__ = [
    "OCCFaceMesh",
    "OCCSubshapeMesh",
    "OCCDetailMesh",
    "is_available",
    "build_detail_mesh",
    "build_detail_mesh_payload",
    "build_canonical_detail_for_type",
    "detail_from_repmap_items",
    "precompute_detail_meshes",
    "proxy_requires_high_detail",
    "mesh_from_detail_mesh",
    "align_detail_mesh_to_base",
]

_DEFAULT_LINEAR_DEF = 0.01  # metres (cap)
_DEFAULT_ANGULAR_DEF = 0.5   # degrees

_PROXY_SHORT_DIM_THRESHOLD = 0.06  # metres
_PROXY_SLENDER_RATIO = 40.0
_PROXY_SMALL_DIAGONAL = 3.0
_PROXY_FACE_COUNT_THRESHOLD = 120
_PROXY_AREA_PER_FACE_THRESHOLD = 0.02


@dataclass
class OCCFaceMesh:
    """Mesh data for a single TopoDS face."""

    face_index: int
    vertices: np.ndarray
    faces: np.ndarray
    material_key: Optional[Any] = None


@dataclass
class OCCSubshapeMesh:
    """Aggregated mesh info for a logical OCC sub-shape (solid/shell)."""

    index: int
    label: str
    shape_type: str
    faces: List[OCCFaceMesh]
    material_key: Optional[Any] = None


@dataclass
class OCCDetailMesh:
    """OCC tessellation payload reused by detail-mode consumers."""

    shape: Any
    subshapes: List[OCCSubshapeMesh]

    @property
    def face_count(self) -> int:
        return sum(len(entry.faces) for entry in self.subshapes)

    @property
    def faces(self) -> List[OCCFaceMesh]:
        flattened: List[OCCFaceMesh] = []
        for entry in self.subshapes:
            flattened.extend(entry.faces)
        return flattened


def is_available() -> bool:
    """Return True once all OCC dependencies have been bootstrapped."""
    return _occ_bootstrap_available()


def why_unavailable() -> str:
    """Return a short reason if OCC detail isn't available."""
    if is_available():
        return "OCC bootstrap already complete"
    bootstrap_occ(strict=False)
    reason = _occ_last_failure()
    if reason:
        return "OCC bootstrap failed:\n  " + reason
    return "OCC bootstrap failed: unknown reason (check logs)"



def _build_canonical_maps(canonical_map: Dict[str, Any]) -> Tuple[Dict[tuple, int], Dict[tuple, Dict[str, int]]]:
    """Build lookup maps from canonical mesh data."""
    coord_to_vid: Dict[tuple, int] = {}
    tri_map: Dict[tuple, Dict[str, int]] = {}
    
    verts = canonical_map.get("vertices")
    faces = canonical_map.get("faces")
    mat_ids = canonical_map.get("material_ids")
    if mat_ids is None:
        mat_ids = []
    item_ids = canonical_map.get("item_ids")
    if item_ids is None:
        item_ids = []
    
    if verts is None or faces is None:
        return coord_to_vid, tri_map
        
    # Tolerance for coordinate matching (should match what's used in _triangulate_face_mapped)
    TOL = 1e-6
    def coord_key(p):
        return tuple((np.round(p / TOL) * TOL).tolist())

    # Build vertex map
    for i, v in enumerate(verts):
        coord_to_vid[coord_key(v)] = i
        
    # Build triangle map
    for i, face in enumerate(faces):
        if len(face) != 3:
            continue
        # Sort vertex indices to make triangle key invariant to winding order (mostly)
        # Actually, winding order matters for normals, but for identification, sorted indices are safer if we assume same vertices.
        # However, we are mapping *canonical* VIDs.
        tri_key = tuple(sorted((int(face[0]), int(face[1]), int(face[2]))))
        
        info = {}
        if i < len(mat_ids):
            info["material_id"] = int(mat_ids[i])
        if i < len(item_ids):
            info["item_id"] = int(item_ids[i])
            
        tri_map[tri_key] = info
        
    return coord_to_vid, tri_map


# -----------------------------------------------------------------------------
# Matrix / Transform Helpers (duplicated from process_ifc to avoid circular dep)
# -----------------------------------------------------------------------------

def _as_float(v, default=0.0):
    try:
        if hasattr(v, "wrappedValue"): return float(v.wrappedValue)
        return float(v)
    except Exception:
        try: return float(default)
        except Exception: return 0.0

def _axis_placement_to_np(placement) -> np.ndarray:
    """Convert IfcAxis2Placement3D to 4x4 numpy matrix."""
    if placement is None:
        return np.eye(4, dtype=float)
    
    loc = getattr(placement, "Location", None)
    axis = getattr(placement, "Axis", None)
    ref_dir = getattr(placement, "RefDirection", None)
    
    ox = _as_float(getattr(loc, "Coordinates", [0,0,0])[0])
    oy = _as_float(getattr(loc, "Coordinates", [0,0,0])[1])
    oz = _as_float(getattr(loc, "Coordinates", [0,0,0])[2])
    
    # Z axis
    if axis and getattr(axis, "DirectionRatios", None):
        z = np.array(axis.DirectionRatios, dtype=float)
        zn = np.linalg.norm(z)
        if zn > 1e-9: z /= zn
        else: z = np.array([0,0,1], dtype=float)
    else:
        z = np.array([0,0,1], dtype=float)
        
    # X axis
    if ref_dir and getattr(ref_dir, "DirectionRatios", None):
        x = np.array(ref_dir.DirectionRatios, dtype=float)
        xn = np.linalg.norm(x)
        if xn > 1e-9: x /= xn
        else: x = np.array([1,0,0], dtype=float)
    else:
        # Arbitrary X if not specified
        if abs(z[2]) < 0.9: x = np.array([0,0,1], dtype=float) # if Z not vertical, use Z as X-ish? No.
        # Standard fallback logic:
        # If Z is vertical (0,0,1), X is (1,0,0).
        # If Z is not vertical, we need an X.
        # Actually, if RefDirection is missing, X is derived from Z.
        # But let's keep it simple: if missing, use (1,0,0) and orthogonalize.
        x = np.array([1,0,0], dtype=float)

    # Orthogonalize X against Z
    # x = x - dot(x,z)*z
    x = x - np.dot(x, z) * z
    xn = np.linalg.norm(x)
    if xn > 1e-9:
        x /= xn
    else:
        # X is parallel to Z? Pick arbitrary
        if abs(z[0]) < 0.9: x = np.array([1,0,0], dtype=float)
        else: x = np.array([0,1,0], dtype=float)
        x = x - np.dot(x, z) * z
        x /= np.linalg.norm(x)
        
    y = np.cross(z, x)
    
    mat = np.eye(4, dtype=float)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[0, 3] = ox
    mat[1, 3] = oy
    mat[2, 3] = oz
    return mat

def _cartesian_transform_to_np(operator) -> np.ndarray:
    """Convert IfcCartesianTransformationOperator3D to 4x4 numpy matrix."""
    if operator is None:
        return np.eye(4, dtype=float)
    
    # Origin
    origin = getattr(operator, "LocalOrigin", None)
    ox = _as_float(getattr(origin, "Coordinates", [0,0,0])[0])
    oy = _as_float(getattr(origin, "Coordinates", [0,0,0])[1])
    oz = _as_float(getattr(origin, "Coordinates", [0,0,0])[2])
    
    # Axis 1 (X)
    axis1 = getattr(operator, "Axis1", None)
    if axis1 and getattr(axis1, "DirectionRatios", None):
        x = np.array(axis1.DirectionRatios, dtype=float)
        xn = np.linalg.norm(x)
        if xn > 1e-9: x /= xn
        else: x = np.array([1,0,0], dtype=float)
    else:
        x = np.array([1,0,0], dtype=float)
        
    # Axis 2 (Y)
    axis2 = getattr(operator, "Axis2", None)
    if axis2 and getattr(axis2, "DirectionRatios", None):
        y = np.array(axis2.DirectionRatios, dtype=float)
        yn = np.linalg.norm(y)
        if yn > 1e-9: y /= yn
        else: y = np.array([0,1,0], dtype=float)
    else:
        y = np.array([0,1,0], dtype=float)
        
    # Axis 3 (Z)
    axis3 = getattr(operator, "Axis3", None)
    if axis3 and getattr(axis3, "DirectionRatios", None):
        z = np.array(axis3.DirectionRatios, dtype=float)
        zn = np.linalg.norm(z)
        if zn > 1e-9: z /= zn
        else: z = np.array([0,0,1], dtype=float)
    else:
        z = np.array([0,0,1], dtype=float)
        
    # Scale
    scale = _as_float(getattr(operator, "Scale", 1.0), 1.0)
    
    mat = np.eye(4, dtype=float)
    mat[:3, 0] = x * scale
    mat[:3, 1] = y * scale
    mat[:3, 2] = z * scale
    mat[0, 3] = ox
    mat[1, 3] = oy
    mat[2, 3] = oz
    return mat

def _repmap_rt_matrix(mapped_item) -> np.ndarray:
    """Return MappingTarget @ MappingOrigin (RepresentationMap frame to product frame)."""
    source = getattr(mapped_item, "MappingSource", None)
    origin_np = _axis_placement_to_np(getattr(source, "MappingOrigin", None)) if source is not None else np.eye(4, dtype=float)
    target_np = _cartesian_transform_to_np(getattr(mapped_item, "MappingTarget", None))
    return target_np @ origin_np

def _apply_transform_to_occ_shape(shape: Any, matrix: np.ndarray) -> Any:
    """Apply a 4x4 numpy matrix transform to a TopoDS_Shape."""
    if not is_available() or shape is None:
        return shape
    try:
        gp_Trsf = sym("gp_Trsf")
        gp_Vec = sym("gp_Vec")
        gp_Mat = sym("gp_Mat")
        BRepBuilderAPI_Transform = sym("BRepBuilderAPI_Transform")
        
        trsf = gp_Trsf()
        
        # Extract rotation/scale part
        m = matrix
        # gp_Mat expects (Row1, Row2, Row3)
        mat = gp_Mat(
            float(m[0,0]), float(m[0,1]), float(m[0,2]),
            float(m[1,0]), float(m[1,1]), float(m[1,2]),
            float(m[2,0]), float(m[2,1]), float(m[2,2])
        )
        vec = gp_Vec(float(m[0,3]), float(m[1,3]), float(m[2,3]))
        
        trsf.SetValues(
            float(m[0,0]), float(m[0,1]), float(m[0,2]), float(m[0,3]),
            float(m[1,0]), float(m[1,1]), float(m[1,2]), float(m[1,3]),
            float(m[2,0]), float(m[2,1]), float(m[2,2]), float(m[2,3])
        )
        
        transformer = BRepBuilderAPI_Transform(shape, trsf, True) # True = copy
        return transformer.Shape()
    except Exception as exc:
        log.warning("Failed to apply transform to OCC shape: %s", exc)
        return shape


def build_detail_mesh(
    shape_obj: Any,
    *,
    linear_deflection: float,
    angular_deflection_rad: float,
    logger: Optional[logging.Logger] = None,
    coord_to_vid: Optional[Dict[tuple, int]] = None,
    tri_map: Optional[Dict[tuple, Dict[str, int]]] = None,
) -> Optional[OCCDetailMesh]:
    """Return OCC meshes grouped by logical sub-shapes, when OCC is available."""
    if not is_available() or shape_obj is None:
        if logger:
            logger.log(1, "Either OCC Module not installed or no valid TopoDS Shape found", exc_info=True)
        return None
    topo_shape = _extract_topods_shape(shape_obj, logger=logger)
    if topo_shape is None:
        return None
    linear_tol = max(float(linear_deflection), 1e-6)
    angular_tol = max(float(angular_deflection_rad), math.radians(0.1))
    if not _perform_meshing(
        topo_shape, 
        linear_tol, 
        angular_tol, 
        logger=logger, 
        ifc_entity=getattr(shape_obj, "id", lambda: None)()
    ):
        return None
    subshape_shapes = _primary_subshape_list(topo_shape)
    subshape_meshes: List[OCCSubshapeMesh] = []
    face_counter = 0
    for idx, (label, subshape) in enumerate(subshape_shapes):
        faces, face_counter = _collect_face_meshes(
            subshape, 
            face_counter,
            coord_to_vid=coord_to_vid,
            tri_map=tri_map
        )
        if faces:
            subshape_meshes.append(
                OCCSubshapeMesh(
                    index=idx,
                    label=label,
                    shape_type=_shape_type_name(subshape),
                    faces=faces,
                )
            )
    if not subshape_meshes:
        faces, _ = _collect_face_meshes(
            topo_shape, 
            0,
            coord_to_vid=coord_to_vid,
            tri_map=tri_map
        )
        if not faces:
            return None
        subshape_meshes.append(
            OCCSubshapeMesh(index=0, label="Shape", shape_type=_shape_type_name(topo_shape), faces=faces)
        )
    log.debug(
        "OCC detail: prepared %d subshape entries for %s",
        len(subshape_meshes),
        _shape_type_name(topo_shape),
    )
    return OCCDetailMesh(shape=topo_shape, subshapes=subshape_meshes)

def _extract_topods_shape(shape_obj: Any, logger: Optional[logging.Logger] = None) -> Optional[Any]:
    """Best-effort conversion of ifcopenshell shapes to TopoDS_Shape."""
    if not is_available():
        return None
    try:
        occ_utils = sym("occ_utils")
    except Exception as exc:
        if logger:
            logger.debug("OCC detail: occ_utils import failed during extraction (%s)", exc)
        return None
    if _is_topods(shape_obj):
        return shape_obj
    for candidate in _iter_occ_candidates(shape_obj):
        try:
            result = occ_utils.create_shape_from_serialization(candidate)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.debug("OCC serialization failed for %s: %s", type(candidate).__name__, exc)
            continue
        if _is_topods(result):
            return result
        if hasattr(result, "geometry") and _is_topods(result.geometry):  # type: ignore[attr-defined]
            return result.geometry  # type: ignore[attr-defined]
    if logger:
        logger.debug("No TopoDS_Shape could be extracted for %s", type(shape_obj).__name__)
    return None


def _iter_occ_candidates(root: Any) -> Iterator[Any]:
    """Yield nested objects that may contain serialised OCC geometry."""
    stack: List[Any] = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in seen or obj is None:
            continue
        seen.add(obj_id)
        yield obj
        for attr in ("data", "geometry"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def _is_topods(candidate: Any) -> bool:
    if candidate is None:
        return False
    if not is_available():
        return False
    try:
        topo_cls = sym("TopoDS_Shape")
    except Exception:
        topo_cls = None
    if topo_cls is not None and isinstance(candidate, topo_cls):
        return True
    return hasattr(candidate, "ShapeType") and hasattr(candidate, "IsNull") and callable(candidate.ShapeType)


def try_inv_4x4(matrix: Any) -> Optional[List[List[float]]]:
    """Attempt to invert a 4x4 matrix; return None on failure."""
    if matrix is None:
        return None
    try:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4):
            return None
        inv = np.linalg.inv(arr)
        return inv.tolist()
    except Exception:
        return None


def mesh_center(vertices: Any) -> Optional[np.ndarray]:
    """Return the centroid of a vertex array (Nx3)."""
    if vertices is None:
        return None
    try:
        arr = np.asarray(vertices, dtype=float)
        if arr.size == 0:
            return None
        return np.mean(arr, axis=0)
    except Exception:
        return None


def _apply_matrix_to_vertices(vertices: np.ndarray, mat16: Any) -> np.ndarray:
    """Apply a flattened 4x4 (row-major) matrix to vertex rows."""
    try:
        mat = np.asarray(mat16, dtype=np.float64).reshape(4, 4)
    except Exception:
        return vertices
    ones = np.ones((vertices.shape[0], 1), dtype=np.float64)
    homo = np.hstack([vertices, ones])
    transformed = homo @ mat.T
    return transformed[:, :3]


def _compose_pre_xform(detail_mesh: OCCDetailMesh, matrix: np.ndarray) -> bool:
    try:
        new_mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    except Exception:
        return False
    existing = getattr(detail_mesh, "pre_xform", None)
    if existing is not None:
        try:
            existing_arr = np.asarray(existing, dtype=np.float64).reshape(4, 4)
            new_mat = new_mat @ existing_arr
        except Exception:
            return False
    detail_mesh.pre_xform = new_mat.tolist()
    return True


def align_detail_mesh_to_base(detail_mesh: OCCDetailMesh, base_mesh: Dict[str, Any]) -> bool:
    """Attach a translation pre_xform so OCC detail shares the base mesh's center."""
    base_center = _mesh_center(base_mesh)
    if base_center is None:
        return False
    detail_flat = mesh_from_detail_mesh(detail_mesh)
    if detail_flat is None:
        return False
    detail_center = _mesh_center(detail_flat)
    if detail_center is None:
        return False
    delta = base_center - detail_center
    if not np.any(np.abs(delta) > 1e-6):
        return False
    translation = np.eye(4, dtype=np.float64)
    translation[0, 3] = float(delta[0])
    translation[1, 3] = float(delta[1])
    translation[2, 3] = float(delta[2])
    return _compose_pre_xform(detail_mesh, translation)


def _scale_detail_mesh(detail_mesh: Optional[OCCDetailMesh], scale: float) -> None:
    if detail_mesh is None:
        return
    if not math.isfinite(scale) or abs(scale - 1.0) < 1e-12:
        return
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    for subshape in subshapes:
        faces = getattr(subshape, "faces", None) or []
        for face in faces:
            verts = getattr(face, "vertices", None)
            if verts is None:
                continue
            try:
                verts_arr = np.asarray(verts, dtype=np.float64)
                verts_arr *= float(scale)
                face.vertices = verts_arr
            except Exception:
                continue


def _require_occ_components(ifc_entity=None) -> None:
    if is_available():
        return
    try:
        bootstrap_occ()
    except ImportError as exc:
        msg = f"Detail mesher unavailable: OCC bindings not loaded for {ifc_entity}: {exc}"
        log.error(msg)
        raise RuntimeError(msg) from exc


_SHAPE_TYPE_NAMES = {
    0: "COMPOUND",
    1: "COMPSOLID",
    2: "SOLID",
    3: "SHELL",
    4: "FACE",
    5: "WIRE",
    6: "EDGE",
    7: "VERTEX",
    8: "SHAPE",
}
_DETAIL_TYPE_WHITELIST = {"COMPOUND", "COMPSOLID", "SOLID", "SHELL", "FACE"}


def _shape_type_name(shape: Any) -> str:
    try:
        shape_type = int(shape.ShapeType())
    except Exception:
        return "<unknown>"
    return _SHAPE_TYPE_NAMES.get(shape_type, str(shape_type))


def _set_linear_deflection(params, value: float) -> None:
    for name in (
        "SetLinearDeflection",
        "SetDeflection",
        "SetMaximalChordalDeviation",
        "SetChordalDeflection",
        "SetMaxChordalDeviation",
    ):
        method = getattr(params, name, None)
        if callable(method):
            method(float(value))
            return
    setattr(params, "linear_deflection", float(value))


def _set_angular_deflection(params, value: float) -> None:
    for name in ("SetAngularDeflection", "SetAngularDeviation", "SetAngularError", "SetMaximalAngle"):
        method = getattr(params, name, None)
        if callable(method):
            method(float(value))
            return
    setattr(params, "angular_deflection", float(value))


def _set_bool_flag(params, candidates: Tuple[str, ...], value: bool) -> None:
    for name in candidates:
        method = getattr(params, name, None)
        if callable(method):
            method(bool(value))
            return
    for name in candidates:
        if hasattr(params, name):
            setattr(params, name, bool(value))
            return





def _get_bounding_box_diag(shape: Any) -> float:
    if not is_available():
        return 1.0
    try:
        Bnd_Box = sym("Bnd_Box")
        BRepBndLib = sym("BRepBndLib")
        box = Bnd_Box()
        box.SetGap(0.0)
        BRepBndLib.Add(shape, box)
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
        dz = float(zmax - zmin)
        diag = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2 + max(dz, 0.0) ** 2)
        return float(diag) if diag > 0 else 1.0
    except Exception:
        return 1.0


def safe_mesh_shape(
    occ_shape,
    *,
    ifc_entity=None,
    base_linear_deflection: float = 0.01,
    base_angular_deflection: float = 0.5,
    logger: Optional[logging.Logger] = None,
):
    active_log = logger or log
    if occ_shape is None or getattr(occ_shape, "IsNull", lambda: True)():
        msg = f"Detail mesher: null OCC shape for {ifc_entity}"
        active_log.error(msg)
        raise RuntimeError(msg)
    if not is_available():
        msg = f"Detail mesher: OCC runtime not available for {ifc_entity}"
        active_log.error(msg)
        raise RuntimeError(msg)

    _require_occ_components(ifc_entity)

    try:
        BRepCheck_Analyzer = sym("BRepCheck_Analyzer")
        ShapeFix_Shape = sym("ShapeFix_Shape")
        BRepBuilderAPI_Sewing = sym("BRepBuilderAPI_Sewing")
        IMeshTools_Parameters = sym("IMeshTools_Parameters")
        BRepMesh_IncrementalMesh = sym("BRepMesh_IncrementalMesh")
    except Exception as exc:
        msg = f"Detail mesher: OCC components unavailable for {ifc_entity}: {exc}"
        active_log.error(msg)
        raise RuntimeError(msg) from exc

    try:
        analyzer = BRepCheck_Analyzer(occ_shape)
        if analyzer is not None and not analyzer.IsValid():
            active_log.info("Detail mesher: invalid B-Rep for %s, running ShapeFix.", ifc_entity)
            fixer = ShapeFix_Shape(occ_shape)
            fixer.Perform()
            fixed_shape = fixer.Shape()
            if fixed_shape is None or fixed_shape.IsNull():
                active_log.warning("Detail mesher: ShapeFix failed for %s", ifc_entity)
                return False, None
            occ_shape = fixed_shape
    except Exception:
        active_log.debug("Detail mesher: BRepCheck analyzer failed for %s", ifc_entity, exc_info=True)

    try:
        TopAbs_SOLID = sym("TopAbs_SOLID")
        TopAbs_SHELL = sym("TopAbs_SHELL")
    except Exception:
        TopAbs_SOLID = None
        TopAbs_SHELL = None

    sewed_performed = False
    if BRepBuilderAPI_Sewing is not None:
        # Optimization: skip sewing if the shape is already a solid or shell
        # Sewing is expensive and mostly needed for compounds of faces.
        should_sew = True
        if TopAbs_SOLID is not None and TopAbs_SHELL is not None:
            try:
                st = occ_shape.ShapeType()
                if st == TopAbs_SOLID or st == TopAbs_SHELL:
                    should_sew = False
            except Exception:
                pass
        
        if should_sew:
            try:
                sewing = BRepBuilderAPI_Sewing(1e-6)
                sewing.Add(occ_shape)
                sewing.Perform()
                sewn = sewing.SewedShape()
                if sewn is not None and not sewn.IsNull():
                    occ_shape = sewn
                sewed_performed = True
            except Exception:
                active_log.debug("Detail mesher: sewing failed for %s", ifc_entity, exc_info=True)

    # Use absolute deflection from settings to ensure consistency with IfcOpenShell/process_ifc
    linear_deflection = base_linear_deflection
    angular_deflection = base_angular_deflection

    # Dynamic deflection logic:
    # Calculate diagonal to determine appropriate scale
    diag = _get_bounding_box_diag(occ_shape)
    
    # Target 0.1% of diagonal as relative deflection
    # But clamp to the absolute setting (don't exceed what user asked for)
    # And clamp to a safe minimum (1e-5)
    relative_deflection = diag * 0.001
    effective_linear = min(linear_deflection, relative_deflection)
    effective_linear = max(effective_linear, 1e-5)

    params = IMeshTools_Parameters()
    _set_linear_deflection(params, effective_linear)
    _set_angular_deflection(params, angular_deflection)
    
    # Enforce absolute deflection (SetRelative=False) to match process_ifc.py intent
    _set_bool_flag(
        params,
        ("SetRelative", "SetRelativeMode", "SetRelativeFlag"),
        False,
    )
    _set_bool_flag(
        params,
        ("SetInParallel", "SetIsInParallel", "SetParallel"),
        True,
    )
    _set_bool_flag(
        params,
        ("SetAllowQualityDecrease", "SetAllowQualityDrop", "SetQualityDecrease"),
        True,
    )

    mesher = None
    success = False
    
    # First meshing attempt
    try:
        mesher = BRepMesh_IncrementalMesh(occ_shape, params)
        success = True
    except Exception as exc:
        active_log.debug("Detail mesher: first pass failed for %s: %s", ifc_entity, exc)

    # Fallback: if failed and we skipped sewing, try sewing now
    if not success and not sewed_performed and BRepBuilderAPI_Sewing is not None:
        active_log.info("Detail mesher: fallback to sewing for %s", ifc_entity)
        try:
            sewing = BRepBuilderAPI_Sewing(1e-6)
            sewing.Add(occ_shape)
            sewing.Perform()
            sewn = sewing.SewedShape()
            if sewn is not None and not sewn.IsNull():
                occ_shape = sewn
                # Retry meshing
                mesher = BRepMesh_IncrementalMesh(occ_shape, params)
                success = True
        except Exception as exc:
             active_log.warning("Detail mesher: fallback sewing/meshing failed for %s: %s", ifc_entity, exc)
             return False, None

    if success and mesher:
        active_log.debug(
            "Detail mesher: meshed %s (diag=%.3f, defl=%.5f [req=%.5f], angle=%.2f)",
            ifc_entity,
            diag,
            effective_linear,
            linear_deflection,
            angular_deflection,
        )
        return True, mesher
    else:
        active_log.warning("Detail mesher: OCC meshing failed for %s", ifc_entity)
        return False, None


def _perform_meshing(
    topo_shape: Any,
    linear_deflection: float,
    angular_deflection: float,
    *,
    logger: Optional[logging.Logger] = None,
    ifc_entity: Any = None,
) -> bool:
    """Run the OCCT mesher once so all faces carry triangulations."""
    try:
        success, _ = safe_mesh_shape(
            topo_shape,
            ifc_entity=ifc_entity,
            base_linear_deflection=linear_deflection,
            base_angular_deflection=angular_deflection,
            logger=logger,
        )
        return success
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.debug("Safe OCC meshing failed: %s", exc)
        return False


def _collect_face_meshes(
    target_shape: Any, 
    start_index: int, 
    *, 
    material_key: Optional[Any] = None,
    coord_to_vid: Optional[Dict[tuple, int]] = None,
    tri_map: Optional[Dict[tuple, Dict[str, int]]] = None
) -> tuple[List[OCCFaceMesh], int]:
    """Collect OCCFaceMesh entries for every face in ``target_shape``."""
    _require_occ_components()
    TopExp_Explorer = sym("TopExp_Explorer")
    TopAbs_FACE = sym("TopAbs_FACE")
    faces: List[OCCFaceMesh] = []
    face_index = start_index
    explorer = TopExp_Explorer(target_shape, TopAbs_FACE)
    while explorer.More():
        try:
            face = explorer.Current()
        except Exception:
            explorer.Next()
            face_index += 1
            continue
        
        # If we have a canonical map, we might split this face into multiple meshes
        if coord_to_vid and tri_map:
            split_meshes = _triangulate_face_mapped(face, coord_to_vid, tri_map)
            if split_meshes:
                for mesh in split_meshes:
                    mesh.face_index = face_index
                    if material_key is not None and getattr(mesh, "material_key", None) is None:
                        mesh.material_key = material_key
                    faces.append(mesh)
            else:
                # Fallback if mapping fails or produces no mesh
                mesh = _triangulate_face(face)
                if mesh is not None:
                    mesh.face_index = face_index
                    if material_key is not None and getattr(mesh, "material_key", None) is None:
                        mesh.material_key = material_key
                    faces.append(mesh)
        else:
            mesh = _triangulate_face(face)
            if mesh is not None:
                mesh.face_index = face_index
                if material_key is not None and getattr(mesh, "material_key", None) is None:
                    mesh.material_key = material_key
                faces.append(mesh)
                
        explorer.Next()
        face_index += 1
    return faces, face_index


def _triangulate_face_mapped(
    face: Any, 
    coord_to_vid: Dict[tuple, int], 
    tri_map: Dict[tuple, Dict[str, int]]
) -> List[OCCFaceMesh]:
    """Triangulate a face and split it by canonical material/item IDs."""
    TopLoc_Location = sym("TopLoc_Location")
    BRep_Tool = sym("BRep_Tool")
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)
    if triangulation is None:
        return []

    def _iter_nodes():
        nodes_fn = getattr(triangulation, "Nodes", None)
        if callable(nodes_fn):
            nodes_handle = nodes_fn()
            if nodes_handle is None:
                return []
            length = nodes_handle.Length()
            return [nodes_handle.Value(i) for i in range(1, length + 1)]
        nb_nodes_fn = getattr(triangulation, "NbNodes", None)
        node_fn = getattr(triangulation, "Node", None)
        if callable(nb_nodes_fn) and callable(node_fn):
            length = int(nb_nodes_fn())
            return [node_fn(i) for i in range(1, length + 1)]
        return []

    def _iter_triangles():
        triangles_fn = getattr(triangulation, "Triangles", None)
        if callable(triangles_fn):
            triangles_handle = triangles_fn()
            if triangles_handle is None:
                return []
            length = triangles_handle.Length()
            return [triangles_handle.Value(i) for i in range(1, length + 1)]
        nb_triangles_fn = getattr(triangulation, "NbTriangles", None)
        triangle_fn = getattr(triangulation, "Triangle", None)
        if callable(nb_triangles_fn) and callable(triangle_fn):
            length = int(nb_triangles_fn())
            return [triangle_fn(i) for i in range(1, length + 1)]
        return []

    node_points = _iter_nodes()
    triangle_handles = _iter_triangles()
    if not node_points or not triangle_handles:
        return []

    use_transform = hasattr(loc, "IsIdentity") and not loc.IsIdentity()
    transform = loc.Transformation() if use_transform else None
    
    # Extract vertices once
    vertices = np.zeros((len(node_points), 3), dtype=np.float64)
    for idx, point in enumerate(node_points):
        if transform is not None:
            try:
                point = point.Transformed(transform)
            except Exception:
                pass
        vertices[idx, :] = (point.X(), point.Y(), point.Z())

    # Group triangles by (material_id, item_id)
    # We need to re-index vertices for each group to keep meshes clean (optional but good)
    # For simplicity, we can just slice the global vertex array if we want, 
    # but usually we want compact vertex arrays for each sub-mesh.
    # Let's just store indices into the full vertex array for now, 
    # and maybe optimize later if needed. But OCCFaceMesh expects a vertex array.
    # So we should probably subset the vertices.

    # Tolerance for coordinate matching
    TOL = 1e-6
    def coord_key(p):
        # We assume vertices are already float64
        return tuple((np.round(p / TOL) * TOL).tolist())

    # Pre-compute canonical VIDs for all OCC vertices
    occ_vid_to_canon_vid = {}
    for i, v in enumerate(vertices):
        k = coord_key(v)
        if k in coord_to_vid:
            occ_vid_to_canon_vid[i] = coord_to_vid[k]

    groups = {}  # (mat_id, item_id) -> list of (i1, i2, i3)

    for tri in triangle_handles:
        try:
            i1, i2, i3 = tri.Get()
            # OCC indices are 1-based
            idx1, idx2, idx3 = int(i1)-1, int(i2)-1, int(i3)-1
        except Exception:
            continue
        
        # Map to canonical triangle
        c1 = occ_vid_to_canon_vid.get(idx1)
        c2 = occ_vid_to_canon_vid.get(idx2)
        c3 = occ_vid_to_canon_vid.get(idx3)
        
        mat_id = -1
        item_id = -1
        
        if c1 is not None and c2 is not None and c3 is not None:
            tri_key = tuple(sorted((c1, c2, c3)))
            if tri_key in tri_map:
                info = tri_map[tri_key]
                mat_id = info.get("material_id", -1)
                item_id = info.get("item_id", -1)
        
        key = (mat_id, item_id)
        if key not in groups:
            groups[key] = []
        groups[key].append((idx1, idx2, idx3))

    results = []
    for (mat_id, item_id), tri_indices in groups.items():
        # Create a subset of vertices for this group
        # Find unique vertex indices used in this group
        used_vids = sorted(list(set(idx for tri in tri_indices for idx in tri)))
        vid_map = {old: new for new, old in enumerate(used_vids)}
        
        sub_vertices = vertices[used_vids]
        sub_faces = np.array([[vid_map[i1], vid_map[i2], vid_map[i3]] for i1, i2, i3 in tri_indices], dtype=np.int32)
        
        # Construct material key (this will be used by semantic splitter later)
        # We can pass a dict or tuple. The semantic splitter expects something it can understand.
        # Currently OCCFaceMesh.material_key is used for grouping.
        # We should probably pass a dict with 'material_id' and 'item_id'.
        
        # Note: The semantic splitter in process_ifc.py uses 'occ_mesh_dict' which comes from 'mesh_from_detail_mesh'.
        # 'mesh_from_detail_mesh' aggregates everything.
        # We need to ensure that the semantic splitter can access this info.
        # The 'OCCFaceMesh' has 'material_key'.
        
        # Let's use a dict as the key, it's flexible.
        mat_key = {"material_id": mat_id, "item_id": item_id}
        
        results.append(OCCFaceMesh(
            face_index=0, # Will be set by caller
            vertices=sub_vertices,
            faces=sub_faces,
            material_key=mat_key
        ))
        
    return results


def _triangulate_face(face: Any) -> Optional[OCCFaceMesh]:
    TopLoc_Location = sym("TopLoc_Location")
    BRep_Tool = sym("BRep_Tool")
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)
    if triangulation is None:
        return None

    def _iter_nodes():
        nodes_fn = getattr(triangulation, "Nodes", None)
        if callable(nodes_fn):
            nodes_handle = nodes_fn()
            if nodes_handle is None:
                return []
            length = nodes_handle.Length()
            return [nodes_handle.Value(i) for i in range(1, length + 1)]
        nb_nodes_fn = getattr(triangulation, "NbNodes", None)
        node_fn = getattr(triangulation, "Node", None)
        if callable(nb_nodes_fn) and callable(node_fn):
            length = int(nb_nodes_fn())
            return [node_fn(i) for i in range(1, length + 1)]
        return []

    def _iter_triangles():
        triangles_fn = getattr(triangulation, "Triangles", None)
        if callable(triangles_fn):
            triangles_handle = triangles_fn()
            if triangles_handle is None:
                return []
            length = triangles_handle.Length()
            return [triangles_handle.Value(i) for i in range(1, length + 1)]
        nb_triangles_fn = getattr(triangulation, "NbTriangles", None)
        triangle_fn = getattr(triangulation, "Triangle", None)
        if callable(nb_triangles_fn) and callable(triangle_fn):
            length = int(nb_triangles_fn())
            return [triangle_fn(i) for i in range(1, length + 1)]
        return []

    node_points = _iter_nodes()
    triangle_handles = _iter_triangles()
    if not node_points or not triangle_handles:
        return None

    use_transform = hasattr(loc, "IsIdentity") and not loc.IsIdentity()
    transform = loc.Transformation() if use_transform else None
    vertices = np.zeros((len(node_points), 3), dtype=np.float64)
    for idx, point in enumerate(node_points):
        if transform is not None:
            try:
                point = point.Transformed(transform)
            except Exception:
                pass
        vertices[idx, :] = (point.X(), point.Y(), point.Z())

    faces_arr = np.zeros((len(triangle_handles), 3), dtype=np.int32)
    for tri_idx, tri in enumerate(triangle_handles):
        try:
            i1, i2, i3 = tri.Get()
        except Exception:  # pragma: no cover - compatibility guard
            indices = [0, 0, 0]
            try:
                tri.Get(indices)  # type: ignore[arg-type]
                i1, i2, i3 = indices
            except Exception:
                continue
        try:
            faces_arr[tri_idx, :] = (int(i1) - 1, int(i2) - 1, int(i3) - 1)
        except Exception:
            continue

    return OCCFaceMesh(face_index=0, vertices=vertices, faces=faces_arr)


def _primary_subshape_list(topo_shape: Any) -> List[tuple[str, Any]]:
    """Return a list of first-level subshapes (prefer solids, then shells, etc.)."""
    collected: List[tuple[str, Any]] = []

    def _flatten_subshape(shape: Any, label: str) -> List[tuple[str, Any]]:
        """Recursively explode COMPOUND nodes, stopping at composite solids/solids."""
        shape_name = _shape_type_name(shape)
        if shape_name not in ("COMPOUND", "COMPSOLID"):
            return [(label, shape)]
        try:
            TopExp_Explorer = sym("TopExp_Explorer")
            TopAbs_COMPSOLID = sym("TopAbs_COMPSOLID")
            TopAbs_SOLID = sym("TopAbs_SOLID")
            TopAbs_SHELL = sym("TopAbs_SHELL")
        except Exception:
            return [(label, shape)]

        had_volumetric = False

        # Prefer composite solids first
        for top_abs, suffix in (
            (TopAbs_COMPSOLID, "CompositeSolid"),
            (TopAbs_SOLID, "Solid"),
            (TopAbs_SHELL, "Shell"),
        ):
            explorer = TopExp_Explorer(shape, top_abs)
            entries: List[tuple[str, Any]] = []
            index = 0
            while explorer.More():
                child = explorer.Current()
                child_label = f"{label}_{suffix}_{index}"
                entries.extend(_flatten_subshape(child, child_label))
                if suffix in ("CompositeSolid", "Solid"):
                    had_volumetric = True
                index += 1
                explorer.Next()
            if entries:
                return entries
        if not had_volumetric:
            log.info("Detail mesh: compound %s contains no solids; returning compound as-is.", label)
        return [(label, shape)]

    occ_utils = None
    try:
        occ_utils = sym("occ_utils")
    except Exception:
        occ_utils = None
    if occ_utils is not None and hasattr(occ_utils, "yield_subshapes"):
        type_buckets: Dict[str, List[tuple[str, Any]]] = {}
        try:
            for subshape in occ_utils.yield_subshapes(topo_shape):
                shape_name = _shape_type_name(subshape)
                if shape_name not in _DETAIL_TYPE_WHITELIST:
                    continue
                bucket = type_buckets.setdefault(shape_name, [])
                base_label = f"{shape_name}_{len(bucket)}"
                bucket.extend(_flatten_subshape(subshape, base_label))
        except Exception as exc:
            log.debug("OCC detail: yield_subshapes failed (%s); falling back to TopExp iterator.", exc)
        if type_buckets:
            for preferred in ("SOLID", "COMPSOLID", "SHELL", "COMPOUND"):
                entries = type_buckets.get(preferred)
                if entries:
                    collected.extend(entries)
                    break
            if not collected:
                # Fallback to whichever bucket exists (preserves previous behaviour)
                for entries in type_buckets.values():
                    collected.extend(entries)
                    break
    if collected:
        log.debug(
            "OCC detail: yield_subshapes produced %d subshapes for %s",
            len(collected),
            _shape_type_name(topo_shape),
        )
        return collected

    try:
        top_abs_solid = sym("TopAbs_SOLID")
        top_abs_compsolid = sym("TopAbs_COMPSOLID")
        top_abs_shell = sym("TopAbs_SHELL")
        top_abs_compound = sym("TopAbs_COMPOUND")
        TopExp_Explorer = sym("TopExp_Explorer")
    except Exception as exc:
        log.debug("OCC detail: TopAbs/TopExp imports failed (%s)", exc)
        return collected

    order = [
        (top_abs_solid, "Solid"),
        (top_abs_compsolid, "CompositeSolid"),
        (top_abs_shell, "Shell"),
        (top_abs_compound, "Compound"),
    ]
    for top_abs, label in order:
        explorer = TopExp_Explorer(topo_shape, top_abs)
        index = 0
        while explorer.More():
            current = explorer.Current()
            collected.extend(_flatten_subshape(current, f"{label}_{index}"))
            index += 1
            explorer.Next()
        if collected:
            log.debug(
                "OCC detail: TopExp fallback produced %d subshapes of type %s for %s",
                len(collected),
                label,
                _shape_type_name(topo_shape),
            )
            return collected

    log.debug("OCC detail: no subshapes discovered for %s", _shape_type_name(topo_shape))
    return []


def _setting_float(settings, key: str, default: float) -> float:
    """Return a float-valued mesh setting with dash/underscore tolerance."""
    if settings is None:
        return float(default)
    try:
        if hasattr(settings, "get"):
            value = settings.get(key)
        else:
            attr = key.replace("-", "_")
            value = getattr(settings, attr, None)
    except Exception:
        value = None
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def build_detail_mesh_payload(
    shape_obj: Any,
    settings,
    *,
    product=None,
    default_linear_def: float = _DEFAULT_LINEAR_DEF,
    default_angular_def: float = _DEFAULT_ANGULAR_DEF,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    material_resolver: Optional[Callable[[Any], Any]] = None,
    reference_shape: Optional[Any] = None,
    unit_scale: float = 1.0,
    canonical_map: Optional[Dict[int, Any]] = None,
    face_style_groups: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[OCCDetailMesh]:
    """Construct OCC face meshes for detail-mode geometry."""
    logref = logger or log
    if settings is None or not is_available():
        return None
    if shape_obj is None and product is None:
        return None

    linear_def = _setting_float(settings, "mesher-linear-deflection", default_linear_def)
    angular_def = _setting_float(settings, "mesher-angular-deflection", default_angular_def)
    try:
        angular_rad = math.radians(float(angular_def))
    except Exception:
        angular_rad = math.radians(default_angular_def)

    # Prepare canonical mapping data if available
    coord_to_vid: Optional[Dict[tuple, int]] = None
    tri_map: Optional[Dict[tuple, Dict[str, int]]] = None
    
    if canonical_map and "vertices" in canonical_map and "faces" in canonical_map:
        try:
            coord_to_vid, tri_map = _build_canonical_maps(canonical_map)
            if logref:
                logref.debug(
                    "OCC detail: prepared canonical map with %d verts and %d triangles for %s",
                    len(coord_to_vid),
                    len(tri_map),
                    getattr(product, "GlobalId", None),
                )
        except Exception as exc:
            if logref:
                logref.debug("OCC detail: failed to build canonical maps: %s", exc)

    def _run_occ_detail(source_obj, source_label: str) -> Optional[OCCDetailMesh]:
        if source_obj is None:
            return None
        detail = build_detail_mesh(
            source_obj,
            linear_deflection=float(linear_def),
            angular_deflection_rad=angular_rad,
            logger=logref,
            coord_to_vid=coord_to_vid,
            tri_map=tri_map,
        )
        if detail is None and logref:
            logref.debug(
                "Detail mode: OCC meshing returned no faces for %s source of %s (guid=%s).",
                source_label,
                _product_name(product) if product is not None else "<unknown>",
                getattr(product, "GlobalId", None) if product is not None else "<n/a>",
            )
        return detail

    def _attach_pre_xform(detail_mesh, primary_shape):
        if detail_mesh is None or product is None:
            return
        candidates: List[Any] = []
        if primary_shape is not None:
            candidates.append(primary_shape)
        if reference_shape is not None and reference_shape is not primary_shape:
            candidates.append(reference_shape)
        for candidate in candidates:
            try:
                from .process_ifc import resolve_absolute_matrix

                abs_mat = resolve_absolute_matrix(candidate, product)
            except Exception:
                abs_mat = None
            abs_inv = try_inv_4x4(abs_mat)
            if abs_inv is not None:
                setattr(detail_mesh, "pre_xform", abs_inv)
                if logref:
                    logref.debug("OCC detail: mesh normalized with inverse abs matrix (source=%s)", type(candidate).__name__)
                return
        if logref:
            logref.debug("OCC detail: abs matrix not invertible or unavailable; writing as-is")

    detail_mesh = None

    # If we need materials but lack a canonical map, prefer the representation-item approach.
    # This ensures we process each item individually (resolving its material) rather than
    # merging them into a monolithic shape where material assignments are lost.
    # This is critical for 'force-occ' workflows where the iterator mesh (and thus canonical map)
    # might be unavailable or unreliable.
    if detail_mesh is None and material_resolver is not None and not canonical_map and product is not None:
        detail_mesh = _build_representation_detail_mesh(
            product,
            settings,
            float(linear_def),
            angular_rad,
            logger=logref,
            detail_level=detail_level,
            material_resolver=material_resolver,
            unit_scale=unit_scale,
            face_style_groups=face_style_groups,
        )

    # Prefer product-local OCC so we stay in the product's native local frame.
    if detail_mesh is None and product is not None:
        product_occ_shape = _create_occ_product_shape(settings, product, logger=logref)
        detail_mesh = _run_occ_detail(product_occ_shape, "product")
        if detail_mesh is not None and product is not None:
             _attach_pre_xform(detail_mesh, product_occ_shape)

    # If that failed, fall back to iterator OCC and attach a normalization matrix if possible.
    if detail_mesh is None and shape_obj is not None:
        detail_mesh = _run_occ_detail(shape_obj, "iterator")
        if detail_mesh is not None and product is not None:
            _attach_pre_xform(detail_mesh, shape_obj)

    if detail_mesh is None and product is not None:
        detail_mesh = _build_representation_detail_mesh(
            product,
            settings,
            float(linear_def),
            angular_rad,
            logger=logref,
            detail_level=detail_level,
            material_resolver=material_resolver,
            unit_scale=unit_scale,
            face_style_groups=face_style_groups,
        )
        # Note: fallback representation mesh doesn't support canonical mapping yet as it processes items individually
        if detail_mesh is not None and logref:
            logref.debug(
                "OCC detail: fallback per-representation extraction succeeded for %s (guid=%s)",
                getattr(product, "is_a", lambda: type(product).__name__)(),
                getattr(product, "GlobalId", None),
            )
    if detail_mesh is None:
        target_name = type(shape_obj).__name__ if shape_obj is not None else "IfcProduct"
        if logref:
            logref.debug("OCC detail: no mesh generated for %s", target_name)
        return None

    if detail_level == "face":
        detail_mesh = _explode_to_faces(detail_mesh)

    subshape_count = len(getattr(detail_mesh, "subshapes", []) or [])
    face_total = detail_mesh.face_count if hasattr(detail_mesh, "face_count") else 0
    _scale_detail_mesh(detail_mesh, unit_scale)
    if logref:
        logref.debug(
            "OCC detail: generated %d subshape mesh(es) totalling %s faces for %s",
            subshape_count,
            face_total,
            type(shape_obj).__name__ if shape_obj is not None else "IfcProduct",
        )
    return detail_mesh


def precompute_detail_meshes(
    ifc_file,
    settings,
    *,
    default_linear_def: float = _DEFAULT_LINEAR_DEF,
    default_angular_def: float = _DEFAULT_ANGULAR_DEF,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    unit_scale: float = 1.0,
) -> Dict[int, OCCDetailMesh]:
    """Precompute OCC detail meshes for every product (detail-mode prepass)."""
    cache: Dict[int, OCCDetailMesh] = {}
    logref = logger or log
    if settings is None or not is_available():
        return cache
    products = ifc_file.by_type("IfcProduct") or []
    total = 0
    for product in products:
        total += 1
        try:
            product_id = int(product.id())
        except Exception:
            product_id = None
        if product_id is None:
            continue
        detail_mesh = build_detail_mesh_payload(
            None,
            settings,
            product=product,
            default_linear_def=default_linear_def,
            default_angular_def=default_angular_def,
            logger=logref,
            detail_level=detail_level,
            unit_scale=unit_scale,
        )
        if detail_mesh is not None:
            cache[product_id] = detail_mesh
    if logref:
        logref.info(
            "Detail prepass: cached OCC meshes for %d / %d product(s).",
            len(cache),
            total,
        )
    return cache


def _iter_representation_items(product) -> Iterator[Tuple[int, int, Any, Optional[np.ndarray]]]:
    """Iterate representation items, expanding IfcMappedItem recursively.
    
    Yields:
        (rep_index, item_index, item, transform_matrix)
    """
    representation = getattr(product, "Representation", None)
    if representation is None:
        return
    reps = getattr(representation, "Representations", None) or []
    
    def _recurse(item, current_xform=None):
        # If item is IfcMappedItem, expand it
        if item.is_a("IfcMappedItem"):
            source = getattr(item, "MappingSource", None)
            mapped_rep = getattr(source, "MappedRepresentation", None) if source else None
            if mapped_rep:
                # Calculate transform
                local_xform = _repmap_rt_matrix(item)
                new_xform = local_xform @ current_xform if current_xform is not None else local_xform
                
                mapped_items = getattr(mapped_rep, "Items", None) or []
                for sub_item in mapped_items:
                    yield from _recurse(sub_item, new_xform)
            return

        # Otherwise yield the item itself
        yield item, current_xform

    for rep_index, rep in enumerate(reps):
        if rep is None:
            continue
        items = getattr(rep, "Items", None) or []
        for item_index, item in enumerate(items):
            if item is None:
                continue
            
            # Expand mapped items
            for expanded_item, xform in _recurse(item):
                yield rep_index, item_index, expanded_item, xform


def _build_representation_detail_mesh(
    product,
    settings,
    linear_def: float,
    angular_rad: float,
    *,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    material_resolver: Optional[Callable[[Any], Any]] = None,
    unit_scale: float = 1.0,
    face_style_groups: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Fallback OCC extraction that tessellates each representation item directly."""
    representation = getattr(product, "Representation", None)
    if representation is None:
        return None
    item_meshes: List[OCCDetailMesh] = []
    face_offset = 0
    for rep_index, item_index, item, xform in _iter_representation_items(product):
        detail_mesh = _detail_mesh_for_item(
            item,
            settings,
            linear_def,
            angular_rad,
            product=product,
            rep_index=rep_index,
            item_index=item_index,
            logger=logger,
            detail_level=detail_level,
            material_resolver=material_resolver,
            unit_scale=unit_scale,
            transform=xform,
            face_style_groups=face_style_groups,
            face_offset=face_offset,
        )
        if detail_mesh is None:
            continue
        # Track cumulative face count for next item
        if hasattr(detail_mesh, 'face_count'):
            face_offset += detail_mesh.face_count
        item_meshes.append(detail_mesh)
    if not item_meshes:
        return None
    if len(item_meshes) == 1:
        _normalize_detail_indices(item_meshes[0])
        return item_meshes[0]
    return _merge_detail_meshes(item_meshes)


def _detail_mesh_for_item(
    item,
    settings,
    linear_def: float,
    angular_rad: float,
    *,
    product=None,
    rep_index: Optional[int] = None,
    item_index: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    material_resolver: Optional[Callable[[Any], Any]] = None,
    unit_scale: float = 1.0,
    transform: Optional[np.ndarray] = None,
):
    if item is None:
        return None
    occ_source = _create_occ_shape(
        settings,
        item,
        product=product,
        rep_index=rep_index,
        item_index=item_index,
        logger=logger,
    )
    if occ_source is None:
        return None
        
    # Apply transform if present (from IfcMappedItem expansion)
    if transform is not None:
        occ_source = _apply_transform_to_occ_shape(occ_source, transform)
        
    detail_mesh = build_detail_mesh(
        occ_source,
        linear_deflection=float(linear_def),
        angular_deflection_rad=float(angular_rad),
        logger=logger or log,
    )


def _describe_rep_item(item) -> str:
    if item is None:
        return "<None>"
    name = getattr(item, "Name", None)
    type_name = item.is_a() if hasattr(item, "is_a") else type(item).__name__
    try:
        step_id = item.id()
    except Exception:
        step_id = getattr(item, "id", lambda: None)()
    label = f"{type_name}"
    if name:
        label += f":{name}"
    if step_id is not None:
        label += f"#{step_id}"
    return label


def _product_name(product) -> str:
    if product is None:
        return "<product>"
    return getattr(product, "Name", None) or getattr(product, "GlobalId", None) or (
        product.is_a() if hasattr(product, "is_a") else "<product>"
    )


def _detail_mesh_for_item(
    item,
    settings,
    linear_def: float,
    angular_rad: float,
    *,
    product=None,
    rep_index: Optional[int] = None,
    item_index: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    material_resolver: Optional[Callable[[Any], Any]] = None,
    unit_scale: float = 1.0,
    transform: Optional[np.ndarray] = None,
    face_style_groups: Optional[Dict[str, Dict[str, Any]]] = None,
    face_offset: int = 0,
):
    if item is None:
        return None
    occ_source = _create_occ_shape(
        settings,
        item,
        product=product,
        rep_index=rep_index,
        item_index=item_index,
        logger=logger,
    )
    if occ_source is None:
        return None
        
    # Apply transform if present (from IfcMappedItem expansion)
    if transform is not None:
        occ_source = _apply_transform_to_occ_shape(occ_source, transform)
        
    detail_mesh = build_detail_mesh(
        occ_source,
        linear_deflection=float(linear_def),
        angular_deflection_rad=float(angular_rad),
        logger=logger or log,
    )
    material_key = material_resolver(item) if material_resolver else None
    
    # Build per-face material mapping from face_style_groups
    face_material_map: Optional[Dict[int, Any]] = None
    if face_style_groups:
        face_material_map = {}
        for token, group_entry in face_style_groups.items():
            material_obj = group_entry.get("material")
            faces = group_entry.get("faces", [])
            if material_obj and hasattr(material_obj, "name"):
                mat_token = material_obj.name
                for face_idx in faces:
                    # Map global face index to item-local face index
                    local_face_idx = face_idx - face_offset
                    # Include faces that belong to this item (face_idx >= face_offset)
                    # We'll check validity when we iterate through actual OCC faces
                    if local_face_idx >= 0:
                        face_material_map[local_face_idx] = mat_token
        
        if logger and face_material_map:
            logger.debug(
                "Detail mesh item: found %d faces with per-face materials for item rep=%s/%s (face_offset=%d, mapped_faces=%s)",
                len(face_material_map),
                rep_index,
                item_index,
                face_offset,
                sorted(face_material_map.keys())[:10],
            )
    
    if detail_mesh is None and logger:
        logger.debug(
            "Detail mode: OCC tessellation yielded no faces for product=%s step=%s item=%s rep=%s/%s",
            getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "<product>",
            getattr(product, "id", lambda: None)(),
            _describe_rep_item(item),
            rep_index,
            item_index,
        )
    elif detail_mesh is not None:
        _apply_material_key(detail_mesh, material_key, face_material_map=face_material_map)
        if detail_level == "face":
            detail_mesh = _explode_to_faces(detail_mesh)
        if product is not None and rep_index is not None and item_index is not None:
            _annotate_detail_subshapes(detail_mesh, product, rep_index, item_index, item, material_key=material_key)
        if logger:
            subshape_count = len(getattr(detail_mesh, "subshapes", None) or [])
            logger.debug(
                "Detail mode: OCC tessellation succeeded for product=%s step=%s item=%s rep=%s/%s subshapes=%d",
                getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "<product>",
                getattr(product, "id", lambda: None)(),
                _describe_rep_item(item),
                rep_index,
                item_index,
                subshape_count,
            )
        _scale_detail_mesh(detail_mesh, unit_scale)
    return detail_mesh


def detail_from_repmap_items(
    type_obj,
    settings,
    *,
    linear_def: float,
    angular_def_rad: float,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    material_resolver: Optional[Callable[[Any], Any]] = None,
    unit_scale: float = 1.0,
) -> Optional[OCCDetailMesh]:
    """Tessellate the type's representation maps (canonical frame) into an OCCDetailMesh."""
    repmaps = getattr(type_obj, "RepresentationMaps", None) or []
    if not repmaps:
        return None
    meshes: List[OCCDetailMesh] = []
    for rep_index, repmap in enumerate(repmaps):
        mapped = getattr(repmap, "MappedRepresentation", None)
        if mapped is None:
            continue
        for item_index, item in enumerate(getattr(mapped, "Items", None) or []):
            dm = _detail_mesh_for_item(
                item,
                settings,
                float(linear_def),
                float(angular_def_rad),
                product=None,
                rep_index=rep_index,
                item_index=item_index,
                logger=logger,
                detail_level=detail_level,
                material_resolver=material_resolver,
                unit_scale=unit_scale,
            )
            if dm is not None:
                meshes.append(dm)
    if not meshes:
        return None
    if len(meshes) == 1:
        _normalize_detail_indices(meshes[0])
        return meshes[0]
    return _merge_detail_meshes(meshes)


def build_canonical_detail_for_type(
    product,
    settings,
    *,
    default_linear_def: float = _DEFAULT_LINEAR_DEF,
    default_angular_def: float = _DEFAULT_ANGULAR_DEF,
    logger: Optional[logging.Logger] = None,
    detail_level: Literal["subshape", "face"] = "subshape",
    material_resolver: Optional[Callable[[Any], Any]] = None,
    unit_scale: float = 1.0,
) -> Optional[OCCDetailMesh]:
    """Return an OCC detail mesh in the type's canonical (repmap) frame."""
    type_obj = None
    for rel in getattr(product, "IsTypedBy", None) or []:
        candidate = getattr(rel, "RelatingType", None)
        if candidate is not None:
            type_obj = candidate
            break
    if type_obj is None:
        return None
    ang_rad = math.radians(float(default_angular_def))
    return detail_from_repmap_items(
        type_obj,
        settings,
        linear_def=float(default_linear_def),
        angular_def_rad=ang_rad,
        logger=logger or log,
        detail_level=detail_level,
        material_resolver=material_resolver,
        unit_scale=unit_scale,
    )


def _create_occ_shape(
    settings,
    element,
    *,
    product=None,
    rep_index: Optional[int] = None,
    item_index: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
):
    try:
        shape_obj = ifcopenshell.geom.create_shape(settings, element)
    except Exception as exc:
        step_id = None
        try:
            step_id = element.id()
        except Exception:
            step_id = getattr(element, "id", lambda: None)()
        if logger:
            logger.debug(
                "Detail OCC fallback: create_shape failed for product=%s step=%s item=%s rep=%s/%s error=%s",
                getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "<product>",
                getattr(product, "id", lambda: None)(),
                _describe_rep_item(element),
                rep_index,
                item_index,
                exc,
            )
        return None
    geometry = getattr(shape_obj, "geometry", shape_obj)
    if not geometry and logger:
        logger.debug(
            "Detail OCC fallback: create_shape returned empty geometry for product=%s item=%s rep=%s/%s",
            getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "<product>",
            _describe_rep_item(element),
            rep_index,
            item_index,
        )
    return geometry or shape_obj


def _create_occ_product_shape(settings, product, *, logger: Optional[logging.Logger] = None):
    try:
        shape_obj = ifcopenshell.geom.create_shape(
            settings,
            product,
        )
    except Exception as exc:
        if logger:
            logger.debug(
                "Detail OCC product fallback failed for guid=%s: %s",
                getattr(product, "GlobalId", None),
                exc,
            )
        return None
    geometry = getattr(shape_obj, "geometry", shape_obj)
    if not geometry and logger:
        logger.debug(
            "Detail OCC product fallback returned empty geometry for guid=%s",
            getattr(product, "GlobalId", None),
        )
    return geometry or shape_obj


def _annotate_detail_subshapes(detail_mesh, product, rep_index: int, item_index: int, item, material_key: Optional[Any] = None):
    """Tag OCC subshape labels so downstream USD meshes remain traceable."""
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    if not subshapes:
        return
    product_label = getattr(product, "Name", None) or getattr(product, "GlobalId", None) or product.is_a()
    item_name = (
        getattr(item, "Name", None)
        or getattr(item, "Description", None)
        or (item.is_a() if hasattr(item, "is_a") else f"Item{item_index}")
    )
    def _clean(text: str) -> str:
        try:
            import re as _re
            token = _re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_") or "Part"
            if len(token) > 64:
                token = f"{token[:48]}_{hash(token) & 0xFFFF:04x}"
            return token
        except Exception:
            return str(text) if text else "Part"

    prefix_core = f"{_clean(product_label)}_rep{rep_index}_item{item_index}_{_clean(item_name)}"
    for local_index, subshape in enumerate(subshapes):
        current = getattr(subshape, "label", f"Subshape_{local_index}")
        shape_type = getattr(subshape, "shape_type", None)
        if shape_type:
            current = f"{_clean(shape_type)}_{current}"
        subshape.label = f"{prefix_core}_{current}"
        if material_key is not None and getattr(subshape, "material_key", None) is None:
            subshape.material_key = material_key
        for face in getattr(subshape, "faces", None) or []:
            if material_key is not None and getattr(face, "material_key", None) is None:
                face.material_key = material_key


def _apply_material_key(detail_mesh, material_key: Optional[Any], face_material_map: Optional[Dict[int, Any]] = None) -> None:
    """Apply material keys to mesh faces.
    
    Args:
        detail_mesh: The OCCDetailMesh to update
        material_key: Single material key to apply to all faces (if face_material_map is None)
        face_material_map: Optional mapping of face indices to material keys for per-face assignment
    """
    if material_key is None and not face_material_map:
        return
    
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    face_index = 0
    
    for subshape in subshapes:
        if getattr(subshape, "material_key", None) is None and material_key is not None:
            subshape.material_key = material_key
        
        for face in getattr(subshape, "faces", None) or []:
            if face_material_map and face_index in face_material_map:
                # Use per-face material assignment from mapping
                face.material_key = face_material_map[face_index]
            elif getattr(face, "material_key", None) is None and material_key is not None:
                # Fall back to single material key
                face.material_key = material_key
            face_index += 1


def _normalize_detail_indices(detail_mesh):
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    for idx, subshape in enumerate(subshapes):
        subshape.index = idx


def _merge_detail_meshes(meshes: List[OCCDetailMesh]) -> Optional[OCCDetailMesh]:
    combined: List[OCCSubshapeMesh] = []
    for mesh in meshes:
        for subshape in getattr(mesh, "subshapes", None) or []:
            label = getattr(subshape, "label", None) or f"Subshape_{len(combined)}"
            combined.append(
                OCCSubshapeMesh(
                    index=len(combined),
                    label=label,
                    shape_type=getattr(subshape, "shape_type", "UNKNOWN"),
                    faces=list(getattr(subshape, "faces", []) or []),
                    material_key=getattr(subshape, "material_key", None),
                )
            )
    if not combined:
        return None
    return OCCDetailMesh(shape=None, subshapes=combined)


def _explode_to_faces(detail_mesh: OCCDetailMesh) -> OCCDetailMesh:
    """Return a copy of detail mesh where each face is exposed as its own subshape."""
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    exploded: List[OCCSubshapeMesh] = []
    for parent in subshapes:
        faces = getattr(parent, "faces", None) or []
        for face in faces:
            label = getattr(parent, "label", "Subshape")
            face_material = getattr(face, "material_key", None)
            subshape_material = getattr(parent, "material_key", None)
            if face_material is None and subshape_material is not None:
                face.material_key = subshape_material
                face_material = subshape_material
            exploded.append(
                OCCSubshapeMesh(
                    index=len(exploded),
                    label=f"{label}_face{face.face_index}",
                    shape_type="Face",
                    faces=[face],
                    material_key=face_material or subshape_material,
                )
            )
    if not exploded:
        return detail_mesh
    return OCCDetailMesh(shape=detail_mesh.shape, subshapes=exploded)


def mesh_from_detail_mesh(detail_mesh: Optional[OCCDetailMesh]) -> Optional[Dict[str, Any]]:
    """Flatten OCC detail meshes into the standard vertex/face dict used by proto pipeline."""
    if detail_mesh is None:
        return None
    faces = getattr(detail_mesh, "faces", None) or []
    if not faces:
        subshapes = getattr(detail_mesh, "subshapes", None) or []
        for entry in subshapes:
            faces.extend(getattr(entry, "faces", None) or [])
        if not faces:
            return None
    vert_arrays: List[np.ndarray] = []
    face_arrays: List[np.ndarray] = []
    material_keys: List[Any] = []
    offset = 0
    for face in faces:
        verts = np.asarray(getattr(face, "vertices", None), dtype=np.float64)
        tris = np.asarray(getattr(face, "faces", None), dtype=np.int64)
        if verts.size == 0 or tris.size == 0:
            continue
        vert_arrays.append(verts)
        face_arrays.append(tris + offset)
        
        # Store material key for each face (triangle)
        # We need to replicate the key for each triangle in this face mesh
        mat_key = getattr(face, "material_key", None)
        num_tris = tris.shape[0]
        material_keys.extend([mat_key] * num_tris)
        
        offset += verts.shape[0]
    if not vert_arrays or not face_arrays:
        return None
    try:
        vertices = np.vstack(vert_arrays)
        faces_arr = np.vstack(face_arrays)
    except Exception:
        return None
    pre_xform = getattr(detail_mesh, "pre_xform", None)
    if pre_xform is not None:
        try:
            vertices = _apply_matrix_to_vertices(vertices, pre_xform)
        except Exception:
            pass
            
    # Extract integer material IDs for standard pipeline compatibility
    material_ids = []
    for k in material_keys:
        if isinstance(k, dict):
            # Use -1 for missing/invalid material ID
            material_ids.append(int(k.get("material_id", -1)))
        elif isinstance(k, int):
            material_ids.append(k)
        else:
            material_ids.append(-1)
            
    return {
        "vertices": vertices, 
        "faces": faces_arr, 
        "material_keys": material_keys,
        "material_ids": material_ids
    }


def proxy_requires_high_detail(stats: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Return True when a proxy-like mesh should request OCC remeshing."""
    reasons: List[str] = []
    face_count = int(stats.get("face_count", 0))
    if face_count <= 0:
        return False, reasons
    shortest = float(stats.get("shortest", 0.0))
    longest = float(stats.get("longest", 0.0))
    diagonal = float(stats.get("diagonal", 0.0))
    area_per_face = float(stats.get("area_per_face", 0.0))

    thin = shortest <= _PROXY_SHORT_DIM_THRESHOLD
    slender = shortest > 0.0 and longest / max(shortest, 1e-6) >= _PROXY_SLENDER_RATIO
    small = diagonal <= _PROXY_SMALL_DIAGONAL
    sparse = face_count <= _PROXY_FACE_COUNT_THRESHOLD or area_per_face >= _PROXY_AREA_PER_FACE_THRESHOLD

    if thin and slender:
        reasons.append(
            f"slender (shortest={shortest:.3f}m longest={longest:.3f}m)"
        )
    if small and sparse:
        reasons.append(
            f"small_sparse (diag={diagonal:.3f}m faces={face_count} area/face={area_per_face:.4f})"
        )

    return (len(reasons) > 0), reasons
