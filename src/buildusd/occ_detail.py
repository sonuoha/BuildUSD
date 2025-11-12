"""OpenCASCADE helpers for detail-mode tessellation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import ifcopenshell

try:
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Shape, topods_Face
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool
    from ifcopenshell.geom import occ_utils as _occ_utils

    _HAVE_OCC = True
except Exception:  # pragma: no cover - optional dependency
    BRepMesh_IncrementalMesh = None  # type: ignore
    TopAbs_FACE = None  # type: ignore
    TopAbs_SOLID = None  # type: ignore
    TopAbs_COMPSOLID = None  # type: ignore
    TopAbs_COMPOUND = None  # type: ignore
    TopAbs_SHELL = None  # type: ignore
    TopExp_Explorer = None  # type: ignore
    TopoDS_Shape = None  # type: ignore
    topods_Face = None  # type: ignore
    TopLoc_Location = None  # type: ignore
    BRep_Tool = None  # type: ignore
    _occ_utils = None  # type: ignore
    _HAVE_OCC = False

log = logging.getLogger(__name__)

__all__ = [
    "OCCFaceMesh",
    "OCCSubshapeMesh",
    "OCCDetailMesh",
    "is_available",
    "build_detail_mesh",
    "build_detail_mesh_payload",
    "precompute_detail_meshes",
    "proxy_requires_high_detail",
    "mesh_from_detail_mesh",
]

_DEFAULT_LINEAR_DEF = 0.005  # metres
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


@dataclass
class OCCSubshapeMesh:
    """Aggregated mesh info for a logical OCC sub-shape (solid/shell)."""

    index: int
    label: str
    shape_type: str
    faces: List[OCCFaceMesh]


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
    """
    Return True when pythonocc + IfcOpenShell occ_utils are importable.
    Uses a live check so it doesn't depend on earlier import order.
    """
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # noqa: F401
        from ifcopenshell.geom import occ_utils  # noqa: F401
        return True
    except Exception:
        return False


def why_unavailable() -> str:
    """Return a short reason if OCC detail isn't available."""
    try:
        from OCC.Core import BRepMesh as _chk_mesh_mod  # noqa: F401
    except Exception as exc:
        return f"OCC.Core import failed: {exc!r}"
    try:
        from ifcopenshell.geom import occ_utils as _chk_occ_utils  # noqa: F401
    except Exception as exc:
        return f"ifcopenshell.geom.occ_utils import failed: {exc!r}"
    return "Unknown: imports succeed here; earlier cached state may have been False"



def build_detail_mesh(
    shape_obj: Any,
    *,
    linear_deflection: float,
    angular_deflection_rad: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[OCCDetailMesh]:
    """Return OCC meshes grouped by logical sub-shapes, when OCC is available."""
    if not is_available() or shape_obj is None:
        return None
    topo_shape = _extract_topods_shape(shape_obj, logger=logger)
    if topo_shape is None:
        return None
    linear_tol = max(float(linear_deflection), 1e-6)
    angular_tol = max(float(angular_deflection_rad), math.radians(0.1))
    if not _perform_meshing(topo_shape, linear_tol, angular_tol, logger=logger):
        return None
    subshape_shapes = _primary_subshape_list(topo_shape)
    subshape_meshes: List[OCCSubshapeMesh] = []
    face_counter = 0
    for idx, (label, subshape) in enumerate(subshape_shapes):
        faces, face_counter = _collect_face_meshes(subshape, face_counter)
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
        faces, _ = _collect_face_meshes(topo_shape, 0)
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
    global _occ_utils
    if _occ_utils is None:
        try:
            from ifcopenshell.geom import occ_utils as _occ_utils  # type: ignore  # noqa: F401
        except Exception as exc:
            if logger:
                logger.debug("OCC detail: occ_utils import failed during extraction (%s)", exc)
            return None
    if _is_topods(shape_obj):
        return shape_obj
    for candidate in _iter_occ_candidates(shape_obj):
        try:
            result = _occ_utils.create_shape_from_serialization(candidate)  # type: ignore[arg-type]
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
    if not is_available() or TopoDS_Shape is None:
        return False
    return isinstance(candidate, TopoDS_Shape)


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


def _perform_meshing(
    topo_shape: Any,
    linear_deflection: float,
    angular_deflection: float,
    *,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Run the OCCT mesher once so all faces carry triangulations."""
    try:
        mesher = BRepMesh_IncrementalMesh(
            topo_shape,
            float(linear_deflection),
            False,
            float(angular_deflection),
            True,
        )
        if hasattr(mesher, "Perform"):
            mesher.Perform()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.debug("BRepMesh_IncrementalMesh failed: %s", exc)
        return False


def _collect_face_meshes(target_shape: Any, start_index: int) -> tuple[List[OCCFaceMesh], int]:
    """Collect OCCFaceMesh entries for every face in ``target_shape``."""
    faces: List[OCCFaceMesh] = []
    face_index = start_index
    explorer = TopExp_Explorer(target_shape, TopAbs_FACE)
    while explorer.More():
        try:
            face = topods_Face(explorer.Current())
        except Exception:
            explorer.Next()
            face_index += 1
            continue
        mesh = _triangulate_face(face)
        if mesh is not None:
            mesh.face_index = face_index
            faces.append(mesh)
        explorer.Next()
        face_index += 1
    return faces, face_index


def _triangulate_face(face: Any) -> Optional[OCCFaceMesh]:
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)
    if triangulation is None:
        return None
    nodes = triangulation.Nodes()
    triangles = triangulation.Triangles()
    if nodes is None or triangles is None or nodes.Length() == 0 or triangles.Length() == 0:
        return None
    use_transform = hasattr(loc, "IsIdentity") and not loc.IsIdentity()
    transform = loc.Transformation() if use_transform else None
    vertices = np.zeros((nodes.Length(), 3), dtype=np.float64)
    for i in range(1, nodes.Length() + 1):
        point = nodes.Value(i)
        if transform is not None:
            point = point.Transformed(transform)
        vertices[i - 1, :] = (point.X(), point.Y(), point.Z())
    faces_arr = np.zeros((triangles.Length(), 3), dtype=np.int32)
    for tri_idx in range(1, triangles.Length() + 1):
        tri = triangles.Value(tri_idx)
        try:
            i1, i2, i3 = tri.Get()
        except Exception:  # pragma: no cover - compatibility guard
            indices = [0, 0, 0]
            tri.Get(indices)  # type: ignore[arg-type]
            i1, i2, i3 = indices
        faces_arr[tri_idx - 1, :] = (int(i1) - 1, int(i2) - 1, int(i3) - 1)
    return OCCFaceMesh(face_index=0, vertices=vertices, faces=faces_arr)


def _primary_subshape_list(topo_shape: Any) -> List[tuple[str, Any]]:
    """Return a list of first-level subshapes (prefer solids, then shells, etc.)."""
    collected: List[tuple[str, Any]] = []
    if hasattr(_occ_utils, "yield_subshapes"):
        try:
            for index, subshape in enumerate(_occ_utils.yield_subshapes(topo_shape)):
                shape_name = _shape_type_name(subshape)
                if shape_name not in _DETAIL_TYPE_WHITELIST:
                    continue
                collected.append((f"{shape_name}_{index}", subshape))
        except Exception as exc:
            log.debug("OCC detail: yield_subshapes failed (%s); falling back to TopExp iterator.", exc)
    if collected:
        log.debug(
            "OCC detail: yield_subshapes produced %d subshapes for %s",
            len(collected),
            _shape_type_name(topo_shape),
        )
        return collected

    order = [
        (TopAbs_SOLID, "Solid"),
        (TopAbs_COMPSOLID, "CompositeSolid"),
        (TopAbs_SHELL, "Shell"),
        (TopAbs_COMPOUND, "Compound"),
    ]
    for top_abs, label in order:
        explorer = TopExp_Explorer(topo_shape, top_abs)
        index = 0
        while explorer.More():
            current = explorer.Current()
            collected.append((f"{label}_{index}", current))
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

    def _run_occ_detail(source_obj, source_label: str) -> Optional[OCCDetailMesh]:
        if source_obj is None:
            return None
        detail = build_detail_mesh(
            source_obj,
            linear_deflection=float(linear_def),
            angular_deflection_rad=angular_rad,
            logger=logref,
        )
        if detail is None and logref:
            logref.debug(
                "Detail mode: OCC meshing returned no faces for %s source of %s (guid=%s).",
                source_label,
                _product_name(product) if product is not None else "<unknown>",
                getattr(product, "GlobalId", None) if product is not None else "<n/a>",
            )
        return detail

    detail_mesh = None
    if shape_obj is not None:
        detail_mesh = _run_occ_detail(shape_obj, "iterator")

    if detail_mesh is None and product is not None:
        product_occ_shape = _create_occ_product_shape(settings, product, logger=logref)
        detail_mesh = _run_occ_detail(product_occ_shape, "product")

    if detail_mesh is None and product is not None:
        detail_mesh = _build_representation_detail_mesh(
            product,
            settings,
            float(linear_def),
            angular_rad,
            logger=logref,
        )
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

    subshape_count = len(getattr(detail_mesh, "subshapes", []) or [])
    face_total = detail_mesh.face_count if hasattr(detail_mesh, "face_count") else 0
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


def _build_representation_detail_mesh(
    product,
    settings,
    linear_def: float,
    angular_rad: float,
    *,
    logger: Optional[logging.Logger] = None,
):
    """Fallback OCC extraction that tessellates each representation item directly."""
    representation = getattr(product, "Representation", None)
    if representation is None:
        return None
    item_meshes: List[OCCDetailMesh] = []
    for rep_index, item_index, item in _iter_representation_items(product):
        detail_mesh = _detail_mesh_for_item(
            item,
            settings,
            linear_def,
            angular_rad,
            product=product,
            rep_index=rep_index,
            item_index=item_index,
            logger=logger,
        )
        if detail_mesh is None:
            continue
        _annotate_detail_subshapes(detail_mesh, product, rep_index, item_index, item)
        item_meshes.append(detail_mesh)
    if not item_meshes:
        return None
    if len(item_meshes) == 1:
        _normalize_detail_indices(item_meshes[0])
        return item_meshes[0]
    return _merge_detail_meshes(item_meshes)


def _iter_representation_items(product):
    representation = getattr(product, "Representation", None)
    if representation is None:
        return
    reps = getattr(representation, "Representations", None) or []
    for rep_index, rep in enumerate(reps):
        if rep is None:
            continue
        items = getattr(rep, "Items", None) or []
        for item_index, item in enumerate(items):
            if item is None:
                continue
            yield rep_index, item_index, item


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
    detail_mesh = build_detail_mesh(
        occ_source,
        linear_deflection=float(linear_def),
        angular_deflection_rad=float(angular_rad),
        logger=logger or log,
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
    elif detail_mesh is not None and logger:
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
    return detail_mesh


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


def _annotate_detail_subshapes(detail_mesh, product, rep_index: int, item_index: int, item):
    """Tag OCC subshape labels so downstream USD meshes remain traceable."""
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    if not subshapes:
        return
    product_label = getattr(product, "GlobalId", None) or getattr(product, "Name", None) or product.is_a()
    item_name = getattr(item, "Name", None) or (item.is_a() if hasattr(item, "is_a") else f"Item{item_index}")
    prefix = f"{product_label}_rep{rep_index}_item{item_index}_{item_name}"
    for local_index, subshape in enumerate(subshapes):
        current = getattr(subshape, "label", f"Subshape_{local_index}")
        subshape.label = f"{prefix}_{current}"


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
                )
            )
    if not combined:
        return None
    return OCCDetailMesh(shape=None, subshapes=combined)


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
    offset = 0
    for face in faces:
        verts = np.asarray(getattr(face, "vertices", None), dtype=np.float64)
        tris = np.asarray(getattr(face, "faces", None), dtype=np.int64)
        if verts.size == 0 or tris.size == 0:
            continue
        vert_arrays.append(verts)
        face_arrays.append(tris + offset)
        offset += verts.shape[0]
    if not vert_arrays or not face_arrays:
        return None
    try:
        vertices = np.vstack(vert_arrays)
        faces_arr = np.vstack(face_arrays)
    except Exception:
        return None
    return {"vertices": vertices, "faces": faces_arr}


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
