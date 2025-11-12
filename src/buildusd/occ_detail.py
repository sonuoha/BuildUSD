"""OpenCASCADE helpers for detail-mode tessellation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import numpy as np

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
]


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
