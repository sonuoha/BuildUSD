"""USD authoring helpers for the buildusd pipeline.

The functions below consume the prototype/instance caches produced by
``process_ifc`` and emit the various USD layers (prototypes, materials,
instances, 2D geometry) expected by ``main.py``.  The code mirrors the behaviour
of the original project but has been pared back to the essentials required by
the new iterator-driven workflow.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from textwrap import dedent
from contextlib import nullcontext
import fnmatch
import hashlib
import json
import re
import logging
import weakref
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .pxr_utils import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from .config.manifest import BasePointConfig, GeodeticCoordinate
from .io_utils import join_path, path_stem, write_text, is_omniverse_path, stat_entry
from .process_ifc import ConversionOptions, CurveWidthRule, InstanceRecord, PrototypeCaches, PrototypeKey
from .utils.matrix_utils import np_to_gf_matrix, scale_matrix_translation_only

try:
    from .ifc_visuals import PBRMaterial
except Exception:  # pragma: no cover - optional dependency safety
    PBRMaterial = None  # type: ignore

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
LOG = logging.getLogger(__name__)

try:
    from pyproj import CRS, Transformer
    _HAVE_PYPROJ = True
    _TRANSFORMER_CACHE: Dict[Tuple[str, str], Transformer] = {}
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYPROJ = False
    _TRANSFORMER_CACHE: Dict[Tuple[str, str], Any] = {}


def _unit_factor(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    return _UNIT_FACTORS.get(str(unit).strip().lower(), 1.0)


def _base_point_components_in_meters(bp: BasePointConfig) -> Tuple[float, float, float, str]:
    unit = bp.unit or "m"
    factor = _unit_factor(unit)
    easting = float(bp.easting) * factor
    northing = float(bp.northing) * factor
    height = float(bp.height) * factor
    return easting, northing, height, unit


def _derive_lonlat_from_base_point(bp: BasePointConfig, projected_crs: Optional[str]) -> Optional[GeodeticCoordinate]:
    if not _HAVE_PYPROJ:
        return None
    source_crs = bp.epsg or projected_crs
    if not source_crs:
        return None
    try:
        key = (source_crs, "EPSG:4326")
        transformer = _TRANSFORMER_CACHE.get(key)
        if transformer is None:
            src = CRS.from_user_input(source_crs)
            dst = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src, dst, always_xy=True)
            _TRANSFORMER_CACHE[key] = transformer
        easting_m, northing_m, height_m, _ = _base_point_components_in_meters(bp)
        longitude, latitude = transformer.transform(easting_m, northing_m)
        return GeodeticCoordinate(longitude=float(longitude), latitude=float(latitude), height=float(height_m))
    except Exception as exc:  # pragma: no cover - logging only
        LOG.debug("Unable to derive lon/lat for CRS %s: %s", source_crs, exc)
        return None

# ---------------- Name helpers ----------------

def sanitize_name(raw_name, fallback=None):
    """Make a USD-legal prim name (deterministic; no time-based suffix)."""
    base = str(raw_name or fallback or "Unnamed")
    try:
        base = base.encode("ascii", "ignore").decode("ascii")
    except Exception:
        base = str(fallback or "Unnamed")
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

    existing_layer = Sdf.Layer.Find(identifier)
    if existing_layer is not None:
        stage = Usd.Stage.Open(existing_layer)
        if stage is None:
            stage = Usd.Stage.CreateNew(identifier)
        else:
            stage.GetRootLayer().Clear()
    else:
        stage = Usd.Stage.CreateNew(identifier)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", float(meters_per_unit))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    return stage


# ---------------- Mesh helpers ----------------

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


def _reindex_for_uv_seams(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    face_vertex_indices: np.ndarray,
    uv_per_corner: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Duplicate vertices at UV seams using vectorized operations."""
    # Combine vertex index and UV coordinates into a unique key for each corner
    # We use a structured array or lexsort to find unique combinations
    
    # Create a composite key array: (original_index, u, v)
    # We need to ensure float UVs are handled correctly (precision issues might occur, but typically exact match is needed for seams)
    
    # Stack data to form rows of [original_index, u, v]
    # original_index is int, u,v are floats. We can view all as float64 or similar for sorting if indices fit.
    # Safer to use structured array or lexsort.
    
    # Let's use lexsort.
    # keys: v, u, original_index
    u = uv_per_corner[:, 0]
    v = uv_per_corner[:, 1]
    orig_idx = face_vertex_indices
    
    # Lexsort keys (last key is primary)
    # We want to group by (orig_idx, u, v)
    # So we sort by v, then u, then orig_idx
    sort_order = np.lexsort((v, u, orig_idx))
    
    # Apply sort
    sorted_orig_idx = orig_idx[sort_order]
    sorted_u = u[sort_order]
    sorted_v = v[sort_order]
    
    # Find unique changes
    # A row is different if any of orig_idx, u, v changed
    diff_orig = np.diff(sorted_orig_idx) != 0
    diff_u = np.diff(sorted_u) != 0
    diff_v = np.diff(sorted_v) != 0
    
    # Combine diffs (boolean OR)
    diff_mask = diff_orig | diff_u | diff_v
    
    # Prepend True for the first element
    unique_mask = np.concatenate(([True], diff_mask))
    
    # These are the unique vertices we need to create
    # The number of unique vertices is the sum of unique_mask
    num_unique = np.sum(unique_mask)
    
    # We need to map from the sorted position to the new unique index
    # cumsum of unique_mask gives the new index for each sorted entry (0-based if we subtract 1)
    new_indices_sorted = np.cumsum(unique_mask) - 1
    
    # Now we need to map back to the original order
    # We can create an array that maps 'sort_order' back to original positions
    # Or simply scatter 'new_indices_sorted' back to their original positions
    new_indices = np.empty_like(face_vertex_indices)
    new_indices[sort_order] = new_indices_sorted
    
    # Extract the unique data to form the new vertex arrays
    # We can just take the values at the unique positions in the sorted arrays
    unique_orig_indices = sorted_orig_idx[unique_mask]
    unique_uvs = np.column_stack((sorted_u[unique_mask], sorted_v[unique_mask]))
    
    new_points = points[unique_orig_indices]
    
    new_normals = None
    if normals is not None:
        new_normals = normals[unique_orig_indices]
        
    return new_points, new_normals, new_indices, unique_uvs


def _material_name_from_entry(entry: Any) -> Optional[str]:
    if entry is None:
        return None
    if PBRMaterial is not None and isinstance(entry, PBRMaterial):
        return entry.name
    if isinstance(entry, dict):
        for key in ("name", "Name", "label", "Label", "description", "Description"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for attr in ("Name", "name", "Description", "ElementName", "Label"):
        if hasattr(entry, attr):
            value = getattr(entry, attr)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _material_name_for_index(index: int, materials: Optional[Sequence[Any]]) -> Optional[str]:
    if materials is None:
        return None
    entry: Any = None
    if isinstance(materials, dict):
        entry = materials.get(index)
        if entry is None:
            entry = materials.get(str(index))
    else:
        try:
            entry = materials[index]
        except Exception:
            entry = None
    return _material_name_from_entry(entry)


def _material_name_from_style_entry(
    entry: Optional[Dict[str, Any]],
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    if not entry:
        return fallback
    for key in ("name", "label", "id", "guid", "shapeAspect", "style_id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    material_obj = entry.get("material")
    name = _material_name_from_entry(material_obj)
    if name:
        return name
    return fallback


def _material_identifier_for_index(index: int, materials: Optional[Sequence[Any]]) -> Any:
    if not materials:
        return index
    try:
        entry = materials[index]
    except Exception:
        return index
    label = _material_name_from_entry(entry)
    if label:
        return (index, label)
    return index


def _subset_token_for_material(material_identifier: Any) -> str:
    """Return a USD-legal prim token for a material/style identifier."""
    if isinstance(material_identifier, tuple) and material_identifier:
        idx = material_identifier[0]
        label = material_identifier[1] if len(material_identifier) > 1 else None
        base = _subset_token_for_material(int(idx))
        label_token = sanitize_name(label, fallback="") if label else ""
        return f"{base}_{label_token}" if label_token else base
    if isinstance(material_identifier, int):
        index = int(material_identifier)
        if index >= 0:
            return f"Material_{index}"
        return f"Material_neg{abs(index)}"
    text = str(material_identifier)
    sanitized = sanitize_name(text, fallback="Style")
    if not sanitized:
        sanitized = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:8]
    return sanitized


def _material_index_from_subset_token(token: str) -> Optional[int]:
    """Recover a material index from a subset token, if it encodes one."""
    if not token.startswith("Material"):
        return None
    match = re.match(r"Material_(neg)?(\d+)", token)
    if not match:
        return None
    value = int(match.group(2))
    if match.group(1):
        value = -value
    return value


def _prepare_mesh_payload(mesh_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not mesh_data:
        return None

    try:
        verts_np = np.asarray(mesh_data.get("vertices"), dtype=np.float32)
        faces_np = np.asarray(mesh_data.get("faces"), dtype=np.int64)
    except Exception as exc:
        LOG.debug("Mesh payload missing vertices/faces: %s", exc)
        return None

    if verts_np.size == 0 or faces_np.size == 0:
        return None

    try:
        verts_np = verts_np.reshape(-1, 3)
    except ValueError:
        LOG.debug("Mesh vertices could not be reshaped to Nx3; skipping mesh.")
        return None

    if faces_np.ndim == 1:
        if len(faces_np) % 3 != 0:
            LOG.debug("Face index array is not divisible by 3; skipping mesh.")
            return None
        faces_np = faces_np.reshape(-1, 3)
    elif faces_np.ndim != 2 or faces_np.shape[1] < 3:
        LOG.debug("Unsupported face array shape %s; skipping mesh.", faces_np.shape)
        return None

    face_vertex_counts = np.full(faces_np.shape[0], faces_np.shape[1], dtype=np.int32)
    face_vertex_indices = faces_np.reshape(-1).astype(np.int32)
    points = verts_np

    normals_interp: Optional[str] = None
    normals_values: Optional[np.ndarray] = None
    normals_src = mesh_data.get("normals")
    if normals_src is not None:
        try:
            normals_np = np.asarray(normals_src, dtype=np.float32).reshape(-1, 3)
            if normals_np.shape[0] == verts_np.shape[0]:
                normals_values = normals_np
                normals_interp = UsdGeom.Tokens.vertex
        except Exception:
            LOG.debug("Unable to process normals; continuing without them.", exc_info=True)

    def _first_available(*names: str) -> Optional[Any]:
        for name in names:
            if name in mesh_data and mesh_data[name] is not None:
                return mesh_data[name]
        return None

    uv_values = _first_available("uvs", "uv_coords", "uvcoordinates", "uv", "texcoords")
    uv_indices = _first_available("uv_indices", "uv_index", "uvs_indices", "uvfaces", "uv_faces")
    st_values: Optional[np.ndarray] = None
    st_interp: Optional[str] = None

    if uv_values is not None:
        try:
            uv_np = np.asarray(uv_values, dtype=np.float32)
            if uv_np.ndim == 1:
                if uv_np.size % 2 != 0:
                    raise ValueError("Flattened UV array length must be even.")
                uv_np = uv_np.reshape(-1, 2)
            elif uv_np.ndim >= 2:
                uv_np = uv_np.reshape(-1, uv_np.shape[-1])[:, :2]
            uv_per_corner: Optional[np.ndarray] = None
            if uv_indices is not None:
                uv_idx_np = np.asarray(uv_indices, dtype=np.int64).flatten()
                if uv_idx_np.size == face_vertex_indices.size:
                    uv_per_corner = uv_np[uv_idx_np]
            else:
                if uv_np.shape[0] == face_vertex_indices.size:
                    uv_per_corner = uv_np
                elif uv_np.shape[0] == verts_np.shape[0]:
                    st_values = uv_np
                    st_interp = UsdGeom.Tokens.vertex

            if uv_per_corner is not None:
                normals_for_reindex = normals_values if normals_interp == UsdGeom.Tokens.vertex else None
                new_points, new_normals, new_indices, new_st = _reindex_for_uv_seams(
                    points, normals_for_reindex, face_vertex_indices, uv_per_corner
                )
                points = new_points
                face_vertex_indices = new_indices
                face_vertex_counts = np.full(new_indices.size // 3, 3, dtype=np.int32)
                st_values = new_st
                st_interp = UsdGeom.Tokens.vertex
                if new_normals is not None:
                    normals_values = new_normals
                    normals_interp = UsdGeom.Tokens.vertex
        except Exception:
            LOG.debug("Unable to process UV data; continuing without primvars:st.", exc_info=True)

    return {
        "points": points,
        "face_vertex_indices": face_vertex_indices,
        "face_vertex_counts": face_vertex_counts,
        "normals": normals_values,
        "normals_interpolation": normals_interp,
        "st": st_values,
        "st_interpolation": st_interp,
    }


def _collect_material_subsets(imageable: UsdGeom.Imageable, family_name: str) -> List[UsdGeom.Subset]:
    # Prefer the filtered API when available (USD 24/25)
    try:
        subsets = UsdGeom.Subset.GetGeomSubsets(
            imageable,
            elementType=UsdGeom.Tokens.face,
            familyName=family_name,
        )
        if subsets:
            return list(subsets)
    except Exception:
        pass

    try:
        subsets = UsdGeom.Subset.GetAllGeomSubsets(imageable)
    except Exception:
        return []
    result: List[UsdGeom.Subset] = []
    for subset in subsets:
        try:
            fam_attr = subset.GetFamilyNameAttr()
            if not fam_attr:
                continue
            value = fam_attr.Get()
            if value is not None and str(value) == family_name:
                result.append(subset)
        except Exception:
            continue
    return result


def _compute_face_components(
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
) -> Tuple[List[Tuple[int, ...]], List[Set[int]]]:
    """Return per-face vertex tuples and adjacency sets."""
    faces: List[Tuple[int, ...]] = []
    adjacency: List[Set[int]] = []
    offset = 0
    edge_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for face_index, count in enumerate(face_vertex_counts):
        verts = tuple(int(v) for v in face_vertex_indices[offset : offset + count])
        offset += count
        faces.append(verts)
        adjacency.append(set())
        if count < 2:
            continue
        for i in range(count):
            a = verts[i]
            b = verts[(i + 1) % count]
            edge = (a, b) if a <= b else (b, a)
            edge_map[edge].append(face_index)
    for connected_faces in edge_map.values():
        if len(connected_faces) < 2:
            continue
        for i in range(len(connected_faces)):
            a = connected_faces[i]
            for j in range(i + 1, len(connected_faces)):
                b = connected_faces[j]
                adjacency[a].add(b)
                adjacency[b].add(a)
    return faces, adjacency


def _face_area_and_normal(points: Sequence[Tuple[float, float, float]], face_vertices: Sequence[int]) -> Tuple[float, Tuple[float, float, float]]:
    """Return (area, normal) for a polygon (fan triangulation)."""
    count = len(face_vertices)
    if count < 3:
        return 0.0, (0.0, 0.0, 0.0)
    p0 = np.array(points[face_vertices[0]], dtype=np.float64)
    area = 0.0
    normal = np.zeros(3, dtype=np.float64)
    for i in range(1, count - 1):
        p1 = np.array(points[face_vertices[i]], dtype=np.float64)
        p2 = np.array(points[face_vertices[i + 1]], dtype=np.float64)
        cross = np.cross(p1 - p0, p2 - p0)
        tri_area = 0.5 * np.linalg.norm(cross)
        area += tri_area
        if tri_area > 0.0:
            normal += cross
    norm = np.linalg.norm(normal)
    if norm > 0.0:
        normal /= norm
    return float(area), (float(normal[0]), float(normal[1]), float(normal[2]))


def _prepare_face_geometry(
    points: Sequence[Tuple[float, float, float]],
    face_vertices: Sequence[Sequence[int]],
) -> _FaceGeometry:
    """Pre-compute normals, centroids, and plane offsets for faces."""
    face_count = len(face_vertices)
    normals = np.zeros((face_count, 3), dtype=np.float64)
    centroids = np.zeros((face_count, 3), dtype=np.float64)
    areas = np.zeros(face_count, dtype=np.float64)
    plane_d = np.zeros(face_count, dtype=np.float64)
    bbox_min = np.full((face_count, 3), np.inf, dtype=np.float64)
    bbox_max = np.full((face_count, 3), -np.inf, dtype=np.float64)

    for idx, verts in enumerate(face_vertices):
        area, normal = _face_area_and_normal(points, verts)
        areas[idx] = area
        normals[idx] = np.asarray(normal, dtype=np.float64)
        if verts:
            coords = np.asarray([points[v] for v in verts], dtype=np.float64)
            centroid = np.mean(coords, axis=0)
            centroids[idx] = centroid
            plane_d[idx] = float(np.dot(normals[idx], centroid))
            bbox_min[idx] = coords.min(axis=0)
            bbox_max[idx] = coords.max(axis=0)
        else:
            centroids[idx] = 0.0
            plane_d[idx] = 0.0
    return _FaceGeometry(
        normals=normals,
        centroids=centroids,
        areas=areas,
        plane_d=plane_d,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def _color_from_material(material: Any) -> Optional[Tuple[float, float, float]]:
    def _try_get(obj: Any, *names: str) -> Optional[Any]:
        for name in names:
            if isinstance(obj, dict):
                if name in obj and obj[name] is not None:
                    return obj[name]
            elif hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    candidate = _try_get(
        material,
        "base_color",
        "baseColor",
        "diffuseColor",
        "DiffuseColor",
        "SurfaceColour",
        "surface_colour",
    )
    if candidate is None:
        surface = _try_get(material, "SurfaceColour", "SurfaceColor")
        if surface is not None:
            r = _try_get(surface, "Red", "red")
            g = _try_get(surface, "Green", "green")
            b = _try_get(surface, "Blue", "blue")
            if all(c is not None for c in (r, g, b)):
                return (float(r), float(g), float(b))
        return None
    try:
        if hasattr(candidate, "__iter__"):
            values = list(candidate)
        else:
            values = [candidate, candidate, candidate]
        if len(values) >= 3:
            return (float(values[0]), float(values[1]), float(values[2]))
    except Exception:
        return None
    return None


def _color_saturation(color: Optional[Tuple[float, float, float]]) -> float:
    if not color:
        return 0.0
    try:
        r, g, b = (float(color[0]), float(color[1]), float(color[2]))
    except Exception:
        return 0.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    return max_c - min_c


def _color_brightness(color: Optional[Tuple[float, float, float]]) -> float:
    if not color:
        return 0.0
    try:
        r, g, b = (float(color[0]), float(color[1]), float(color[2]))
    except Exception:
        return 0.0
    return (r + g + b) / 3.0

# ---------- NEW helpers: robust per-component banding & centroids ----------
def _face_centroid(points: Sequence[Tuple[float,float,float]], face_vertices: Sequence[int]) -> Tuple[float,float,float]:
    if not face_vertices:
        return (0.0,0.0,0.0)
    xs = ys = zs = 0.0
    n = float(len(face_vertices))
    for v in face_vertices:
        x,y,z = points[v]
        xs += float(x); ys += float(y); zs += float(z)
    return (xs/n, ys/n, zs/n)

def _local_z_bands(
    points: Sequence[Tuple[float,float,float]],
    faces: Sequence[Tuple[int,...]],
    comp_min_z: float,
    comp_max_z: float,
    top_frac: float = 0.18,
    bot_frac: float = 0.12,
) -> Tuple[set[int], set[int]]:
    """Return (top_faces, bottom_faces) decided in the component's *local* bbox."""
    if comp_max_z <= comp_min_z + 1e-9:
        return set(), set()
    height = comp_max_z - comp_min_z
    z_top = comp_min_z + (1.0 - top_frac) * height
    z_bot = comp_min_z + bot_frac * height
    top_idx: set[int] = set()
    bot_idx: set[int] = set()
    for fi, verts in enumerate(faces):
        cx, cy, cz = _face_centroid(points, verts)
        if cz >= z_top:
            top_idx.add(fi)
        elif cz <= z_bot:
            bot_idx.add(fi)
    return top_idx, bot_idx


def _resolve_component_styles(
    material_ids: Sequence[int],
    style_groups: Dict[str, Dict[str, Any]],
    style_tokens: Dict[str, str],
    *,
    points: Optional[Sequence[Tuple[float, float, float]]] = None,
    face_vertex_counts: Optional[Sequence[int]] = None,
    face_vertex_indices: Optional[Sequence[int]] = None,
    materials: Optional[Sequence[Any]] = None,
    cluster_styles: bool = False,
) -> Tuple[Dict[str, List[int]], Dict[str, str], Dict[str, str]]:
    """Return mapping of style tokens to face indices (and cluster base map)."""
    face_count = len(material_ids)
    if not face_count or not style_tokens:
        return {}, {}, {}

    if (
        points is None
        or face_vertex_counts is None
        or face_vertex_indices is None
        or len(face_vertex_counts) != face_count
    ):
        # Without topology data we can only return the original style groups.
        result: Dict[str, List[int]] = {}
        for key, entry in style_groups.items():
            token = style_tokens.get(key)
            if not token:
                continue
            faces = [int(idx) for idx in (entry.get("faces") or []) if 0 <= int(idx) < face_count]
            if faces:
                result[token] = faces
        return result, {}, {}

    style_color_map: Dict[str, Optional[Tuple[float, float, float]]] = {}
    for key, entry in style_groups.items():
        token = style_tokens.get(key)
        if not token:
            continue
        style_color_map[token] = _color_from_material(entry.get("material"))

    style_saturation_map: Dict[str, float] = {
        token: _color_saturation(color) for token, color in style_color_map.items()
    }
    sorted_styles_by_saturation = sorted(
        style_saturation_map.items(), key=lambda item: item[1], reverse=True
    )
    high_saturation_threshold = 0.18
    primary_high_saturation_token = next(
        (token for token, sat in sorted_styles_by_saturation if sat >= high_saturation_threshold),
        None,
    )
    neutral_style_token = (
        min(style_saturation_map.items(), key=lambda item: item[1])[0]
        if style_saturation_map
        else None
    )
    aluminium_style_token: Optional[str] = None
    for token, color in style_color_map.items():
        if color is None:
            continue
        brightness = _color_brightness(color)
        saturation = style_saturation_map.get(token, 0.0)
        if brightness >= 0.75 and saturation <= 0.15:
            aluminium_style_token = token
            break

    material_style_stats: Dict[int, Tuple[str, int]] = {}
    for key, entry in style_groups.items():
        token = style_tokens.get(key)
        if not token:
            continue
        id_counts: Counter[int] = Counter()
        for idx in entry.get("faces") or []:
            try:
                face_index = int(idx)
            except Exception:
                continue
            if 0 <= face_index < face_count:
                id_counts[int(material_ids[face_index])] += 1
        if not id_counts:
            continue
        dominant_id, dominant_count = id_counts.most_common(1)[0]
        existing = material_style_stats.get(dominant_id)
        if existing is None or dominant_count > existing[1]:
            material_style_stats[dominant_id] = (token, dominant_count)

    material_style_lookup: Dict[int, str] = {
        material_id: token for material_id, (token, _) in material_style_stats.items()
    }

    material_color_cache: Dict[int, Optional[Tuple[float, float, float]]] = {}

    def _material_color_for_id(material_id: int) -> Optional[Tuple[float, float, float]]:
        if material_id in material_color_cache:
            return material_color_cache[material_id]
        color: Optional[Tuple[float, float, float]] = None
        if materials is not None:
            try:
                mat = None
                if isinstance(materials, dict):
                    mat = materials.get(material_id)
                    if mat is None:
                        mat = materials.get(str(material_id))
                else:
                    if 0 <= material_id < len(materials):
                        mat = materials[material_id]
                if mat is not None:
                    color = _color_from_material(mat)
            except Exception:
                color = None
        material_color_cache[material_id] = color
        return color

    face_vertices, adjacency = _compute_face_components(face_vertex_counts, face_vertex_indices)
    face_geometry = _prepare_face_geometry(points, face_vertices)
    face_normals = face_geometry.normals
    face_centroids = face_geometry.centroids
    face_areas = face_geometry.areas
    face_plane_d = face_geometry.plane_d
    face_bbox_min = face_geometry.bbox_min
    face_bbox_max = face_geometry.bbox_max

    face_styles: List[Optional[str]] = [None] * face_count
    face_directions: List[Optional[str]] = [None] * face_count
    for key, entry in style_groups.items():
        token = style_tokens.get(key)
        if not token:
            continue
        for idx in entry.get("faces") or []:
            try:
                face_index = int(idx)
            except Exception:
                continue
            if 0 <= face_index < face_count:
                face_styles[face_index] = token

    style_normals: Dict[str, np.ndarray] = {}
    style_area_map: Dict[str, float] = {}
    if points is not None:
        for face_index, token in enumerate(face_styles):
            if token is None:
                continue
            area = float(face_areas[face_index])
            if area <= 1e-9:
                continue
            normal_vec = np.asarray(face_normals[face_index], dtype=np.float64)
            if normal_vec.size != 3:
                continue
            vec = normal_vec * area
            if token in style_normals:
                style_normals[token] += vec
                style_area_map[token] = style_area_map.get(token, 0.0) + area
            else:
                style_normals[token] = vec
                style_area_map[token] = area
        for token, vec in list(style_normals.items()):
            norm = np.linalg.norm(vec)
            if norm > 0.0:
                style_normals[token] = vec / norm
            else:
                style_normals[token] = vec

    style_up_token: Optional[str] = None
    style_up_score = 0.0
    style_front_token: Optional[str] = None
    style_front_score = 0.0
    for token, vec in style_normals.items():
        try:
            z_score = abs(float(vec[2]))
            if z_score > style_up_score:
                style_up_score = z_score
                style_up_token = token
            xy_score = max(abs(float(vec[0])), abs(float(vec[1])))
            if xy_score > style_front_score:
                style_front_score = xy_score
                style_front_token = token
        except Exception:
            continue

    components: List[Dict[str, Any]] = []
    visited = [False] * face_count
    for start_face in range(face_count):
        if visited[start_face]:
            continue
        material_id = int(material_ids[start_face])
        queue = [start_face]
        faces_in_component: List[int] = []
        area = 0.0
        normal_acc = np.zeros(3, dtype=np.float64)
        bounds_min = np.full(3, np.inf)
        bounds_max = np.full(3, -np.inf)
        bbox_initialized = False
        style_presence: Dict[str, int] = defaultdict(int)
        neighbor_styles: Dict[str, int] = defaultdict(int)
        centroid_sum = np.zeros(3, dtype=np.float64)
        centroid_weight = 0.0
        face_metrics: List[Tuple[float, np.ndarray]] = []
        face_details: Dict[int, Dict[str, np.ndarray]] = {}

        while queue:
            face_idx = queue.pop()
            if visited[face_idx]:
                continue
            if int(material_ids[face_idx]) != material_id:
                continue
            visited[face_idx] = True
            faces_in_component.append(face_idx)
            face_area = float(face_areas[face_idx])
            normal_vec_np = np.asarray(face_normals[face_idx], dtype=np.float64)
            centroid_vec = np.asarray(face_centroids[face_idx], dtype=np.float64)
            if face_area > 1e-9 and normal_vec_np.size == 3:
                area += face_area
                normal_acc += normal_vec_np * face_area
                centroid_sum += centroid_vec * face_area
                centroid_weight += face_area
                face_metrics.append((face_area, normal_vec_np))
                face_details[face_idx] = {
                    "area": face_area,
                    "normal": normal_vec_np,
                    "centroid": centroid_vec,
                }
            bbox_face_min = face_bbox_min[face_idx]
            bbox_face_max = face_bbox_max[face_idx]
            if np.isfinite(bbox_face_min).all() and np.isfinite(bbox_face_max).all():
                if not bbox_initialized:
                    bounds_min = np.array(bbox_face_min, copy=True)
                    bounds_max = np.array(bbox_face_max, copy=True)
                    bbox_initialized = True
                else:
                    bounds_min = np.minimum(bounds_min, bbox_face_min)
                    bounds_max = np.maximum(bounds_max, bbox_face_max)
            token = face_styles[face_idx]
            if token:
                style_presence[token] += 1
            for nb in adjacency[face_idx]:
                if not visited[nb] and int(material_ids[nb]) == material_id:
                    queue.append(nb)
                else:
                    neighbor_token = face_styles[nb]
                    if neighbor_token:
                        neighbor_styles[neighbor_token] += 1

        if not faces_in_component:
            continue
        if not bbox_initialized:
            bounds_min = np.zeros(3, dtype=np.float64)
            bounds_max = np.zeros(3, dtype=np.float64)
        if np.linalg.norm(normal_acc) > 0.0:
            normal_acc = normal_acc / np.linalg.norm(normal_acc)

        min_bounds = tuple(float(v) for v in bounds_min)
        max_bounds = tuple(float(v) for v in bounds_max)
        extents = tuple(float(max_bounds[i] - min_bounds[i]) for i in range(3))

        if centroid_weight > 0.0:
            centroid = centroid_sum / centroid_weight
        elif faces_in_component:
            centroid = np.asarray(face_centroids[faces_in_component[0]], dtype=np.float64)
        else:
            centroid = np.zeros(3, dtype=np.float64)

        normal_variation = 0.0
        if face_metrics and area > 0.0 and np.linalg.norm(normal_acc) > 0.0:
            normal_variation = sum(
                face_area * (1.0 - float(abs(np.dot(normal_acc, face_normal))))
                for face_area, face_normal in face_metrics
            ) / area

        orientation_abs = (
            float(abs(normal_acc[0])),
            float(abs(normal_acc[1])),
            float(abs(normal_acc[2])),
        )

        dominant_style_token = None
        if style_presence:
            dominant_style_token = max(style_presence.items(), key=lambda item: item[1])[0]

        components.append(
            {
                "faces": faces_in_component,
                "material_id": material_id,
                "area": area,
                "normal": tuple(float(v) for v in normal_acc),
                "bounds_min": min_bounds,
                "bounds_max": max_bounds,
                "extents": extents,
                "centroid": tuple(float(v) for v in centroid),
                "normal_variation": float(normal_variation),
                "orientation_abs": orientation_abs,
                "style_presence": style_presence,
                "neighbor_styles": neighbor_styles,
                "style_token": dominant_style_token,
                "face_details": face_details,
            }
        )

    def _style_fallback_for_material(material_id: int) -> Optional[str]:
        best_token: Optional[str] = None
        best_distance = float("inf")
        base_color = _material_color_for_id(material_id)
        preferred_token = material_style_lookup.get(material_id)
        if base_color is None and preferred_token is not None:
            return preferred_token
        for key, entry in style_groups.items():
            token = style_tokens.get(key)
            if not token:
                continue
            mat = entry.get("material")
            style_color = None
            if mat is not None:
                try:
                    style_color = tuple(float(c) for c in getattr(mat, "base_color"))
                except Exception:
                    style_color = _color_from_material(mat)
            if style_color is None:
                continue
            if base_color is None:
                # Without a base color fall back to the first style encountered.
                return token if best_token is None else best_token
            dist = sum((float(a) - float(b)) ** 2 for a, b in zip(style_color, base_color))
            if dist < best_distance:
                best_distance = dist
                best_token = token
        return best_token

    # Remove global Z dependence - classify per-component using its *local* height.

    area_threshold = 1.0
    saturation_threshold = 0.15
    orientation_override_area = 5.0
    orientation_override_sat = 0.2
    color_match_threshold = 0.01
    horizontal_area_threshold = 4.0
    horizontal_normal_threshold = 0.6
    vertical_area_threshold = 5.0
    vertical_normal_threshold = 0.3
    trim_thickness_threshold = 0.08
    trim_length_threshold = 0.6
    min_override_area = 2.0
    panel_area_threshold = 2.5
    trim_mid_threshold = 0.25
    handle_length_threshold = 0.4         # meters - min long edge of a handle bar
    handle_thickness_threshold = 0.06     # meters - thin section
    handle_mid_threshold = 0.15
    handle_area_threshold = 0.6
    handle_aspect_threshold = 8.0         # very elongated
    roof_brightness_threshold = 0.35
    front_panel_area_threshold = panel_area_threshold
    front_panel_normal_variation_threshold = 0.25
    front_panel_brightness_threshold = 0.15
    preserve_style_area_threshold = 0.65
    top_zone_fraction = 0.18
    bottom_zone_fraction = 0.10

    # Aggregate global Z bounds for mesh-relative heuristics
    global_min_z = float("inf")
    global_max_z = float("-inf")
    directional_components: List[Dict[str, Any]] = []

    def _rebuild_component_subset(
        base_component: Dict[str, Any],
        subset_faces: List[int],
        direction_label: str,
    ) -> Optional[Dict[str, Any]]:
        if not subset_faces:
            return None
        face_details = base_component.get("face_details") or {}
        material_id = base_component["material_id"]
        area = 0.0
        normal_acc = np.zeros(3, dtype=np.float64)
        bounds_min = np.full(3, np.inf)
        bounds_max = np.full(3, -np.inf)
        centroid_sum = np.zeros(3, dtype=np.float64)
        centroid_weight = 0.0
        face_metrics: List[Tuple[float, np.ndarray]] = []
        style_presence: Dict[str, int] = defaultdict(int)
        neighbor_styles: Dict[str, int] = defaultdict(int)
        sub_face_details: Dict[int, Dict[str, np.ndarray]] = {}

        for face_idx in subset_faces:
            detail = face_details.get(face_idx)
            if detail:
                face_area = float(detail.get("area", 0.0))
                normal_vec_np = np.asarray(detail.get("normal", np.zeros(3)), dtype=np.float64)
                tri_centroid = np.asarray(detail.get("centroid", np.zeros(3)), dtype=np.float64)
            else:
                face_area = float(face_areas[face_idx])
                normal_vec_np = np.asarray(face_normals[face_idx], dtype=np.float64)
                tri_centroid = np.asarray(face_centroids[face_idx], dtype=np.float64)

            if face_area <= 1e-9 or normal_vec_np.size != 3:
                continue

            bbox_face_min = face_bbox_min[face_idx]
            bbox_face_max = face_bbox_max[face_idx]
            if np.isfinite(bbox_face_min).all() and np.isfinite(bbox_face_max).all():
                bounds_min = np.minimum(bounds_min, bbox_face_min)
                bounds_max = np.maximum(bounds_max, bbox_face_max)

            sub_face_details[face_idx] = {
                "area": face_area,
                "normal": normal_vec_np,
                "centroid": tri_centroid,
            }
            area += face_area
            normal_acc += normal_vec_np * face_area
            centroid_sum += tri_centroid * face_area
            centroid_weight += face_area
            face_metrics.append((face_area, normal_vec_np))

            token = face_styles[face_idx]
            if token:
                style_presence[token] += 1

        if area <= 0.0:
            return None

        normal_len = np.linalg.norm(normal_acc)
        if normal_len > 0.0:
            orientation_abs = tuple(float(abs(v)) for v in normal_acc / normal_len)
        else:
            orientation_abs = (0.0, 0.0, 0.0)

        if centroid_weight > 0.0:
            centroid = tuple(float(v) for v in (centroid_sum / centroid_weight))
        else:
            centroid = tuple(float(v) for v in centroid_sum)

        normal_variation = 0.0
        if face_metrics and normal_len > 0.0:
            unit_normal = normal_acc / normal_len
            normal_variation = sum(
                face_area * (1.0 - float(abs(np.dot(unit_normal, face_normal))))
                for face_area, face_normal in face_metrics
            ) / area

        bounds_min_t = tuple(float(v) for v in bounds_min)
        bounds_max_t = tuple(float(v) for v in bounds_max)
        extents_vec = np.maximum(bounds_max - bounds_min, 0.0)
        extents = tuple(float(v) for v in extents_vec)

        style_token = None
        if style_presence:
            style_token = max(style_presence.items(), key=lambda item: item[1])[0]

        subset_set = set(subset_faces)
        for face_idx in subset_faces:
            for nb in adjacency[face_idx]:
                if nb in subset_set:
                    continue
                neighbor_token = face_styles[nb]
                if neighbor_token:
                    neighbor_styles[neighbor_token] += 1

        return {
            "faces": subset_faces,
            "material_id": material_id,
            "area": area,
            "normal": tuple(float(v) for v in normal_acc),
            "bounds_min": bounds_min_t,
            "bounds_max": bounds_max_t,
            "extents": extents,
            "centroid": centroid,
            "normal_variation": float(normal_variation),
            "orientation_abs": orientation_abs,
            "style_presence": style_presence,
            "neighbor_styles": neighbor_styles,
            "style_token": style_token,
            "face_details": sub_face_details,
            "direction_label": direction_label,
        }

    for component in components:
        face_details = component.get("face_details")
        if points is None or not face_details:
            component.pop("face_details", None)
            directional_components.append(component)
            continue

        bounds_min = np.asarray(component["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(component["bounds_max"], dtype=np.float64)
        extents = np.asarray(component["extents"], dtype=np.float64)
        edge_tol = 0.2
        bins: Dict[str, List[int]] = defaultdict(list)

        for face_idx in component["faces"]:
            detail = face_details.get(face_idx)
            if detail:
                area = float(detail.get("area", 0.0))
                normal_vec = np.asarray(detail.get("normal", np.zeros(3)), dtype=np.float64)
                centroid_vec = np.asarray(detail.get("centroid", np.zeros(3)), dtype=np.float64)
            else:
                area = float(face_areas[face_idx])
                normal_vec = np.asarray(face_normals[face_idx], dtype=np.float64)
                centroid_vec = np.asarray(face_centroids[face_idx], dtype=np.float64)
            normal_len = np.linalg.norm(normal_vec)
            if area <= 0.0 or normal_len <= 1e-5:
                bins["INTERIOR"].append(face_idx)
                continue
            axis = int(np.argmax(np.abs(normal_vec)))
            axis_label = ("X", "Y", "Z")[axis]
            sign_positive = normal_vec[axis] >= 0.0
            span = extents[axis]
            rel = 0.5 if span <= 1e-6 else (centroid_vec[axis] - bounds_min[axis]) / max(span, 1e-6)
            near_min = rel <= edge_tol
            near_max = rel >= (1.0 - edge_tol)

            if axis_label == "Z":
                if span <= 1e-4:
                    direction = "TOP" if sign_positive else "BOTTOM"
                elif sign_positive and near_max:
                    direction = "TOP"
                elif (not sign_positive) and near_min:
                    direction = "BOTTOM"
                else:
                    direction = "HORIZONTAL"
            else:
                if sign_positive and near_max:
                    direction = f"POS_{axis_label}"
                elif (not sign_positive) and near_min:
                    direction = f"NEG_{axis_label}"
                else:
                    direction = f"VERT_{axis_label}"
            bins[direction].append(face_idx)

        if len(bins) <= 1:
            component.pop("face_details", None)
            directional_components.append(component)
            continue

        for direction, subset in bins.items():
            rebuilt = _rebuild_component_subset(component, subset, direction)
            if rebuilt:
                directional_components.append(rebuilt)

    if directional_components:
        components = directional_components

    for component in components:
        bounds_min = component.get("bounds_min")
        bounds_max = component.get("bounds_max")
        if not bounds_min or not bounds_max or len(bounds_min) < 3 or len(bounds_max) < 3:
            continue
        try:
            global_min_z = min(global_min_z, float(bounds_min[2]))
            global_max_z = max(global_max_z, float(bounds_max[2]))
        except Exception:
            continue
    if global_min_z == float("inf"):
        global_min_z = 0.0
        global_max_z = 0.0
    global_height = max(global_max_z - global_min_z, 1e-6)

    # ---- Global priors: major(by coverage), panel(by saturation), neutral(by min saturation) ----
    face_count_f = float(max(1, len(material_ids)))
    coverage_by_token: Dict[str, float] = {}
    for tok, entry in style_groups.items():
        tkn = style_tokens.get(tok)
        if not tkn:
            continue
        cov = len(entry.get("faces") or [])
        coverage_by_token[tkn] = cov / face_count_f
    # Major by coverage
    major_token: Optional[str] = None
    major_cov = -1.0
    for tkn, cov in coverage_by_token.items():
        if cov > major_cov:
            major_cov, major_token = cov, tkn
    # Panel by saturation (the most colourful present)
    panel_token: Optional[str] = None
    panel_sat = -1.0
    for tkn, col in style_color_map.items():
        sat = _color_saturation(col)
        if sat > panel_sat:
            panel_sat, panel_token = sat, tkn
    # Neutral by minimal saturation
    neutral_token: Optional[str] = None
    neutral_sat = 10.0
    for tkn, col in style_color_map.items():
        sat = _color_saturation(col)
        if sat < neutral_sat:
            neutral_sat, neutral_token = sat, tkn
    # thresholds
    PANEL_SAT = 0.18      # what we consider panel-like colour
    FRONT_MAJ_COV = 0.45  # min coverage hint to prefer panel on fronts

    for component in components:
        faces_in_component = component["faces"]
        chosen_token: Optional[str] = None
        chosen_from_style = False
        material_id = component["material_id"]
        direction_label = component.get("direction_label") or ""
        component.pop("direction_label", None)

        def _assign_faces(token_value: Optional[str], faces_list: List[int]) -> None:
            if token_value is None:
                return
            for face_idx in faces_list:
                face_styles[face_idx] = token_value
                face_directions[face_idx] = direction_label or "NA"

        material_color = _material_color_for_id(material_id)
        preferred_token = material_style_lookup.get(material_id)
        material_brightness = _color_brightness(material_color)

        component.pop("face_details", None)

        extents = component.get("extents", (0.0, 0.0, 0.0))
        min_extent = min(extents) if extents else 0.0
        max_extent = max(extents) if extents else 0.0
        normal_vec = np.asarray(component["normal"], dtype=np.float64)
        abs_normal_z = float(abs(normal_vec[2]))
        is_horizontal = abs_normal_z >= horizontal_normal_threshold
        is_vertical = abs_normal_z <= vertical_normal_threshold
        sorted_extents = sorted(extents)
        mid_extent = sorted_extents[1] if len(sorted_extents) >= 2 else max_extent
        bounds_min = component.get("bounds_min", (0.0, 0.0, 0.0))
        bounds_max = component.get("bounds_max", (0.0, 0.0, 0.0))
        min_z = float(bounds_min[2]) if len(bounds_min) >= 3 else 0.0
        max_z = float(bounds_max[2]) if len(bounds_max) >= 3 else 0.0
        local_height = max(1e-6, max_z - min_z)
        centroid_z = float(component.get("centroid", (0.0, 0.0, 0.0))[2]) if len(component.get("centroid", (0.0, 0.0, 0.0))) >= 3 else (min_z + max_z) * 0.5
        rel_local = (centroid_z - min_z) / local_height
        near_top = False
        near_bottom = False
        if direction_label == "TOP":
            near_top = True
        elif direction_label == "BOTTOM":
            near_bottom = True
        else:
            near_top = rel_local >= (1.0 - top_zone_fraction)
            near_bottom = rel_local <= bottom_zone_fraction
        rel_global = (centroid_z - global_min_z) / max(global_height, 1e-6)
        is_top_band = direction_label == "TOP" or rel_global >= 0.80
        is_bottom_band = direction_label == "BOTTOM" or rel_global <= 0.20
        orientation_abs = component.get("orientation_abs", (0.0, 0.0, 0.0))
        normal_variation = float(component.get("normal_variation", 0.0))
        thin_trim = (
            min_extent <= trim_thickness_threshold
            and mid_extent <= trim_length_threshold
            and component["area"] <= area_threshold * 12.0
        )

        axis_tag = direction_label.split("_")[1] if "_" in direction_label else ""
        is_vertical_direction = direction_label.startswith("POS_") or direction_label.startswith("NEG_") or direction_label.startswith("VERT_")
        is_horizontal_direction = direction_label in ("TOP", "BOTTOM", "HORIZONTAL", "")

        # ---------- HANDLE RULE (aspect ratio + small area) ----------
        is_handle_component = (
            max_extent / max(1e-6, min_extent) >= handle_aspect_threshold
            and component["area"] <= handle_area_threshold
            and (is_vertical or abs_normal_z <= 0.3)
        )
        if is_handle_component:
            handle_token = preferred_token or aluminium_style_token or neutral_token or chosen_token
            if handle_token:
                chosen_token = handle_token
                _assign_faces(chosen_token, faces_in_component)
                LOG.debug("Handle forced -> %s", chosen_token)
                continue

        if thin_trim:
            desired_token = chosen_token
            if preferred_token:
                desired_token = preferred_token
                chosen_from_style = True
            elif desired_token is None and neutral_style_token:
                desired_token = neutral_style_token
            if desired_token:
                chosen_token = desired_token
                _assign_faces(chosen_token, faces_in_component)
                continue

        is_bottom_trim = (
            is_horizontal
            and component["area"] >= min_override_area
            and near_bottom
            and mid_extent <= trim_mid_threshold
        )

        auto_panel_applied = False
        is_roof_panel = (
            (direction_label == "TOP")
            or (
                is_horizontal_direction
                and component["area"] >= horizontal_area_threshold
                and (near_top or is_top_band)
                and material_brightness >= roof_brightness_threshold
            )
        )
        is_front_panel = (
            (is_vertical_direction and axis_tag in ("X", "Y"))
            or (
                is_vertical
                and (orientation_abs[0] >= 0.6 or orientation_abs[1] >= 0.6)
            )
        )
        is_front_panel = (
            is_front_panel
            and component["area"] >= front_panel_area_threshold
            and normal_variation <= front_panel_normal_variation_threshold
            and material_brightness >= front_panel_brightness_threshold
        )

        roof_candidate = None
        if panel_token and panel_sat >= PANEL_SAT:
            roof_candidate = panel_token
        elif style_up_token:
            roof_candidate = style_up_token

        if is_roof_panel and roof_candidate:
            chosen_token = roof_candidate
            auto_panel_applied = True

        front_candidate = None
        if panel_token and panel_sat >= PANEL_SAT:
            front_candidate = panel_token
        elif style_front_token:
            front_candidate = style_front_token

        if (
            is_front_panel
            and not near_bottom
            and front_candidate
            and coverage_by_token.get(front_candidate, 0.0) >= FRONT_MAJ_COV
        ):
            chosen_token = front_candidate
            auto_panel_applied = True

        chosen_color = style_color_map.get(chosen_token) if chosen_token else None
        color_match_value = bool(
            chosen_token
            and material_color is not None
            and chosen_color is not None
            and sum((float(material_color[i]) - float(chosen_color[i])) ** 2 for i in range(3)) < color_match_threshold
            and _color_saturation(chosen_color) > saturation_threshold
        )
        component["color_match"] = color_match_value
        style_coverage = 0.0
        if chosen_token:
            coverage_count = component["style_presence"].get(chosen_token, 0)
            style_coverage = float(coverage_count) / float(len(faces_in_component) or 1)

        # --- Choose a token using priority rules / overrides ---
        color_match = component.get("color_match", False)

        # Prefer explicit style on component if good coverage; else apply heuristics
        if chosen_token is None and style_coverage >= 0.50 and component.get("style_token"):
            chosen_token = component["style_token"]
            chosen_from_style = True

        # --- RULE 1 (pre): bottom-band guard (prefer neutral on bottom band) ---
        if chosen_token is None and (is_bottom_trim or thin_trim or near_bottom):
            chosen_token = neutral_token or component.get("style_token")

        # --- RULE 2: roof assertion - top horizontals inherit panel style when present ---
        if chosen_token is None and is_roof_panel and roof_candidate:
            chosen_token = roof_candidate
            chosen_from_style = False
            auto_panel_applied = True

        # --- RULE 3: front preference - large verticals (not bottom) prefer panel ---
        if (
            chosen_token is None
            and is_front_panel
            and not near_bottom
            and front_candidate
            and coverage_by_token.get(front_candidate, 0.0) >= FRONT_MAJ_COV
        ):
            chosen_token = front_candidate
            chosen_from_style = False
            auto_panel_applied = True

        # --- FINAL VETO: never leave bottom-band/trim/very-thin painted as a coloured panel ---
        if (near_bottom or is_bottom_band or is_bottom_trim or thin_trim) and chosen_token:
            if _color_saturation(style_color_map.get(chosen_token)) >= PANEL_SAT:
                chosen_token = neutral_token or component.get("style_token") or chosen_token

        if (direction_label == "TOP" or is_top_band) and roof_candidate:
            if chosen_token is None or chosen_token == neutral_token:
                chosen_token = roof_candidate
                auto_panel_applied = True
                chosen_from_style = False

        if chosen_token:
            coverage_count = component["style_presence"].get(chosen_token, 0)
            style_coverage = float(coverage_count) / float(len(faces_in_component) or 1)
        final_style_coverage = style_coverage

        allow_override = (
            bool(chosen_token)
            and not chosen_from_style
            and not color_match
            and component["area"] >= min_override_area
            and not auto_panel_applied
            and style_coverage < preserve_style_area_threshold
        )

        if allow_override:
            component_normal = np.asarray(component["normal"], dtype=np.float64)
            comp_norm_len = np.linalg.norm(component_normal)
            if comp_norm_len > 0.0:
                component_normal = component_normal / comp_norm_len
            else:
                component_normal = None
            chosen_color = style_color_map.get(chosen_token)
            chosen_sat = _color_saturation(chosen_color)
            if chosen_sat < saturation_threshold and component["area"] > area_threshold:
                candidate_token = chosen_token
                candidate_score = 0.0
                neighbor_area_threshold = 1.0
                for neighbor_token, neighbor_count in component["neighbor_styles"].items():
                    if style_area_map.get(neighbor_token, 0.0) < neighbor_area_threshold:
                        continue
                    neighbor_color = style_color_map.get(neighbor_token)
                    neighbor_sat = _color_saturation(neighbor_color)
                    if neighbor_sat <= chosen_sat:
                        continue
                    if component_normal is not None:
                        neighbor_normal = style_normals.get(neighbor_token)
                        if neighbor_normal is not None:
                            orientation = float(np.dot(component_normal, neighbor_normal))
                            if abs(orientation) < 0.4:
                                continue
                    score = neighbor_sat * float(neighbor_count)
                    if score > candidate_score:
                        candidate_score = score
                        candidate_token = neighbor_token
                if candidate_token != chosen_token and candidate_score > 0.0:
                    chosen_token = candidate_token
                elif candidate_token == chosen_token or candidate_score == 0.0:
                    # Fall back to globally available styles with compatible orientation.
                    global_candidate = chosen_token
                    global_score = 0.0
                    for token, color in style_color_map.items():
                        if token == chosen_token:
                            continue
                        token_sat = _color_saturation(color)
                        if token_sat <= chosen_sat:
                            continue
                        if component_normal is not None:
                            token_normal = style_normals.get(token)
                        orientation_ok = True
                        if component_normal is not None and token_normal is not None:
                            orientation = float(np.dot(component_normal, token_normal))
                            if abs(orientation) < 0.6:
                                if (
                                    style_area_map.get(token, 0.0) >= orientation_override_area
                                    and token_sat - chosen_sat > orientation_override_sat
                                ):
                                    orientation_ok = True
                                else:
                                    orientation_ok = False
                        if not orientation_ok:
                            continue
                        area_weight = style_area_map.get(token, 1.0)
                        score = token_sat * area_weight
                        if score > global_score:
                            global_score = score
                            global_candidate = token
                    if global_candidate != chosen_token and global_score > 0.0:
                        chosen_token = global_candidate

        if chosen_token:
            chosen_color = style_color_map.get(chosen_token)
            color_match = bool(
                material_color is not None
                and chosen_color is not None
                and sum((float(material_color[i]) - float(chosen_color[i])) ** 2 for i in range(3))
                < color_match_threshold
                and _color_saturation(chosen_color) > saturation_threshold
            )
            final_style_coverage = float(
                component["style_presence"].get(chosen_token, 0)
            ) / float(len(faces_in_component) or 1)
        else:
            final_style_coverage = 0.0
            color_match = False

        LOG.debug(
            "Component classify: mat=%s faces=%d area=%.3f horiz=%s vert=%s dir=%s z=[%.3f, %.3f] near_top=%.2f near_bottom=%s -> %s (from_style=%s cov=%.2f panelChoice=%s panelSat=%.2f)",
            material_id,
            len(faces_in_component),
            float(component.get("area", 0.0)),
            bool(is_horizontal),
            bool(is_vertical),
            direction_label or "NA",
            min_z,
            max_z,
            float(is_top_band),
            str(near_bottom),
            chosen_token,
            chosen_from_style,
            final_style_coverage,
            panel_token,
            panel_sat,
        )

        if chosen_token:
            _assign_faces(chosen_token, faces_in_component)

    mapping: Dict[str, List[int]] = defaultdict(list)
    direction_counters: Dict[str, Counter[str]] = defaultdict(Counter)
    cluster_lookup: Dict[str, str] = {}
    cluster_direction_map: Dict[str, str] = {}
    for idx, token in enumerate(face_styles):
        if token is None:
            continue
        mapping[token].append(idx)
        direction_value = face_directions[idx] or ""
        direction_counters[token][direction_value] += 1

    def _sanitize_direction_label(label: Optional[str]) -> str:
        if not label:
            return "NA"
        return sanitize_name(label, fallback="NA")

    for token, counter in direction_counters.items():
        if counter:
            cluster_direction_map[token] = _sanitize_direction_label(counter.most_common(1)[0][0])

    if cluster_styles and points is not None and face_vertices is not None and len(mapping) > 1:
        face_set_adjacency = adjacency

        def _cluster_faces_by_plane(face_list: List[int]) -> List[List[int]]:
            if not face_list:
                return []
            face_set = set(face_list)
            visited: Set[int] = set()
            components_local: List[List[int]] = []
            for fid in face_list:
                if fid in visited:
                    continue
                stack = [fid]
                comp: List[int] = []
                while stack:
                    fcur = stack.pop()
                    if fcur in visited or fcur not in face_set:
                        continue
                    visited.add(fcur)
                    comp.append(fcur)
                    for nb in face_set_adjacency[fcur]:
                        if nb in face_set and nb not in visited:
                            stack.append(nb)
                if comp:
                    components_local.append(comp)

            PLANE_D_TOL = 0.005
            N_DOT_TOL = 0.98
            clustered: List[List[int]] = []

            for comp in components_local:
                planes: List[Tuple[np.ndarray, float, List[int]]] = []
                for fid in comp:
                    n = np.asarray(face_normals[fid], dtype=float)
                    norm = np.linalg.norm(n)
                    if norm < 1e-9:
                        continue
                    n /= norm
                    d = float(face_plane_d[fid])
                    placed = False
                    for entry in planes:
                        ref_n, ref_d, lst = entry
                        if abs(np.dot(ref_n, n)) >= N_DOT_TOL and abs(ref_d - d) <= PLANE_D_TOL:
                            lst.append(fid)
                            placed = True
                            break
                    if not placed:
                        planes.append([n, d, [fid]])
                for _, _, faces_cluster in planes:
                    if faces_cluster:
                        clustered.append(faces_cluster)
            return clustered or [face_list]

        clustered_mapping: Dict[str, List[int]] = {}
        for token, faces in mapping.items():
            if not faces:
                continue
            clusters = _cluster_faces_by_plane(faces)
            if len(clusters) <= 1:
                clustered_mapping[token] = faces
            else:
                cluster_direction_map.pop(token, None)
                for ci, face_subset in enumerate(clusters):
                    dir_counter = Counter(face_directions[f] or "" for f in face_subset)
                    if dir_counter:
                        dir_raw = dir_counter.most_common(1)[0][0]
                    else:
                        dir_raw = ""
                    dir_label = _sanitize_direction_label(dir_raw)
                    clustered_name = f"{token}_C{ci}_{dir_label}"
                    clustered_mapping[clustered_name] = face_subset
                    cluster_lookup[clustered_name] = token
                    cluster_direction_map[clustered_name] = dir_label
        if clustered_mapping:
            mapping = clustered_mapping

    return mapping, cluster_lookup, cluster_direction_map


def _ensure_material_subsets(
    mesh: UsdGeom.Mesh,
    material_ids: Optional[List[int]],
    face_count: int,
    *,
    style_groups: Optional[Dict[str, Dict[str, Any]]] = None,
    materials: Optional[Sequence[Any]] = None,
    points: Optional[Sequence[Tuple[float, float, float]]] = None,
    face_vertex_counts: Optional[Sequence[int]] = None,
    face_vertex_indices: Optional[Sequence[int]] = None,
    subset_token_map: Optional[Dict[str, str]] = None,
) -> bool:
    prim = mesh.GetPrim()
    stage = prim.GetStage()
    imageable = UsdGeom.Imageable(prim)
    family_name = "materialBind"
    element_token = UsdGeom.Tokens.face

    existing = _collect_material_subsets(imageable, family_name)
    for subset in existing:
        stage.RemovePrim(subset.GetPrim().GetPath())

    try:
        UsdGeom.Subset.SetFamilyType(imageable, family_name, UsdGeom.Tokens.nonOverlapping)
    except Exception:
        pass

    created_subsets = False
    stats: List[Tuple[str, int]] = []
    token_to_style_key: Dict[str, str] = {}
    style_faces_handled: Set[int] = set()

    if style_groups:
        for key, entry in style_groups.items():
            raw_faces = entry.get("faces") or []
            if not raw_faces:
                continue
            valid_faces: List[int] = []
            for value in raw_faces:
                try:
                    face_index = int(value)
                except Exception:
                    continue
                if 0 <= face_index < face_count:
                    valid_faces.append(face_index)
            if not valid_faces:
                continue
            subset_token = _subset_token_for_material(key)
            token_to_style_key[subset_token] = key
            if subset_token_map is not None:
                subset_token_map[key] = subset_token
            subset_path = mesh.GetPath().AppendChild(subset_token)
            subset = UsdGeom.Subset.Define(stage, subset_path)
            try:
                UsdGeom.Subset.SetElementType(subset, element_token)
            except Exception:
                try:
                    subset.CreateElementTypeAttr().Set(element_token)
                except AttributeError:
                    subset.GetPrim().CreateAttribute("elementType", Sdf.ValueTypeNames.Token).Set(str(element_token))
            indices_attr = subset.GetIndicesAttr()
            if not indices_attr:
                indices_attr = subset.CreateIndicesAttr()
            indices_attr.Set(Vt.IntArray(valid_faces))
            prim = subset.GetPrim()
            material_name = _material_name_from_style_entry(entry, fallback=str(key))
            _set_subset_material_name(prim, material_name)
            prim.CreateAttribute("ifc:sourceStyleKey", Sdf.ValueTypeNames.String).Set(str(key))
            prim.CreateAttribute("ifc:sourceStyleToken", Sdf.ValueTypeNames.String).Set(subset_token)
            if material_ids:
                mids = sorted({int(material_ids[i]) for i in valid_faces if 0 <= int(i) < face_count})
                if len(mids) == 1:
                    prim.CreateAttribute("ifc:baseMaterialIndex", Sdf.ValueTypeNames.Int).Set(mids[0])
                elif mids:
                    prim.CreateAttribute("ifc:baseMaterialIndices", Sdf.ValueTypeNames.IntArray).Set(Vt.IntArray(mids))
            try:
                subset.CreateFamilyNameAttr().Set(family_name)
            except Exception:
                subset.GetPrim().CreateAttribute("familyName", Sdf.ValueTypeNames.String).Set(family_name)
            created_subsets = True
            stats.append((str(subset_path), len(valid_faces)))
            for face_int in valid_faces:
                style_faces_handled.add(face_int)

    if material_ids and len(material_ids) == face_count:
        indices_by_material: Dict[int, List[int]] = defaultdict(list)
        for face_index, material_index in enumerate(material_ids):
            if face_index in style_faces_handled:
                continue
            indices_by_material[int(material_index)].append(int(face_index))
        if not style_groups and len(indices_by_material) == 1:
            # Uniform material with no style information: rely on the mesh-level
            # binding instead of authoring a redundant GeomSubset.
            indices_by_material.clear()

        for material_index, face_indices in sorted(indices_by_material.items()):
            if not face_indices:
                continue
            subset_token = _subset_token_for_material(_material_identifier_for_index(material_index, materials))
            subset_path = mesh.GetPath().AppendChild(subset_token)
            subset = UsdGeom.Subset.Define(stage, subset_path)
            try:
                UsdGeom.Subset.SetElementType(subset, element_token)
            except Exception:
                try:
                    subset.CreateElementTypeAttr().Set(element_token)
                except AttributeError:
                    subset.GetPrim().CreateAttribute("elementType", Sdf.ValueTypeNames.Token).Set(str(element_token))
            indices_attr = subset.GetIndicesAttr()
            if not indices_attr:
                indices_attr = subset.CreateIndicesAttr()
            indices_attr.Set(Vt.IntArray(face_indices))
            # Traceability for numeric subsets
            subset_prim = subset.GetPrim()
            material_name = _material_name_for_index(material_index, materials)
            _set_subset_material_name(subset_prim, material_name)
            subset_prim.CreateAttribute("ifc:sourceMaterialIndex", Sdf.ValueTypeNames.Int).Set(int(material_index))
            try:
                subset.CreateFamilyNameAttr().Set(family_name)
            except Exception:
                subset.GetPrim().CreateAttribute("familyName", Sdf.ValueTypeNames.String).Set(family_name)
            created_subsets = True
            stats.append((str(subset_path), len(face_indices)))

    if LOG.isEnabledFor(logging.DEBUG) and stats:
        total_faces = sum(count for _, count in stats)
        LOG.debug(
            "Material subset coverage for %s: subsets=%d totalFaces=%d/%d",
            mesh.GetPath(),
            len(stats),
            total_faces,
            face_count,
        )
        for subset_path, count in stats:
            LOG.debug("  Subset %s -> %d faces", subset_path, count)

    return created_subsets


def write_usd_mesh(stage, parent_path, mesh_name, mesh_data, abs_mat=None,
                   material_ids=None, style_groups=None, materials=None, stage_meters_per_unit=1.0,
                   scale_matrix_translation=False):
    """Author a Mesh prim beneath ``parent_path`` with the supplied data."""
    mesh_path = Sdf.Path(parent_path).AppendChild(mesh_name)
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    if isinstance(mesh_data, tuple) and len(mesh_data) >= 2:
        verts, faces = mesh_data[:2]
        mesh_dict: Dict[str, Any] = {
            "vertices": np.asarray(verts, dtype=np.float32),
            "faces": np.asarray(faces, dtype=np.int64),
        }
    elif isinstance(mesh_data, dict):
        mesh_dict = mesh_data
    else:
        mesh_dict = {}

    payload = _prepare_mesh_payload(mesh_dict)
    if payload is None:
        return mesh

    points_attr = Vt.Vec3fArray(len(payload["points"]))
    for i, (x, y, z) in enumerate(payload["points"]):
        points_attr[i] = Gf.Vec3f(float(x), float(y), float(z))
    mesh.CreatePointsAttr(points_attr)

    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in payload["face_vertex_indices"]]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(c) for c in payload["face_vertex_counts"]]))

    normals_values = payload.get("normals")
    if normals_values:
        normals_attr = Vt.Vec3fArray(len(normals_values))
        for i, (nx, ny, nz) in enumerate(normals_values):
            normals_attr[i] = Gf.Vec3f(float(nx), float(ny), float(nz))
        mesh.CreateNormalsAttr(normals_attr)
        normals_interp = payload.get("normals_interpolation")
        if normals_interp:
            mesh.SetNormalsInterpolation(normals_interp)

    st_values = payload.get("st")
    st_interp = payload.get("st_interpolation")
    if st_values and st_interp:
        st_attr = Vt.TexCoord2fArray(len(st_values))
        for i, (u, v) in enumerate(st_values):
            st_attr[i] = (float(u), float(v))
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        st_primvar = primvars_api.CreatePrimvar("st", st_interp, Sdf.ValueTypeNames.TexCoord2fArray)
        st_primvar.Set(st_attr)

    if abs_mat is not None:
        gf = np_to_gf_matrix(abs_mat)
        if scale_matrix_translation and stage_meters_per_unit != 1.0:
            gf = scale_matrix_translation_only(gf, 1.0 / float(stage_meters_per_unit))
        xf = UsdGeom.Xformable(mesh)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(gf)

    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    face_count = len(payload["face_vertex_counts"])
    subsets_use_styles = False
    payload_points = payload.get("points")
    payload_counts = payload.get("face_vertex_counts")
    payload_indices = payload.get("face_vertex_indices")

    subset_token_map: Optional[Dict[str, str]] = {} if style_groups else None

    if material_ids:
        mat_attr = Vt.IntArray([int(i) for i in material_ids])
        mesh.GetPrim().CreateAttribute("ifc:materialIds", Sdf.ValueTypeNames.IntArray).Set(mat_attr)
        subsets_use_styles = _ensure_material_subsets(
            mesh,
            list(mat_attr),
            face_count,
            style_groups=style_groups,
            materials=materials,
            points=payload_points,
            face_vertex_counts=payload_counts,
            face_vertex_indices=payload_indices,
            subset_token_map=subset_token_map,
        )
    else:
        subsets_use_styles = _ensure_material_subsets(
            mesh,
            [],
            face_count,
            style_groups=style_groups,
            materials=materials,
            points=payload_points,
            face_vertex_counts=payload_counts,
            face_vertex_indices=payload_indices,
            subset_token_map=subset_token_map,
        )
    if subset_token_map and style_groups:
        for key, token in subset_token_map.items():
            entry = style_groups.get(key)
            if entry is not None:
                entry["subset_token"] = token

    # If a single style group covers the entire mesh, author displayColor for quick previews
    try:
        if style_groups and subsets_use_styles:
            for entry in style_groups.values():
                faces = entry.get("faces") or []
                material = entry.get("material")
                if not faces or len(faces) != face_count:
                    continue
                display = getattr(material, "base_color", None) or getattr(material, "display_color", None)
                if not display and isinstance(material, dict):
                    display = (
                        material.get("diffuseColor")
                        or material.get("DiffuseColor")
                        or material.get("diffuse_color")
                        or material.get("Color")
                        or material.get("colour")
                    )
                if not display:
                    continue
                display_values = display[:3] if hasattr(display, "__getitem__") else (display, display, display)
                display_tuple = tuple(float(c) for c in display_values)
                color = Vt.Vec3fArray([Gf.Vec3f(*_normalize_color(display_tuple))])
                UsdGeom.Gprim(mesh.GetPrim()).CreateDisplayColorAttr().Set(color)
                break
    except Exception:
        pass

    return mesh


def _author_occ_detail_meshes(
    stage: Usd.Stage,
    detail_root_path: Sdf.Path,
    detail_mesh: Any,
    *,
    meters_per_unit: float,
    abs_mat: Optional[Sequence[Sequence[float]]] = None,
) -> List[Tuple[UsdGeom.Mesh, List[Any]]]:
    """Author one Mesh per OCC subshape beneath ``detail_root_path``."""
    subshapes = getattr(detail_mesh, "subshapes", None) or []
    faces = getattr(detail_mesh, "faces", None) or []
    if not subshapes and faces:
        subshapes = [
            {"index": idx, "label": f"Face_{idx}", "shape_type": "FACE", "faces": [face]}
            for idx, face in enumerate(faces)
        ]
    if not subshapes:
        LOG.debug("Detail mesh authoring skipped: no subshapes or faces found for %s", detail_root_path)
        return []

    detail_root = UsdGeom.Xform.Define(stage, detail_root_path)
    detail_root.ClearXformOpOrder()
    authored: List[Tuple[UsdGeom.Mesh, List[Any]]] = []
    used_names: Set[str] = set()

    composed_abs = abs_mat
    try:
        pre_xform = getattr(detail_mesh, "pre_xform", None)
    except Exception:
        pre_xform = None
    if pre_xform is not None:
        try:
            pre_arr = np.asarray(pre_xform, dtype=float).reshape(4, 4)
            if composed_abs is None:
                composed_abs = pre_arr.tolist()
            else:
                base_arr = np.asarray(composed_abs, dtype=float).reshape(4, 4)
                composed_abs = (pre_arr @ base_arr).tolist()
        except Exception:
            composed_abs = abs_mat

    def _combine_faces(face_entries: List[Any]) -> Tuple[Optional[Dict[str, Any]], List[Any]]:
        verts_list: List[np.ndarray] = []
        faces_list: List[np.ndarray] = []
        offset = 0
        face_materials: List[Any] = []
        for entry in face_entries:
            verts = getattr(entry, "vertices", None)
            tris = getattr(entry, "faces", None)
            if verts is None or tris is None:
                continue
            verts_np = np.asarray(verts, dtype=np.float64)
            tris_np = np.asarray(tris, dtype=np.int64)
            if verts_np.size == 0 or tris_np.size == 0:
                continue
            verts_list.append(verts_np)
            faces_list.append(tris_np + offset)
            face_token = getattr(entry, "material_key", None)
            face_materials.extend([face_token] * tris_np.shape[0])
            offset += verts_np.shape[0]
        if not verts_list or not faces_list:
            return None, []
        return (
            {
                "vertices": np.vstack(verts_list),
                "faces": np.vstack(faces_list),
            },
            face_materials,
        )

    for entry in subshapes:
        faces_for_subshape = getattr(entry, "faces", None) or []
        mesh_data, face_materials = _combine_faces(faces_for_subshape)
        if mesh_data is None:
            LOG.debug(
                "Detail mesh subshape %s produced no triangles (label=%s)",
                getattr(entry, "index", len(authored)),
                getattr(entry, "label", "<unnamed>"),
            )
            continue
        label = getattr(entry, "label", None) or f"Subshape_{getattr(entry, 'index', len(authored))}"
        sanitized = _sanitize_identifier(label, fallback="Subshape")
        candidate = sanitized
        counter = 1
        while candidate in used_names:
            candidate = f"{sanitized}_{counter}"
            counter += 1
        used_names.add(candidate)
        mesh = write_usd_mesh(
            stage,
            detail_root_path,
            candidate,
            mesh_data,
            abs_mat=composed_abs,
            stage_meters_per_unit=meters_per_unit,
        )
        verts_np = np.asarray(mesh_data.get("vertices"), dtype=np.float64)
        faces_np = np.asarray(mesh_data.get("faces"), dtype=np.int64)
        LOG.debug(
            "Detail mesh: authored %s with %d vertices / %d faces",
            detail_root_path.AppendChild(candidate),
            verts_np.shape[0],
            faces_np.shape[0],
        )
        authored.append((mesh, face_materials))
    return authored


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


def _prepare_writable_layer(layer_path: PathLike) -> Sdf.Layer:
    """
    Return a cleared, editable layer matching ``layer_path`` without tripping
    Sdf's unique-identifier constraint when rerunning inside the same process.
    """
    identifier = _layer_identifier(layer_path)
    existing = Sdf.Layer.Find(identifier)
    if existing is not None:
        existing.Clear()
        return existing
    return Sdf.Layer.CreateNew(identifier)


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
        if getattr(proto, "detail_mesh", None) is not None:
            yield PrototypeKey(kind="repmap_detail", identifier=rep_id), proto
    for digest, proto in sorted(caches.hashes.items()):
        yield PrototypeKey(kind="hash", identifier=digest), proto


def _prototype_from_key(caches: PrototypeCaches, key: Optional[PrototypeKey]) -> Optional[Any]:
    """Return the prototype record referenced by ``key``."""
    if key is None:
        return None
    if key.kind in ("repmap", "repmap_detail"):
        try:
            identifier = int(key.identifier)
        except Exception:
            return None
        return caches.repmaps.get(identifier)
    identifier_str = str(key.identifier)
    return caches.hashes.get(identifier_str)


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
    proto_layer = _prepare_writable_layer(layer_path)

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    proto_root = Sdf.Path("/World/__Prototypes")
    proto_root_detail = Sdf.Path("/World/__PrototypesDetail")
    proto_paths: Dict[PrototypeKey, Sdf.Path] = {}
    used_names: Dict[str, int] = {}
    detail_scope = getattr(options, "detail_scope", "all") or "all"
    detail_mode_enabled = bool(
        getattr(options, "detail_mode", False) or getattr(options, "enable_high_detail_remesh", False)
    )
    stage_meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)

    def _unique_name(base: str) -> str:
        c = used_names.get(base, 0)
        used_names[base] = c + 1
        return base if c == 0 else f"{base}_{c}"

    with Usd.EditContext(stage, proto_layer):
        UsdGeom.Scope.Define(stage, proto_root)
        UsdGeom.Scope.Define(stage, proto_root_detail)

        for key, proto in _iter_prototypes(caches):
            mesh_payload = getattr(proto, "mesh", None)
            detail_payload = getattr(proto, "detail_mesh", None)
            has_detail = bool(detail_mode_enabled and detail_payload and getattr(detail_payload, "faces", None))

            if not mesh_payload and not has_detail:
                continue

            # Name the prototype Xform
            if key.kind in ("repmap", "repmap_detail"):
                base = _name_for_repmap(proto)
                if key.kind == "repmap_detail":
                    base = f"{base}_Detail"
            else:
                base = _name_for_hash(proto)
            prim_name = _unique_name(base)

            parent_scope = proto_root_detail if key.kind == "repmap_detail" else proto_root
            proto_path = parent_scope.AppendChild(prim_name)
            proto_paths[key] = proto_path

            # Define prototype Xform and mark as 'guide' (no direct render)
            x = UsdGeom.Xform.Define(stage, proto_path)
            x.ClearXformOpOrder()
            proto_prim = x.GetPrim()
            # DO NOT set instanceable on the prototype
            UsdGeom.Imageable(proto_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

            # Author the mesh as a child "Geom"
            full_material_ids = list(getattr(proto, "material_ids", []) or [])
            full_style_groups = getattr(proto, "style_face_groups", None)
            full_materials = list(getattr(proto, "materials", []) or [])
            semantic_parts_base = getattr(proto, "semantic_parts", {}) or {}
            semantic_parts_detail = getattr(proto, "semantic_parts_detail", {}) or {}

            # Semantic parts on the base prototype take precedence and short-circuit
            if semantic_parts_base and key.kind != "repmap_detail":
                geom_parent = proto_path.AppendChild("Geom")
                for label, part in semantic_parts_base.items():
                    mesh_name = sanitize_name(label, fallback="Part")
                    part_mesh_dict = {
                        "vertices": np.asarray(part.get("vertices"), dtype=np.float32),
                        "faces": np.asarray(part.get("faces"), dtype=np.int64),
                    }
                    part_material_ids = part.get("material_ids") or []
                    write_usd_mesh(
                        stage,
                        geom_parent,
                        mesh_name,
                        part_mesh_dict,
                        abs_mat=None,
                        material_ids=part_material_ids if part_material_ids else None,
                        materials=full_materials,
                        style_groups=part.get("style_groups") or {},
                    )
                continue

            if has_detail and key.kind == "repmap_detail":
                detail_parent = proto_path.AppendChild("Geom")
                if semantic_parts_detail:
                    for label, part in semantic_parts_detail.items():
                        mesh_name = sanitize_name(label, fallback="Part")
                        part_mesh_dict = {
                            "vertices": np.asarray(part.get("vertices"), dtype=np.float32),
                            "faces": np.asarray(part.get("faces"), dtype=np.int64),
                        }
                        part_material_ids = part.get("material_ids") or []
                        write_usd_mesh(
                            stage,
                            detail_parent,
                            mesh_name,
                            part_mesh_dict,
                            abs_mat=None,
                            material_ids=part_material_ids if part_material_ids else None,
                            style_groups=part.get("style_groups") or {},
                            materials=full_materials,
                            stage_meters_per_unit=stage_meters_per_unit,
                        )
                else:
                    _author_occ_detail_meshes(
                        stage,
                        detail_parent,
                        detail_payload,
                        meters_per_unit=stage_meters_per_unit,
                    )
            elif key.kind != "repmap_detail":
                if semantic_parts_base:
                    geom_parent = proto_path.AppendChild("Geom")
                    for label, part in semantic_parts_base.items():
                        mesh_name = sanitize_name(label, fallback="Part")
                        part_mesh_dict = {
                            "vertices": np.asarray(part.get("vertices"), dtype=np.float32),
                            "faces": np.asarray(part.get("faces"), dtype=np.int64),
                        }
                        part_material_ids = part.get("material_ids") or []
                        write_usd_mesh(
                            stage,
                            geom_parent,
                            mesh_name,
                            part_mesh_dict,
                            abs_mat=None,
                            material_ids=part_material_ids if part_material_ids else None,
                            style_groups=part.get("style_groups") or {},
                            materials=full_materials,
                            stage_meters_per_unit=stage_meters_per_unit,
                        )
                else:
                    write_usd_mesh(
                        stage,
                        proto_path,
                        "Geom",
                        mesh_payload,
                        abs_mat=None,
                        material_ids=full_material_ids,
                        style_groups=full_style_groups,
                        materials=full_materials,
                        stage_meters_per_unit=stage_meters_per_unit,
                    )

    return proto_layer, proto_paths



# ---------------- Materials (kept minimal; wiring points preserved) ----------------

def _prototype_materials(caches: PrototypeCaches, key: PrototypeKey) -> Any:
    if key.kind == "repmap":
        proto = caches.repmaps.get(int(key.identifier))
    else:
        proto = caches.hashes.get(str(key.identifier))
    if not proto:
        return {}
    materials = getattr(proto, "materials", None)
    if isinstance(materials, dict):
        return materials
    if isinstance(materials, (list, tuple)):
        return list(materials)
    if materials is None:
        return {}
    return materials



def _prototype_style_groups(caches: PrototypeCaches, key: PrototypeKey) -> Dict[str, Dict[str, Any]]:
    if key.kind == "repmap":
        proto = caches.repmaps.get(int(key.identifier))
    else:
        proto = caches.hashes.get(str(key.identifier))

    if not proto:
        return {}
    groups = getattr(proto, "style_face_groups", None)
    if not groups:
        return {}
    return groups

def _merge_style_face_groups(
    instance_groups: Optional[Dict[str, Dict[str, Any]]],
    prototype_groups: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Deep-merge instance + prototype style groups into a single view:
      - Instance faces/material override prototype when present.
      - Prototype fills any missing fields/entries.
      - Faces are normalized to a unique, sorted list of ints.
    """
    inst = dict(instance_groups or {})
    proto = dict(prototype_groups or {})
    if not inst and not proto:
        return {}
    keys = set(inst.keys()) | set(proto.keys())
    merged: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        I = dict(inst.get(key) or {})
        P = dict(proto.get(key) or {})
        faces_src = I.get("faces", P.get("faces"))
        if faces_src is not None:
            try:
                faces = sorted({int(i) for i in faces_src})
            except Exception:
                faces = [int(i) for i in faces_src]  # best effort
        else:
            faces = None
        material = I.get("material", P.get("material"))
        entry: Dict[str, Any] = {}
        if faces is not None:
            entry["faces"] = faces
        if material is not None:
            entry["material"] = material
        # carry through helpful tags when present
        for extra in ("name", "label", "id", "guid", "shapeAspect", "style_id"):
            if extra in I:
                entry[extra] = I[extra]
            elif extra in P:
                entry[extra] = P[extra]
        merged[key] = entry
    return merged
def _iter_material_entries(materials: Any) -> Iterable[Tuple[Any, Any]]:
    if isinstance(materials, dict):
        return materials.items()
    if isinstance(materials, (list, tuple)):
        return enumerate(materials)
    if materials is None:
        return []
    return [("__value__", materials)]


@dataclass(frozen=True)
class _MaterialProperties:
    name: str
    display_color: Tuple[float, float, float]
    opacity: float
    # NEW: PBR scalars carried through from IFC
    metallic: float = 0.0
    roughness: float = 0.5
    emissive_color: Optional[Tuple[float, float, float]] = None
    # NEW: optional texture slots
    base_color_tex: Optional[str] = None
    normal_tex: Optional[str] = None
    orm_tex: Optional[str] = None


@dataclass(frozen=True)
class _FaceGeometry:
    normals: np.ndarray       # (F, 3)
    centroids: np.ndarray     # (F, 3)
    areas: np.ndarray         # (F,)
    plane_d: np.ndarray       # (F,)
    bbox_min: np.ndarray      # (F, 3)
    bbox_max: np.ndarray      # (F, 3)


@dataclass
class MaterialLibrary:
    signature_to_path: Dict[Tuple[Any, ...], Sdf.Path]


# ---------- helpers used by binding ----------
_MATERIAL_NAME_CACHE: "weakref.WeakKeyDictionary[Usd.Stage, Dict[str, Sdf.Path]]" = weakref.WeakKeyDictionary()


def _mesh_has_geom_subsets(stage: Usd.Stage, mesh_path: Optional[Sdf.Path]) -> bool:
    """
    Return True iff the composed prim at mesh_path has authored GeomSubset children.
    """
    if mesh_path is None:
        return False
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim:
        return False
    for child in mesh_prim.GetChildren():
        if child.GetTypeName() == "GeomSubset":
            return True
    return False


def _material_name_key(name: Any) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    return text.lower()


def _register_material_name(stage: Optional[Usd.Stage], name: Any, path: Optional[Sdf.Path]) -> None:
    if stage is None or path is None:
        return
    key = _material_name_key(name)
    if key is None:
        return
    cache = _MATERIAL_NAME_CACHE.setdefault(stage, {})
    cache[key] = path


def _material_path_by_name(stage: Optional[Usd.Stage], name: Any) -> Optional[Sdf.Path]:
    if stage is None:
        return None
    key = _material_name_key(name)
    if key is None:
        return None
    cache = _MATERIAL_NAME_CACHE.get(stage)
    if cache and key in cache:
        return cache[key]
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Material":
            continue
        attr = prim.GetAttribute("ifc:materialName")
        if not attr or not attr.HasAuthoredValue():
            continue
        try:
            value = attr.Get()
        except Exception:
            continue
        if _material_name_key(value) == key:
            path = prim.GetPath()
            cache = _MATERIAL_NAME_CACHE.setdefault(stage, {})
            cache[key] = path
            return path
    return None


def _set_subset_material_name(subset_prim: Usd.Prim, material_name: Optional[str], *, overwrite: bool = False) -> None:
    if subset_prim is None or material_name is None:
        return
    text = str(material_name).strip()
    if not text:
        return
    attr = subset_prim.GetAttribute("ifc:materialName")
    if attr and attr.HasAuthoredValue() and not overwrite:
        return
    subset_prim.CreateAttribute("ifc:materialName", Sdf.ValueTypeNames.String, custom=True).Set(text)


def _subset_indices_from_attrs(subset_prim: Usd.Prim) -> List[int]:
    indices: Set[int] = set()

    def _collect(attr_name: str) -> None:
        attr = subset_prim.GetAttribute(attr_name)
        if not attr or not attr.HasAuthoredValue():
            return
        try:
            value = attr.Get()
        except Exception:
            return
        if value is None:
            return
        if isinstance(value, (list, tuple, Vt.IntArray)):
            for item in value:
                try:
                    indices.add(int(item))
                except Exception:
                    continue
        else:
            try:
                indices.add(int(value))
            except Exception:
                return

    for name in ("ifc:sourceMaterialIndex", "ifc:baseMaterialIndex", "ifc:baseMaterialIndices"):
        _collect(name)
    return sorted(indices)


def _update_material_metadata(
    stage: Usd.Stage,
    material_path: Sdf.Path,
    *,
    material_name: Optional[str] = None,
    source_indices: Optional[Iterable[int]] = None,
) -> None:
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim:
        return
    if material_name:
        text = str(material_name).strip()
        if text:
            material_prim.CreateAttribute("ifc:materialName", Sdf.ValueTypeNames.String, custom=True).Set(text)
            _register_material_name(stage, text, material_path)
    new_values: Set[int] = set()
    if source_indices:
        for idx in source_indices:
            try:
                new_values.add(int(idx))
            except Exception:
                continue
    if new_values:
        attr = material_prim.GetAttribute("ifc:sourceMaterialIndices")
        existing: Set[int] = set()
        if attr and attr.HasAuthoredValue():
            try:
                existing_values = attr.Get()
                for value in existing_values:
                    try:
                        existing.add(int(value))
                    except Exception:
                        continue
            except Exception:
                existing = set()
        combined = sorted(existing.union(new_values))
        target_attr = attr or material_prim.CreateAttribute(
            "ifc:sourceMaterialIndices", Sdf.ValueTypeNames.IntArray, custom=True
        )
        target_attr.Set(Vt.IntArray(combined))
        if len(combined) == 1:
            material_prim.CreateAttribute("ifc:sourceMaterialIndex", Sdf.ValueTypeNames.Int, custom=True).Set(
                int(combined[0])
            )


def _propagate_subset_metadata_to_material(
    stage: Usd.Stage,
    subset_prim: Usd.Prim,
    material_path: Sdf.Path,
) -> None:
    if subset_prim is None or material_path is None:
        return
    material_name: Optional[str] = None
    name_attr = subset_prim.GetAttribute("ifc:materialName")
    if name_attr and name_attr.HasAuthoredValue():
        try:
            material_name = name_attr.Get()
        except Exception:
            material_name = None
    source_indices = _subset_indices_from_attrs(subset_prim)
    _update_material_metadata(stage, material_path, material_name=material_name, source_indices=source_indices)


def _material_name_from_usd_material(material: UsdShade.Material) -> Optional[str]:
    if not material:
        return None
    prim = material.GetPrim()
    if not prim:
        return None
    attr = prim.GetAttribute("ifc:materialName")
    if attr and attr.HasAuthoredValue():
        try:
            return str(attr.Get()).strip()
        except Exception:
            return None
    return None


def _bind_material_from_ifc_name(stage: Usd.Stage, prim: Usd.Prim) -> Optional[Sdf.Path]:
    if prim is None:
        return None
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if binding_api.GetDirectBinding().GetMaterial():
        return None
    attr = prim.GetAttribute("ifc:materialName")
    if not attr or not attr.HasAuthoredValue():
        return None
    try:
        value = attr.Get()
    except Exception:
        return None
    if not value:
        return None
    material_path = _material_path_by_name(stage, value)
    if not material_path:
        return None
    material = UsdShade.Material.Get(stage, material_path)
    if not material:
        return None
    binding_api.Bind(material)
    prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(str(material_path))
    _propagate_subset_metadata_to_material(stage, prim, material_path)
    return material_path


def _bind_ifc_named_materials(stage: Usd.Stage, mesh_path: Optional[Sdf.Path]) -> None:
    if stage is None or mesh_path is None:
        return
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim:
        return
    _bind_material_from_ifc_name(stage, mesh_prim)
    for child in mesh_prim.GetChildren():
        if child.GetTypeName() == "GeomSubset":
            _bind_material_from_ifc_name(stage, child)


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
    for attr_set in (
        ("Red", "Green", "Blue"),
        ("R", "G", "B"),
        ("red", "green", "blue"),
        ("r", "g", "b"),
    ):
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


def _material_properties_from_pbr(
    material: Any,
    fallback_name: str,
) -> Optional[_MaterialProperties]:
    if PBRMaterial is not None and isinstance(material, PBRMaterial):
        base_color = _normalize_color(tuple(float(c) for c in material.base_color[:3]))
        opacity = max(0.0, min(1.0, float(material.opacity)))
        emissive = None
        if material.emissive and max(material.emissive) > 0.0:
            emissive = _normalize_color(tuple(float(c) for c in material.emissive[:3]))
        name = material.name or fallback_name
        return _MaterialProperties(
            name=name,
            display_color=base_color,
            opacity=opacity,
            emissive_color=emissive,
            metallic=float(getattr(material, "metallic", 0.0)),
            roughness=float(getattr(material, "roughness", 0.5)),
            base_color_tex=getattr(material, "base_color_tex", None),
            normal_tex=getattr(material, "normal_tex", None),
            orm_tex=getattr(material, "orm_tex", None),
        )
    if isinstance(material, dict):
        base = material.get("base_color")
        if base is not None:
            try:
                base_color = _normalize_color(tuple(float(c) for c in base[:3]))
                opacity = max(0.0, min(1.0, float(material.get("opacity", 1.0))))
                emissive_value = material.get("emissive")
                emissive = None
                if emissive_value is not None:
                    emissive = _normalize_color(tuple(float(c) for c in emissive_value[:3]))
                name = str(material.get("name", fallback_name))
                return _MaterialProperties(
                    name=name,
                    display_color=base_color,
                    opacity=opacity,
                    emissive_color=emissive,
                    metallic=float(material.get("metallic", 0.0)),
                    roughness=float(material.get("roughness", 0.5)),
                    base_color_tex=material.get("base_color_tex"),
                    normal_tex=material.get("normal_tex"),
                    orm_tex=material.get("orm_tex"),
                )
            except Exception:
                pass
    return None


def _bind_materials_on_prototype_mesh(
    stage: Usd.Stage,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    caches: PrototypeCaches,
    library: MaterialLibrary,
) -> None:
    """Bind style-derived materials on prototype subset prims."""

    for key, proto in _iter_prototypes(caches):
        mesh_root = proto_paths.get(key)
        if not mesh_root:
            continue
        detail_payload = getattr(proto, "detail_mesh", None)
        if detail_payload and getattr(detail_payload, "faces", None):
            continue
        mesh_path = mesh_root.AppendChild("Geom")
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim:
            continue

        style_groups = getattr(proto, "style_face_groups", None) or {}
        if not style_groups:
            continue

        numeric_materials: Dict[int, Sdf.Path] = {}
        proto_label = getattr(proto, "type_name", None) or getattr(proto, "name", None) or "Prototype"
        for idx, mat in enumerate(getattr(proto, "materials", None) or []):
            props = _extract_material_properties(mat, prototype_label=proto_label, index=idx)
            if props is None:
                LOG.debug(
                    "Prototype %s material[%d] produced no usable properties (raw type=%s)",
                    mesh_root,
                    idx,
                    type(mat).__name__,
                )
                continue
            sig = _material_signature(props)
            mat_path = library.signature_to_path.get(sig)
            LOG.debug(
                "Prototype %s material[%d] -> signature %s -> %s",
                mesh_root,
                idx,
                sig,
                mat_path,
            )
            if mat_path:
                numeric_materials[idx] = mat_path

        material_ids_attr = mesh_prim.GetAttribute("ifc:materialIds")
        material_ids_seq: List[int] = []
        if material_ids_attr and material_ids_attr.IsValid():
            try:
                raw_ids = material_ids_attr.Get()
                if raw_ids:
                    material_ids_seq = list(raw_ids)
            except Exception:
                material_ids_seq = []
        LOG.debug("Prototype %s materialIds count: %d", mesh_path, len(material_ids_seq))

        style_tokens: Set[str] = set()
        style_token_to_material: Dict[str, Sdf.Path] = {}
        material_id_to_style_path: Dict[int, Sdf.Path] = {}
        material_conflicts_reported: Set[int] = set()
        style_material_path: Optional[Sdf.Path] = None

        for token, entry in style_groups.items():
            faces = entry.get("faces") or []
            mat_obj = entry.get("material")
            if not faces or not mat_obj:
                continue

            props = _material_properties_from_pbr(mat_obj, fallback_name=token)
            if props is None:
                continue
            signature = _material_signature(props)
            material_path = library.signature_to_path.get(signature)
            if material_path is None:
                continue

            subset_token = _subset_token_for_material(token)
            if style_material_path is None:
                style_material_path = material_path
            style_tokens.add(subset_token)
            style_token_to_material[subset_token] = material_path

            if material_ids_seq:
                for face_index in faces:
                    try:
                        face_int = int(face_index)
                    except Exception:
                        continue
                    if 0 <= face_int < len(material_ids_seq):
                        material_id = int(material_ids_seq[face_int])
                        existing = material_id_to_style_path.get(material_id)
                        if existing is None:
                            material_id_to_style_path[material_id] = material_path
                        elif existing != material_path and material_id not in material_conflicts_reported:
                            LOG.debug(
                                "Material ID %d already mapped to %s; skipping alternate style %s for prototype %s",
                                material_id,
                                existing,
                                material_path,
                                mesh_root,
                            )
                            material_conflicts_reported.add(material_id)

        cluster_binding: Dict[str, Sdf.Path] = {}
        for subset_token, material_path in style_token_to_material.items():
            cluster_binding[subset_token] = material_path
            subset_path = mesh_path.AppendChild(subset_token)
            subset_prim = stage.GetPrimAtPath(subset_path)
            if not subset_prim:
                continue

            material = UsdShade.Material.Get(stage, material_path)
            if not material:
                continue

            UsdShade.MaterialBindingAPI(subset_prim).Bind(material)
            LOG.debug("Bound style subset %s -> %s", subset_path, material_path)

        try:
            subsets = UsdGeom.Subset.GetGeomSubsets(
                UsdGeom.Imageable(mesh_prim),
                elementType=UsdGeom.Tokens.face,
            )
        except Exception:
            subsets = []
        subset_list = subsets or []

        if cluster_binding:
            for subset in subset_list:
                name = subset.GetPrim().GetName()
                if name in cluster_binding:
                    continue
                base_attr = subset.GetPrim().GetAttribute("ifc:baseStyleToken")
                base_token_val = None
                if base_attr and base_attr.HasAuthoredValue():
                    try:
                        base_token_val = str(base_attr.Get())
                    except Exception:
                        base_token_val = None
                if base_token_val:
                    mat_path = cluster_binding.get(base_token_val)
                    if mat_path:
                        material = UsdShade.Material.Get(stage, mat_path)
                        if material:
                            UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)
                            style_tokens.add(name)
                            LOG.debug("Bound clustered style subset %s -> %s", subset.GetPath(), mat_path)
                            continue
                for prefix, mat_path in cluster_binding.items():
                    if not name.startswith(prefix + "_C"):
                        continue
                    material = UsdShade.Material.Get(stage, mat_path)
                    if not material:
                        continue
                    UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)
                    style_tokens.add(name)
                    LOG.debug("Bound clustered style subset %s -> %s", subset.GetPath(), mat_path)
                    break

        for subset in subset_list:
            name = subset.GetPrim().GetName()
            if name in style_tokens:
                continue
            material_index = _material_index_from_subset_token(name)
            if material_index is None:
                continue
            material_path = material_id_to_style_path.get(material_index)
            rebound_from_style = False
            if material_path is None:
                material_path = numeric_materials.get(material_index)
            else:
                rebound_from_style = True
            if material_path is None and style_material_path is not None:
                material_path = style_material_path
            if not material_path:
                continue
            material = UsdShade.Material.Get(stage, material_path)
            if not material:
                continue
            UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)
            LOG.debug(
                "Bound numeric subset %s -> %s (%s)",
                subset.GetPath(),
                material_path,
                "style" if rebound_from_style else "library",
            )
            try:
                indices = subset.GetIndicesAttr().Get()
                if not indices:
                    LOG.debug("Numeric subset %s had no indices after binding.", subset.GetPath())
            except Exception:
                pass

        imageable = UsdGeom.Imageable(mesh_prim)
        try:
            if not UsdGeom.Subset.ValidateFamily(
                imageable,
                elementType=UsdGeom.Tokens.face,
                familyName="materialBind",
            ):
                LOG.warning(
                    "Material subset validation failed on prototype %s; face assignment may be incomplete.",
                    mesh_path,
                )
        except Exception:
            LOG.debug("Unable to validate material subsets for %s", mesh_path, exc_info=True)


def _extract_material_properties(
    material: Any,
    *,
    prototype_label: str,
    index: int,
) -> Optional[_MaterialProperties]:
    """Derive a usable material definition from an ifcopenshell geometry material."""
    fallback_name = f"{prototype_label}_Material_{index + 1}"
    pbr_props = _material_properties_from_pbr(material, fallback_name)
    if pbr_props is not None:
        return pbr_props

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
        name_value = fallback_name

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
        if isinstance(color_source, dict):
            nested = color_source.get("diffuse") or color_source.get("Diffuse")
            if isinstance(nested, dict):
                color = _resolve_color_value(
                    nested.get("colour")
                    or nested.get("color")
                    or nested.get("albedo")
                    or nested.get("rgb")
                    or nested
                )
        if color is None and isinstance(material, dict):
            diffuse_block = material.get("diffuse") or material.get("Diffuse")
            if isinstance(diffuse_block, dict):
                color = _resolve_color_value(
                    diffuse_block.get("colour")
                    or diffuse_block.get("color")
                    or diffuse_block.get("albedo")
                    or diffuse_block.get("rgb")
                    or diffuse_block
                )
        if color is None and hasattr(material, "diffuse"):
            diffuse_attr = getattr(material, "diffuse")
            color = _resolve_color_value(
                getattr(diffuse_attr, "colour", None)
                or getattr(diffuse_attr, "color", None)
                or diffuse_attr
            )
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
    """Unique key for material de-dup; include PBR scalars so Metal/Rough variants don’t collapse."""
    emissive = tuple(round(c, 5) for c in (props.emissive_color or ())) or None
    return (
        tuple(round(c, 5) for c in props.display_color),
        round(props.opacity, 5),
        round(float(getattr(props, "metallic", 0.0)), 5),
        round(float(getattr(props, "roughness", 0.5)), 5),
        emissive,
        sanitize_name(props.name, fallback="Material").lower(),
    )


def _author_preview_surface(stage: Usd.Stage, material_path: Sdf.Path, props: _MaterialProperties) -> UsdShade.Material:
    """
    Author a UsdPreviewSurface network and expose PBR controls via material inputs.
    This allows downstream edits on </World/Materials/...>.inputs:* and wires textures when available.
    """
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")

    # Material inputs (defaults)
    in_base = material.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f)
    in_base.Set(Gf.Vec3f(*props.display_color))
    in_opacity = material.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    in_opacity.Set(float(props.opacity))
    in_metal = material.CreateInput("metallic", Sdf.ValueTypeNames.Float)
    in_metal.Set(float(getattr(props, "metallic", 0.0)))
    in_rough = material.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    in_rough.Set(float(getattr(props, "roughness", 0.5)))
    in_emiss = None
    if props.emissive_color:
        in_emiss = material.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        in_emiss.Set(Gf.Vec3f(*props.emissive_color))

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(in_base)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(in_opacity)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(in_metal)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(in_rough)
    if in_emiss:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(in_emiss)

    st_reader: Optional[UsdShade.Shader] = None

    def _ensure_st_reader() -> UsdShade.Shader:
        nonlocal st_reader
        if st_reader:
            return st_reader
        st_reader = UsdShade.Shader.Define(stage, material_path.AppendChild("Primvar_st"))
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        return st_reader

    def _connect_uv_texture(slot: str, asset_path: Optional[str], channel: str = "rgb") -> None:
        if not asset_path:
            return
        tex = UsdShade.Shader.Define(stage, material_path.AppendChild(f"Tex_{slot}"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(asset_path)))
        reader = _ensure_st_reader()
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader, "result")
        target = material.GetInput(slot) or material.CreateInput(slot, Sdf.ValueTypeNames.Token)
        target.ConnectToSource(tex, channel)

    if getattr(props, "base_color_tex", None):
        _connect_uv_texture("baseColor", props.base_color_tex, "rgb")
    if getattr(props, "orm_tex", None):
        _connect_uv_texture("roughness", props.orm_tex, "g")
        _connect_uv_texture("metallic", props.orm_tex, "b")
    if getattr(props, "normal_tex", None):
        in_norm = material.GetInput("normal") or material.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(in_norm)
        tex = UsdShade.Shader.Define(stage, material_path.AppendChild("Tex_normal"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(props.normal_tex)))
        reader = _ensure_st_reader()
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader, "result")
        in_norm.ConnectToSource(tex, "rgb")

    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))

    prim = material.GetPrim()
    prim.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f, custom=True).Set(Gf.Vec3f(*props.display_color))
    prim.CreateAttribute("displayOpacity", Sdf.ValueTypeNames.Float, custom=True).Set(float(props.opacity))
    if props.name:
        text = str(props.name).strip()
        if text:
            prim.CreateAttribute("ifc:materialName", Sdf.ValueTypeNames.String, custom=True).Set(text)
            _register_material_name(stage, text, material_path)
    return material


def _apply_display_color_to_prim(prim: Usd.Prim, color: Optional[Gf.Vec3f]) -> None:
    if color is None or prim is None:
        return
    attr = prim.CreateAttribute("displayColor", Sdf.ValueTypeNames.Color3f, custom=True)
    attr.Set(color)


def _resolve_instance_materials(
    record: InstanceRecord,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    material_library: Optional[MaterialLibrary],
) -> Tuple[Dict[Any, Sdf.Path], Optional[Gf.Vec3f], Dict[str, Dict[str, Any]]]:
    materials_container: Any = record.materials if getattr(record, "materials", None) else None
    label = sanitize_name(record.name, fallback=f"Ifc_{record.step_id}")
    if not materials_container and record.prototype is not None:
        materials_container = _prototype_materials(caches, record.prototype)
        proto_path = proto_paths.get(record.prototype) if proto_paths else None
        if proto_path:
            label = sanitize_name(proto_path.name, fallback=label)
    resolved_paths: Dict[Any, Sdf.Path] = {}
    display_color: Optional[Gf.Vec3f] = None
    style_groups: Dict[str, Dict[str, Any]] = {}

    def _apply_style(style_obj: Any, source: str) -> None:
        nonlocal display_color
        if style_obj is None:
            return
        props = _material_properties_from_pbr(style_obj, label)
        if props is None:
            return
        if display_color is None:
            display_color = Gf.Vec3f(*props.display_color)
        if material_library is None:
            return
        signature = _material_signature(props)
        material_path = material_library.signature_to_path.get(signature)
        if material_path:
            resolved_paths.setdefault(f"__style__{source}", material_path)
            if "ARX" in label.upper() or "PROP" in label.upper():
                LOG.info(
                    "IFC style material applied for %s (%s): %s",
                    label,
                    source,
                    props,
                )

    _apply_style(getattr(record, "style_material", None), "instance")
    if record.prototype is not None:
        _apply_style(_prototype_materials(caches, record.prototype), "prototype")

    # Merge instance-level style groups over prototype groups so the
    # resolved view always reflects "how this instance presents".
    proto_groups = _prototype_style_groups(caches, record.prototype) if record.prototype is not None else {}
    inst_groups = record.style_face_groups or {}
    style_groups = _merge_style_face_groups(inst_groups, proto_groups)

    for entry_index, (material_key, material_value) in enumerate(_iter_material_entries(materials_container)):
        props = _extract_material_properties(material_value, prototype_label=label, index=entry_index)
        if props is not None and display_color is None:
            display_color = Gf.Vec3f(*props.display_color)
        if "ARX" in label.upper() or "PROP" in label.upper():
            try:
                LOG.info(
                    "IFC material debug for %s idx=%d raw=%s props=%s",
                    label,
                    material_key,
                    material_value,
                    props,
                )
            except Exception:
                LOG.info("IFC material debug for %s idx=%s (unprintable material)", label, material_key)
        if material_library is None or props is None:
            continue
        signature = _material_signature(props)
        material_path = material_library.signature_to_path.get(signature)
        if material_path:
            resolved_paths[material_key] = material_path

    if style_groups and material_library is not None:
        for key, entry in style_groups.items():
            material_obj = entry.get("material")
            if material_obj is None:
                continue
            props = _material_properties_from_pbr(material_obj, label)
            if props is None:
                continue
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(
                    "Style group %s (style_id=%s) -> PBR name=%s base=%s opacity=%.4f roughness=%.4f metallic=%.4f",
                    key,
                    entry.get("style_id"),
                    props.name,
                    tuple(round(c, 4) for c in props.display_color),
                    props.opacity,
                    props.roughness,
                    props.metallic,
                )
            signature = _material_signature(props)
            material_path = material_library.signature_to_path.get(signature)
            if material_path:
                resolved_paths.setdefault(key, material_path)
                # Prefer a style color for previews when available.
                if display_color is None:
                    display_color = Gf.Vec3f(*props.display_color)

    return resolved_paths, display_color, style_groups


def _apply_material_bindings_to_prim(
    stage: Usd.Stage,
    prim: Usd.Prim,
    resolved_materials: Dict[Any, Sdf.Path],
    *,
    style_groups: Optional[Dict[str, Dict[str, Any]]] = None,
    mesh_path: Optional[Sdf.Path] = None,
) -> Optional[Sdf.Path]:
    if not resolved_materials:
        if mesh_path is not None:
            _bind_ifc_named_materials(stage, mesh_path)
        return None
    items = list(resolved_materials.items())
    if not items:
        if mesh_path is not None:
            _bind_ifc_named_materials(stage, mesh_path)
        return None

    def _style_primary_key() -> Optional[Any]:
        for key in resolved_materials:
            if isinstance(key, str) and key.strip("'\"").startswith("__style__"):
                return key
        return None

    primary_key = _style_primary_key()
    primary_path = resolved_materials.get(primary_key) if primary_key is not None else None

    if primary_path is None:
        numeric_items = [item for item in items if isinstance(item[0], int)]
        if numeric_items:
            primary_key, primary_path = numeric_items[0]
        else:
            primary_key, primary_path = items[0]

    primary_material = UsdShade.Material.Get(stage, primary_path)
    if not primary_material:
        if mesh_path is not None:
            _bind_ifc_named_materials(stage, mesh_path)
        return None
    mesh_has_subsets = _mesh_has_geom_subsets(stage, mesh_path)
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if mesh_path is None or not mesh_has_subsets:
        binding_api.Bind(primary_material)
        prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(str(primary_path))
        _set_subset_material_name(prim, _material_name_from_usd_material(primary_material))
        _bind_ifc_named_materials(stage, mesh_path)
        return primary_path

    bound_subset_keys: Set[Any] = set()
    subset_bound = False

    def _bind_subset(key: Any, material_path: Sdf.Path) -> None:
        nonlocal subset_bound
        if key in bound_subset_keys:
            return
        subset_token = None
        if isinstance(key, str) and style_groups:
            entry = style_groups.get(key)
            if entry is not None:
                subset_token = entry.get("subset_token")
        if not subset_token:
            subset_token = _subset_token_for_material(key)
        subset_path = mesh_path.AppendChild(subset_token)
        existing = stage.GetPrimAtPath(subset_path)
        if not existing or existing.GetTypeName() != "GeomSubset":
            return
        subset_prim = stage.OverridePrim(subset_path)
        subset_material = UsdShade.Material.Get(stage, material_path)
        if not subset_material:
            return
        UsdShade.MaterialBindingAPI(subset_prim).Bind(subset_material)
        subset_name = _material_name_from_usd_material(subset_material)
        if subset_name:
            _set_subset_material_name(subset_prim, subset_name, overwrite=False)
        subset_prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(str(material_path))
        _propagate_subset_metadata_to_material(stage, subset_prim, material_path)
        bound_subset_keys.add(key)
        subset_bound = True

    style_keys = {
        key for key in resolved_materials if isinstance(key, str) and key.strip("'\"").startswith("__style__")
    }

    for key, material_path in items:
        if key == primary_key or key in style_keys:
            continue
        _bind_subset(key, material_path)

    if style_groups:
        for key, entry in style_groups.items():
            material_path = resolved_materials.get(key)
            if material_path is None:
                continue
            _bind_subset(key, material_path)

    subset_bound = subset_bound or _bind_style_subsets(stage, mesh_path, style_groups, resolved_materials)

    if subset_bound:
        try:
            binding_api.UnbindAllBindings()
        except Exception:
            pass
        _bind_ifc_named_materials(stage, mesh_path)
        return None

    binding_api.Bind(primary_material)
    prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(str(primary_path))
    _set_subset_material_name(prim, _material_name_from_usd_material(primary_material))
    _bind_ifc_named_materials(stage, mesh_path)
    return primary_path


def _canonical_material_key(k: Any) -> Any:
    """Normalize various material key shapes into a consistent, hashable form.

    Supported inputs seen in the pipeline:
      - int (material index)
      - tuple like (index, label)
      - dict (commonly {'material_id':..., 'item_id':..., ...}) -> material_id normalized to int
      - strings (style tokens)
      - PBRMaterial or dict material object -> try to extract stable identifier
    """
    # None stays None
    if k is None:
        return None
    # ints and simple scalars pass through
    if isinstance(k, (int, str)):
        return k
    # Already a tuple -> keep as-is (common: (index,label))
    if isinstance(k, tuple):
        return k
    # dict -> tuple of sorted items (stable ordering)
    if isinstance(k, dict):
        try:
            # try to prioritise material_id / item_id if present
            if "material_id" in k:
                mid = k.get("material_id")
                try:
                    mid_val = int(mid)
                except Exception:
                    mid_val = mid
                # OCC detail meshes emit {'material_id': X, 'item_id': Y}; normalise to the same slot key used by iterator meshes.
                return mid_val
            if "item_id" in k:
                iid = k.get("item_id")
                try:
                    iid_val = int(iid)
                except Exception:
                    iid_val = iid
                return ("dict", ("item_id", iid_val))
            # fallback: sorted items tuple
            return ("dict",) + tuple(sorted((str(a), k[a]) for a in k.keys()))
        except Exception:
            return tuple(sorted(k.items()))
    # try to extract index/Name from objects (e.g. PBRMaterial, ifcopenshell material)
    try:
        # PBRMaterial-like
        name = getattr(k, "name", None)
        if name:
            return ("name", str(name))
        # numeric id accessor
        if hasattr(k, "id"):
            try:
                return ("id", int(k.id()))
            except Exception:
                pass
    except Exception:
        pass
    # Fallback to str(key)
    return ("str", str(k))


def _bind_detail_materials(
    stage: Usd.Stage,
    mesh_geom: UsdGeom.Mesh,
    face_material_keys: Sequence[Any],
    resolved_materials: Dict[Any, Sdf.Path],
) -> None:
    mesh_path = mesh_geom.GetPath()
    if not resolved_materials:
        _bind_ifc_named_materials(stage, mesh_path)
        return

    keys = list(face_material_keys or [])

    # Normalize keys (handle dicts from OCC detail) into canonical forms
    normalized_keys = [_canonical_material_key(k) for k in keys]

    # Build canonical lookup for resolved_materials
    resolved_canonical: Dict[Any, Sdf.Path] = {}
    for rk, rp in resolved_materials.items():
        ck = _canonical_material_key(rk)
        # Avoid overwriting earlier entries—keep first mapping (deterministic)
        if ck not in resolved_canonical:
            resolved_canonical[ck] = rp

    valid_keys = [k for k in normalized_keys if k in resolved_canonical]
    
    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug(
            "Detail material binding: mesh=%s face_keys_sample=%s resolved_keys_sample=%s",
            mesh_path,
            normalized_keys[:5],
            list(resolved_canonical.keys())[:5]
        )
        # If no matches, also log ALL keys for debugging
        if not valid_keys:
            LOG.debug(
                "No material key matches for mesh %s. All normalized_keys: %s. All resolved_keys: %s",
                mesh_path,
                set(normalized_keys),
                set(resolved_canonical.keys())
            )
    
    if not valid_keys:
        LOG.debug("No valid material keys after canonicalization; skipping detail material binding for %s", mesh_path)
        _bind_ifc_named_materials(stage, mesh_path)
        return

    # preserve order while removing duplicates
    order: List[Any] = []
    seen: Set[Any] = set()
    for k in valid_keys:
        if k in seen:
            continue
        seen.add(k)
        order.append(k)

    if len(order) == 0:
        _bind_ifc_named_materials(stage, mesh_path)
        return

    binding_api = UsdShade.MaterialBindingAPI(mesh_geom)
    binding_api.UnbindAllBindings()

    primary_key = order[0]
    primary_material = UsdShade.Material.Get(stage, resolved_canonical.get(primary_key))
    if not primary_material:
        LOG.debug("Primary material not found for key %s on mesh %s", primary_key, mesh_path)
        _bind_ifc_named_materials(stage, mesh_path)
        return

    binding_api.Bind(primary_material)
    mesh_prim = mesh_geom.GetPrim()
    primary_material_path = resolved_canonical.get(primary_key)
    if primary_material_path:
        mesh_prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(
            str(primary_material_path)
        )
    _set_subset_material_name(mesh_prim, _material_name_from_usd_material(primary_material))

    family_name = "materialBind"
    face_lookup: Dict[Any, List[int]] = {}
    for face_index, nk in enumerate(normalized_keys):
        face_lookup.setdefault(nk, []).append(face_index)

    # create subsets for each canonical key present
    for ck, indices in face_lookup.items():
        # Skip creating a subset for the primary material if it covers everything or is handled by default
        if ck == primary_key and len(order) == 1:
            continue

        mat_path = resolved_canonical.get(ck)
        if not mat_path:
            continue
        try:
            subset_token = _subset_token_for_material(ck)
            
            # Use CreateUniqueGeomSubset for per-face material binding
            subset = UsdGeom.Subset.CreateUniqueGeomSubset(
                UsdGeom.Imageable(mesh_prim),
                indices,
                familyName=family_name,
                elementType=UsdGeom.Tokens.face,
                subsetName=subset_token,
            )
            
            subset_material = UsdShade.Material.Get(stage, mat_path)
            if subset_material:
                subset_prim = subset.GetPrim()
                mat_name = _material_name_from_usd_material(subset_material)
                _set_subset_material_name(subset_prim, mat_name)
                UsdShade.MaterialBindingAPI(subset_prim).Bind(subset_material)
                subset_path_val = subset_material.GetPath()
                subset_prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(
                    str(subset_path_val)
                )
                _propagate_subset_metadata_to_material(stage, subset_prim, subset_path_val)
        except Exception as exc:
            LOG.debug("Failed to create subset for material key %s on %s: %s", ck, mesh_path, exc)

    try:
        UsdGeom.Subset.SetFamilyType(mesh_geom, family_name, UsdGeom.Tokens.nonOverlapping)
    except Exception:
        pass
    _bind_ifc_named_materials(stage, mesh_path)


def _bind_style_subsets(
    stage: Usd.Stage,
    mesh_path: Optional[Sdf.Path],
    style_groups: Optional[Dict[str, Dict[str, Any]]],
    resolved_materials: Dict[Any, Sdf.Path],
) -> bool:
    if mesh_path is None or not style_groups:
        return False
    bound = False
    for key, entry in style_groups.items():
        material_path = resolved_materials.get(key)
        if not material_path:
            continue
        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            continue
        subset_token = (entry or {}).get("subset_token") or _subset_token_for_material(key)
        subset_path = mesh_path.AppendChild(subset_token)
        existing = stage.GetPrimAtPath(subset_path)
        if not existing or existing.GetTypeName() != "GeomSubset":
            continue
        subset_prim = stage.OverridePrim(subset_path)
        UsdShade.MaterialBindingAPI(subset_prim).Bind(material)
        subset_name = _material_name_from_usd_material(material)
        if subset_name:
            _set_subset_material_name(subset_prim, subset_name, overwrite=False)
        subset_prim.CreateAttribute("ifc:materialPrim", Sdf.ValueTypeNames.String, custom=True).Set(str(material_path))
        _propagate_subset_metadata_to_material(stage, subset_prim, material_path)
        bound = True
    return bound


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

    Materials are authored once per unique signature and returned as a lookup
    table so instances can bind the appropriate previews at author time.
    """
    material_layer = _prepare_writable_layer(layer_path)
    log = logging.getLogger(__name__)

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    proto_sub_path = _sublayer_identifier(proto_layer, layer_path)
    if proto_sub_path not in proto_layer.subLayerPaths:
        proto_layer.subLayerPaths.append(proto_sub_path)

    material_root = Sdf.Path("/World/Materials")
    signature_to_path: Dict[Tuple[Any, ...], Sdf.Path] = {}
    used_names: Dict[str, int] = {}

    def _material_sources() -> Iterable[Tuple[str, Any]]:
        for record in caches.instances.values():
            label = sanitize_name(record.name, fallback="Instance")
            materials = record.materials if getattr(record, "materials", None) else None
            if materials:
                yield label, materials
            style_groups = getattr(record, "style_face_groups", None) or {}
            if style_groups:
                group_materials = {
                    key: entry.get("material")
                    for key, entry in style_groups.items()
                    if entry.get("material") is not None
                }
                if group_materials:
                    yield f"{label}_styles", group_materials
            style = getattr(record, "style_material", None)
            if style is not None:
                yield f"{label}_style", {"style": style}
        for key, proto in _iter_prototypes(caches):
            materials = _prototype_materials(caches, key)
            proto_path = proto_paths.get(key) if proto_paths else None
            label_source = proto_path.name if proto_path else getattr(proto, "type_name", None) or getattr(proto, "name", None)
            label = sanitize_name(label_source, fallback="Prototype")
            if materials:
                yield label, materials
            style_groups = _prototype_style_groups(caches, key)
            if style_groups:
                group_materials = {
                    key_name: entry.get("material")
                    for key_name, entry in style_groups.items()
                    if entry.get("material") is not None
                }
                if group_materials:
                    yield f"{label}_styles", group_materials
            style_candidate = getattr(proto, "style_material", None)
            if style_candidate is not None:
                yield f"{label}_style", {"style": style_candidate}

    with Usd.EditContext(stage, material_layer):
        UsdGeom.Scope.Define(stage, material_root)

        for label, materials in _material_sources():
            for entry_index, (material_key, material_value) in enumerate(_iter_material_entries(materials)):
                props = _extract_material_properties(material_value, prototype_label=label, index=entry_index)
                if props is None:
                    continue
                signature = _material_signature(props)
                if signature in signature_to_path:
                    continue
                base_name = sanitize_name(props.name, fallback=f"{label}_Material")
                token = _unique_name(base_name, used_names)
                material_prim_path = material_root.AppendChild(token)
                _author_preview_surface(stage, material_prim_path, props)
                signature_to_path[signature] = material_prim_path
                log.debug("Authored material %s for signature %s", material_prim_path, signature)

    library = MaterialLibrary(signature_to_path)
    # Do not bind materials on /World/__Prototypes. We want all style/material
    # opinions authored per-instance so the same prototype can appear with different looks.
    return material_layer, library


# ---------------- Instance authoring ----------------

def author_instance_layer(
    stage: Usd.Stage,
    caches: PrototypeCaches,
    proto_paths: Dict[PrototypeKey, Sdf.Path],
    material_library: Optional[MaterialLibrary],
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
    :param material_library: Lookup of resolved materials authored on the material layer.
    :param layer_path: The path to the layer to be created.
    :param base_name: The base name of the layer.
    :param options: Optional conversion options.
    :return: The authored layer.
    """
    inst_layer = _prepare_writable_layer(layer_path)

    root_layer = stage.GetRootLayer()
    root_sub_path = _sublayer_identifier(root_layer, layer_path)
    if root_sub_path not in root_layer.subLayerPaths:
        root_layer.subLayerPaths.append(root_sub_path)

    inst_root_name = _sanitize_identifier(f"{base_name}_Instances", fallback="Instances")
    inst_root = Sdf.Path(f"/World/{inst_root_name}")
    setattr(caches, "instance_layer", inst_layer)
    setattr(caches, "instance_root_path", inst_root)

    from collections import defaultdict
    name_counters: Dict[Sdf.Path, Dict[str, int]] = defaultdict(dict)
    hierarchy_nodes_by_step: Dict[int, Sdf.Path] = {}
    hierarchy_nodes_by_label: Dict[Tuple[Sdf.Path, str], Sdf.Path] = {}

    detail_scope = getattr(options, "detail_scope", "all") or "all"
    detail_mode_enabled = bool(
        getattr(options, "detail_mode", False) or getattr(options, "enable_high_detail_remesh", False)
    )
    stage_meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)

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
            has_prototype = record.prototype is not None
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
                try:
                    inst_prim.CreateAttribute("ifc:GlobalId", Sdf.ValueTypeNames.String).Set(str(record.guid))
                except Exception:
                    pass
            if record.step_id is not None:
                try:
                    inst_prim.CreateAttribute("ifc:StepId", Sdf.ValueTypeNames.Int).Set(int(record.step_id))
                except Exception:
                    try:
                        inst_prim.CreateAttribute("ifc:StepId", Sdf.ValueTypeNames.String).Set(str(record.step_id))
                    except Exception:
                        pass

            if getattr(record, "ifc_path", None):
                try:
                    inst_prim.CreateAttribute("ifc:sourcePath", Sdf.ValueTypeNames.String).Set(str(record.ifc_path))
                except Exception:
                    pass

            if record.transform is not None:
                xform.AddTransformOp().Set(np_to_gf_matrix(record.transform))

            if getattr(options, "convert_metadata", True) and getattr(record, "attributes", None):
                _author_instance_attributes(inst_prim, record.attributes)

            resolved_materials, resolved_color, style_groups = _resolve_instance_materials(
                record, caches, proto_paths, material_library
            )
            material_ids = list(record.material_ids or [])

            detail_payload = getattr(record, "detail_mesh", None)
            has_detail_mesh = bool(
                detail_mode_enabled and detail_payload and getattr(detail_payload, "faces", None)
            )

            semantic_parts = getattr(record, "semantic_parts", None) or {}

            # Case: semantic sub-parts (no OCC detail mesh)
            if semantic_parts and not has_detail_mesh:
                geom_root = inst_path.AppendChild("Geom")
                for label, part in semantic_parts.items():
                    mesh_name = sanitize_name(label, fallback="Part")
                    part_mesh_dict = {
                        "vertices": np.asarray(part.get("vertices"), dtype=np.float32),
                        "faces": np.asarray(part.get("faces"), dtype=np.int64),
                    }
                    part_material_ids = part.get("material_ids") or []
                    mesh_geom = write_usd_mesh(
                        stage,
                        geom_root,
                        mesh_name,
                        part_mesh_dict,
                        abs_mat=None,
                        material_ids=part_material_ids if part_material_ids else None,
                        materials=list(getattr(record, "materials", []) or []),
                        stage_meters_per_unit=stage_meters_per_unit,
                        style_groups=part.get("style_groups") or {},
                    )
                    if resolved_materials:
                        _apply_material_bindings_to_prim(
                            stage,
                            mesh_geom.GetPrim(),
                            resolved_materials,
                            style_groups=style_groups,
                            mesh_path=mesh_geom.GetPath(),
                        )
                    if resolved_color is not None:
                        try:
                            mesh_gprim = UsdGeom.Gprim(mesh_geom.GetPrim())
                            mesh_gprim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([resolved_color]))
                        except Exception:
                            pass
                if resolved_color is not None:
                    _apply_display_color_to_prim(inst_prim, resolved_color)
                continue

            if has_detail_mesh:
                detail_root = inst_path.AppendChild("Geom")
                detail_entries = _author_occ_detail_meshes(
                    stage,
                    detail_root,
                    detail_payload,
                    meters_per_unit=stage_meters_per_unit,
                )
                if resolved_materials:
                    for mesh_geom, face_materials in detail_entries:
                        _bind_detail_materials(
                            stage,
                            mesh_geom,
                            face_materials,
                            resolved_materials,
                        )
                if resolved_color is not None:
                    for mesh_geom, _ in detail_entries:
                        try:
                            mesh_gprim = UsdGeom.Gprim(mesh_geom.GetPrim())
                            mesh_gprim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([resolved_color]))
                        except Exception:
                            pass
                continue

            # Per-instance mesh fallback (non-instanced geometry)
            if record.mesh:
                mesh_geom = write_usd_mesh(
                    stage,
                    inst_path,
                    "Geom",
                    record.mesh,
                    abs_mat=None,
                    material_ids=material_ids,
                    materials=list(getattr(record, "materials", []) or []),
                    stage_meters_per_unit=stage_meters_per_unit,
                    style_groups=style_groups,
                )
                if resolved_materials:
                    _apply_material_bindings_to_prim(
                        stage,
                        inst_prim,
                        resolved_materials,
                        style_groups=style_groups,
                        mesh_path=mesh_geom.GetPath(),
                    )
                if resolved_color is not None:
                    _apply_display_color_to_prim(inst_prim, resolved_color)
                    try:
                        mesh_gprim = UsdGeom.Gprim(mesh_geom.GetPrim())
                        mesh_gprim.CreateDisplayColorAttr().Set(Vt.Vec3fArray([resolved_color]))
                    except Exception:
                        pass
                continue

            # Prototype reference
            if record.prototype is None:
                continue
            proto_path = proto_paths.get(record.prototype)
            if proto_path is None:
                continue

            proto_detail = _prototype_from_key(caches, record.prototype)
            proto_detail_mesh = getattr(proto_detail, "detail_mesh", None) if proto_detail else None
            proto_has_detail = bool(proto_detail_mesh and getattr(proto_detail_mesh, "faces", None))

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
            mesh_child_path = ref_path.AppendChild("Geom")
            mesh_child_override = stage.OverridePrim(mesh_child_path)
            UsdGeom.Imageable(mesh_child_override).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)

            # If this instance supplies its own style groups and/or per-face material ids,
            # re-author the GeomSubsets on the *instance override* so faces match the instance.
            try:
                mesh_geom = UsdGeom.Mesh(mesh_child_override)
                counts_val = mesh_geom.GetFaceVertexCountsAttr().Get() or []
                face_count_for_inst = len(counts_val)
            except Exception:
                face_count_for_inst = 0
            if face_count_for_inst > 0 and (style_groups or material_ids):
                _ensure_material_subsets(
                    mesh_geom,
                    material_ids if material_ids else [],
                    face_count_for_inst,
                    style_groups=style_groups if style_groups else {},
                    materials=list(getattr(record, "materials", []) or []),
                )

            if resolved_materials:
                # Bind on the *instance* override path, not on the prototype.
                # This makes per-instance style/material changes stick.
                instance_mesh_path = None if proto_has_detail else mesh_child_path
                _apply_material_bindings_to_prim(
                    stage,
                    inst_prim,
                    resolved_materials,
                    style_groups=style_groups,
                    mesh_path=instance_mesh_path,
                )
            if resolved_color is not None:
                _apply_display_color_to_prim(inst_prim, resolved_color)
                try:
                    mesh_child_override.CreateAttribute(
                        "displayColor", Sdf.ValueTypeNames.Color3f, custom=True
                    ).Set(resolved_color)
                except Exception:
                    pass

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

    geom_layer = _prepare_writable_layer(layer_path)

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
                curve_count = max(1, len(counts))
                widths = Vt.FloatArray(curve_count)
                for idx in range(curve_count):
                    widths[idx] = float(width_value)
                basis.CreateWidthsAttr(widths)
                basis.SetWidthsInterpolation(UsdGeom.Tokens.uniform)

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
    shared_site_base_point: Optional[BasePointConfig] = None,
    anchor_mode: Optional[Literal["local", "site", "basepoint", "shared_site"]] = None,
    align_axes_to_map: bool = False,
    enable_geo_anchor: bool = True,  # kept for signature compatibility (unused)
    projected_crs: Optional[str] = None,
    lonlat: Optional[GeodeticCoordinate] = None,
    **_unused,
) -> None:
    """Anchor metadata writer.

    NEW BEHAVIOUR:
      * Does NOT apply any transform ops to /World or the instance roots.
      * All basepoint/map offsets are already baked into instance matrices.
      * This function only writes anchor / geolocation attributes.
    """

    if anchor_mode is None:
        return

    mode_key = anchor_mode.strip().lower() if isinstance(anchor_mode, str) else str(anchor_mode).strip().lower()
    if not mode_key:
        return

    alias_map = {
        "local": "local",
        "site": "site",
        "basepoint": "local",
        "shared_site": "site",
        "none": None,
    }
    mode_normalised = alias_map.get(mode_key)
    if mode_normalised is None:
        LOG.debug("Unknown anchor_mode '%s'; skipping stage anchoring", anchor_mode)
        return

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        world_xf = UsdGeom.Xform.Define(stage, "/World")
        world_prim = world_xf.GetPrim()
    # NOTE: do NOT ClearXformOpOrder or add any transform ops here anymore.
    world_xf = UsdGeom.Xformable(world_prim)

    stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    if stage_meters_per_unit <= 0.0:
        stage_meters_per_unit = 1.0

    zero_base_point = BasePointConfig(easting=0.0, northing=0.0, height=0.0, unit="m")
    shared_site_resolved = (
        shared_site_base_point.with_fallback(zero_base_point)
        if shared_site_base_point
        else zero_base_point
    )
    base_point_resolved = (
        base_point.with_fallback(shared_site_resolved) if base_point else shared_site_resolved
    )

    effective_mode = mode_normalised
    if effective_mode == "local" and base_point is None:
        effective_mode = "site"
    anchor_bp = shared_site_resolved if effective_mode == "site" else base_point_resolved

    anchor_easting_m, anchor_northing_m, anchor_height_m, anchor_unit = _base_point_components_in_meters(anchor_bp)
    shared_easting_m, shared_northing_m, shared_height_m, shared_unit = _base_point_components_in_meters(
        shared_site_resolved
    )
    (
        model_bp_easting_m,
        model_bp_northing_m,
        model_bp_height_m,
        model_bp_unit,
    ) = _base_point_components_in_meters(base_point_resolved)

    map_conv = getattr(caches, "map_conversion", None)
    translation_stage = Gf.Vec3d(0.0, 0.0, 0.0)
    rotation_deg = 0.0

    if map_conv is not None:
        scale = float(getattr(map_conv, "scale", 1.0) or 1.0)
        eastings_m = float(getattr(map_conv, "eastings", 0.0) or 0.0) * scale
        northings_m = float(getattr(map_conv, "northings", 0.0) or 0.0) * scale
        ortho_height_m = float(getattr(map_conv, "orthogonal_height", 0.0) or 0.0) * scale
        d_e = anchor_easting_m - eastings_m
        d_n = anchor_northing_m - northings_m
        ax, ay = map_conv.normalized_axes()
        local_x = ax * d_e + ay * d_n
        local_y = -ay * d_e + ax * d_n
        local_z = anchor_height_m - ortho_height_m
        # These are now metadata only; no transform ops applied.
        translation_stage = Gf.Vec3d(
            -local_x / stage_meters_per_unit,
            -local_y / stage_meters_per_unit,
            -local_z / stage_meters_per_unit,
        )
        if align_axes_to_map:
            try:
                rotation_deg = float(map_conv.rotation_degrees())
            except Exception:  # pragma: no cover - defensive
                rotation_deg = 0.0
    else:
        translation_stage = Gf.Vec3d(
            -anchor_easting_m / stage_meters_per_unit,
            -anchor_northing_m / stage_meters_per_unit,
            -anchor_height_m / stage_meters_per_unit,
        )

    # NOTE: anchor_matrix is no longer used to transform any prims; we keep only metadata.
    base_geodetic = _derive_lonlat_from_base_point(base_point_resolved, projected_crs)
    site_geodetic = _derive_lonlat_from_base_point(shared_site_resolved, projected_crs)
    if lonlat is not None:
        if effective_mode == "site":
            site_geodetic = lonlat
        else:
            base_geodetic = lonlat
    anchor_geodetic = site_geodetic if effective_mode == "site" else base_geodetic

    attr_values: List[Tuple[str, Any, Any]] = []

    def _append_attr(name: str, value_type: Any, value: Any) -> None:
        if value is None:
            return
        attr_values.append((name, value_type, value))

    _append_attr("ifc:stageAnchorMode", Sdf.ValueTypeNames.String, effective_mode)
    _append_attr(
        "ifc:stageAnchorTranslation",
        Sdf.ValueTypeNames.Double3,
        tuple(float(translation_stage[i]) for i in range(3)),
    )
    _append_attr("ifc:stageAnchorRotationDegrees", Sdf.ValueTypeNames.Double, float(rotation_deg))
    _append_attr(
        "ifc:stageAnchorBasePointMeters",
        Sdf.ValueTypeNames.Double3,
        (float(anchor_easting_m), float(anchor_northing_m), float(anchor_height_m)),
    )
    _append_attr("ifc:stageAnchorBasePointUnit", Sdf.ValueTypeNames.String, anchor_unit or "")
    _append_attr(
        "ifc:stageAnchorModelBasePointMeters",
        Sdf.ValueTypeNames.Double3,
        (float(model_bp_easting_m), float(model_bp_northing_m), float(model_bp_height_m)),
    )
    _append_attr("ifc:stageAnchorModelBasePointUnit", Sdf.ValueTypeNames.String, model_bp_unit or "")
    _append_attr(
        "ifc:stageAnchorSharedSiteMeters",
        Sdf.ValueTypeNames.Double3,
        (float(shared_easting_m), float(shared_northing_m), float(shared_height_m)),
    )
    _append_attr("ifc:stageAnchorSharedSiteUnit", Sdf.ValueTypeNames.String, shared_unit or "")
    _append_attr("ifc:stageAnchorProjectedCRS", Sdf.ValueTypeNames.String, projected_crs or "")
    if map_conv is not None:
        _append_attr(
            "ifc:stageAnchorMapScale",
            Sdf.ValueTypeNames.Double,
            float(getattr(map_conv, "scale", 1.0) or 1.0),
        )

    if anchor_geodetic is not None:
        _append_attr("ifc:longitude", Sdf.ValueTypeNames.Double, float(anchor_geodetic.longitude))
        _append_attr("ifc:latitude", Sdf.ValueTypeNames.Double, float(anchor_geodetic.latitude))
        if anchor_geodetic.height is not None:
            _append_attr("ifc:ellipsoidalHeight", Sdf.ValueTypeNames.Double, float(anchor_geodetic.height))
        _append_attr("CesiumLongitude", Sdf.ValueTypeNames.Double, float(anchor_geodetic.longitude))
        _append_attr("CesiumLatitude", Sdf.ValueTypeNames.Double, float(anchor_geodetic.latitude))
        if anchor_geodetic.height is not None:
            _append_attr("CesiumHeight", Sdf.ValueTypeNames.Double, float(anchor_geodetic.height))

    if base_geodetic is not None:
        _append_attr("ifc:stageAnchorModelLongitude", Sdf.ValueTypeNames.Double, float(base_geodetic.longitude))
        _append_attr("ifc:stageAnchorModelLatitude", Sdf.ValueTypeNames.Double, float(base_geodetic.latitude))
        if base_geodetic.height is not None:
            _append_attr(
                "ifc:stageAnchorModelEllipsoidalHeight",
                Sdf.ValueTypeNames.Double,
                float(base_geodetic.height),
            )
        _append_attr("CesiumModelLongitude", Sdf.ValueTypeNames.Double, float(base_geodetic.longitude))
        _append_attr("CesiumModelLatitude", Sdf.ValueTypeNames.Double, float(base_geodetic.latitude))
        if base_geodetic.height is not None:
            _append_attr("CesiumModelHeight", Sdf.ValueTypeNames.Double, float(base_geodetic.height))

    if site_geodetic is not None:
        _append_attr("ifc:stageAnchorSharedSiteLongitude", Sdf.ValueTypeNames.Double, float(site_geodetic.longitude))
        _append_attr("ifc:stageAnchorSharedSiteLatitude", Sdf.ValueTypeNames.Double, float(site_geodetic.latitude))
        if site_geodetic.height is not None:
            _append_attr(
                "ifc:stageAnchorSharedSiteEllipsoidalHeight",
                Sdf.ValueTypeNames.Double,
                float(site_geodetic.height),
            )
        _append_attr("CesiumSharedSiteLongitude", Sdf.ValueTypeNames.Double, float(site_geodetic.longitude))
        _append_attr("CesiumSharedSiteLatitude", Sdf.ValueTypeNames.Double, float(site_geodetic.latitude))
        if site_geodetic.height is not None:
            _append_attr("CesiumSharedSiteHeight", Sdf.ValueTypeNames.Double, float(site_geodetic.height))

    attr_names_to_cleanup = [name for name, _, _ in attr_values]

    instance_root_paths: List[Sdf.Path] = []
    recorded_root = getattr(caches, "instance_root_path", None)
    if recorded_root:
        if isinstance(recorded_root, Sdf.Path):
            instance_root_paths.append(recorded_root)
        else:
            try:
                instance_root_paths.append(Sdf.Path(str(recorded_root)))
            except Exception:
                LOG.debug("Unable to coerce recorded instance root path %r into Sdf.Path", recorded_root)

    if not instance_root_paths:
        for child in world_prim.GetChildren():
            if child.GetTypeName() == "Xform" and child.GetName().endswith("_Instances"):
                instance_root_paths.append(child.GetPath())

    instance_layer = getattr(caches, "instance_layer", None)
    applied_to_instance = False

    def _apply_to_prim(prim: Usd.Prim, layer: Optional[Sdf.Layer]) -> None:
        # NOTE: no transform ops are authored here; we only set attributes.
        context = Usd.EditContext(stage, layer) if layer is not None else nullcontext()
        with context:
            for name, value_type, value in attr_values:
                attr = prim.CreateAttribute(name, value_type)
                attr.Set(value)

    for root_path in instance_root_paths:
        prim = stage.GetPrimAtPath(root_path)
        if not prim:
            continue
        _apply_to_prim(prim, instance_layer)
        applied_to_instance = True

    if not applied_to_instance:
        LOG.debug("Instance root not found; applying geo anchor metadata on /World fallback.")
        _apply_to_prim(world_prim, None)
    else:
        # When we have an instance root, move the metadata off /World and onto the instance layer.
        root_layer = stage.GetRootLayer()
        cleanup_context = Usd.EditContext(stage, root_layer) if root_layer is not None else nullcontext()
        with cleanup_context:
            for name in attr_names_to_cleanup:
                if world_prim.HasProperty(name):
                    try:
                        world_prim.RemoveProperty(name)
                    except Exception:
                        LOG.debug("Unable to remove world attribute %s during anchor relocation", name)


# ---------------- Federated master view - stub (retain signature) ----------------

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
    try:
        mesh_child_path = target_path.AppendChild("Geom")
        mesh_override = stage.OverridePrim(mesh_child_path)
        UsdGeom.Imageable(mesh_override).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
    except Exception:
        pass
    stage.GetRootLayer().Save()
    return stage.GetRootLayer()
