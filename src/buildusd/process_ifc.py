"""IFC geometry inventory and prototype construction utilities.

This module wraps a single iterator-driven pass over an IFC file and produces the
data needed by the USD writer.  It handles:

* computing stable prototype keys (IfcType + mesh fingerprint + tessellation settings)
* resolving world transforms and spatial hierarchy for every instance
* extracting optional annotation/2D geometry and map anchor metadata
* filtering out hidden layers and auxiliary elements (e.g. IfcOpeningElement)

Most helpers are thin translations of the logic we previously maintained across a
larger codebase.  Docstrings below call out the subtle pieces so future tweaks do not
trip over the same unit-conversion or placement traps again.
"""

from __future__ import annotations
import logging
from collections import Counter
import multiprocessing
import numpy as np
import ifcopenshell
import os
import math
import re
from typing import Dict, Optional, Union, Literal, List, Tuple, Any, TYPE_CHECKING, Iterable, Set
from dataclasses import dataclass, field, replace
import hashlib

from .ifc_visuals import (
    PBRMaterial,
    build_material_for_product,
    extract_face_style_groups,
    get_face_styles,
    pbr_from_surface_style,
)
from . import semantic_subcomponents
from . import occ_detail
from .process_ifc_2d import AnnotationCurve, AnnotationHooks, extract_annotation_curves

if TYPE_CHECKING:
    from .occ_detail import OCCDetailMesh


def _clone_materials(source):
    if source is None:
        return None
    if isinstance(source, dict):
        cloned: Dict[Any, Any] = {}
        for key, value in source.items():
            if hasattr(value, "copy"):
                try:
                    cloned[key] = value.copy()
                except Exception:
                    cloned[key] = value
            else:
                cloned[key] = value
        return cloned
    if isinstance(source, (list, tuple)):
        return [value for value in source]
    return source


def _clone_style_groups(groups):
    if not groups:
        return {}
    cloned: Dict[str, Dict[str, Any]] = {}
    for key, entry in groups.items():
        material = entry.get("material")
        faces = [int(idx) for idx in entry.get("faces", [])]
        cloned[key] = {"material": material, "faces": faces}
        if "style_id" in entry:
            cloned[key]["style_id"] = entry["style_id"]
    return cloned

_STYLE_RENDER_TOKEN_RE = re.compile(r"IfcSurfaceStyleRendering[-_](\d+)", re.IGNORECASE)
_RENDER_STYLE_CACHE: Dict[int, Dict[int, ifcopenshell.entity_instance]] = {}


def _render_id_from_style_name(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    match = _STYLE_RENDER_TOKEN_RE.search(str(name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _pbr_from_ifc_style(style_obj, fallback_name: str) -> PBRMaterial:
    """Convert an ifcopenshell style wrapper into a simple PBR material."""
    base_color = (0.8, 0.8, 0.8)
    try:
        diffuse = getattr(style_obj, "diffuse", None)
        if diffuse is not None:
            base_color = (
                float(getattr(diffuse, "r", base_color[0])),
                float(getattr(diffuse, "g", base_color[1])),
                float(getattr(diffuse, "b", base_color[2])),
            )
    except Exception:
        pass
    opacity = 1.0
    try:
        transparency = getattr(style_obj, "transparency", None)
        if transparency is not None:
            opacity = max(0.0, min(1.0, 1.0 - float(transparency)))
    except Exception:
        pass
    roughness = 0.5
    try:
        spec = getattr(style_obj, "specular", None)
        if spec is not None:
            comps = [
                float(getattr(spec, attr, 0.5))
                for attr in ("r", "g", "b")
            ]
            level = max(comps)
            roughness = max(0.05, 1.0 - min(0.95, level))
    except Exception:
        pass
    base_name = fallback_name or getattr(style_obj, "name", None)
    sanitized = sanitize_name(base_name, fallback=fallback_name or "Material")
    return PBRMaterial(name=sanitized, base_color=base_color, opacity=opacity, roughness=roughness)


def _render_style_map(ifc_file) -> Dict[int, ifcopenshell.entity_instance]:
    cache_key = id(ifc_file)
    mapping = _RENDER_STYLE_CACHE.get(cache_key)
    if mapping is not None:
        return mapping
    mapping = {}
    try:
        for style in ifc_file.by_type("IfcSurfaceStyle"):
            for element in getattr(style, "Styles", []) or []:
                if element is None or not element.is_a("IfcSurfaceStyleRendering"):
                    continue
                try:
                    mapping[int(element.id())] = style
                except Exception:
                    continue
    except Exception:
        pass
    _RENDER_STYLE_CACHE[cache_key] = mapping
    return mapping


def _normalize_geom_materials(materials, face_style_groups, ifc_file):
    """Map iterator-provided style wrappers to friendly PBR materials."""
    if not isinstance(materials, (list, tuple)):
        return materials
    style_lookup: Dict[int, PBRMaterial] = {}
    for entry in (face_style_groups or {}).values():
        style_id = entry.get("style_id")
        mat = entry.get("material")
        if style_id is None or mat is None or not isinstance(mat, PBRMaterial):
            continue
        try:
            style_lookup[int(style_id)] = mat
        except Exception:
            continue
    render_map = _render_style_map(ifc_file)
    converted: List[PBRMaterial] = []
    for idx, entry in enumerate(materials):
        if isinstance(entry, PBRMaterial):
            converted.append(entry)
            continue
        render_name = getattr(entry, "name", None)
        render_id = _render_id_from_style_name(render_name)
        style_entity = render_map.get(render_id) if render_id is not None else None
        style_id = None
        style_name = None
        if style_entity is not None:
            try:
                style_id = int(style_entity.id())
            except Exception:
                style_id = None
            style_name = getattr(style_entity, "Name", None)
        if style_id is not None:
            existing = style_lookup.get(style_id)
            if existing is not None:
                converted.append(existing)
                continue
        fallback = sanitize_name(style_name, fallback=f"Material_{idx}")
        if style_entity is not None:
            pbr = pbr_from_surface_style(style_entity, model=ifc_file, preferred_name=fallback)
            converted.append(pbr)
            if style_id is not None:
                style_lookup.setdefault(style_id, pbr)
            continue
        pbr = _pbr_from_ifc_style(entry, fallback)
        converted.append(pbr)
        if style_id is not None:
            style_lookup.setdefault(style_id, pbr)
    return converted


def _pbr_signature(mat: Any) -> Optional[Tuple]:
    """Stable signature for PBRMaterial comparison (rounded floats)."""
    if not isinstance(mat, PBRMaterial):
        return None
    def _round3(val):
        try:
            return round(float(val), 6)
        except Exception:
            return None
    base = tuple(_round3(c) for c in getattr(mat, "base_color", ()))
    emissive = tuple(_round3(c) for c in getattr(mat, "emissive", ()))
    return (
        sanitize_name(getattr(mat, "name", None), fallback="Material"),
        base,
        _round3(getattr(mat, "opacity", None)),
        _round3(getattr(mat, "metallic", None)),
        _round3(getattr(mat, "roughness", None)),
        emissive,
        getattr(mat, "base_color_tex", None),
        getattr(mat, "normal_tex", None),
        getattr(mat, "orm_tex", None),
    )


def _log_semantic_attempt(product, mesh_dict, face_style_groups, materials, *, label: str) -> None:
    """Emit a concise debug line before running semantic splitting."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    face_count = 0
    try:
        faces = mesh_dict.get("faces") if isinstance(mesh_dict, dict) else None
        if faces is not None:
            face_count = int(np.asarray(faces).shape[0])
    except Exception:
        face_count = 0
    try:
        aspect_names = [
            _norm(getattr(a, "Name", None) or getattr(a, "Description", None) or getattr(a, "Identification", None))
            for a in (getattr(product, "HasShapeAspects", []) or [])
        ]
        aspect_names = [a for a in aspect_names if a]
    except Exception:
        aspect_names = []
    try:
        rep_ids = []
        rep_container = getattr(product, "Representation", None)
        reps = getattr(rep_container, "Representations", []) or []
        for rep in reps:
            ident = getattr(rep, "RepresentationIdentifier", None)
            if ident:
                rep_ids.append(str(ident))
        rep_ids = list(dict.fromkeys(rep_ids))
    except Exception:
        rep_ids = []
    style_keys = list((face_style_groups or {}).keys())
    log.debug(
        "Semantic split [%s]: prod=%s guid=%s faces=%s style_groups=%d rep_ids=%s aspects=%s materials=%d",
        label,
        product.is_a() if hasattr(product, "is_a") else "<IfcProduct>",
        getattr(product, "GlobalId", None),
        face_count,
        len(style_keys),
        rep_ids,
        aspect_names,
        len(materials or []),
    )


def _face_group_map(groups: Dict[str, Dict[str, Any]]) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    """Index face-style groups by face tuple for comparison."""
    mapping: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for token, entry in (groups or {}).items():
        faces = tuple(sorted(int(idx) for idx in entry.get("faces", []) if idx is not None))
        if not faces:
            continue
        mapping[faces] = entry
    return mapping


def _variant_suffix(kind: str, group_map: Dict[Tuple[int, ...], Dict[str, Any]], materials: Any) -> str:
    h = hashlib.sha1()
    h.update(kind.encode("utf-8"))
    for faces in sorted(group_map.keys()):
        h.update(str(faces).encode("utf-8"))
        h.update(str(_pbr_signature(group_map[faces].get("material"))).encode("utf-8"))
    if not group_map:
        for mat in materials or []:
            h.update(str(_pbr_signature(mat)).encode("utf-8"))
    return h.hexdigest()[:8]


def _variantize_material(mat: Any, suffix: str) -> Any:
    if not isinstance(mat, PBRMaterial):
        return mat
    name = sanitize_name(getattr(mat, "name", None), fallback="Material")
    return replace(mat, name=sanitize_name(f"{name}_var_{suffix}", fallback=name))


def _variantize_materials(materials: Any, suffix: str):
    if not isinstance(materials, (list, tuple)):
        return materials
    return [ _variantize_material(mat, suffix) for mat in materials ]


def _variantize_face_groups(groups: Dict[str, Dict[str, Any]], suffix: str) -> Dict[str, Dict[str, Any]]:
    if not groups:
        return {}
    variant: Dict[str, Dict[str, Any]] = {}
    for token, entry in groups.items():
        new_entry = dict(entry)
        if "material" in new_entry:
            new_entry["material"] = _variantize_material(new_entry["material"], suffix)
        variant[token] = new_entry
    return variant


def _instance_variant_kind(proto: Optional[MeshProto], materials: Any, face_style_groups: Dict[str, Dict[str, Any]], style_material: Optional[PBRMaterial]) -> Tuple[Optional[str], Optional[str]]:
    """Compare instance material/style with prototype to decide variant kind and suffix."""
    if proto is None:
        return None, None
    proto_groups = getattr(proto, "style_face_groups", None) or {}
    inst_groups = face_style_groups or {}
    proto_map = _face_group_map(proto_groups)
    inst_map = _face_group_map(inst_groups)

    if proto_map or inst_map:
        if set(proto_map.keys()) != set(inst_map.keys()):
            suffix = _variant_suffix("material", inst_map, materials)
            return "material", suffix
        style_diff = False
        for faces, inst_entry in inst_map.items():
            proto_entry = proto_map.get(faces)
            if proto_entry is None:
                style_diff = True
                break
            if _pbr_signature(inst_entry.get("material")) != _pbr_signature(proto_entry.get("material")):
                style_diff = True
                break
        if style_diff:
            suffix = _variant_suffix("style", inst_map, materials)
            return "style", suffix

    proto_materials = getattr(proto, "materials", None) or []
    inst_materials = materials or []
    if proto_materials or inst_materials:
        if len(proto_materials) != len(inst_materials):
            suffix = _variant_suffix("material", inst_map, inst_materials)
            return "material", suffix
        mat_diff = any(
            _pbr_signature(im) != _pbr_signature(pm)
            for im, pm in zip(inst_materials, proto_materials)
        )
        if mat_diff:
            suffix = _variant_suffix("style", inst_map, inst_materials)
            return "style", suffix

    # Consider object-level style material differences.
    proto_style_sig = _pbr_signature(getattr(proto, "style_material", None))
    inst_style_sig = _pbr_signature(style_material)
    if inst_style_sig is not None and proto_style_sig is not None and inst_style_sig != proto_style_sig:
        suffix = _variant_suffix("stylemat", inst_map, inst_materials)
        return "style", suffix

    return None, None

def _build_object_scope_detail_mesh(
    ctx: PrototypeBuildContext,
    product,
    *,
    material_resolver,
    reference_shape=None,
    canonical_map=None,
) -> Optional["OCCDetailMesh"]:
    if ctx.object_scope_settings is None:
        return None
    log.info(
        "Detail scope object: building OCC mesh for %s name=%s guid=%s",
        product.is_a() if hasattr(product, "is_a") else "<product>",
        getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "<unnamed>",
        getattr(product, "GlobalId", None),
    )
    try:
        shape_obj = ifcopenshell.geom.create_shape(ctx.object_scope_settings, product)
    except Exception as exc:
        log.warning(
            "Object-scope detail: failed to create OCC shape for %s guid=%s: %s",
            product.is_a() if hasattr(product, "is_a") else "<product>",
            getattr(product, "GlobalId", None),
            exc,
        )
        return None
    geometry = getattr(shape_obj, "geometry", shape_obj)
    if geometry is None:
        log.warning(
            "Object-scope detail: OCC geometry empty for %s guid=%s",
            product.is_a() if hasattr(product, "is_a") else "<product>",
            getattr(product, "GlobalId", None),
        )
        return None
    return occ_detail.build_detail_mesh_payload(
        geometry,
        ctx.object_scope_settings,
        product=product,
        default_linear_def=_DEFAULT_LINEAR_DEF,
        default_angular_def=_DEFAULT_ANGULAR_DEF,
        logger=log,
        detail_level=ctx.detail_level,
        material_resolver=material_resolver,
        reference_shape=reference_shape,
        canonical_map=canonical_map,
    )


def _build_detail_material_resolver(
    ifc_file,
    product,
    style_token_by_style_id: Dict[int, str],
    has_instance_style: bool,
    default_material_key: Optional[Any],
):
    face_styles_cache: Optional[Dict[int, List[Any]]] = None

    def _resolver(item) -> Optional[Any]:
        nonlocal face_styles_cache
        if face_styles_cache is None:
            try:
                face_styles_cache = get_face_styles(ifc_file, product) or {}
            except Exception:
                face_styles_cache = {}
        styles = face_styles_cache.get(id(item)) if face_styles_cache else None
        if styles:
            for style in styles:
                try:
                    style_id = int(style.id())
                except Exception:
                    style_id = None
                if style_id is not None:
                    token = style_token_by_style_id.get(style_id)
                    if token:
                        return token
        if has_instance_style:
            return "__style__instance"
        return default_material_key

    return _resolver

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

log = logging.getLogger(__name__)
_OCC_WARNING_EMITTED = False

# ---------------------------------
# Thread count helper
# ---------------------------------

def _resolve_threads(env_var="IFC_GEOM_THREADS", minimum=1):
    """Determine how many geometry threads to request from ifcopenshell."""
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
_BASE_GEOM_SETTINGS: Dict[str, Any] = {
    "use-world-coords": False,
    "weld-vertices": True,
    "disable-opening-subtractions": False,
    "apply-default-materials": True,
    "mesher-linear-deflection": 0.005,
    "mesher-angular-deflection": 0.5,
    # NEW high-level controls (interpreted by GeometrySettingsManager)
    # Model offset in *model units*; applied according to offset-type.
    "model-offset": (0.0, 0.0, 0.0),
    # "positive" → offset added; "negative" → supplied offset is negated.
    "offset-type": "negative",
    # Model rotation stored as quaternion (x, y, z, w). We accept axis+angle
    # input in degrees and convert it to quaternion in the settings manager.
    "model-rotation": (0.0, 0.0, 0.0, 1.0),
}


_HIGH_DETAIL_OVERRIDES: Dict[str, Any] = {
    "mesher-linear-deflection": 0.0015,
    "mesher-angular-deflection": 0.2,
}

_DEFAULT_LINEAR_DEF = float(_BASE_GEOM_SETTINGS["mesher-linear-deflection"])
_DEFAULT_ANGULAR_DEF = float(_BASE_GEOM_SETTINGS["mesher-angular-deflection"])


class GeometrySettingsManager:
    """
    Manage a coordinated trio of ifcopenshell geometry settings objects:

        - default: iterator / “normal” mode (no OCC)
        - occ:     same as default, but with Python OpenCascade enabled
        - remesh:  OCC-enabled, with optional finer deflection overrides

    Core behaviour
    --------------
    * Settings are initialised from a common base (e.g. _BASE_GEOM_SETTINGS).
    * An optional remesh_overrides dict adjusts only the remesh settings.
    * set(key, value) broadcasts to all settings unless a specific target
      ("default" | "iterator" | "occ" | "remesh") is requested.
    * get(key, source=...) reads a value from any of the managed settings.
    * High-level keys "model-offset", "offset-type", "model-rotation" are
      tracked in Python and not required to exist on ifcopenshell settings.
    """

    def __init__(
        self,
        *,
        base_settings: Mapping[str, Any],
        remesh_overrides: Optional[Mapping[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        import ifcopenshell.geom  # local import

        self._log = logger or log

        # High-level state
        self._offset_raw: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._offset_type: str = str(base_settings.get("offset-type", "positive")).lower()
        if self._offset_type == "negative":
            self._offset_signed = (-0.0, -0.0, -0.0)
        else:
            self._offset_signed = self._offset_raw
        self._rotation_quat: Tuple[float, float, float, float] = tuple(
            float(v) for v in base_settings.get("model-rotation", (0.0, 0.0, 0.0, 1.0))
        )

        # Underlying ifcopenshell settings
        self._default = ifcopenshell.geom.settings()
        self._occ = ifcopenshell.geom.settings()
        self._remesh = ifcopenshell.geom.settings()

        # Apply base config to all three
        for s in (self._default, self._occ, self._remesh):
            _apply_geom_settings(s, dict(base_settings))

        # OCC toggles (never on default, always on occ/remesh)
        self._occ_available = _enable_occ(self._occ)
        if remesh_overrides:
            remesh_base = dict(base_settings)
            remesh_base.update(remesh_overrides)
            _apply_geom_settings(self._remesh, remesh_base)
        self._remesh_occ_available = _enable_occ(self._remesh)

        # Re-apply the initial model offset using the current offset-type
        # so the underlying settings carry the signed value.
        self._set_model_offset(base_settings.get("model-offset", (0.0, 0.0, 0.0)))
        self._apply_offset_to_settings()

    # ---------------- properties ----------------

    @property
    def default_settings(self):
        """Settings for the main iterator / normal mode (no OCC)."""
        return self._default

    @property
    def iterator_settings(self):
        return self._default

    @property
    def occ_settings(self):
        """Settings with Python OpenCascade toggled on."""
        return self._occ

    @property
    def remesh_settings(self):
        """High-detail remesh settings (OCC + finer deflections)."""
        return self._remesh

    @property
    def occ_available(self) -> bool:
        return bool(self._occ_available)

    @property
    def remesh_occ_available(self) -> bool:
        return bool(self._remesh_occ_available)

    # ---------------- helpers for high-level keys ----------------

    def _set_offset_type(self, value: Any) -> None:
        text = str(value).strip().lower()
        if text in {"positive", "pos", "+"}:
            self._offset_type = "positive"
        elif text in {"negative", "neg", "-"}:
            self._offset_type = "negative"
        else:
            # keep previous, but log
            self._log.debug("GeometrySettingsManager: unknown offset-type %r", value)
            return

        # Recompute signed offset whenever the type changes
        self._update_signed_offset()
        self._apply_offset_to_settings()

    def _set_model_offset(self, value: Any) -> Tuple[float, float, float]:
        try:
            xs = list(value)
        except Exception:
            xs = [0.0, 0.0, 0.0]
        xyz = [float(xs[i]) if i < len(xs) else 0.0 for i in range(3)]
        self._offset_raw = (xyz[0], xyz[1], xyz[2])
        self._update_signed_offset()
        return self._offset_signed

    def _update_signed_offset(self) -> None:
        """Derive signed offset from raw offset and offset_type without double-applying."""
        ox, oy, oz = self._offset_raw
        if self._offset_type == "negative":
            self._offset_signed = (-ox, -oy, -oz)
        else:
            self._offset_signed = (ox, oy, oz)

    def _apply_offset_to_settings(self) -> None:
        """Push the resolved signed offset into all managed settings objects."""
        for s in (self._default, self._occ, self._remesh):
            try:
                s.set("model-offset", self._offset_signed)
            except Exception:
                try:
                    s.set("MODEL_OFFSET", self._offset_signed)
                except Exception:
                    pass

    @staticmethod
    def _axis_angle_deg_to_quat(axis: Sequence[float], angle_deg: float) -> Tuple[float, float, float, float]:
        """Convert an axis + angle (degrees) into an (x,y,z,w) quaternion."""
        ax = list(axis)
        if len(ax) < 3:
            ax = [1.0, 0.0, 0.0]
        x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
        length = math.sqrt(x * x + y * y + z * z) or 1.0
        x /= length
        y /= length
        z /= length
        half = math.radians(float(angle_deg)) * 0.5
        s = math.sin(half)
        c = math.cos(half)
        return (x * s, y * s, z * s, c)

    def _set_model_rotation(self, value: Any) -> None:
        """
        Accept either:
          - axis + angle in degrees: (ax, ay, az, angle_deg)
          - or a raw quaternion: (x, y, z, w)
        """
        try:
            xs = list(value)
        except Exception:
            xs = [0.0, 0.0, 0.0, 1.0]

        if len(xs) == 4:
            # Heuristic: if last value looks like a large magnitude, treat as degrees
            angle_deg = float(xs[3])
            if abs(angle_deg) > 2.0 * math.pi:
                quat = self._axis_angle_deg_to_quat(xs[:3], angle_deg)
            else:
                # Assume already quaternion
                quat = (float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3]))
        else:
            quat = (0.0, 0.0, 0.0, 1.0)
        self._rotation_quat = quat

    # ---------------- public API: set / get ----------------

    def set(self, key: str, value: Any, *, target: Optional[str] = None) -> None:
        """
        Set a tessellation option.

        - target=None (default) → broadcast to all (default/occ/remesh)
        - target="default" or "iterator" → default only
        - target="occ" → occ settings only
        - target="remesh" → remesh settings only

        High-level keys:
        - "model-offset"  → stored in manager, broadcast as raw value
        - "offset-type"   → "positive" / "negative"
        - "model-rotation"→ axis+angle(deg) or quaternion; stored as quat
        """
        # Handle high-level keys first
        lk = key.lower()
        if lk == "offset-type":
            self._set_offset_type(value)
        elif lk == "model-offset":
            # Apply sign according to current offset-type before broadcasting
            value = self._set_model_offset(value)
        elif lk == "model-rotation":
            self._set_model_rotation(value)

        if target is None:
            targets = (self._default, self._occ, self._remesh)
        else:
            t = target.lower()
            if t in {"default", "iterator"}:
                targets = (self._default,)
            elif t == "occ":
                targets = (self._occ,)
            elif t == "remesh":
                targets = (self._remesh,)
            else:
                self._log.debug("GeometrySettingsManager.set: unknown target %r", target)
                return

        for s in targets:
            try:
                s.set(key, value)
            except Exception:
                try:
                    alt_key = str(key).replace("-", "_").upper()
                    s.set(alt_key, value)
                except Exception:
                    # silently ignore unknown keys on the underlying settings
                    pass

    def get(self, key: str, *, source: str = "default") -> Any:
        """
        Get a tessellation option from one of the managed settings objects.

        source: "default" | "iterator" | "occ" | "remesh"

        High-level keys:
        - "model-offset" → returns signed offset according to offset-type
        - "offset-type"  → returns "positive" or "negative"
        - "model-rotation" → quaternion (x, y, z, w)
        """
        lk = key.lower()
        if lk == "model-offset":
            return self._offset_signed
        if lk == "offset-type":
            return self._offset_type
        if lk == "model-rotation":
            return self._rotation_quat

        src = source.lower()
        if src in {"default", "iterator"}:
            settings_obj = self._default
        elif src == "occ":
            settings_obj = self._occ
        else:  # "remesh" or anything else → remesh by default
            settings_obj = self._remesh

        try:
            if hasattr(settings_obj, "get"):
                return settings_obj.get(key)
            return getattr(settings_obj, key, None)
        except Exception:
            return None


_HIGH_DETAIL_CLASSES = {
    "IFCSTAIR",
    "IFCSTAIRFLIGHT",
    "IFCRAILING",
    "IFCPLATE",
    "IFCMEMBER",
    "IFCBEAM",
}

_BREP_FALLBACK_CLASSES = {
    "IFCSTAIR",
    "IFCSTAIRFLIGHT",
    "IFCRAILING",
    "IFCPLATE",
    "IFCMEMBER",
}

_OCC_ALIGNMENT_THRESHOLD = 1.0  # metres; larger deltas suggest double transforms


def _apply_geom_settings(settings, overrides: Dict[str, Any]) -> None:
    """Apply iterator settings while tolerating different schema naming conventions."""
    for key, value in overrides.items():
        try:
            settings.set(key, value)
        except Exception:
            try:
                alt_key = str(key).replace("-", "_").upper()
                settings.set(alt_key, value)
            except Exception:
                pass


def _force_local_coordinates(settings) -> None:
    """Ensure geometry is emitted relative to the IFC local placement."""
    for key in ("use-world-coords", "USE_WORLD_COORDS"):
        try:
            settings.set(key, False)
            return
        except Exception:
            continue


def _enable_occ(settings) -> bool:
    """Best-effort helper to flip on OCCT-backed shape creation.

    Returns True when the toggle succeeded, False when Python OpenCascade is unavailable.
    """
    global _OCC_WARNING_EMITTED
    last_exc: Optional[Exception] = None
    for key in ("use-python-opencascade", "USE_PYTHON_OPENCASCADE"):
        try:
            settings.set(key, True)
            return True
        except Exception as exc:
            last_exc = exc
            continue
    if not _OCC_WARNING_EMITTED:
        detail = f" ({last_exc})" if last_exc else ""
        log.warning(
            "Python OpenCascade bindings are unavailable; OCC detail meshes will be skipped%s. "
            "Install pythonocc-core (e.g. pip install pythonocc-core==7.8.1) to enable detail mode.",
            detail,
        )
        _OCC_WARNING_EMITTED = True
    return False


def _disable_occ(settings) -> None:
    """Best-effort helper to force iterator settings to avoid OCC."""
    for key in ("use-python-opencascade", "USE_PYTHON_OPENCASCADE"):
        try:
            settings.set(key, False)
        except Exception:
            continue


def _settings_fingerprint(settings) -> str:
    """Return a stable hash describing tessellation settings used for a shape."""
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
    return hashlib.md5("|".join(fp_vals).encode("utf-8")).hexdigest()


def _remesh_product_geometry(product, settings, *, use_brep: bool = False):
    """Create a fresh geometry shape for a product using the supplied settings."""
    try:
        kwargs = {"use_python_opencascade": True} if use_brep else {}
        return ifcopenshell.geom.create_shape(settings, product, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(
            "High detail remesh failed for %s (brep=%s): %s",
            getattr(product, "GlobalId", None),
            use_brep,
            exc,
        )
        return None


def _mesh_stats(mesh: Dict[str, Any]) -> Optional[Dict[str, float]]:
    verts = mesh.get("vertices")
    faces = mesh.get("faces")
    if verts is None or faces is None:
        return None
    try:
        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
    except Exception:
        return None
    dims = maxs - mins
    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
    diagonal = float(np.linalg.norm(dims))
    longest = float(np.max(dims)) if dims.size else 0.0
    shortest = float(np.min(dims)) if dims.size else 0.0
    face_count = int(faces.shape[0]) if hasattr(faces, "shape") else len(faces)
    area = float(2.0 * (dx * dy + dx * dz + dy * dz))
    area_per_face = area / max(face_count, 1)
    return {
        "dims": (dx, dy, dz),
        "diagonal": diagonal,
        "longest": longest,
        "shortest": shortest,
        "face_count": face_count,
        "area": area,
        "area_per_face": area_per_face,
    }


def _mesh_center(mesh: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not mesh:
        return None
    verts = mesh.get("vertices")
    if verts is None or getattr(verts, "size", 0) == 0:
        return None
    try:
        mins = np.min(verts, axis=0)
        maxs = np.max(verts, axis=0)
        return (mins + maxs) * 0.5
    except Exception:
        return None


def _occ_mesh_misaligned(base_mesh: Optional[Dict[str, Any]], occ_mesh: Dict[str, Any]) -> bool:
    """Return True when the OCC mesh looks geolocated relative to the base iterator mesh."""
    if not base_mesh:
        return False
    base_center = _mesh_center(base_mesh)
    occ_center = _mesh_center(occ_mesh)
    if base_center is None or occ_center is None:
        return False
    try:
        delta = float(np.linalg.norm(occ_center - base_center))
    except Exception:
        return False
    return delta > _OCC_ALIGNMENT_THRESHOLD


def round_tuple_list(xs: Iterable[Iterable[float]], tol: int = 9) -> Tuple[Tuple[float, ...], ...]:
    """Round nested iterables to improve deterministic hashing."""
    return tuple(tuple(round(float(v), tol) for v in x) for x in xs)

def stable_mesh_hash(verts: Any, faces: Any) -> str:
    """Hash mesh topology ignoring float noise and container type."""
    if hasattr(verts, "tolist"): verts_iter = verts.tolist()
    else: verts_iter = list(verts)
    if hasattr(faces, "tolist"): faces_iter = faces.tolist()
    else: faces_iter = list(faces)
    rverts = round_tuple_list(verts_iter, tol=9)
    rfaces = tuple(tuple(int(i) for i in f) for f in faces_iter)
    h = hashlib.sha256(); h.update(str(rverts).encode("utf-8")); h.update(b"|"); h.update(str(rfaces).encode("utf-8"))
    return h.hexdigest()

def sanitize_name(raw_name: Optional[str], fallback: Optional[str] = None) -> str:
    """Return a USD-friendly identifier derived from IFC names."""
    base = str(raw_name or fallback or "Unnamed")
    if base.strip().lower() == "undefined" or not base.strip():
        base = str(fallback or "Material")
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
    """Prototype metadata for an IfcTypeObject-backed mesh."""

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
    style_face_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    count: int = 0
    style_material: Optional[PBRMaterial] = None
    detail_mesh: Optional["OCCDetailMesh"] = None
    semantic_parts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    semantic_parts_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class HashProto:
    """Prototype metadata keyed only by geometry hash (occurrence fallback)."""

    digest: str
    name: Optional[str] = None
    type_name: Optional[str] = None
    type_guid: Optional[str] = None
    mesh: Optional[dict] = None
    materials: List[Any] = field(default_factory=list)
    material_ids: List[int] = field(default_factory=list)
    signature: Optional[Tuple[Any, ...]] = None
    canonical_frame: Optional[Tuple[float, ...]] = None
    style_face_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    settings_fp: Optional[str] = None
    count: int = 0
    style_material: Optional[PBRMaterial] = None
    detail_mesh: Optional["OCCDetailMesh"] = None

@dataclass(frozen=True)
class CurveWidthRule:
    """User-configurable rule describing a desired annotation curve width."""

    width: float
    unit: Optional[str] = None
    layer: Optional[str] = None
    curve: Optional[str] = None
    hierarchy: Optional[str] = None
    step_id: Optional[int] = None

@dataclass
class ConversionOptions:
    """High-level knobs used while harvesting IFC geometry."""

    enable_instancing: bool = True
    enable_hash_dedup: bool = True
    convert_metadata: bool = True
    enable_high_detail_remesh: bool = False
    manifest: Optional['ConversionManifest'] = None
    curve_width_rules: Tuple[CurveWidthRule, ...] = tuple()
    anchor_mode: Optional[Literal["local", "site"]] = None
    split_topology_by_material: bool = False
    detail_scope: Literal["none", "all", "object"] = "none"
    detail_level: Literal["subshape", "face"] = "subshape"
    detail_object_ids: Tuple[int, ...] = tuple()
    detail_object_guids: Tuple[str, ...] = tuple()
    enable_instance_material_variants: bool = True
    enable_semantic_subcomponents: bool = False
    semantic_tokens: Dict[str, List[str]] = field(default_factory=dict)
    force_occ: bool = False
    # Optional overrides for geometry settings (fed from anchoring logic)
    model_offset: Optional[Tuple[float, float, float]] = None
    model_offset_type: Optional[str] = None

@dataclass(frozen=True)
class PrototypeKey:
    """Stable dictionary key distinguishing repmap/detail/hash prototypes."""

    kind: Literal["repmap", "repmap_detail", "hash"]
    identifier: Union[int, str]

@dataclass
class MapConversionData:
    """Lightweight snapshot of IfcMapConversion parameters."""

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
    style_material: Optional[PBRMaterial] = None
    style_face_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detail_mesh: Optional["OCCDetailMesh"] = None
    semantic_parts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ifc_path: Optional[str] = None

@dataclass
class PrototypeCaches:
    repmaps: Dict[int, MeshProto]
    repmap_counts: Counter
    hashes: Dict[str, HashProto]
    step_keys: Dict[int, PrototypeKey]
    instances: Dict[int, InstanceRecord]
    annotations: Dict[int, AnnotationCurve] = field(default_factory=dict)
    map_conversion: Optional[MapConversionData] = None


@dataclass
class PrototypeBuildContext:
    """Hold mutable state and shared settings for a single IFC traversal."""

    ifc_file: Any
    options: ConversionOptions
    settings: Any
    settings_fp_base: str
    settings_fp_high: str
    settings_fp_brep: str
    enable_high_detail: bool
    high_detail_settings: Optional[Any] = None
    detail_mesh_cache: Dict[int, "OCCDetailMesh"] = field(default_factory=dict)
    repmaps: Dict[int, MeshProto] = field(default_factory=dict)
    repmap_counts: Counter = field(default_factory=Counter)
    hashes: Dict[str, HashProto] = field(default_factory=dict)
    step_keys: Dict[int, PrototypeKey] = field(default_factory=dict)
    instances: Dict[int, InstanceRecord] = field(default_factory=dict)
    hierarchy_cache: Dict[int, Tuple[Tuple[str, Optional[int]], ...]] = field(default_factory=dict)
    annotation_hooks: Optional[AnnotationHooks] = field(default=None)
    detail_scope: Literal["none", "all", "object"] = "none"
    detail_level: Literal["subshape", "face"] = "subshape"
    detail_object_ids: Set[int] = field(default_factory=set)
    detail_object_guids: Set[str] = field(default_factory=set)
    user_high_detail: bool = False
    object_scope_settings: Optional[Any] = None
    fallback_settings_no_occ: Optional[Any] = None
    occ_settings: Optional[Any] = None

    @classmethod
    def build(cls, ifc_file, options: Optional[ConversionOptions]) -> "PrototypeBuildContext":
        if options is None:
            options = ConversionOptions()

        # Normalise detail scope / level
        detail_scope = getattr(options, "detail_scope", "none") or "none"
        if detail_scope not in ("none", "all", "object"):
            detail_scope = "none"

        detail_level = getattr(options, "detail_level", "subshape") or "subshape"
        explicit_semantic = bool(getattr(options, "enable_semantic_subcomponents", False))

        # "subcomponents" is a semantic request but uses same OCC detail level
        if detail_level.lower() in ("subcomponent", "subcomponents"):
            detail_level = "subshape"
            explicit_semantic = True

        if detail_level not in ("subshape", "face"):
            detail_level = "subshape"

        # Any non-"none" detail_scope implies semantics + detail
        if detail_scope != "none":
            explicit_semantic = True

        options.enable_semantic_subcomponents = explicit_semantic

        # Decode explicit object ids / guids
        detail_object_ids: Set[int] = set()
        for value in getattr(options, "detail_object_ids", tuple()) or tuple():
            try:
                detail_object_ids.add(int(value))
            except Exception:
                continue

        detail_object_guids: Set[str] = {
            str(value).strip().upper()
            for value in (getattr(options, "detail_object_guids", tuple()) or tuple())
            if str(value).strip()
        }

        detail_mode_requested = detail_scope == "all"
        user_high_detail = bool(getattr(options, "enable_high_detail_remesh", False))
        force_occ = getattr(options, "force_occ", False)

        if force_occ:
            # Force OCC requires high detail generation (enforced by CLI/conversion)
            
            if detail_scope in ("none", "all"):
                log.info("Force OCC enabled: applying OCC pipeline to ALL products (Global Run).")
                detail_scope = "all"
                # We also disable semantic subcomponents implicitly for the global run if they weren't explicitly requested?
                # The user said: "bypass the semantic decoposition step".
                # My logic in build_prototypes handles the bypass.
            elif detail_scope == "object":
                log.info("Force OCC enabled: bypassing semantic splitting for detailed objects.")

        # Instantiate our unified settings manager
        base_geom_settings = dict(_BASE_GEOM_SETTINGS)
        if getattr(options, "model_offset", None) is not None:
            base_geom_settings["model-offset"] = tuple(getattr(options, "model_offset"))
        if getattr(options, "model_offset_type", None):
            base_geom_settings["offset-type"] = str(getattr(options, "model_offset_type"))

        settings_manager = GeometrySettingsManager(
            base_settings=base_geom_settings,
            remesh_overrides=_HIGH_DETAIL_OVERRIDES if user_high_detail else None,
            logger=log,
        )
        occ_settings = settings_manager.occ_settings

        # Iterator uses default settings (no OCC); this is always the primary pipeline.
        settings = settings_manager.iterator_settings

        # High-detail remesh: we only use the remesh settings as a per-product fallback
        # (via _remesh_product_geometry / occ_detail), never as a global prepass.
        enable_high_detail = bool(user_high_detail)
        if not enable_high_detail:
            log.info("High-detail remeshing is DISABLED; using iterator tessellation only.")
        else:
            log.info("High-detail remeshing is ENABLED: remesh settings will be used as a per-product fallback.")

        if detail_scope == "object":
            log.info(
                "Detail scope 'object': iterator + semantic subcomponents run first; "
                "OCC detail will be attempted only for selected objects where needed."
            )
        elif detail_scope == "all":
            log.info(
                "Detail scope 'all': iterator + semantic subcomponents run for all products; "
                "OCC detail will only be used as a fallback when semantic splitting fails."
            )

        fine_remesh_settings = settings_manager.remesh_settings if enable_high_detail else None

        # OCC detail meshes are now built lazily per product (in build_prototypes)
        # after semantic subcomponents have been attempted. We do not run a global
        # OCC prepass here anymore.
        detail_mesh_cache: Dict[int, "OCCDetailMesh"] = {}


        # Fingerprints
        settings_fp_base = _settings_fingerprint(settings)
        if enable_high_detail and fine_remesh_settings is not None:
            settings_fp_high = _settings_fingerprint(fine_remesh_settings)
            settings_fp_brep = hashlib.md5(
                f"{settings_fp_high}|brep".encode("utf-8")
            ).hexdigest()
        else:
            settings_fp_high = settings_fp_base
            settings_fp_brep = settings_fp_base

        annotation_hooks = AnnotationHooks(
            entity_on_hidden_layer=_entity_on_hidden_layer,
            collect_spatial_hierarchy=_collect_spatial_hierarchy,
            entity_label=_entity_label,
            object_placement_to_np=_object_placement_to_np,
            context_to_np=_context_to_np,
            mapping_item_transform=_mapping_item_transform,
            extract_curve_points=_extract_curve_points,
            transform_points=_transform_points,
        )

        object_scope_settings = None
        if detail_scope == "object":
            # Object-scope detail: use OCC settings with local coordinates enforced
            try:
                object_scope_settings = occ_settings
                _force_local_coordinates(object_scope_settings)
            except Exception:
                object_scope_settings = None

        return cls(
            ifc_file=ifc_file,
            options=options,
            settings=settings,
            settings_fp_base=settings_fp_base,
            settings_fp_high=settings_fp_high,
            settings_fp_brep=settings_fp_brep,
            enable_high_detail=enable_high_detail,
            high_detail_settings=fine_remesh_settings,
            detail_mesh_cache=detail_mesh_cache,
            repmaps={},
            repmap_counts=Counter(),
            hashes={},
            step_keys={},
            instances={},
            hierarchy_cache={},
            annotation_hooks=annotation_hooks,
            detail_scope=detail_scope,
            detail_level=detail_level,
            detail_object_ids=detail_object_ids,
            detail_object_guids=detail_object_guids,
            user_high_detail=user_high_detail,
            object_scope_settings=object_scope_settings,
            occ_settings=occ_settings,
        )


    def wants_detail_for_product(self, step_id: Optional[int], guid: Optional[str]) -> bool:
        if self.detail_scope != "object":
            return False
        if step_id is not None and step_id in self.detail_object_ids:
            return True
        if guid:
            guid_upper = str(guid).strip().upper()
            if guid_upper and guid_upper in self.detail_object_guids:
                return True
        return False

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
    """Convert an IfcAxis2Placement to a 4×4 USD matrix in meters."""
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
    """Recursively compose nested IfcLocalPlacement into a single matrix."""
    if obj_placement is None: return Gf.Matrix4d(1)
    local = axis2placement_to_matrix(getattr(obj_placement, "RelativePlacement", None), length_to_m)
    parent = compose_object_placement(getattr(obj_placement, "PlacementRelTo", None), length_to_m)
    return parent * local

def gf_to_tuple16(gf: Gf.Matrix4d):
    """Flatten a Matrix4d into a row-major 16-element tuple."""
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
    """Collect property sets and quantities for a product (and its type)."""
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
    """Normalise an ifcopenshell triangulation into numpy arrays."""
    g = getattr(geom, "geometry", geom)
    if hasattr(g, "verts"): v = np.array(g.verts, dtype=float).reshape(-1,3)
    elif hasattr(g, "coordinates"): v = np.array(g.coordinates, dtype=float).reshape(-1,3)
    elif hasattr(g, "vertices"): v = np.array(g.vertices, dtype=float).reshape(-1,3)
    else: raise AttributeError("No verts/coordinates/vertices found")
    if hasattr(g, "faces"): f = np.array(g.faces, dtype=int).reshape(-1,3)
    elif hasattr(g, "triangles"): f = np.array(g.triangles, dtype=int).reshape(-1,3)
    elif hasattr(g, "indices"): f = np.array(g.indices, dtype=int).reshape(-1,3)
    else: raise AttributeError("No faces/triangles/indices found")
    payload: Dict[str, Any] = {"vertices": v, "faces": f}
    normals = getattr(g, "normals", None)
    if normals is not None:
        try:
            payload["normals"] = np.array(normals, dtype=float).reshape(-1, 3)
        except Exception:
            pass
    uv_values = None
    for attr in ("uvs", "UVs", "uv_coords", "uvcoordinates", "uv", "texcoords"):
        if hasattr(g, attr):
            candidate = getattr(g, attr)
            if candidate is not None:
                try:
                    array = np.asarray(candidate)
                except Exception:
                    continue
                if array.size:
                    uv_values = array
                    break
    if uv_values is not None:
        payload["uvs"] = uv_values
    for attr in ("uv_indices", "uv_index", "uvs_indices", "uvfaces", "uv_faces"):
        if hasattr(g, attr):
            candidate = getattr(g, attr)
            if candidate is None:
                continue
            try:
                array = np.asarray(candidate, dtype=int)
            except Exception:
                continue
            if array.size:
                payload["uv_indices"] = array
                break
    return payload

# Spatial hierarchy (unchanged)

def _entity_label(entity) -> str:
    """Return a user-friendly label for hierarchy breadcrumb nodes."""
    for attr in ("Name","LongName","Description"):
        value = getattr(entity, attr, None)
        if isinstance(value, str) and value.strip(): return value.strip()
    label = entity.is_a() if hasattr(entity, "is_a") else "IfcEntity"
    try: step_id = entity.id()
    except Exception: step_id = None
    return f"{label}_{step_id}" if step_id is not None else label

def _resolve_spatial_parent(element):
    """Walk containment/decomposition relationships to find parent."""
    for rel in (getattr(element, "ContainedInStructure", None) or []):
        parent = getattr(rel, "RelatingStructure", None)
        if parent is not None: return parent
    for rel in (getattr(element, "Decomposes", None) or []):
        parent = getattr(rel, "RelatingObject", None)
        if parent is not None: return parent
    return None

def _collect_spatial_hierarchy(product) -> Tuple[Tuple[str, Optional[int]], ...]:
    """Build the tuple of (label, step_id) from site/project down to product."""
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
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)
def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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


def extract_map_conversion(ifc_file) -> Optional[MapConversionData]:
    """Return the preferred coordinate operation (map conversion or rigid)."""

    def _map_data_from_conversion(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "Eastings", 0.0), 0.0),
            northings=_as_float(getattr(op, "Northings", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "OrthogonalHeight", 0.0), 0.0),
            x_axis_abscissa=_as_float(getattr(op, "XAxisAbscissa", 1.0), 1.0),
            x_axis_ordinate=_as_float(getattr(op, "XAxisOrdinate", 0.0), 0.0),
            scale=_as_float(getattr(op, "Scale", 1.0), 1.0) or 1.0,
        )

    def _map_data_from_rigid(op) -> MapConversionData:
        return MapConversionData(
            eastings=_as_float(getattr(op, "FirstCoordinate", 0.0), 0.0),
            northings=_as_float(getattr(op, "SecondCoordinate", 0.0), 0.0),
            orthogonal_height=_as_float(getattr(op, "Height", 0.0), 0.0),
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        )

    best_data: Optional[MapConversionData] = None
    for ctx in ifc_file.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op is None or not hasattr(op, "is_a"):
                continue
            data: Optional[MapConversionData] = None
            if op.is_a("IfcMapConversion"):
                data = _map_data_from_conversion(op)
            elif op.is_a("IfcRigidOperation"):
                data = _map_data_from_rigid(op)
            if data is None:
                continue
            if (
                getattr(ctx, "ContextType", None) == "Model"
                and getattr(ctx, "CoordinateSpaceDimension", None) == 3
            ):
                return data
            if best_data is None:
                best_data = data
    return best_data



# Absolute/world transform resolver (same signature used by process_usd)

def resolve_absolute_matrix(shape, element) -> Optional[Tuple[float, ...]]:
    """Return the absolute/world 4×4 for the iterator result.

    Priority:
      1) iterator's per-piece matrix (covers MappingTarget × MappingOrigin × placements)
      2) ifcopenshell.util.shape.get_shape_matrix(shape) (robust fallback)
      3) composed IfcLocalPlacement (last resort)

    The order mirrors ifcopenshell's behaviour and keeps instancing stable even
    when representation maps are nested.  Identity matrices are left intact so
    USD always has an explicit transform op.
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


def _tuple16_to_np(mat16) -> Optional[np.ndarray]:
    """Return a 4x4 numpy array for a flattened matrix tuple, or None if invalid."""
    if mat16 is None:
        return None
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
    except Exception:
        return None
    return arr


def _is_affine_invertible_tuple(mat16, *, atol: float = 1e-9) -> bool:
    """Heuristic check that a flattened 4x4 matrix represents an invertible affine transform."""
    arr = _tuple16_to_np(mat16)
    if arr is None:
        return False
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=atol):
        return False
    try:
        det = float(np.linalg.det(arr[:3, :3]))
    except Exception:
        return False
    return abs(det) > atol

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


def _point_list_entry_to_tuple(entry) -> Tuple[float, float, float]:
    """Best-effort conversion for CoordList tuples (supports 2D/3D lists)."""
    if entry is None:
        return (0.0, 0.0, 0.0)
    if isinstance(entry, (list, tuple)):
        seq = entry
    else:
        if hasattr(entry, "wrappedValue"):
            seq = entry.wrappedValue
        elif hasattr(entry, "CoordList"):
            seq = getattr(entry, "CoordList") or []
        elif hasattr(entry, "Coordinates"):
            seq = getattr(entry, "Coordinates") or []
        else:
            try:
                seq = list(entry)
            except Exception:
                seq = []
    x = _as_float(seq[0] if len(seq) > 0 else 0.0)
    y = _as_float(seq[1] if len(seq) > 1 else 0.0)
    z = _as_float(seq[2] if len(seq) > 2 else 0.0)
    return (x, y, z)


def _indexed_polycurve_points(curve) -> List[Tuple[float, float, float]]:
    """Expand an IfcIndexedPolyCurve into explicit XYZ points."""
    point_list = getattr(curve, "Points", None)
    if point_list is None:
        return []
    coord_list = getattr(point_list, "CoordList", None)
    if coord_list is None:
        coord_list = getattr(point_list, "CoordinatesList", None)
    coords: List[Tuple[float, float, float]] = []
    if coord_list:
        for entry in coord_list:
            coords.append(_point_list_entry_to_tuple(entry))
    if not coords:
        return []

    segments = list(getattr(curve, "Segments", None) or [])
    if not segments:
        return coords

    result: List[Tuple[float, float, float]] = []

    def _append_index(raw_index) -> None:
        try:
            idx = int(raw_index)
        except Exception:
            return
        idx -= 1  # IfcPositiveInteger is 1-based
        if idx < 0 or idx >= len(coords):
            return
        point = coords[idx]
        if result and result[-1] == point:
            return
        result.append(point)

    for segment in segments:
        if segment is None:
            continue
        if hasattr(segment, "wrappedValue"):
            indices = segment.wrappedValue
        elif hasattr(segment, "Points"):
            indices = getattr(segment, "Points", None) or ()
        elif isinstance(segment, (list, tuple)):
            indices = segment
        else:
            try:
                indices = list(segment)
            except Exception:
                indices = (segment,)
        if not indices:
            continue
        # Some representations wrap the tuple once more (e.g. [(1,2,3)]).
        if len(indices) == 1 and isinstance(indices[0], (list, tuple)):
            indices = indices[0]
        for raw_index in indices:
            if isinstance(raw_index, (list, tuple)):
                for nested in raw_index:
                    _append_index(nested)
            else:
                _append_index(raw_index)

    return result if result else coords


def _object_placement_to_np(obj_placement) -> np.ndarray:
    """Compose an IfcObjectPlacement into a numpy 4x4 matrix."""
    try:
        gf_matrix = compose_object_placement(obj_placement, length_to_m=1.0)
    except Exception:
        return np.eye(4, dtype=float)
    return _gf_matrix_to_np(gf_matrix)

def _cartesian_transform_to_np(op) -> np.ndarray:
    """Convert an IfcCartesianTransformationOperator into a 4×4 matrix."""
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


def _rigid_operation_to_np(op) -> np.ndarray:
    """Build a 4x4 translation matrix from an IfcRigidOperation."""
    tx = _as_float(getattr(op, "FirstCoordinate", 0.0), 0.0)
    ty = _as_float(getattr(op, "SecondCoordinate", 0.0), 0.0)
    tz = _as_float(getattr(op, "Height", 0.0), 0.0)
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat

def _context_to_np(ctx) -> np.ndarray:
    """Compose nested representation contexts into a single matrix."""
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
            try:
                if op.is_a("IfcMapConversion"):
                    transform = transform @ _map_conversion_to_np(op)
                elif op.is_a("IfcRigidOperation"):
                    transform = transform @ _rigid_operation_to_np(op)
            except Exception:
                continue
        current = getattr(current, "ParentContext", None)
    return transform


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if text in {"F", "FALSE", "NO", "0"}:
        return False
    if text in {"T", "TRUE", "YES", "1"}:
        return True
    return None


def _layer_is_hidden(layer) -> bool:
    if layer is None:
        return False
    for attr in ("LayerOn", "LayerVisible", "LayerVisibility"):
        if _to_bool(getattr(layer, attr, None)) is False:
            return True
    for attr in ("LayerFrozen", "LayerBlocked"):
        if _to_bool(getattr(layer, attr, None)) is True:
            return True
    return False


def _entity_on_hidden_layer(entity) -> bool:
    """Return True if the entity or any of its representations are hidden."""
    seen: set[int] = set()

    def _collect(obj):
        if obj is None:
            return
        for layer in getattr(obj, "LayerAssignments", None) or []:
            if layer is None:
                continue
            try:
                lid = int(layer.id())
            except Exception:
                lid = id(layer)
            if lid in seen:
                continue
            seen.add(lid)
            yield layer

    for layer in _collect(entity):
        if _layer_is_hidden(layer):
            return True

    rep = getattr(entity, "Representation", None)
    if rep is not None:
        for layer in _collect(rep):
            if _layer_is_hidden(layer):
                return True
        for representation in getattr(rep, "Representations", []) or []:
            for layer in _collect(representation):
                if _layer_is_hidden(layer):
                    return True
            for item in getattr(representation, "Items", []) or []:
                for layer in _collect(item):
                    if _layer_is_hidden(layer):
                        return True
    return False

def _extract_curve_points(item) -> List[Tuple[float, float, float]]:
    """Traverse curve/curve-set items and gather 3D points."""
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
    if hasattr(item, "is_a") and item.is_a("IfcIndexedPolyCurve"):
        return _indexed_polycurve_points(item)
    return []

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

def _build_recursive_ifc_mesh(product, settings, logger=None):
    """
    Recursively decompose a product's representation to generate a granular mesh.
    
    This traverses IfcMappedItem and IfcBooleanResult to reach leaf items (Extrusions, etc.),
    generating separate meshes for each leaf and tagging them with the leaf's Item ID.
    This allows semantic splitting to distinguish parts even within mapped representations.
    """
    import ifcopenshell.util.placement
    import ifcopenshell.geom

    # Simple cache to avoid re-processing the same product/representation multiple times
    # Key: product.id()
    if not hasattr(_build_recursive_ifc_mesh, "_cache"):
        _build_recursive_ifc_mesh._cache = {}
    
    prod_id = product.id()
    if prod_id in _build_recursive_ifc_mesh._cache:
        return _build_recursive_ifc_mesh._cache[prod_id]

    # Collect all leaf items with their accumulated transforms
    leaves = []
    
    def _process_item(item, parent_xf):
        if item.is_a("IfcMappedItem"):
            source = item.MappingSource
            target = item.MappingTarget
            
            # Convert target operator to matrix
            try:
                # This function handles IfcAxis2Placement3D/2D and IfcCartesianTransformationOperator3D/2D
                # Note: get_operator_matrix is available in recent ifcopenshell versions
                xf = ifcopenshell.util.placement.get_operator_matrix(target)
            except (AttributeError, Exception):
                # Fallback: try to construct it manually or assume identity if failing
                # For now, let's assume identity to be safe, but log warning
                xf = np.eye(4)
                # We could try to use get_local_placement if it was a placement, but it's an operator.

            # Combine with parent transform
            combined_xf = parent_xf @ xf
            
            # Recurse into source representation
            if source.MappedRepresentation:
                for sub_item in source.MappedRepresentation.Items:
                    _process_item(sub_item, combined_xf)
                    
        elif item.is_a("IfcBooleanResult"):
            # Treat BooleanResult as a leaf to preserve the boolean operation result,
            # unless we explicitly want to decompose operands (which might be geometrically invalid).
            leaves.append((item, parent_xf))
            
        else:
            # Leaf item (Extrusion, Brep, etc.)
            leaves.append((item, parent_xf))

    # Start traversal
    reps = []
    if hasattr(product, "is_a"):
        if product.is_a("IfcProduct"):
             if product.Representation:
                 reps = product.Representation.Representations
        elif product.is_a("IfcShapeRepresentation"):
             reps = [product]
        elif product.is_a("IfcProductDefinitionShape"):
             reps = product.Representations
    
    if not reps:
        return None

    for rep in reps:
        for item in rep.Items:
            _process_item(item, np.eye(4))

    if not leaves:
        return None

    # Generate meshes for leaves
    all_verts = []
    all_faces = []
    all_mat_ids = []
    all_item_ids = []
    
    materials_list = []
    material_cache = {} # name -> index

    total_verts = 0
    
    for item, xf in leaves:
        try:
            # Create shape for the leaf item using the same settings
            shape = ifcopenshell.geom.create_shape(settings, item)
            if not shape:
                continue
                
            # Extract geometry
            verts = shape.geometry.verts # flat list of floats
            faces = shape.geometry.faces # flat list of ints
            
            if not verts or not faces:
                continue
                
            # Convert to numpy
            verts_np = np.array(verts).reshape(-1, 3)
            faces_np = np.array(faces).reshape(-1, 3)
            
            # Apply transform
            # xf is 4x4, verts is Nx3
            verts_h = np.hstack([verts_np, np.ones((verts_np.shape[0], 1))])
            verts_transformed = (verts_h @ xf.T)[:, :3]
            
            # Offset faces
            faces_offset = faces_np + total_verts
            
            # Handle materials
            item_mat_ids = shape.geometry.material_ids
            if not item_mat_ids:
                item_mat_ids = [-1] * faces_np.shape[0]
            
            # Map item materials to our global list
            mapped_mat_ids = []
            for m_id in item_mat_ids:
                if m_id < 0:
                    mapped_mat_ids.append(-1)
                    continue
                    
                if m_id < len(shape.geometry.materials):
                    mat = shape.geometry.materials[m_id]
                    # Use name as key (or id if available, but these are wrappers)
                    # mat.name is usually reliable
                    mat_key = mat.name
                    if mat_key not in material_cache:
                        material_cache[mat_key] = len(materials_list)
                        materials_list.append(mat)
                    mapped_mat_ids.append(material_cache[mat_key])
                else:
                    mapped_mat_ids.append(-1)

            # Store data
            all_verts.append(verts_transformed)
            all_faces.append(faces_offset)
            all_mat_ids.extend(mapped_mat_ids)
            # Tag every face with the leaf item ID
            all_item_ids.extend([item.id()] * faces_np.shape[0])
            
            total_verts += verts_transformed.shape[0]
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to process leaf item {item.id()}: {e}")
            continue

    if not all_verts:
        _build_recursive_ifc_mesh._cache[prod_id] = None
        return None

    # Combine all
    final_verts = np.vstack(all_verts)
    final_faces = np.vstack(all_faces)
    final_mat_ids = np.array(all_mat_ids, dtype=np.int32)
    final_item_ids = np.array(all_item_ids, dtype=np.int32)
    
    result = {
        "vertices": final_verts,
        "faces": final_faces,
        "material_ids": final_mat_ids,
        "item_ids": final_item_ids,
        "materials": materials_list
    }
    _build_recursive_ifc_mesh._cache[prod_id] = result
    return result

def build_prototypes(ifc_file, options: ConversionOptions, ifc_path: Optional[str] = None) -> PrototypeCaches:
    """Run the ifcopenshell iterator and build prototype/instance caches.

    The iterator is evaluated exactly once.  Geometry is grouped by
    ``(IfcTypeObject id, mesh hash, tessellation settings)`` so USD can emit
    instanced prototypes without re-tessellating representation maps.  The
    function also records per-instance transforms, attribute dictionaries, 2-D
    annotation curves, and optional map conversion metadata.
    """
    options = options or ConversionOptions()
    if options.enable_hash_dedup:
        log.info(
            "Hash-based prototype de-duplication is ENABLED (experimental). "
            "Identical tessellated occurrences will share hashed prototypes; "
            "set enable_hash_dedup=False to opt out if issues arise."
        )
    ctx = PrototypeBuildContext.build(ifc_file, options)
    log.info(
        "Detail configuration: scope=%s level=%s object_ids=%d object_guids=%d",
        ctx.detail_scope,
        ctx.detail_level,
        len(ctx.detail_object_ids),
        len(ctx.detail_object_guids),
    )

    settings = ctx.settings
    high_detail_settings = ctx.high_detail_settings
    enable_high_detail = ctx.enable_high_detail
    detail_mesh_cache = ctx.detail_mesh_cache
    repmaps = ctx.repmaps
    repmap_counts = ctx.repmap_counts
    hashes = ctx.hashes
    step_keys = ctx.step_keys
    instances = ctx.instances
    hierarchy_cache = ctx.hierarchy_cache
    occ_settings = ctx.occ_settings
    settings_fp_base = ctx.settings_fp_base
    settings_fp_high = ctx.settings_fp_high
    settings_fp_brep = ctx.settings_fp_brep
    annotation_hooks = ctx.annotation_hooks
    if getattr(options, "enable_semantic_subcomponents", False):
        enable_high_detail = False
        high_detail_settings = None
        detail_mesh_cache = {}

    iterator = ifcopenshell.geom.iterator(settings, ifc_file, threads)

    if not iterator.initialize():
        # Fallback: still return caches for downstream authoring (empty)
        return PrototypeCaches(
            repmaps=repmaps,
            repmap_counts=repmap_counts,
            hashes=hashes,
            step_keys=step_keys,
            instances=instances,
            annotations=extract_annotation_curves(ifc_file, hierarchy_cache, annotation_hooks),
            map_conversion=extract_map_conversion(ifc_file),
        )

    import time
    start_time = time.perf_counter()
    processed_count = 0

    while True:
        processed_count += 1
        if processed_count % 100 == 0:
            elapsed = time.perf_counter() - start_time
            log.info(f"Processed {processed_count} products in {elapsed:.2f}s ({processed_count/elapsed:.1f} products/s)")

        shape = iterator.get()
        if shape is None:
            if not iterator.next():
                break
            continue
        try:
            step_id = int(getattr(shape, "id"))
        except Exception:
            data = getattr(shape, "data", None)
            step_id = getattr(data, "id", None) if data is not None else None
            if step_id is None and hasattr(shape, "geometry"):
                step_id = getattr(shape.geometry, "id", None)
            if step_id is not None:
                try:
                    step_id = int(step_id)
                except Exception:
                    step_id = None
        if step_id is None:
            log.debug("Iterator returned shape without step id; skipping.")
            if not iterator.next():
                break
            continue

        product = ifc_file.by_id(step_id)
        if product is None:
            if not iterator.next():
                break
            continue

        if hasattr(product, "is_a") and product.is_a("IfcOpeningElement"):
            if not iterator.next():
                break
            continue

        if _entity_on_hidden_layer(product):
            if not iterator.next():
                break
            continue

        geom = getattr(shape, "geometry", None)
        if geom is None:
            if not iterator.next():
                break
            continue

        materials = _clone_materials(getattr(geom, "materials", None))
        material_ids = list(getattr(geom, "material_ids", []) or [])
        try:
            style_material = build_material_for_product(product)
        except Exception:
            style_material = None
        face_style_groups = extract_face_style_groups(product) or {}
        if materials:
            materials = _normalize_geom_materials(materials, face_style_groups, ifc_file)
        style_token_by_style_id: Dict[int, str] = {}
        for token, entry in face_style_groups.items():
            style_id = entry.get("style_id")
            if style_id is None:
                continue
            try:
                style_token_by_style_id[int(style_id)] = token
            except Exception:
                continue
        default_material_key = material_ids[0] if material_ids else None
        detail_material_resolver = _build_detail_material_resolver(
            ifc_file,
            product,
            style_token_by_style_id,
            bool(style_material),
            default_material_key,
        )
        settings_fp_current = settings_fp_base
        skip_occ_detail = False

        product_class = product.is_a() if hasattr(product, "is_a") else None
        product_class_upper = product_class.upper() if isinstance(product_class, str) else None
        product_name = getattr(product, "Name", None) or getattr(product, "GlobalId", None) or "<unnamed>"
        guid_for_log = getattr(product, "GlobalId", None)

        try:
            product_id_int = int(product.id())
        except Exception:
            product_id_int = None

        type_ref = None
        type_guid = None
        type_name = None
        type_id: Optional[int] = None
        for rel in (getattr(product, "IsTypedBy", []) or []):
            rel_type = getattr(rel, "RelatingType", None)
            if rel_type is None:
                continue
            type_ref = rel_type
            type_guid = getattr(rel_type, "GlobalId", None)
            type_name = getattr(rel_type, "Name", None)
            try:
                type_id = int(rel_type.id())
            except Exception:
                type_id = None
            break

        type_info: Optional[MeshProto] = None
        type_detail_mesh: Optional["OCCDetailMesh"] = None
        if type_id is not None:
            info = repmaps.get(type_id)
            if info is None:
                info = MeshProto(
                    repmap_id=type_id,
                    type_name=type_name,
                    type_class=type_ref.is_a() if hasattr(type_ref, "is_a") else None,
                    type_guid=type_guid,
                    repmap_index=None,
                    style_material=style_material,
                    style_face_groups=_clone_style_groups(face_style_groups),
                )
                repmaps[type_id] = info
            else:
                if info.type_name is None and type_name is not None:
                    info.type_name = type_name
                if info.type_class is None and hasattr(type_ref, "is_a"):
                    info.type_class = type_ref.is_a()
                if info.type_guid is None and type_guid is not None:
                    info.type_guid = type_guid
                if info.style_material is None and style_material is not None:
                    info.style_material = style_material
                if not info.style_face_groups and face_style_groups:
                    info.style_face_groups = _clone_style_groups(face_style_groups)
            if (
                ctx.enable_high_detail
                and ctx.detail_scope in ("all", "object")
                and occ_detail.is_available()
                and getattr(info, "detail_mesh", None) is None
            ):
                try:
                    info.detail_mesh = occ_detail.build_canonical_detail_for_type(
                        product,
                        ctx.high_detail_settings or ctx.settings,
                        default_linear_def=_DEFAULT_LINEAR_DEF,
                        default_angular_def=_DEFAULT_ANGULAR_DEF,
                        logger=log,
                        detail_level=ctx.detail_level,
                        material_resolver=detail_material_resolver,
                    )
                    if info.detail_mesh:
                        log.debug(
                            "Cached canonical OCC detail for type step=%s name=%s",
                            type_id,
                            getattr(type_ref, "Name", None)
                            or (type_ref.is_a() if hasattr(type_ref, "is_a") else "<type>"),
                        )
                except Exception:
                    log.debug("Canonical OCC detail build failed for type step=%s", type_id, exc_info=True)
            type_info = info
            type_detail_mesh = getattr(info, "detail_mesh", None)

        primary_key: Optional[PrototypeKey] = None

        def _mesh_from_geom(candidate_geom):
            try:
                tri = triangulated_to_dict(candidate_geom)
                if tri["faces"].size == 0:
                    return None
                return tri
            except Exception:
                return None

        detail_object_match = ctx.wants_detail_for_product(product_id_int, guid_for_log)
        detail_mesh = detail_mesh_cache.get(product_id_int) if product_id_int is not None else None

        mesh_hash = None
        mesh_dict_base = _mesh_from_geom(geom)
        mesh_dict = mesh_dict_base
        mesh_stats = _mesh_stats(mesh_dict_base) if mesh_dict_base is not None else None
        face_count = int(mesh_dict["faces"].shape[0]) if mesh_dict is not None and "faces" in mesh_dict else 0
        if mesh_dict_base is not None:
            try:
                expected_corners = int(mesh_dict_base["faces"].size)
            except Exception:
                expected_corners = None
            try:
                uv_data = extract_uvs_for_product(product)
            except Exception:
                uv_data = None
            if uv_data and expected_corners and len(uv_data.uvs) == expected_corners:
                mesh_dict_base["uvs"] = uv_data.uvs
                mesh_dict_base.pop("uv_indices", None)
        if face_count and material_ids and len(material_ids) != face_count:
            geom_attrs = [name for name in dir(geom) if name.startswith("material") or name.endswith("counts")]
            log.debug(
                "Material mismatch for %s (%s): faces=%d material_ids=%d geom_attrs=%s style=%s styledFaces=%d",
                getattr(product, "GlobalId", None),
                product.is_a() if hasattr(product, "is_a") else "IfcProduct",
                face_count,
                len(material_ids),
                geom_attrs,
                getattr(style_material, "name", None),
                sum(len(entry.get("faces", [])) for entry in face_style_groups.values()),
            )
        detail_reasons: List[str] = []
        detail_mode = "base"
        detail_mesh_data: Optional["OCCDetailMesh"] = detail_mesh
        detail_source_obj: Any = shape
        detail_settings_obj = None

        need_fine_remesh = False
        allow_brep_fallback = False
        force_occ = getattr(options, "force_occ", False)
        need_occ_detail = False
        defer_occ_detail = bool(getattr(options, "enable_semantic_subcomponents", False)) and not force_occ

        if force_occ and ctx.detail_scope != "object":
            need_occ_detail = True
            detail_reasons.append("force_occ")

        if detail_object_match:
            log.info(
                "Detail scope object hit: %s name=%s guid=%s step=%s",
                product_class_upper or product_class or "<unknown>",
                product_name,
                guid_for_log,
                step_id,
            )
            need_occ_detail = True
            detail_mode = "object"
            # When semantic subcomponents are enabled we *defer* the OCC detail
            # build for object-scope detail until after semantic splitting has
            # been attempted. This preserves the intended pipeline:
            #   iterator → semantic split → OCC detail (fallback) → iterator mesh.
            #
            # If semantic splitting is disabled entirely, we can still build the
            # object-scope OCC detail eagerly as before.
            if not getattr(options, "enable_semantic_subcomponents", False):
                detail_mesh_data = _build_object_scope_detail_mesh(
                    ctx,
                    product,
                    material_resolver=detail_material_resolver,
                    reference_shape=shape,
                )
                if detail_mesh_data is None:
                    log.warning(
                        "Detail scope object: OCC mesh unavailable for %s guid=%s step=%s name=%s",
                        product_class_upper or product_class or "<unknown>",
                        getattr(product, "GlobalId", None),
                        step_id,
                        product_name,
                    )
                elif type_detail_mesh is not None and type_id is not None:
                    # Reuse canonical type-level OCC detail when available.
                    primary_key = PrototypeKey(kind="repmap_detail", identifier=type_id)
                    detail_mesh_data = type_detail_mesh
        else:
            if ctx.detail_scope == "all":
                need_occ_detail = True
                detail_reasons.append("scope_all")

            if enable_high_detail and ctx.user_high_detail and product_class_upper:
                if product_class_upper in _HIGH_DETAIL_CLASSES and "class_match" not in detail_reasons:
                    need_fine_remesh = True
                    detail_reasons.append("class_match")
                if product_class_upper in _BREP_FALLBACK_CLASSES:
                    allow_brep_fallback = True

            if (
                enable_high_detail
                and ctx.user_high_detail
                and not need_fine_remesh
                and product_class_upper == "IFCBUILDINGELEMENTPROXY"
                and mesh_stats is not None
            ):
                proxy_needs_high_detail, proxy_reasons = occ_detail.proxy_requires_high_detail(mesh_stats)
                if proxy_needs_high_detail:
                    need_fine_remesh = True
                    detail_reasons.extend(proxy_reasons)

            if enable_high_detail and need_fine_remesh and fine_remesh_settings is not None:
                remeshed = _remesh_product_geometry(product, fine_remesh_settings)
                if remeshed is not None:
                    high_geom = getattr(remeshed, "geometry", remeshed)
                    high_mesh = _mesh_from_geom(high_geom)
                    if high_mesh is not None:
                        geom = high_geom
                        mesh_dict = high_mesh
                        mesh_stats = _mesh_stats(high_mesh)
                        settings_fp_current = settings_fp_high
                        materials = _clone_materials(getattr(remeshed, "materials", materials) or materials)
                        material_ids = list(getattr(remeshed, "material_ids", material_ids) or material_ids)
                        detail_mode = "high"
                        detail_source_obj = remeshed
                    else:
                        mesh_dict = mesh_dict_base
                        mesh_stats = _mesh_stats(mesh_dict_base) if mesh_dict_base is not None else None
                        detail_reasons.append("high_detail_failed_empty_geometry")
                else:
                    mesh_dict = mesh_dict_base
                    mesh_stats = _mesh_stats(mesh_dict_base) if mesh_dict_base is not None else None
                    detail_reasons.append("high_detail_remesh_failed")

            if (
                enable_high_detail
                and fine_remesh_settings is not None
                and (mesh_dict is None or mesh_dict["faces"].size == 0)
                and allow_brep_fallback
            ):
                detail_reasons.append("brep_fallback_triggered")
                remeshed_brep = _remesh_product_geometry(product, fine_remesh_settings, use_brep=True)
                if remeshed_brep is not None:
                    brep_geom = getattr(remeshed_brep, "geometry", remeshed_brep)
                    brep_mesh = _mesh_from_geom(brep_geom)
                    if brep_mesh is not None:
                        geom = brep_geom
                        mesh_dict = brep_mesh
                        mesh_stats = _mesh_stats(brep_mesh)
                        settings_fp_current = settings_fp_brep
                        materials = _clone_materials(getattr(remeshed_brep, "materials", materials) or materials)
                        material_ids = list(getattr(remeshed_brep, "material_ids", material_ids) or material_ids)
                        detail_mode = "brep"
                        detail_source_obj = remeshed_brep
                    else:
                        detail_reasons.append("brep_fallback_empty_geometry")
                        mesh_dict = mesh_dict_base
                        mesh_stats = _mesh_stats(mesh_dict_base) if mesh_dict_base is not None else None
                else:
                    detail_reasons.append("brep_fallback_failed")
                    mesh_dict = mesh_dict_base
                    mesh_stats = _mesh_stats(mesh_dict_base) if mesh_dict_base is not None else None

            if need_occ_detail and detail_mesh_data is None and occ_settings is not None and not defer_occ_detail:
                detail_settings_obj = occ_settings
                # Generate canonical map from iterator mesh (Mode 1)
                canonical_map = None
                if mesh_dict_base and "vertices" in mesh_dict_base and "faces" in mesh_dict_base:
                    try:
                        # Reuse iterator data for canonical mapping
                        # We need item IDs too, but iterator might not provide them directly in mesh_dict_base
                        # However, we can use material_ids as a proxy or if we have item_ids available
                        # The user's script uses ish.get_faces_representation_item_ids(g)
                        # Here 'shape.geometry' is the ifcopenshell triangulation object 'g'
                        
                        g = shape.geometry
                        c_verts = mesh_dict_base["vertices"]
                        c_faces = mesh_dict_base["faces"]
                        c_mat_ids = material_ids if material_ids else []
                        
                        # Try to get item IDs if possible
                        c_item_ids = []
                        try:
                            import ifcopenshell.util.shape as ish
                            c_item_ids = ish.get_faces_representation_item_ids(g)
                        except Exception:
                            pass
                        
                        canonical_map = {
                            "vertices": c_verts,
                            "faces": c_faces,
                            "material_ids": c_mat_ids,
                            "item_ids": c_item_ids
                        }
                    except Exception as exc:
                        log.debug("Failed to build canonical map for %s: %s", product, exc)

                detail_mesh_data = occ_detail.build_detail_mesh_payload(
                    detail_source_obj,
                    detail_settings_obj,
                    product=product,
                    default_linear_def=_DEFAULT_LINEAR_DEF,
                    default_angular_def=_DEFAULT_ANGULAR_DEF,
                    logger=log,
                    detail_level=ctx.detail_level,
                    material_resolver=detail_material_resolver,
                    reference_shape=shape,
                    canonical_map=canonical_map,
                )
                if detail_mesh_data is None:
                    log.debug(
                        "Detail mode: OCC mesh unavailable for %s guid=%s step=%s name=%s",
                        product_class_upper or product_class or "<unknown>",
                        getattr(product, "GlobalId", None),
                        step_id,
                        product_name,
                    )

        if detail_mesh_data is not None:
            subshape_entries = getattr(detail_mesh_data, "subshapes", None) or []
            subshape_count = len(subshape_entries)
            face_total = getattr(detail_mesh_data, "face_count", None)
            log.debug(
                "Detail mode: OCC mesh ready for %s guid=%s step=%s name=%s | subshapes=%d faces=%s",
                product_class_upper or product_class or "<unknown>",
                getattr(product, "GlobalId", None),
                step_id,
                product_name,
                subshape_count,
                face_total if face_total is not None else "n/a",
            )
            if subshape_entries:
                for entry in subshape_entries:
                    label = getattr(entry, "label", "<subshape>")
                    shape_type = getattr(entry, "shape_type", "<unknown>")
                    faces_list = getattr(entry, "faces", None) or []
                    log.debug(
                        "  ↳ Subshape %s (%s) faces=%d",
                        label,
                        shape_type,
                        len(faces_list),
                    )
            occ_mesh_dict = occ_detail.mesh_from_detail_mesh(detail_mesh_data)
            if occ_mesh_dict is not None:
                if _occ_mesh_misaligned(mesh_dict_base, occ_mesh_dict):
                    base_center = _mesh_center(mesh_dict_base)
                    occ_center = _mesh_center(occ_mesh_dict)
                    delta_val = (
                        float(np.linalg.norm(occ_center - base_center))
                        if base_center is not None and occ_center is not None
                        else float("nan")
                    )
                    log.warning(
                        "Detail mode: OCC mesh appears geolocated (center delta %.2fm) for %s guid=%s; keeping iterator mesh.",
                        delta_val,
                        product_class_upper or product_class or "<unknown>",
                        getattr(product, "GlobalId", None),
                    )
                elif (
                    mesh_dict is None
                    or "faces" not in mesh_dict
                    or getattr(mesh_dict["faces"], "size", 0) == 0
                ):
                    mesh_dict = occ_mesh_dict
                    mesh_dict_base = occ_mesh_dict
                    mesh_stats = _mesh_stats(occ_mesh_dict)
                    if detail_settings_obj is high_detail_settings and high_detail_settings is not None:
                        settings_fp_current = settings_fp_high

        # -------------------------------------------------------------
        # Mesh + hash (robust, with salvage + explicit logging)
        # -------------------------------------------------------------
        if mesh_dict is None:
            # Try to salvage a mesh directly from the iterator geometry
            # to avoid silently dropping this element.
            source_mesh = None
            raw_geom = getattr(shape, "geometry", None)
            if raw_geom is not None:
                # Newer ifcopenshell may expose a convenient as_dict()
                try:
                    as_dict = getattr(raw_geom, "as_dict", None)
                    if callable(as_dict):
                        source_mesh = as_dict()
                except Exception:
                    source_mesh = None

                # Fallback: build a minimal dict from verts/faces if present
                if source_mesh is None and hasattr(raw_geom, "verts") and hasattr(raw_geom, "faces"):
                    try:
                        source_mesh = {
                            "vertices": np.array(raw_geom.verts, dtype=float).reshape(-1, 3),
                            "faces": np.array(raw_geom.faces, dtype=int).reshape(-1, 3),
                        }
                    except Exception:
                        source_mesh = None

            if source_mesh is not None:
                mesh_dict = source_mesh
                mesh_dict_base = source_mesh
                mesh_stats = _mesh_stats(source_mesh)
            else:
                log.debug(
                    "Dropping %s (GlobalId=%s, step=%s): no triangulated mesh "
                    "from iterator or raw geometry.",
                    product.is_a() if hasattr(product, "is_a") else "<IfcProduct>",
                    getattr(product, "GlobalId", "unknown"),
                    step_id,
                )
                if not iterator.next():
                    break
                continue

        try:
            mesh_hash = stable_mesh_hash(mesh_dict["vertices"], mesh_dict["faces"])
        except Exception as exc:
            log.debug(
                "Dropping %s (GlobalId=%s, step=%s): stable_mesh_hash failed (%s).",
                product.is_a() if hasattr(product, "is_a") else "<IfcProduct>",
                getattr(product, "GlobalId", "unknown"),
                step_id,
                exc,
            )
            if not iterator.next():
                break
            continue

        guid_for_log = getattr(product, "GlobalId", None)
        if enable_high_detail and not skip_occ_detail:
            if detail_mode in ("high", "brep"):
                dims = mesh_stats.get("dims") if mesh_stats else None
                faces_logged = mesh_stats.get("face_count") if mesh_stats else None
                log.info(
                    "High-detail tessellation (%s) applied to %s step=%s guid=%s reasons=%s dims=%s faces=%s",
                    detail_mode,
                    product_class_upper or product_class or "<unknown>",
                    step_id,
                    guid_for_log,
                    ", ".join(detail_reasons) if detail_reasons else "unspecified",
                    f"({dims[0]:.3f},{dims[1]:.3f},{dims[2]:.3f})" if dims else "n/a",
                    faces_logged,
                )
            elif needs_high_detail and detail_mode == "base":
                log.debug(
                    "High-detail tessellation requested but fell back to iterator mesh for %s step=%s guid=%s; reasons=%s",
                    product_class_upper or product_class or "<unknown>",
                    step_id,
                    guid_for_log,
                    ", ".join(detail_reasons) if detail_reasons else "unspecified",
                )

        # World transform
        xf_tuple = resolve_absolute_matrix(shape, product)

        instance_mesh: Optional[Dict[str, Any]] = None

        if type_info is not None and type_id is not None and mesh_hash is not None:
            info = type_info
            if info.mesh is None:
                info.mesh = mesh_dict
                info.mesh_hash = mesh_hash
                info.settings_fp = settings_fp_current
                if not info.materials and materials:
                    info.materials = _clone_materials(materials)
                if not info.material_ids and material_ids:
                    info.material_ids = list(material_ids)
                if not info.style_face_groups and face_style_groups:
                    info.style_face_groups = _clone_style_groups(face_style_groups)
                if getattr(options, "enable_semantic_subcomponents", False):
                    try:
                        _log_semantic_attempt(
                            product,
                            mesh_dict,
                            face_style_groups,
                            materials,
                            label="proto/base",
                        )
                        info.semantic_parts = semantic_subcomponents.split_product_by_semantic_roles(
                            ifc_file=ifc_file,
                            product=product,
                            mesh_dict=mesh_dict,
                            material_ids=material_ids,
                            face_style_groups=face_style_groups,
                            materials=materials,
                            semantic_tokens=options.semantic_tokens,
                        ) or {}
                    except Exception:
                        info.semantic_parts = {}
                if info.detail_mesh is None and detail_mesh_data is not None and ctx.detail_scope != "object":
                    info.detail_mesh = detail_mesh_data
                    if getattr(options, "enable_semantic_subcomponents", False):
                        try:
                            occ_mesh_dict = occ_detail.mesh_from_detail_mesh(detail_mesh_data)
                        except Exception:
                            occ_mesh_dict = None
                        if occ_mesh_dict:
                            try:
                                _log_semantic_attempt(
                                    product,
                                    occ_mesh_dict,
                                    face_style_groups,
                                    materials,
                                    label="proto/detail",
                                )
                                info.semantic_parts_detail = semantic_subcomponents.split_product_by_semantic_roles(
                                    ifc_file=ifc_file,
                                    product=product,
                                    mesh_dict=occ_mesh_dict,
                                    material_ids=material_ids,
                                    face_style_groups=face_style_groups,
                                    materials=materials,
                                    semantic_tokens=options.semantic_tokens,
                                ) or {}
                            except Exception:
                                info.semantic_parts_detail = {}
                info.count = 0

            if (
                info.mesh is not None
                and info.mesh_hash == mesh_hash
                and info.settings_fp == settings_fp_current
            ):
                info.count += 1
                repmap_counts[type_id] += 1
                if primary_key is None:
                    primary_key = PrototypeKey(kind="repmap", identifier=type_id)

            if info.detail_mesh is None and detail_mesh_data is not None and ctx.detail_scope != "object":
                info.detail_mesh = detail_mesh_data

        if primary_key is None:
            if options.enable_hash_dedup and mesh_hash is not None:
                digest = f"{mesh_hash}|{settings_fp_current}"
                bucket = hashes.get(digest)
                if bucket is None:
                    bucket = HashProto(
                        digest=digest,
                        name=sanitize_name(getattr(product, "Name", None), fallback=product.is_a() if hasattr(product, "is_a") else None),
                        type_name=get_type_name(product),
                        type_guid=getattr(product, "GlobalId", None),
                        mesh=mesh_dict,
                        materials=_clone_materials(materials),
                        material_ids=list(material_ids),
                        canonical_frame=None,
                        settings_fp=settings_fp_current,
                        count=0,
                        style_material=style_material,
                        style_face_groups=_clone_style_groups(face_style_groups),
                        detail_mesh=detail_mesh_data if ctx.detail_scope != "object" else None,
                    )
                    hashes[digest] = bucket
                bucket.count += 1
                if not bucket.materials and materials: bucket.materials = _clone_materials(materials)
                if not bucket.material_ids and material_ids: bucket.material_ids = list(material_ids)
                if bucket.mesh is None: bucket.mesh = mesh_dict
                if not bucket.style_face_groups and face_style_groups:
                    bucket.style_face_groups = _clone_style_groups(face_style_groups)
                if bucket.style_material is None and style_material is not None:
                    bucket.style_material = style_material
                if not bucket.style_face_groups and face_style_groups:
                    bucket.style_face_groups = _clone_style_groups(face_style_groups)
                if bucket.canonical_frame is None and xf_tuple is not None and _is_affine_invertible_tuple(xf_tuple):
                    bucket.canonical_frame = xf_tuple
                if bucket.detail_mesh is None and detail_mesh_data is not None and ctx.detail_scope != "object":
                    bucket.detail_mesh = detail_mesh_data
                primary_key = PrototypeKey(kind="hash", identifier=digest)
            else:
                instance_mesh = mesh_dict

        detail_mesh_for_instance: Optional["OCCDetailMesh"] = detail_mesh_data
        if skip_occ_detail:
            detail_mesh_for_instance = None
        if primary_key is not None:
            prototype_bucket = None
            if primary_key.kind in ("repmap", "repmap_detail"):
                prototype_bucket = repmaps.get(primary_key.identifier)
            elif primary_key.kind == "hash":
                prototype_bucket = hashes.get(primary_key.identifier)
            if prototype_bucket is not None:
                if detail_mesh_for_instance is None:
                    detail_mesh_for_instance = prototype_bucket.detail_mesh
        else:
            detail_mesh_for_instance = detail_mesh_data

        if (
            getattr(options, "enable_instance_material_variants", True)
            and primary_key is not None
            and primary_key.kind in ("repmap", "repmap_detail")
        ):
            proto = repmaps.get(primary_key.identifier)
            variant_kind, variant_suffix = _instance_variant_kind(proto, materials, face_style_groups, style_material)
            if variant_suffix:
                materials = _variantize_materials(materials, variant_suffix)
                face_style_groups = _variantize_face_groups(face_style_groups, variant_suffix)
                if style_material is not None:
                    style_material = _variantize_material(style_material, variant_suffix)

        # Semantic sub-component selection (prototype-driven when available)
        semantic_parts: Dict[str, Dict[str, Any]] = {}
        
        # Determine if we should attempt semantic splitting
        should_attempt_semantic = getattr(options, "enable_semantic_subcomponents", False)
        if getattr(options, "force_occ", False):
            # If force_occ is on, we skip semantic ONLY if this object is targeted for OCC detail
            if ctx.detail_scope == "all":
                should_attempt_semantic = False
            elif ctx.detail_scope == "object" and detail_object_match:
                should_attempt_semantic = False
        
        if product_id_int in (461286, 577409, 573013):
            log.info(f"DEBUG: Product {product_id_int} match={detail_object_match} force_occ={getattr(options, 'force_occ', False)} scope={ctx.detail_scope} semantic={should_attempt_semantic} ids={ctx.detail_object_ids}")

        if should_attempt_semantic:
            if primary_key is not None:
                proto = None
                if primary_key.kind in ("repmap", "repmap_detail"):
                    proto = repmaps.get(primary_key.identifier)
                elif primary_key.kind == "hash":
                    proto = hashes.get(primary_key.identifier)
                
                if proto is not None:
                    # Check if prototype needs granular upgrade (if missing semantic parts)
                    if not getattr(proto, "semantic_parts", None):
                        # Try to upgrade prototype with granular mesh
                        rep_map_source = None
                        if product.Representation:
                            for rep in product.Representation.Representations:
                                for item in rep.Items:
                                    if item.is_a("IfcMappedItem"):
                                        # Check if this mapped item matches the prototype identifier
                                        # For repmap, identifier is usually the ID of the mapped representation
                                        source_rep = item.MappingSource.MappedRepresentation
                                        if source_rep and source_rep.id() == primary_key.identifier:
                                            rep_map_source = source_rep
                                            break
                                if rep_map_source: break
                        
                        if rep_map_source:
                            try:
                                rec_mesh = _build_recursive_ifc_mesh(rep_map_source, settings, logger=log)
                                if rec_mesh:
                                    # Normalize materials
                                    rec_materials = _normalize_geom_materials(rec_mesh["materials"], {}, ifc_file)
                                    
                                    # Split it using instance properties
                                    parts = semantic_subcomponents.split_product_by_semantic_roles(
                                        ifc_file=ifc_file,
                                        product=product,
                                        mesh_dict=rec_mesh,
                                        material_ids=rec_mesh["material_ids"],
                                        face_style_groups={}, # Use material IDs
                                        materials=rec_materials,
                                    )
                                    if parts:
                                        proto.semantic_parts = parts
                            except Exception:
                                pass

                    if detail_mesh_for_instance is not None and getattr(proto, "semantic_parts_detail", None):
                        semantic_parts = getattr(proto, "semantic_parts_detail", {}) or {}
                    else:
                        semantic_parts = getattr(proto, "semantic_parts", {}) or {}
                    
                    # If force_occ is enabled, we must ignore prototype semantic parts for targeted objects
                    if getattr(options, "force_occ", False):
                        if ctx.detail_scope == "all":
                            semantic_parts = {}
                        elif ctx.detail_scope == "object" and detail_object_match:
                            semantic_parts = {}
            elif mesh_dict is not None:
                try:
                    # Attempt recursive build for granular items
                    recursive_mesh = _build_recursive_ifc_mesh(product, settings, logger=log)
                    
                    target_mesh = recursive_mesh if recursive_mesh else mesh_dict
                    target_mat_ids = recursive_mesh["material_ids"] if recursive_mesh else material_ids
                    
                    target_materials = materials
                    if recursive_mesh:
                        target_materials = _normalize_geom_materials(recursive_mesh["materials"], {}, ifc_file)
                    
                    effective_style_groups = face_style_groups if not recursive_mesh else {}

                    _log_semantic_attempt(
                        product,
                        target_mesh,
                        face_style_groups,
                        materials,
                        label="instance",
                    )
                    semantic_parts = semantic_subcomponents.split_product_by_semantic_roles(
                        ifc_file=ifc_file,
                        product=product,
                        mesh_dict=target_mesh,
                        material_ids=target_mat_ids,
                        face_style_groups=effective_style_groups,
                        materials=target_materials,
                    ) or {}
                except Exception:
                    semantic_parts = {}
                if semantic_parts:
                    instance_mesh = None
                    skip_occ_detail = True
                    detail_mesh_data = None
            # If semantic splitting produced nothing and OCC detail was requested, build it now.
            if not semantic_parts and need_occ_detail and detail_mesh_for_instance is None:
                detail_mesh_data = None
                
                # Generate canonical map from iterator mesh (Mode 1)
                canonical_map = None
                if mesh_dict_base and "vertices" in mesh_dict_base and "faces" in mesh_dict_base:
                    try:
                        # Reuse iterator data for canonical mapping
                        g = shape.geometry
                        c_verts = mesh_dict_base["vertices"]
                        c_faces = mesh_dict_base["faces"]
                        c_mat_ids = material_ids if material_ids else []
                        
                        # Try to get item IDs if possible
                        c_item_ids = []
                        try:
                            import ifcopenshell.util.shape as ish
                            c_item_ids = ish.get_faces_representation_item_ids(g)
                        except Exception:
                            pass
                            
                        canonical_map = {
                            "vertices": c_verts,
                            "faces": c_faces,
                            "material_ids": c_mat_ids,
                            "item_ids": c_item_ids
                        }
                    except Exception:
                        pass

                if ctx.detail_scope == "object" and detail_object_match:
                    if ctx.object_scope_settings is not None:
                        detail_settings_obj = ctx.object_scope_settings
                        detail_mesh_data = _build_object_scope_detail_mesh(
                            ctx,
                            product,
                            material_resolver=detail_material_resolver,
                            reference_shape=shape,
                            canonical_map=canonical_map,
                        )
                        if detail_mesh_data is None:
                            log.debug(
                                "Detail scope object (post-semantic): OCC mesh unavailable for %s guid=%s step=%s name=%s",
                                product_class_upper or product_class or "<unknown>",
                                getattr(product, "GlobalId", None),
                                step_id,
                                product_name,
                            )
                        elif type_detail_mesh is not None and type_id is not None:
                            primary_key = PrototypeKey(kind="repmap_detail", identifier=type_id)
                            detail_mesh_data = type_detail_mesh
                elif occ_settings is not None:
                    detail_settings_obj = occ_settings

                    detail_mesh_data = occ_detail.build_detail_mesh_payload(
                        detail_source_obj,
                        detail_settings_obj,
                        product=product,
                        default_linear_def=_DEFAULT_LINEAR_DEF,
                        default_angular_def=_DEFAULT_ANGULAR_DEF,
                        logger=log,
                        detail_level=ctx.detail_level,
                        material_resolver=detail_material_resolver,
                        reference_shape=shape,
                        canonical_map=canonical_map,
                    )
                    if detail_mesh_data is None:
                        log.debug(
                            "Detail mode (post-semantic): OCC mesh unavailable for %s guid=%s step=%s name=%s",
                            product_class_upper or product_class or "<unknown>",
                            getattr(product, "GlobalId", None),
                            step_id,
                            product_name,
                        )

                detail_mesh_for_instance = detail_mesh_data

        # If we have no prototype, no per-instance mesh, no detail, and no semantic parts,
        # we still record an InstanceRecord so the element appears in the cache/scene
        # (it will just be an empty Xform in USD). This avoids "silent" drops.
        if primary_key is None and instance_mesh is None and not semantic_parts and detail_mesh_for_instance is None:
            log.debug(
                "Instance for %s (GlobalId=%s, step=%s) has no prototype, mesh, detail or semantic parts; "
                "recording as geometry-less instance.",
                product.is_a() if hasattr(product, "is_a") else "<IfcProduct>",
                getattr(product, "GlobalId", "unknown"),
                step_id,
            )

        # Build InstanceRecord
        # step_id already computed above
        if primary_key is not None:
            step_keys[step_id] = primary_key

        fallback = getattr(product, "GlobalId", None) or str(getattr(product, "id", lambda: step_id)())
        name = sanitize_name(getattr(product, "Name", None), fallback=fallback)
        try: product_id = product.id()
        except Exception: product_id = None

        proto_delta_tuple: Optional[Tuple[float, ...]] = None
        instance_transform_tuple = xf_tuple
        if options.enable_instancing and primary_key is not None and primary_key.kind == "hash":
            bucket = hashes.get(primary_key.identifier)
            if bucket:
                if bucket.canonical_frame is None and xf_tuple is not None and _is_affine_invertible_tuple(xf_tuple):
                    bucket.canonical_frame = xf_tuple
                canonical_frame = bucket.canonical_frame
                if canonical_frame is not None:
                    instance_transform_tuple = canonical_frame
                if canonical_frame is not None and xf_tuple is not None:
                    try:
                        canonical_np = _tuple16_to_np(canonical_frame)
                        xf_np = _tuple16_to_np(xf_tuple)
                        if canonical_np is None or xf_np is None:
                            raise ValueError("invalid transform data")
                        delta_np = np.matmul(np.linalg.inv(canonical_np), xf_np)
                        if not np.allclose(delta_np, np.eye(4), atol=1e-8):
                            proto_delta_tuple = tuple(float(delta_np[i, j]) for i in range(4) for j in range(4))
                    except Exception:
                        bucket.canonical_frame = None
                        instance_transform_tuple = xf_tuple
                        proto_delta_tuple = None

        attributes = collect_instance_attributes(product) if options.convert_metadata else {}

        key = product_id if product_id is not None else id(product)
        hierarchy_nodes = hierarchy_cache.get(key) or _collect_spatial_hierarchy(product)
        hierarchy_cache[key] = hierarchy_nodes


 
        guid = getattr(product, "GlobalId", None)
        instances[step_id] = InstanceRecord(
            step_id=step_id,
            product_id=product_id,
            prototype=primary_key,
            name=name,
            transform=instance_transform_tuple,
            material_ids=list(material_ids),
            materials=_clone_materials(materials),
            attributes=attributes,
            prototype_delta=proto_delta_tuple,
            hierarchy=hierarchy_nodes,
            mesh=instance_mesh,
            guid=guid,
            style_material=style_material,
            style_face_groups=_clone_style_groups(face_style_groups),
            detail_mesh=detail_mesh_for_instance,
            semantic_parts=semantic_parts,
            ifc_path=ifc_path,
        )

        if not iterator.next():
            break

    return PrototypeCaches(
        #Add the it break here
        repmaps=repmaps,
        repmap_counts=repmap_counts,
        hashes=hashes,
        step_keys=step_keys,
        instances=instances,
        annotations=extract_annotation_curves(ifc_file, hierarchy_cache, annotation_hooks),
        map_conversion=extract_map_conversion(ifc_file),
    )

