"""Frame resolution helpers for robust anchoring.

This module exists to answer one question reliably for arbitrary IFC exports:

    *Are the authored coordinates already in a global/survey frame, or are they
     local (near a project base point / internal origin)?*

Once we know that, we can compute a correct ifcopenshell `model-offset` for any
desired anchor point (local PBP or shared survey point), without adding extra
USD transforms.

Design goals
------------
* Fast: must run before meshing/prototype build.
* Robust: tolerate exporter differences (Revit, Civil3D, Tekla, etc.).
* Conservative: never raise; fall back safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple


GeomSpace = Literal["local", "global"]


@dataclass(frozen=True)
class GeometrySpaceResult:
    space: GeomSpace
    max_abs_translation_m: float
    sample_count: int


def _ifc_length_to_meters(ifc) -> float:
    """Best-effort IFC model length unit -> meters scale."""
    try:
        from ifcopenshell.util.unit import calculate_unit_scale  # type: ignore

        scale = float(calculate_unit_scale(ifc))
        if scale > 0:
            return scale
    except Exception:
        pass
    return 1.0


def _placement_location_xyz(obj_placement) -> Optional[Tuple[float, float, float]]:
    """Extract translation by walking the PlacementRelTo chain (approx absolute)."""
    if obj_placement is None:
        return None
    x = y = z = 0.0
    found = False
    cur = obj_placement
    depth = 0
    while cur is not None and depth < 32:
        rel = getattr(cur, "RelativePlacement", None)
        loc = getattr(rel, "Location", None) if rel is not None else None
        coords = getattr(loc, "Coordinates", None) if loc is not None else None
        if coords:
            try:
                xs = list(coords) + [0.0, 0.0, 0.0]
                x += float(xs[0])
                y += float(xs[1])
                z += float(xs[2])
                found = True
            except Exception:
                pass
        cur = getattr(cur, "PlacementRelTo", None)
        depth += 1
    if not found:
        return None
    return x, y, z


def classify_ifc_geometry_space(
    ifc: Any,
    *,
    global_threshold_m: float = 1.0e5,
    sample_limit: int = 400,
) -> GeometrySpaceResult:
    """Classify authored placement translations as LOCAL vs GLOBAL.

    Heuristic:
      - Sample up to `sample_limit` product placements.
      - Compute max(|x|,|y|,|z|) translation in meters.
      - If max_abs >= global_threshold_m => GLOBAL, else LOCAL.

    This works well because:
      - Local/PBP models have translations in tens/hundreds/thousands.
      - MGA/UTM survey frames have translations in hundreds of thousands/millions.
    """
    length_to_m = _ifc_length_to_meters(ifc)

    max_abs = 0.0
    count = 0

    def _iter_products():
        # Prefer IfcProduct; fall back to IfcElement/IfcBuildingElement
        for typename in ("IfcProduct", "IfcElement", "IfcBuildingElement"):
            try:
                items = ifc.by_type(typename) or []
                if items:
                    return items
            except Exception:
                continue
        return []

    # Signal from map conversion (if present) to catch files that author local coords + map conversion.
    try:
        contexts = ifc.by_type("IfcGeometricRepresentationContext") or []
    except Exception:
        contexts = []
    for ctx in contexts:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            try:
                if not hasattr(op, "is_a"):
                    continue
                if op.is_a("IfcMapConversion"):
                    scale = float(getattr(op, "Scale", 1.0) or 1.0)
                    eastings = float(getattr(op, "Eastings", 0.0) or 0.0) * scale
                    northings = float(getattr(op, "Northings", 0.0) or 0.0) * scale
                    height = float(getattr(op, "OrthogonalHeight", 0.0) or 0.0) * scale
                    max_abs = max(
                        max_abs,
                        abs(eastings * length_to_m),
                        abs(northings * length_to_m),
                        abs(height * length_to_m),
                    )
                elif op.is_a("IfcRigidOperation"):
                    first = float(getattr(op, "FirstCoordinate", 0.0) or 0.0)
                    second = float(getattr(op, "SecondCoordinate", 0.0) or 0.0)
                    height = float(getattr(op, "Height", 0.0) or 0.0)
                    max_abs = max(
                        max_abs,
                        abs(first * length_to_m),
                        abs(second * length_to_m),
                        abs(height * length_to_m),
                    )
            except Exception:
                continue

    for product in _iter_products():
        if count >= sample_limit:
            break
        try:
            place = getattr(product, "ObjectPlacement", None)
        except Exception:
            continue
        xyz = _placement_location_xyz(place)
        if xyz is None:
            continue
        x, y, z = (
            abs(xyz[0] * length_to_m),
            abs(xyz[1] * length_to_m),
            abs(xyz[2] * length_to_m),
        )
        max_abs = max(max_abs, x, y, z)
        count += 1

    # Default to local if we have no signal; it is safer for jitter.
    space: GeomSpace = "global" if max_abs >= float(global_threshold_m) else "local"
    return GeometrySpaceResult(
        space=space, max_abs_translation_m=max_abs, sample_count=count
    )


def compute_model_offset(
    *,
    geom_space: Any,
    anchor_mode: str,
    pbp: Any,
    shared_site: Any,
    bp_to_m: Callable[[Any], Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """Compute the raw ifcopenshell model-offset (meters) for offset-type='negative'.

    Interpretation:
      stage_pos = authored_pos - model_offset

    Definitions:
      - pbp: site-local base point (per-file/project base point)
      - shared_site: federated/survey anchor point (shared across sites)
      - anchor_mode: 'local' => stage origin represents PBP
                     'site'  => stage origin represents shared_site

    If geometry is GLOBAL (already in survey frame):
      model_offset = desired_anchor

    If geometry is LOCAL (authored around PBP):
      model_offset = desired_anchor - PBP
      (so stage_pos = v - (A - P) = v + (P - A))
    """
    mode = (anchor_mode or "local").strip().lower()
    desired_anchor = pbp if mode == "local" else shared_site

    p_m = bp_to_m(pbp)
    a_m = bp_to_m(desired_anchor)

    # Normalize geom_space object
    space = getattr(geom_space, "space", geom_space)
    space = str(space).strip().lower()
    if space not in ("local", "global"):
        # conservative fallback: treat as local
        space = "local"

    if space == "global":
        return (float(a_m[0]), float(a_m[1]), float(a_m[2]))

    # local geometry around PBP
    return (
        float(a_m[0] - p_m[0]),
        float(a_m[1] - p_m[1]),
        float(a_m[2] - p_m[2]),
    )
