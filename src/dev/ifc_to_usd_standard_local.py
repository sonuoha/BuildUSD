import math
import re
import time
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional, Dict, Tuple


import ifcopenshell
import ifcopenshell.geom
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf, Vt

# Optional: EPSG transforms (WGS84 <-> GDA2020 / MGA Zone 55 : EPSG:7855)
try:
    from pyproj import CRS, Transformer
    _HAVE_PYPROJ = True
except Exception:
    _HAVE_PYPROJ = False

# ifcopenshell util for robust matrices (mapped/type geometry)
try:
    from ifcopenshell.util import shape as ifc_shape_util
    _HAVE_IFC_UTIL_SHAPE = True
except Exception:
    _HAVE_IFC_UTIL_SHAPE = False


# ---------------- Matrix helpers ----------------

def np_to_gf_matrix(mat_data):
    """Convert a 16-element tuple or 4x4 numpy array to Gf.Matrix4d (row-major)."""
    if isinstance(mat_data, tuple) and len(mat_data) == 16:
        np_mat = np.array(mat_data, dtype=float).reshape(4, 4)
    elif isinstance(mat_data, np.ndarray) and mat_data.shape == (4, 4):
        np_mat = mat_data.astype(float)
    else:
        raise ValueError(f"Invalid matrix data: {mat_data}")

    gf_mat = Gf.Matrix4d()
    for i in range(4):
        for j in range(4):
            gf_mat[i, j] = np_mat[i, j]
    return gf_mat


def _extract_translation_safe(mat: Gf.Matrix4d) -> Gf.Vec3d:
    """Return translation from a 4x4 matrix robustly across USD builds."""
    try:
        return mat.ExtractTranslation()
    except Exception:
        pass
    try:
        r3 = mat[3]  # Gf.Vec4d
        return Gf.Vec3d(float(r3[0]), float(r3[1]), float(r3[2]))
    except Exception:
        pass
    try:
        return Gf.Vec3d(float(mat[0][3]), float(mat[1][3]), float(mat[2][3]))
    except Exception:
        pass
    return Gf.Vec3d(0.0, 0.0, 0.0)


def scale_matrix_translation_only(gfmat, scale):
    """Scale translation only (used if stage units != meters and you opt in)."""
    if scale == 1.0:
        return Gf.Matrix4d(gfmat)
    out = Gf.Matrix4d(gfmat)
    t = _extract_translation_safe(out)
    out.SetTranslateOnly(Gf.Vec3d(t[0]*scale, t[1]*scale, t[2]*scale))
    return out


def gf_to_tuple16(gf: Gf.Matrix4d):
    """Row-major 16-tuple from a Gf.Matrix4d."""
    return tuple(gf[i, j] for i in range(4) for j in range(4))


def _is_identity16(mat16, atol=1e-10):
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
        return np.allclose(arr, np.eye(4), atol=atol)
    except Exception:
        return False


# ---------------- IFC iteration ----------------

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

                # ----- absolute matrix resolution -----
                mat = None
                tr = getattr(shape, "transformation", None)
                if tr is not None and hasattr(tr, "matrix"):
                    mat = tuple(tr.matrix)
                    if _is_identity16(mat):
                        mat = None

                if mat is None and _HAVE_IFC_UTIL_SHAPE:
                    try:
                        gm = ifc_shape_util.get_shape_matrix(shape)  # list or 4x4
                        gm = np.array(gm, dtype=float).reshape(4, 4)
                        mat = tuple(gm.flatten().tolist())
                        if _is_identity16(mat):
                            mat = None
                    except Exception:
                        mat = None

                if mat is None:
                    try:
                        place = getattr(element, "ObjectPlacement", None)
                        gf = compose_object_placement(place, length_to_m=1.0)
                        mat = gf_to_tuple16(gf)
                    except Exception:
                        mat = None
                # --------------------------------------

                yield element, faces, verts, mat, materials, material_ids

            except Exception as e:
                if element is not None:
                    print(f"⚠️ Skipping element {getattr(element,'id',lambda: '?')()} due to error: {e}")
                else:
                    print(f"⚠️ Iterator error: {e}")
            if not it.next():
                break


# ---------------- USD authoring ----------------

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
        print(f"⚠️ No vertices for {mesh_name}; creating empty mesh.")

    if faces:
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in faces]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(faces)//3)))
    else:
        print(f"⚠️ No faces for {mesh_name}; mesh will be empty.")

    if abs_mat is not None:
        try:
            gf = np_to_gf_matrix(abs_mat)
            if scale_matrix_translation and stage_meters_per_unit != 1.0:
                gf = scale_matrix_translation_only(gf, 1.0/float(stage_meters_per_unit))
            xf = UsdGeom.Xformable(mesh)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(gf)
        except Exception as e:
            print(f"⚠️ Invalid matrix for {mesh_name}: {e}")

    if material_ids:
        mesh.GetPrim().CreateAttribute("ifc:materialIds", Sdf.ValueTypeNames.IntArray)\
            .Set(Vt.IntArray([int(i) for i in material_ids]))
    return mesh


# ---------------- Cesium alignment utilities ----------------

def _as_float(v, default=0.0):
    try:
        if hasattr(v, "wrappedValue"):
            return float(v.wrappedValue)
        return float(v)
    except Exception:
        return float(default)


def axis2placement_to_matrix(place, length_to_m=1.0):
    """Compose a 4x4 from IfcAxis2Placement2D/3D (location+axes)."""
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
    y = Gf.Cross(z, x).GetNormalized()
    x = Gf.Cross(y, z).GetNormalized()

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



def find_map_conversion_strict(ifc): 
    """Prefer IfcMapConversion from 3D 'Model' context. Fallback to first."""
    best = None
    for ctx in ifc.by_type("IfcGeometricRepresentationContext") or []:
        ops = getattr(ctx, "HasCoordinateOperation", None) or []
        for op in ops:
            if op.is_a("IfcMapConversion"):
                if (getattr(ctx, "ContextType", None) == "Model"
                    and getattr(ctx, "CoordinateSpaceDimension", None) == 3):
                    return op
                best = best or op
    return best


def local_to_map_ENH(local_xyh, map_conv):
    """
    Convert local (x,y,h) → map (E,N,H) using IfcMapConversion:
      [E;N] = [E0;N0] + (1/Scale) * [[ a, -b],[ b,  a]] * [x;y];  H = H0 + h
    """
    x, y, h = local_xyh
    a = _as_float(getattr(map_conv, "XAxisAbscissa", 1.0), 1.0)
    b = _as_float(getattr(map_conv, "XAxisOrdinate", 0.0), 0.0)
    s = _as_float(getattr(map_conv, "Scale", 1.0), 1.0) or 1.0
    E0 = _as_float(getattr(map_conv, "Eastings", 0.0), 0.0)
    N0 = _as_float(getattr(map_conv, "Northings", 0.0), 0.0)
    H0 = _as_float(getattr(map_conv, "OrthogonalHeight", 0.0), 0.0)
    norm = math.hypot(a, b)
    if norm > 0:
        a /= norm; b /= norm
    E = E0 + ( a * x - b * y) / s
    N = N0 + ( b * x + a * y) / s
    H = H0 + h
    return (E, N, H)


def map_to_local_XY(ENH, map_conv):
    """Inverse: map (E,N,*) → local (x,y,*)."""
    E, N, _ = ENH
    a = _as_float(getattr(map_conv, "XAxisAbscissa", 1.0), 1.0)
    b = _as_float(getattr(map_conv, "XAxisOrdinate", 0.0), 0.0)
    s = _as_float(getattr(map_conv, "Scale", 1.0), 1.0) or 1.0
    E0 = _as_float(getattr(map_conv, "Eastings", 0.0), 0.0)
    N0 = _as_float(getattr(map_conv, "Northings", 0.0), 0.0)
    norm = math.hypot(a, b)
    if norm > 0:
        a /= norm; b /= norm
    dE = (E - E0)
    dN = (N - N0)
    x =  s * ( a * dE + b * dN)
    y =  s * (-b * dE + a * dN)
    return (x, y)


def ifc_compound_to_degrees(comp):
    """
    IfcSite RefLatitude/RefLongitude: [deg,min,sec,μas] → decimal degrees.
    Returns None if input is None or invalid.
    
    Args:
        comp: IfcCompoundPlaneAngleMeasure (list/tuple of 1-4 numbers)
    Returns:
        Decimal degrees (float) or None.
    
    """
    if not comp:
        return None
    vals = list(comp)
    d = _as_float(vals[0], 0.0)
    m = _as_float(vals[1], 0.0) if len(vals) > 1 else 0.0
    s = _as_float(vals[2], 0.0) if len(vals) > 2 else 0.0
    micro = _as_float(vals[3], 0.0) if len(vals) > 3 else 0.0
    sign = 1.0 if d >= 0 else -1.0
    sec_total = abs(s) + abs(micro) * 1e-6
    deg = abs(d) + abs(m) / 60.0 + sec_total / 3600.0
    return sign * deg


def compute_mga55_from_site_latlon(site):
    """WGS84 lat/lon → GDA2020 / MGA55 (EPSG:7855) Easting/Northing (m)."""
    if not _HAVE_PYPROJ:
        return None
    lat = ifc_compound_to_degrees(getattr(site, "RefLatitude", None))
    lon = ifc_compound_to_degrees(getattr(site, "RefLongitude", None))
    if lat is None or lon is None:
        return None
    t = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(7855), always_xy=True)
    E, N = t.transform(lon, lat)
    H = _as_float(getattr(site, "RefElevation", 0.0), 0.0)
    return (E, N, H)


def set_cesium_anchor_attrs(stage, lat_deg, lon_deg, height_m):
    root = stage.GetPrimAtPath("/World")
    root.CreateAttribute("cesium:anchor:latitude",  Sdf.ValueTypeNames.Double).Set(float(lat_deg))
    root.CreateAttribute("cesium:anchor:longitude", Sdf.ValueTypeNames.Double).Set(float(lon_deg))
    root.CreateAttribute("cesium:anchor:height",    Sdf.ValueTypeNames.Double).Set(float(height_m))


def upsert_world_transform(stage, mat4d, op_name_suffix="geoAlign"):
    world = stage.GetPrimAtPath("/World")
    xf = UsdGeom.Xformable(world)
    existing = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == f"xformOp:transform:{op_name_suffix}":
            existing = op; break
    if existing:
        existing.Set(mat4d)
    else:
        xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, op_name_suffix).Set(mat4d)


# ---------- SP normalization (units) ----------

def _unit_label(scale):
    # meters-per-input-unit
    return {1.0: "m", 0.1: "dm", 0.01: "cm", 0.001: "mm"}.get(scale, f"×{scale}")


def _normalize_coords_to_meters(e_raw, n_raw, h_raw=0.0, unit_hint="auto"):
    """
    Normalize SP Easting/Northing/Height to meters.
    unit_hint: 'auto' | 'm' | 'mm' | 'cm' | 'dm'
    Returns (E_m, N_m, H_m, scale_used), where scale_used = meters-per-input-unit.
    """
    e = _as_float(e_raw, 0.0)
    n = _as_float(n_raw, 0.0)
    h = _as_float(h_raw, 0.0)

    # explicit hint
    if isinstance(unit_hint, str) and unit_hint.lower() in {"m","meter","meters"}:
        return e, n, h, 1.0
    if isinstance(unit_hint, str) and unit_hint.lower() in {"mm","millimeter","millimeters"}:
        return e*0.001, n*0.001, h*0.001, 0.001
    if isinstance(unit_hint, str) and unit_hint.lower() in {"cm","centimeter","centimeters"}:
        return e*0.01, n*0.01, h*0.01, 0.01
    if isinstance(unit_hint, str) and unit_hint.lower() in {"dm","decimeter","decimeters"}:
        return e*0.1, n*0.1, h*0.1, 0.1

    # auto-detect to plausible UTM/MGA meter ranges
    candidates = [1.0, 0.1, 0.01, 0.001]
    def plausible(e_m, n_m):
        return (1e5 <= abs(e_m) <= 1e6) and (1e6 <= abs(n_m) <= 1e7)
    for s in candidates:
        if plausible(e*s, n*s):
            return e*s, n*s, h*s, s

    # fallback: minimise range error
    def range_error(e_m, n_m):
        e_ok = 0.0 if 1e5 <= abs(e_m) <= 1e6 else min(abs(abs(e_m)-1e5), abs(abs(e_m)-1e6))
        n_ok = 0.0 if 1e6 <= abs(n_m) <= 1e7 else min(abs(abs(n_m)-1e6), abs(abs(n_m)-1e7))
        return e_ok + n_ok
    best_s = min(candidates, key=lambda s: range_error(e*s, n*s))
    return e*best_s, n*best_s, h*best_s, best_s


def compute_align_matrix_from_mapconv(map_conv, stage_meters_per_unit,
                                      E_sp_m, N_sp_m, Z_sp_m=0.0, align_axes=True):
    """Translate SP to origin (and optionally rotate local XY to EN)."""
    x_sp, y_sp = map_to_local_XY((E_sp_m, N_sp_m, Z_sp_m), map_conv)
    meters_to_stage = 1.0 / float(stage_meters_per_unit if stage_meters_per_unit else 1.0)
    T = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-x_sp * meters_to_stage,
                                            -y_sp * meters_to_stage,
                                            -Z_sp_m * meters_to_stage))
    if not align_axes:
        return T
    a = _as_float(getattr(map_conv, "XAxisAbscissa", 1.0), 1.0)
    b = _as_float(getattr(map_conv, "XAxisOrdinate", 0.0), 0.0)
    norm = math.hypot(a, b)
    if norm > 0:
        a /= norm; b /= norm
    phi_deg = math.degrees(math.atan2(b, a))
    R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), phi_deg))
    return R * T


def apply_world_alignment_with_SP(
    ifc,
    stage,
    sp_e_mm, sp_n_mm,
    sp_lat_deg, sp_lon_deg,
    sp_h_mm=0.0,
    align_axes_to_EN=False,      # keep OFF until translation is verified
    sp_unit_hint="auto"          # 'auto' | 'm' | 'mm' | 'cm' | 'dm'
):
    """
    ALWAYS uses SP to build the inverse world shift.
      - Normalize SP (E,N,H) to meters (auto or hinted).
      - If IfcMapConversion exists: (E,N,H m) → local (x,y,*) and translate by −(x,y,H).
      - Else: assume local XY≈map EN and translate by −(E,N,H) (approx).
    Writes a single double-precision transform on /World and sets Cesium anchor.
    """
    # 0) normalize input to meters
    E_sp_m, N_sp_m, H_sp_m, scale_used = _normalize_coords_to_meters(
        sp_e_mm, sp_n_mm, sp_h_mm, unit_hint=sp_unit_hint
    )
    print(f"🧭 SP normalization: interpreted units={_unit_label(scale_used)}, "
          f"SP (m)=({E_sp_m:.3f}, {N_sp_m:.3f}, {H_sp_m:.3f})")

    stage_mpu = float(stage.GetMetadata("metersPerUnit") or 1.0)
    meters_to_stage = 1.0 / stage_mpu

    map_conv = find_map_conversion_strict(ifc)
    if map_conv:
        x_sp, y_sp = map_to_local_XY((E_sp_m, N_sp_m, H_sp_m), map_conv)
        print(f"✅ MapConversion in IFC. SP local xy = ({x_sp:.3f}, {y_sp:.3f}); H={H_sp_m:.3f} m")
    else:
        x_sp, y_sp = E_sp_m, N_sp_m
        print("⚠️ No IfcMapConversion. Using SP EN as local XY (approx).")

    T = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(-x_sp * meters_to_stage, -y_sp * meters_to_stage, -H_sp_m * meters_to_stage)
    )

    M = T
    if align_axes_to_EN and map_conv:
        a = _as_float(getattr(map_conv, "XAxisAbscissa", 1.0), 1.0)
        b = _as_float(getattr(map_conv, "XAxisOrdinate", 0.0), 0.0)
        norm = math.hypot(a, b)
        if norm > 0:
            a /= norm; b /= norm
        phi_deg = math.degrees(math.atan2(b, a))
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), phi_deg))
        M = R * T
        print(f"↻ Axis alignment enabled: rotating Z by {phi_deg:.6f}°")

    upsert_world_transform(stage, M, op_name_suffix="geoAlign")
    set_cesium_anchor_attrs(stage, sp_lat_deg, sp_lon_deg, H_sp_m)
    print(f"📍 Cesium anchor set at lat={sp_lat_deg}, lon={sp_lon_deg}, h={H_sp_m} m")
    print("   /World ‘geoAlign’ authored using SP inverse shift.")


# ---------------- Conversion (baseline-accurate) ----------------

def convert_ifc_to_usd(
    ifc_path: Union[str, Path],
    usd_path: Optional[Union[str, Path]] = None,
    meters_per_unit: float = 1.0,
    convert_back_units: bool = False,
    # Cesium/SP alignment (OFF by default)
    align_with_survey_point: bool = False,
    sp_e_mm: Optional[float] = None,
    sp_n_mm: Optional[float] = None,
    sp_lat_deg: Optional[float] = None,
    sp_lon_deg: Optional[float] = None,
    sp_h_mm: float = 0.0,
    align_axes_to_EN: bool = False,
    sp_unit_hint: str = "auto",
) -> None:

    """
    Baseline-accurate IFC → USD:
      - USE_WORLD_COORDS=False
      - author local points
      - set per-mesh absolute matrix returned by the iterator
      - no scaling unless stage units != meters (disabled here)
    If ifc_path is a directory, every *.ifc inside is converted into
    an adjacent 'output/<stem>.usda' (or into the directory 'usd_path' if usd_path is a directory).
    """
    p = Path(ifc_path)

    def _process_one_file(ifc_file_path: Path, out_file_path: Path):
        print(f"⏳ Processing IFC: {ifc_file_path}")
        ifc = ifcopenshell.open(str(ifc_file_path))

        # Iterator settings: local verts + absolute transforms; data in meters by default
        settings = ifcopenshell.geom.settings()
        settings.set("USE_WORLD_COORDS", False)
        settings.set("convert-back-units", bool(convert_back_units))  # default False → meters out

        # Create stage per file
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        stage = create_usd_stage(out_file_path, meters_per_unit)
        world = "/World"

        # Group under IFC type (no transforms on parent)
        type_groups = defaultdict(list)
        for element, faces, verts, abs_mat, materials, material_ids in generate_mesh_data(ifc, settings):
            if not verts or not faces:
                continue
            etype = element.is_a()
            raw_name = getattr(element, "Name", None)
            fallback = getattr(element, "GlobalId", None) or str(element.id())
            prim_name = sanitize_name(raw_name, fallback=fallback)
            type_groups[etype].append((prim_name, faces, verts, abs_mat, material_ids))

        for etype, items in type_groups.items():
            type_path = f"{world}/{re.sub(r'[^A-Za-z0-9_]', '_', etype)[:63]}"
            UsdGeom.Xform.Define(stage, type_path)
            for prim_name, faces, verts, abs_mat, material_ids in items:
                write_usd_mesh(
                    stage, type_path, prim_name,
                    verts, faces,
                    abs_mat=abs_mat,
                    material_ids=material_ids,
                    stage_meters_per_unit=meters_per_unit,
                    scale_matrix_translation=False
                )

        # Optional: Cesium/SP alignment
        if align_with_survey_point:
            missing = [k for k, v in dict(
                sp_e_mm=sp_e_mm, sp_n_mm=sp_n_mm,
                sp_lat_deg=sp_lat_deg, sp_lon_deg=sp_lon_deg).items() if v is None]
            if missing:
                print(f"⚠️ align_with_survey_point=True but missing inputs: {missing}. Skipping SP alignment.")
            else:
                apply_world_alignment_with_SP(
                    ifc, stage,
                    sp_e_mm=sp_e_mm, sp_n_mm=sp_n_mm,
                    sp_lat_deg=sp_lat_deg, sp_lon_deg=sp_lon_deg,
                    sp_h_mm=sp_h_mm,
                    align_axes_to_EN=align_axes_to_EN,
                    sp_unit_hint=sp_unit_hint
                )

        stage.Save()
        print(f"✅ USD written: {out_file_path}")

    # Directory or file?
    if p.is_dir():
        # if usd_path is a directory, place outputs there; else default to p/'output'
        out_dir = Path(usd_path) if (usd_path and Path(usd_path).is_dir()) else (p / "output")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() == ".ifc":
                out_file = out_dir / (f.stem + ".usda")
                _process_one_file(f, out_file)
    else:
        in_file = p
        if usd_path is None:
            out_file = in_file.parent / "output" / (in_file.stem + ".usda")
        else:
            out_file = Path(usd_path)
            if out_file.is_dir():
                out_file = out_file / (in_file.stem + ".usda")
        _process_one_file(in_file, out_file)


# --------- Inspector (robust across USD builds) ---------

def inspect_world_alignment(usd_path: Union[str, Path]) -> None:
    stage = Usd.Stage.Open(usd_path)
    root  = stage.GetPrimAtPath("/World")
    if not root:
        print("❌ /World not found")
        return

    mpu = stage.GetMetadata("metersPerUnit")
    print(f"Stage metersPerUnit: {mpu}")

    xf  = UsdGeom.Xformable(root)
    ops = xf.GetOrderedXformOps()
    print("xformOpOrder:", [op.GetOpName() for op in ops])

    local_tf = xf.GetLocalTransformation()
    world_mat = local_tf[0] if isinstance(local_tf, tuple) else local_tf

    if isinstance(world_mat, Gf.Matrix4d):
        t = _extract_translation_safe(world_mat)
        print("Current /World local translation (stage units):", (float(t[0]), float(t[1]), float(t[2])))
    else:
        print("⚠️ Unexpected GetLocalTransformation() return type:", type(world_mat))

    for a in ("cesium:anchor:latitude","cesium:anchor:longitude","cesium:anchor:height"):
        attr = root.GetAttribute(a)
        print(a, "=", (attr.Get() if attr and attr.HasAuthoredValueOpinion() else None))


# --------- Optional: compute lon/lat from MGA55 E/N (mm) ---------

def mga55_en_to_wgs84_lonlat_deg(
    e_raw: Union[float, int, str],
    n_raw: Union[float, int, str],
    unit_hint: Optional[str] = "auto"
) -> Tuple[float, float]:
    """
    Convert MGA55 (GDA2020, EPSG:7855) Easting/Northing to WGS84 lon/lat (degrees),
    robustly handling input units.

    Args:
        e_raw, n_raw: Easting/Northing in m, dm, cm, or mm.
        unit_hint: 'auto' (default), 'm', 'dm', 'cm', or 'mm' to override detection.

    Returns:
        (lon_deg, lat_deg)
    """
    if not _HAVE_PYPROJ:
        raise RuntimeError("pyproj is required to convert MGA55 to WGS84. Install with: pip install pyproj")

    e_m, n_m, h_m, scale_used = _normalize_coords_to_meters(e_raw, n_raw, unit_hint=unit_hint)
    print(f"[mga55_en_to_wgs84] Using units scale={scale_used} → meters; E={e_m}, N={n_m}")

    t = Transformer.from_crs(CRS.from_epsg(7855), CRS.from_epsg(4326), always_xy=True)
    lon, lat = t.transform(e_m, n_m)  # (x,y) -> (lon, lat)
    return float(lon), float(lat)


# ---------------- Example runner ----------------

if __name__ == "__main__":
    # Change these as needed
    cwd = Path(__file__).resolve().parent / "data"
    # Single file example:
    #20400-WEA-S00-IFC-EA-000100
    #SRL-WPD-TVC-UTU8-MOD-CTU-BUW-000001
    #20400-WEA-S00-IFC-EA-000100_PBP
    #SRL-WPD-TVC-XBWS-MOD-SGE-BUW-000001_PBP

    #ifc_file = cwd / "SRL-WPD-TVC-UTU8-MOD-CTU-BUW-000001.ifc"
    ifc_file = cwd / "20400-WEA-S00-IFC-EA-000100_PBP.ifc"

    # Directory example (convert every *.ifc inside):
    # ifc_file = Path(r"C:\Users\samue\Tanech Pty Ltd\Projects - Documents\Assemble-sonuoha\dev_tests")

    outdir = cwd / "output"; outdir.mkdir(parents=True, exist_ok=True)
    usd_out = outdir / (ifc_file.stem + ".usda")

    if not ifc_file.exists():
        raise FileNotFoundError(f"IFC file not found: {ifc_file}")

    # Survey Point in MGA55 (mm)
    SP_E_MM = 332_942_836.9
    SP_N_MM = 5_801_895_432.4
    SP_H_MM = 0.0

    # Prefer computing lon/lat from E/N so values are authoritative
    if _HAVE_PYPROJ:
        SP_LON, SP_LAT = mga55_en_to_wgs84_lonlat_deg(SP_E_MM, SP_N_MM, unit_hint="auto")
        print(f"Derived SP WGS84: lon={SP_LON:.7f}, lat={SP_LAT:.7f}")
    else:
        # Fallback constants (replace if needed)
        SP_LAT  = -37.91515794
        SP_LON  = 145.09951462
        print("⚠️ pyproj not available; using fallback SP lon/lat.")

    """convert_ifc_to_usd(
        ifc_file,
        usd_out,
        meters_per_unit=1.0,
        convert_back_units=False,
        align_with_survey_point=True,
        sp_e_mm=SP_E_MM,                # can be mm/cm/dm/m; auto-detected
        sp_n_mm=SP_N_MM,
        sp_lat_deg=SP_LAT,
        sp_lon_deg=SP_LON,
        sp_h_mm=SP_H_MM,
        align_axes_to_EN=False,         # first verify translation is perfect
        sp_unit_hint="auto"             # or "mm"/"m"/"cm"/"dm" if you want to force it
    ) 
    """
    ass_easting = float(318197.2518)  # 3318197.2502 m
    ass_northing = float(5815160.4723)  # 5815160.4696 m
    # Prefer computing lon/lat from E/N so values are authoritative
    if _HAVE_PYPROJ:
        as_long, as_lat = mga55_en_to_wgs84_lonlat_deg(ass_easting, ass_northing, unit_hint="auto")
        print(f"Derived PBP WGS84 for Assemble: lon={as_long:.7f}, lat={as_lat:.7f}")
    else:
        # Fallback constants (replace if needed)
        ass_SP_LAT  = -37.792849
        ass_SP_LON  = 144.935172
        print("⚠️ pyproj not available; using fallback SP lon/lat.")
    #BURWOOD
    bur_pbp_n = 5809101.4680
    bur_pbp_e = 333800.4900
    bur_pbp_h = 0.0
    bur_pbp_attn = 277.6100
    if not _HAVE_PYPROJ:
        # Fallback constants (replace if needed)
        SP_LAT  = -37.85040191094061
        SP_LON  = 145.1109260732928
        print("⚠️ pyproj not available; using fallback SP lon/lat.")
    else:
        SP_LON, SP_LAT = mga55_en_to_wgs84_lonlat_deg(bur_pbp_e, bur_pbp_n)
        print(f"Derived SP WGS84: lon={SP_LON:.7f}, lat={SP_LAT:.7f}")

    convert_ifc_to_usd(
        ifc_file,
        meters_per_unit=1.0,
        convert_back_units=False,
        align_with_survey_point=True,
        sp_e_mm=ass_easting,                # can be mm/cm/dm/m; auto-detected
        sp_n_mm=ass_northing,
        sp_lat_deg=as_lat,
        sp_lon_deg=as_long,
        align_axes_to_EN=True,  
        sp_unit_hint="auto"             # or "mm"/"m"/"cm"/"dm" if you want to force it
    )


    # Optional: inspect the world alignment that was authored
    # inspect_world_alignment(str(usd_out))
# 20400-WEA-S00-IFC-EA-000100
