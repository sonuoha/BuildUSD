
import re
import time
from pxr import Usd, UsdGeom, Sdf, Gf, Vt
from src.utils.matrix_utils import np_to_gf_matrix, scale_matrix_translation_only

from collections import Counter



# 1. ---------------------------- USD Utilities ----------------------------
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
