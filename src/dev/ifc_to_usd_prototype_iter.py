# Fix the SyntaxError (a stray colon in the commented argparse block).
from pathlib import Path
from textwrap import dedent


# Create an end-to-end script that discovers prototypes from instances:
# - RepMap-based prototypes from IfcMappedItem.MappingSource (LOCAL tessellation)
# - Unmapped prototypes by hashing LOCAL IfcShapeRepresentation meshes (dedupe)
# - Writes a USD with /__Prototypes only
# - Emits a manifest CSV and clear console summary with names derived from type names when available
from textwrap import dedent
from pathlib import Path

#script_path = Path("/mnt/data/ifc_prototypes_from_instances.py")


"""
ifc_prototypes_from_instances.py  (iterator-integrated)

Discover and build **prototype USD meshes only** from an IFC, using an instance-driven approach,
now powered by the geometry **iterator** for discovery.

1) Iterate all product representations using ifcopenshell.geom.iterator in **LOCAL** coords.
   - If a rep uses IfcMappedItem, cache its IfcRepresentationMap id (true type geometry).
   - If a rep has no mapped items, take the iterator's tessellated LOCAL mesh, hash triangles,
     and group by hash as a fallback prototype.

2) Author a USD stage with ONLY /__Prototypes:
   - For each unique IfcRepresentationMap id seen, tessellate rm.MappedRepresentation (LOCAL) once
     and author a prototype. If direct tessellation fails, derive from an occurrence via unapplying
     Placement * MappingTarget * MappingOrigin.
   - For each fallback hash, author a single prototype (first seen mesh).

3) Emit a manifest CSV with per-prototype metadata and counts, and print a clear summary.

Usage:
  pip install ifcopenshell numpy pxr
  python ifc_prototypes_from_instances.py --ifc model.ifc --usd prototypes.usda \
      [--disable-openings] [--body-only] [--manifest manifest.csv] [--hash-precision 6] [--max-print 50]

Notes:
  - Iterator runs with USE_WORLD_COORDS=False for LOCAL prototype meshes.
  - 'body-only' filters representations to identifiers typically used for solid geometry.
"""

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Set

import numpy as np

import ifcopenshell
import ifcopenshell.geom
from ifcopenshell.entity_instance import entity_instance

from pxr import Usd, UsdGeom, Sdf, Gf

BODY_IDENTIFIERS  = {
    "body",
    "facetation",
    "tessellation",
    "surfacemodel",
    "sweptsolid",
    "advancedsweptsolid",
    "brep",
    "advancedbrep",
    "clipping",
    "csg"
}


# ----------------------------
# Helpers
# ----------------------------

def sanitize_name(name: str) -> str:
    if not name:
        return "Unnamed"
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Unnamed"

def triangulated_to_dict(shape_geom) -> dict:
    verts = np.array(shape_geom.verts, dtype=float).reshape(-1, 3)
    faces = np.array(shape_geom.faces, dtype=int).reshape(-1, 3)
    return {"vertices": verts, "faces": faces}

def mesh_hash(mesh: dict, precision: int = 6) -> str:
    verts = np.round(mesh["vertices"], precision)
    uniq, inv = np.unique(verts, axis=0, return_inverse=True)
    faces = inv[mesh["faces"]]
    faces_canon = np.sort(np.sort(faces, axis=1), axis=0)
    h = hashlib.sha256()
    h.update(uniq.tobytes()); h.update(faces_canon.tobytes())
    return h.hexdigest()

def author_mesh(stage: Usd.Stage, path: Sdf.Path, mesh: dict, double_sided=True):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr([Gf.Vec3f(*p) for p in mesh["vertices"]])
    prim.CreateFaceVertexCountsAttr([3] * len(mesh["faces"]))
    prim.CreateFaceVertexIndicesAttr(mesh["faces"].astype(int).flatten().tolist())
    if double_sided:
        prim.CreateDoubleSidedAttr(True)
    return prim

def safe_set(settings, key, value):
    try:
        settings.set(key, value); return True
    except Exception:
        alts = []
        if "-" in key: alts.append(key.replace("-", "_").upper())
        if "_" in key: alts.append(key.replace("_", "-").lower())
        for k in alts:
            try:
                settings.set(k, value); return True
            except Exception:
                continue
    return False

# ----------------------------
# Records
# ----------------------------

@dataclass
class RepMapProtoInfo:
    repmap_id: int
    type_guid: Optional[str] = None
    type_name: Optional[str] = None
    type_class: Optional[str] = None
    repmap_index: Optional[int] = None
    count_instances: int = 0

@dataclass
class FallbackProtoInfo:
    hash: str
    count_instances: int = 0
    example_class: Optional[str] = None
    example_object_type: Optional[str] = None
    example_name: Optional[str] = None
    mesh: Optional[dict] = None

# ----------------------------
# Builder
# ----------------------------

class PrototypesFromInstances:
    def __init__(self, ifc_path: str, usd_path: str,
                 disable_openings: bool = False,
                 body_only: bool = False,
                 hash_precision: int = 6,
                 max_print: int = 50,
                 manifest_csv: Optional[str] = None):

        self.ifc = ifcopenshell.open(ifc_path)
        self.usd_path = usd_path
        self.body_only = body_only
        self.hash_precision = int(hash_precision)
        self.max_print = max_print
        self.manifest_csv = manifest_csv

        # LOCAL tessellation (iterator will use this too)
        self.settings = ifcopenshell.geom.settings()
        #self.settings.set("USE_PYTHON_OPENCASCADE", True)
        safe_set(self.settings, "weld-vertices", True)
        safe_set(self.settings, "use-world-coords", False)  # LOCAL for prototypes
        safe_set(self.settings, "disable-opening-subtractions", bool(disable_openings))
        #self.settings.set("SEW_SHELLS", True)
        safe_set(self.settings, "apply-default-materials", False)

        # USD stage
        self.stage = Usd.Stage.CreateNew(self.usd_path)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(self.stage, Sdf.Path("/__Prototypes"))

        # Pre-index repmap -> (type, idx)
        self.repmap_to_type: Dict[int, Tuple[entity_instance, int]] = {}
        for t in self.ifc.by_type("IfcTypeProduct"):
            repmaps = getattr(t, "RepresentationMaps", None) or []
            for idx, rm in enumerate(repmaps):
                self.repmap_to_type[rm.id()] = (t, idx)

        # Discovered
        self.repmap_infos: Dict[int, RepMapProtoInfo] = {}
        self.fallback_infos: Dict[str, FallbackProtoInfo] = {}

        # Errors
        self.tessellation_fail_repmap: List[int] = []
        self.tessellation_fail_unmapped: int = 0

    def _is_body_like(self, rep) -> bool:
        if not self.body_only:
            return True
        ident = (getattr(rep, "RepresentationIdentifier", None) or "").lower()
        return ident in BODY_IDENTIFIERS

    # ---------- derive prototype mesh from an occurrence using this repmap ----------
    def _derive_mesh_from_occurrence(self, repmap):
        """
        Derive a prototype mesh from an occurrence using this repmap.

        Iterate all IFC products to find one that uses this repmap. Then, tessellate the
        occurrence rep (LOCAL coords) and build the transform chain to unapply the repmap
        transform. Return the derived prototype mesh in local coordinates.

        Returns
        -------
        dict
            A dictionary containing the derived prototype mesh in local coordinates.
        """

        import numpy as np

        def _norm(v):
            """
            Normalize a 3-element vector v to have unit length.
            
            If the vector has zero length, return the original vector (to avoid NaNs).
            
            Parameters
            ----------
            v : numpy.ndarray
                The vector to be normalized.

            Returns
            -------
            numpy.ndarray
                The normalized vector.
            """
            n = np.linalg.norm(v); return v / n if n else v

        def axis2placement3d_to_matrix(ax):
            Z = np.array([0,0,1.0]); X = np.array([1.0,0,0])
            if getattr(ax,"Axis",None):
                Z = np.array(ax.Axis.DirectionRatios, float)
            if getattr(ax,"RefDirection",None):
                X = np.array(ax.RefDirection.DirectionRatios, float)
            Z=_norm(Z); X=_norm(X); Y=_norm(np.cross(Z,X)); X=_norm(np.cross(Y,Z))
            O = np.array(ax.Location.Coordinates, float)
            M = np.eye(4); M[:3,0]=X; M[:3,1]=Y; M[:3,2]=Z; M[:3,3]=O; return M

        def local_placement_to_matrix(lp):
            if not lp: return np.eye(4)
            M = local_placement_to_matrix(getattr(lp,"PlacementRelTo",None))
            return M @ axis2placement3d_to_matrix(lp.RelativePlacement)

        def cartop_to_matrix(op):
            if not op: return np.eye(4)
            M = axis2placement3d_to_matrix(op)
            s = float(op.Scale or 1.0)
            sx=s; sy=float(getattr(op,"Scale2",s) or s); sz=float(getattr(op,"Scale3",s) or s)
            S = np.diag([sx,sy,sz,1.0]); R=M.copy(); R[:3,3]=0; T=np.eye(4); T[:3,3]=M[:3,3]
            return T @ R @ S

        # Find one product that uses this repmap
        for prod in self.ifc.by_type("IfcProduct"):
            repC = getattr(prod, "Representation", None)
            if not repC: continue
            for rep in (repC.Representations or []):
                items = rep.Items or []
                for it in items:
                    if not it.is_a("IfcMappedItem"): continue
                    if it.MappingSource != repmap: continue

                    # Tessellate the occurrence rep (LOCAL coords)
                    sh = ifcopenshell.geom.create_shape(self.settings, rep)
                    geom = sh.geometry
                    verts = np.array(geom.verts, float).reshape(-1,3)
                    faces = np.array(geom.faces, int).reshape(-1,3)

                    # Build transform chain and unapply
                    P = local_placement_to_matrix(prod.ObjectPlacement) if prod.ObjectPlacement else np.eye(4)
                    Tm = cartop_to_matrix(it.MappingTarget) if it.MappingTarget else np.eye(4)
                    O  = axis2placement3d_to_matrix(repmap.MappingOrigin) if repmap.MappingOrigin else np.eye(4)
                    M  = P @ Tm @ O

                    Minv = np.linalg.inv(M)
                    v_h = np.c_[verts, np.ones((len(verts),1))]
                    verts_local = (v_h @ Minv.T)[:, :3]
                    return {"vertices": verts_local, "faces": faces}
        return None

    def discover(self):
        """Iterator-driven discovery: collect repmaps & unmapped fallback meshes in LOCAL coords."""
        total_steps = 0
        it = ifcopenshell.geom.iterator(self.settings, self.ifc)
        if not it.initialize():
            raise RuntimeError("Iterator initialization failed")

        while it.next():
            total_steps += 1
            shape_result = it.get()                      # <- shape result for this step
            if shape_result is None:
                continue

            geom = shape_result.geometry                 # tessellated geometry
            rep = getattr(geom, "representation", None)  # IfcShapeRepresentation
            if not rep or not rep.is_a("IfcShapeRepresentation"):
                continue
            if not self._is_body_like(rep):
                continue

            # Resolve the IfcProduct from the shape result id
            product = None
            try:
                product = self.ifc.by_id(shape_result.id)
            except Exception:
                pass

            items = rep.Items or []
            mapped = [mi for mi in items if mi.is_a("IfcMappedItem")]
            if mapped:
                for mi in mapped:
                    rm = mi.MappingSource
                    rmid = rm.id()
                    info = self.repmap_infos.get(rmid)
                    if info is None:
                        t, idx = self.repmap_to_type.get(rmid, (None, None))
                        info = RepMapProtoInfo(
                            repmap_id=rmid,
                            type_guid=getattr(t, "GlobalId", None) if t else None,
                            type_name=getattr(t, "Name", None) if t else None,
                            type_class=t.is_a() if t else None,
                            repmap_index=idx
                        )
                        self.repmap_infos[rmid] = info
                    info.count_instances += 1
            else:
                # Unmapped: use iterator's tessellated LOCAL mesh and hash it
                try:
                    mesh = triangulated_to_dict(geom)
                    digest = mesh_hash(mesh, precision=self.hash_precision)
                    fb = self.fallback_infos.get(digest)
                    if fb is None:
                        fb = FallbackProtoInfo(
                            hash=digest,
                            count_instances=1,
                            example_class=product.is_a() if product else None,
                            example_object_type=getattr(product, "ObjectType", None) if product else None,
                            example_name=getattr(product, "Name", None) if product else None,
                            mesh=mesh
                        )
                        self.fallback_infos[digest] = fb
                    else:
                        fb.count_instances += 1
                except Exception as e:
                    self.tessellation_fail_unmapped += 1
                    print(f"[Iterator unmapped] tessellation failed: {type(e).__name__}: {e}")

        print(f"Iterator discovery complete: steps={total_steps}")
        print(f"  Unique repmaps from instances: {len(self.repmap_infos)}")
        print(f"  Unique unmapped-hash prototypes: {len(self.fallback_infos)}")
        if self.tessellation_fail_unmapped:
            print(f"  Unmapped tessellation failures: {self.tessellation_fail_unmapped}")


    def build_prototypes(self):
        # RepMap-based prototypes
        for rmid, info in self.repmap_infos.items():
            rm = self.ifc.by_id(rmid)
            rep = getattr(rm, "MappedRepresentation", None)
            if not rep or not self._is_body_like(rep):
                continue

            mesh = None
            try:
                sh = ifcopenshell.geom.create_shape(self.settings, rep)
                mesh = triangulated_to_dict(sh.geometry)
            except Exception as e:
                # Fallback: derive from an occurrence that uses this repmap
                mesh = self._derive_mesh_from_occurrence(rm)
                if mesh is None:
                    self.tessellation_fail_repmap.append(rmid)
                    print(f"[RepMap #{rmid}] tessellation failed: {type(e).__name__}: {e}")
                    continue

            base = sanitize_name(info.type_name) if info.type_name else f"RepMap_{rmid}"
            suffix = []
            if info.type_class: suffix.append(sanitize_name(info.type_class))
            if info.type_guid:  suffix.append(info.type_guid)
            if info.repmap_index is not None: suffix.append(f"i{info.repmap_index}")
            name = "_".join([base] + suffix)

            path = Sdf.Path("/__Prototypes").AppendPath(name)
            author_mesh(self.stage, path, mesh)

        # Fallback prototypes (from iterator unmapped cache)
        for digest, fb in self.fallback_infos.items():
            mesh = fb.mesh
            if mesh is None:
                continue
            guess = fb.example_object_type or fb.example_name or fb.example_class or "Unmapped"
            base = sanitize_name(guess)
            name = f"{base}_Fallback_{digest[:12]}"
            path = Sdf.Path("/__Prototypes").AppendPath(name)
            author_mesh(self.stage, path, mesh)

        self.stage.Save()

    def write_manifest(self):
        if not self.manifest_csv:
            return
        with open(self.manifest_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kind","usd_prim_name","repmap_id_or_hash","type_class","type_name","type_guid","repmap_index","instance_count"])
            prototypes = self.stage.GetPrimAtPath("/__Prototypes")
            for child in prototypes.GetChildren():
                name = child.GetName()
                matched = False
                for rmid, info in self.repmap_infos.items():
                    expected_base = sanitize_name(info.type_name) if info.type_name else f"RepMap_{rmid}"
                    parts = [expected_base]
                    if info.type_class: parts.append(sanitize_name(info.type_class))
                    if info.type_guid:  parts.append(info.type_guid)
                    if info.repmap_index is not None: parts.append(f"i{info.repmap_index}")
                    expected_name = "_".join(parts)
                    if expected_name == name:
                        w.writerow(["repmap", name, rmid, info.type_class or "", info.type_name or "", info.type_guid or "", info.repmap_index if info.repmap_index is not None else "", info.count_instances])
                        matched = True
                        break
                if matched:
                    continue
                if "_Fallback_" in name:
                    hash_suffix = name.split("_Fallback_")[-1]
                    for h, fb in self.fallback_infos.items():
                        if h.startswith(hash_suffix):
                            w.writerow(["fallback", name, h, fb.example_class or "", fb.example_object_type or fb.example_name or "", "", "", fb.count_instances])
                            matched = True
                            break
                if not matched:
                    w.writerow(["unknown", name, "", "", "", "", "", ""])

    def print_summary(self):
        n_repmap = len(self.repmap_infos)
        n_fallback = len(self.fallback_infos)
        print("\n=== Prototype Summary ===")
        print(f"RepMap prototypes: {n_repmap}")
        printed = 0
        for rmid, info in sorted(self.repmap_infos.items(), key=lambda kv: -kv[1].count_instances):
            base = sanitize_name(info.type_name) if info.type_name else f"RepMap_{rmid}"
            parts = [base]
            if info.type_class: parts.append(sanitize_name(info.type_class))
            if info.type_guid:  parts.append(info.type_guid)
            if info.repmap_index is not None: parts.append(f"i{info.repmap_index}")
            name = "_".join(parts)
            print(f"  - {name}  (repmap #{rmid}, instances={info.count_instances})")
            printed += 1
            if printed >= self.max_print:
                break

        print(f"\nFallback prototypes (deduped from unmapped): {n_fallback}")
        printed = 0
        for h, fb in sorted(self.fallback_infos.items(), key=lambda kv: -kv[1].count_instances):
            guess = fb.example_object_type or fb.example_name or fb.example_class or "Unmapped"
            nm = f"{sanitize_name(guess)}_Fallback_{h[:12]}"
            print(f"  - {nm}  (instances={fb.count_instances})")
            printed += 1
            if printed >= self.max_print:
                break

        if self.tessellation_fail_repmap:
            print(f"\nRepMap tessellation failures: {len(self.tessellation_fail_repmap)} → ids: {self.tessellation_fail_repmap}")
        if self.tessellation_fail_unmapped:
            print(f"Unmapped tessellation failures (skipped): {self.tessellation_fail_unmapped}")

        total = n_repmap + n_fallback
        print(f"\nTOTAL prototypes authored: {total} (repmap={n_repmap}, fallback={n_fallback})")

# ----------------------------
# CLI
# ----------------------------

def main():
    cwd = Path.cwd()
    input_path = cwd.joinpath('data').resolve()
    filename = "SRL-WPD-TVC-UTU8-MOD-CTU-BUW-000001.ifc"
    ifc_path = input_path.joinpath(filename).resolve()

    output_path = input_path.joinpath('output').resolve()
    #usd_path = output_path.joinpath(ifc_path.stem + '.usda')
    usd_path = output_path.joinpath('prototypes.usda').resolve()
    manifest_path = output_path.joinpath('manifest.csv').resolve()

    '''ap = argparse.ArgumentParser()
    ap.add_argument("--ifc", required=True, help="Path to IFC file")
    ap.add_argument("--usd", required=True, help="Output USD/USDA path")
    ap.add_argument("--disable-openings", action="store_true", help="Prototype meshes without opening subtractions")
    ap.add_argument("--body-only", action="store_true", help="Keep only 'Body-like' representations")
    ap.add_argument("--manifest", default=None, help="Write a CSV manifest of prototypes")
    ap.add_argument("--hash-precision", type=int, default=6, help="Rounding precision for dedupe hashing")
    ap.add_argument("--max-print", type=int, default 50, help="Max prototype names to print in summary")
    args = ap.parse_args()

    builder = PrototypesFromInstances(
        ifc_path=args.ifc,
        usd_path=args.usd,
        disable_openings=args.disable_openings,
        body_only=args.body_only,
        hash_precision=args.hash_precision,
        max_print=args.max_print,
        manifest_csv=args.manifest
    )
    '''
    builder = PrototypesFromInstances(
        ifc_path=str(ifc_path),
        usd_path=str(usd_path),
        body_only=True,
        max_print=50,
        manifest_csv=str(manifest_path)
    )

    builder.discover()
    builder.build_prototypes()
    builder.write_manifest()
    builder.print_summary()

if __name__ == "__main__":
    main()