from __future__ import annotations

#Import standard libraries
import logging
import numpy as np

from ..pxr_utils import Gf


# Set up logging
log = logging.getLogger(__name__)

# Optional: EPSG transforms (WGS84 <-> GDA2020 / MGA Zone 55 : EPSG:7855)
try:
    from pyproj import CRS, Transformer
    _HAVE_PYPROJ = True
except Exception:
    _HAVE_PYPROJ = False

# Optional: ifcopenshell util for robust matrices (mapped/type geometry)
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
    """
    Check if a 4x4 matrix (16-element tuple or list) is an identity matrix.
    
    Parameters
    ----------
    mat16 : tuple or list of 16 floats
        The matrix to check.
    atol : float, optional
        The absolute tolerance to use when checking for equality.
        Defaults to 1e-10.
    
    Returns
    -------
    bool
        True if the matrix is an identity matrix, False otherwise.
    """
    try:
        arr = np.array(mat16, dtype=float).reshape(4, 4)
        return np.allclose(arr, np.eye(4), atol=atol)
    except Exception:
        return False

