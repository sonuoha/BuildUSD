from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from .io_utils import is_omniverse_path
from .kit_runtime import ensure_kit, shutdown_kit

PathLike = Union[str, Path]

_MODE: Optional[str] = None  # "kit" or "offline"
_DEFAULT_MODE_ENV = os.environ.get("IFC_CONVERTER_DEFAULT_USD_MODE", "kit").strip().lower()
_DEFAULT_MODE = _DEFAULT_MODE_ENV if _DEFAULT_MODE_ENV in {"kit", "offline"} else "kit"
_PREFERRED_MODE: Optional[str] = _DEFAULT_MODE
_PXR_CACHE: Dict[str, object] = {}


def set_preferred_mode(mode: Optional[str]) -> None:
    """Force the USD backend mode ('kit' or 'offline')."""
    global _PREFERRED_MODE
    if mode is None:
        _PREFERRED_MODE = _DEFAULT_MODE
        return
    if mode not in {"kit", "offline"}:
        raise ValueError(f"Unsupported USD mode '{mode}'")
    _PREFERRED_MODE = mode


def _normalize_paths(paths: Iterable[PathLike]) -> Sequence[str]:
    normalized: list[str] = []
    for p in paths:
        if p is None:
            continue
        if isinstance(p, Path):
            normalized.append(p.as_posix())
        else:
            normalized.append(str(p))
    return normalized


def _determine_mode(required_paths: Iterable[PathLike]) -> str:
    if _PREFERRED_MODE:
        return _PREFERRED_MODE
    normalized = _normalize_paths(required_paths)
    if any(is_omniverse_path(p) for p in normalized):
        return "kit"
    return "offline"


def initialize_usd(required_paths: Iterable[PathLike] = (), preferred_mode: Optional[str] = None) -> str:
    """Initialize pxr bindings using Kit or standalone USD based on the requested mode."""
    global _MODE, _PXR_CACHE

    if preferred_mode is not None:
        set_preferred_mode(preferred_mode)

    mode = _determine_mode(required_paths)
    if _MODE == mode and _PXR_CACHE:
        return _MODE

    _teardown_current_mode()

    if mode == "kit":
        _unload_pxr_modules()
        ensure_kit(("omni.client", "omni.usd"))

    _PXR_CACHE = {}
    _MODE = mode
    return _MODE


def _teardown_current_mode() -> None:
    global _MODE, _PXR_CACHE
    if _MODE == "kit":
        shutdown_kit()
    _unload_pxr_modules()
    _PXR_CACHE = {}
    _MODE = None


def get_pxr_module(name: str):
    """Return the pxr.<name> module using the currently active mode."""
    if name not in _PXR_CACHE:
        initialize_usd()
        _PXR_CACHE[name] = importlib.import_module(f"pxr.{name}")
    return _PXR_CACHE[name]


def get_pxr_package():
    """Return the pxr package root for the active mode."""
    if "__pxr__" not in _PXR_CACHE:
        initialize_usd()
        _PXR_CACHE["__pxr__"] = importlib.import_module("pxr")
    return _PXR_CACHE["__pxr__"]


def shutdown_usd_context() -> None:
    """Shut down the active USD backend (and Kit, if running)."""
    _teardown_current_mode()
def _unload_pxr_modules() -> None:
    """Remove pxr-related modules from sys.modules to avoid ABI clashes when switching backends."""
    for name in list(sys.modules):
        if name == "pxr" or name.startswith("pxr.") or name in {"Usd", "UsdGeom", "UsdShade", "UsdUtils", "Sdf", "Gf", "Vt"}:
            sys.modules.pop(name, None)
