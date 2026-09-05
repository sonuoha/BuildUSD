from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from .kit_runtime import ensure_kit, shutdown_kit

_MODE: Optional[str] = None  # "kit" or "offline"
_PXR_CACHE: Dict[str, object] = {}
_DLL_DIR_HANDLES: dict[str, object] = {}
_DEFAULT_ENV = (
    os.environ.get("BUILDUSD_DEFAULT_USD_MODE")
    or os.environ.get("IFC_CONVERTER_DEFAULT_USD_MODE")
    or "kit"
)
_DEFAULT_MODE = _DEFAULT_ENV.strip().lower()
if _DEFAULT_MODE not in {"kit", "offline"}:
    _DEFAULT_MODE = "kit"


def get_mode() -> Optional[str]:
    """Return the current USD mode ('kit' or 'offline')."""
    global _MODE
    return _MODE


# buildusd/usd_context.py  (replace initialize_usd with this)
def initialize_usd(*, offline: Optional[bool] = None) -> str:
    """Prepare USD bindings.

    - If `offline` is None, keep whatever mode we're already in (or default to _DEFAULT_MODE on first call).
    - If mode is unchanged, do nothing (idempotent).
    """
    global _MODE, _PXR_CACHE

    # Decide desired mode: preserve current unless explicitly overridden.
    if offline is None:
        desired = _MODE or _DEFAULT_MODE
    else:
        desired = "offline" if offline else "kit"

    # If already in the desired mode, do nothing (idempotent).
    if _MODE == desired:
        return _MODE

    # Switching modes: teardown previous and (re)initialize as needed
    _teardown()

    _clear_pxr_modules()
    if desired == "offline":
        _prepare_offline_usd_dll_path()
    if desired == "kit":
        # Start Kit with the extensions we rely on for Nucleus/Usd
        ensure_kit(("omni.client", "omni.usd"))

    _MODE = desired
    _PXR_CACHE = {}
    return _MODE


def _prepare_offline_usd_dll_path() -> None:
    """Make usd-exchange native DLLs discoverable before pxr imports.

    The usd-exchange wheel sets PXR_USD_WINDOWS_DLL_PATH from pxr.__init__, but
    USD's plugin loader also needs the DLL directory on PATH when loading file
    format plugins such as usd_usd.dll.
    """
    if os.name != "nt":
        return
    candidates: list[Path] = []
    configured = os.environ.get("PXR_USD_WINDOWS_DLL_PATH")
    if configured:
        candidates.append(Path(configured))
    try:
        spec = importlib.util.find_spec("pxr")
    except Exception:
        spec = None
    if spec and spec.submodule_search_locations:
        pxr_dir = Path(next(iter(spec.submodule_search_locations)))
        candidates.append((pxr_dir / ".." / "usd_exchange.libs").resolve())

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    normalized_path_parts = {
        str(Path(part).resolve()).lower() for part in path_parts if part.strip()
    }
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if not resolved.exists():
            continue
        resolved_text = str(resolved)
        if resolved_text.lower() not in normalized_path_parts:
            os.environ["PATH"] = resolved_text + os.pathsep + os.environ.get("PATH", "")
            normalized_path_parts.add(resolved_text.lower())
        dll_key = resolved_text.lower()
        if hasattr(os, "add_dll_directory") and dll_key not in _DLL_DIR_HANDLES:
            try:
                _DLL_DIR_HANDLES[dll_key] = os.add_dll_directory(resolved_text)
            except Exception:
                pass


def _teardown() -> None:
    global _MODE, _PXR_CACHE
    if _MODE == "kit":
        shutdown_kit()
    _clear_pxr_modules()
    _PXR_CACHE = {}
    _MODE = None


def _clear_pxr_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "pxr"
            or name.startswith("pxr.")
            or name in {"Usd", "UsdGeom", "UsdShade", "UsdUtils", "Sdf", "Gf", "Vt"}
        ):
            sys.modules.pop(name, None)


def get_pxr_module(name: str):
    if name not in _PXR_CACHE:
        if _MODE is None:
            initialize_usd()
        _PXR_CACHE[name] = importlib.import_module(f"pxr.{name}")
    return _PXR_CACHE[name]


def get_pxr_package():
    if "__pxr__" not in _PXR_CACHE:
        if _MODE is None:
            initialize_usd()
        _PXR_CACHE["__pxr__"] = importlib.import_module("pxr")
    return _PXR_CACHE["__pxr__"]


def shutdown_usd_context() -> None:
    _teardown()
