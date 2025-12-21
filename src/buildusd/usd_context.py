from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Optional

from .kit_runtime import ensure_kit, shutdown_kit

_MODE: Optional[str] = None  # "kit" or "offline"
_PXR_CACHE: Dict[str, object] = {}
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
    if desired == "kit":
        # Start Kit with the extensions we rely on for Nucleus/Usd
        ensure_kit(("omni.client", "omni.usd"))

    _MODE = desired
    _PXR_CACHE = {}
    return _MODE


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
