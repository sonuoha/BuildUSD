from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Optional

from .kit_runtime import ensure_kit, shutdown_kit

_MODE: Optional[str] = None  # "kit" or "offline"
_PXR_CACHE: Dict[str, object] = {}
_DEFAULT_MODE = os.environ.get("IFC_CONVERTER_DEFAULT_USD_MODE", "kit").strip().lower()
if _DEFAULT_MODE not in {"kit", "offline"}:
    _DEFAULT_MODE = "kit"

def get_mode() -> Optional[str]:
    """Return the current USD mode ('kit' or 'offline')."""
    global _MODE
    return _MODE

def initialize_usd(*, offline: Optional[bool] = None) -> str:
    """Prepare USD bindings. Default = Kit; use offline=True to stay with standalone pxr."""
    global _MODE, _PXR_CACHE

    # If caller didn't specify, respect CURRENT mode (or env default), don't flip.
    desired = (
        "offline" if offline is True
        else "kit" if offline is False
        else (_MODE or _DEFAULT_MODE)   # <— important: no implicit switch
    )

    # Idempotent: if already in desired mode, do nothing.
    if _MODE == desired:
        return _MODE

    # Only tear down when actually switching modes.
    _teardown()

    if desired == "kit":
        _clear_pxr_modules()
        ensure_kit(("omni.client", "omni.usd"))
    else:
        _clear_pxr_modules()

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
        if name == "pxr" or name.startswith("pxr.") or name in {"Usd", "UsdGeom", "UsdShade", "UsdUtils", "Sdf", "Gf", "Vt"}:
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
