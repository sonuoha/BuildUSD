from __future__ import annotations

from importlib import import_module
from typing import Dict

from .kit_runtime import ensure_kit

_MODULE_CACHE: Dict[str, object] = {}


def require_pxr_module(name: str):
    """Ensure Kit (with omni.usd) is running and return pxr.<name>."""

    ensure_kit(("omni.client", "omni.usd"))
    if name not in _MODULE_CACHE:
        module = import_module(f"pxr.{name}")
        _MODULE_CACHE[name] = module
    return _MODULE_CACHE[name]


def require_pxr_attribute(name: str):
    """Return attribute (module or symbol) exposed from the pxr package root."""

    ensure_kit(("omni.client", "omni.usd"))
    if "__package__" not in _MODULE_CACHE:
        package = import_module("pxr")
        _MODULE_CACHE["__package__"] = package
    else:
        package = _MODULE_CACHE["__package__"]
    return getattr(package, name)
