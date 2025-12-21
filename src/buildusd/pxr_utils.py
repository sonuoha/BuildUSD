from __future__ import annotations

from typing import Any

from .usd_context import (
    get_pxr_module,
    get_pxr_package,
    initialize_usd,
    shutdown_usd_context,
)


def require_pxr_module(name: str) -> Any:
    initialize_usd()
    return get_pxr_module(name)


def require_pxr_attribute(name: str) -> Any:
    initialize_usd()
    package = get_pxr_package()
    return getattr(package, name)


class _ModuleProxy:
    def __init__(self, module_name: str):
        self._module_name = module_name

    def _module(self):
        return require_pxr_module(self._module_name)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._module(), item)

    def __dir__(self):
        return dir(self._module())


Gf = _ModuleProxy("Gf")
Sdf = _ModuleProxy("Sdf")
Usd = _ModuleProxy("Usd")
UsdGeom = _ModuleProxy("UsdGeom")
UsdShade = _ModuleProxy("UsdShade")
Vt = _ModuleProxy("Vt")

__all__ = [
    "Gf",
    "Sdf",
    "Usd",
    "UsdGeom",
    "UsdShade",
    "Vt",
    "require_pxr_module",
    "require_pxr_attribute",
    "shutdown_usd_context",
    "initialize_usd",
]
