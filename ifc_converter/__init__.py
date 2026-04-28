"""Compatibility shim so legacy `ifc_converter` imports still resolve."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_replacement = import_module("src.buildusd")

_skip = {"__name__", "__loader__", "__package__", "__spec__", "__path__", "__file__"}
globals().update({k: v for k, v in vars(_replacement).items() if k not in _skip})

__doc__ = getattr(_replacement, "__doc__", __doc__)
__all__ = getattr(_replacement, "__all__", [])
__path__ = getattr(_replacement, "__path__", [])
__spec__ = getattr(_replacement, "__spec__", None)
__file__ = getattr(_replacement, "__file__", __file__)
__package__ = __name__
