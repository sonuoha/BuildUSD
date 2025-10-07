"""Thin compatibility shim so `python -m ifc_converter` works without installation."""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.ifc_converter import *  # noqa: F401,F403

