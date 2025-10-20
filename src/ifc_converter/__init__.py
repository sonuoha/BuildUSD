from pathlib import Path
import sys

# Ensure the package is importable even when repository root is not on sys.path.
_SRC_PARENT = Path(__file__).resolve().parents[1]
if str(_SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(_SRC_PARENT))

from .config.manifest import ConversionManifest
from .main import ConversionCancelledError, ConversionResult, ConversionOptions, convert
from .process_ifc import CurveWidthRule

__all__ = [
    "convert",
    "ConversionOptions",
    "ConversionResult",
    "ConversionManifest",
    "CurveWidthRule",
    "ConversionCancelledError",
]
