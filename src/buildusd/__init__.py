from pathlib import Path
import sys

# Ensure the package is importable even when repository root is not on sys.path.
_SRC_PARENT = Path(__file__).resolve().parents[1]
if str(_SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(_SRC_PARENT))

from . import api
from .api import (
    AnchorMode,
    AnchorModeSetting,
    ConversionDefaults,
    ConversionSettings,
    FederationDefaults,
    FederationSettings,
    apply_stage_anchor_transform,
    convert,
    federate_stages,
    CONVERSION_DEFAULTS,
    FEDERATION_DEFAULTS,
    DEFAULT_CONVERSION_OPTIONS,
)
from .config.manifest import ConversionManifest
from .federation_orchestrator import FederationTask
from .federation_builder import AnchorInfo, build_federated_stage, validate_federation
from .main import ConversionCancelledError, ConversionOptions, ConversionResult
from .process_ifc import CurveWidthRule

__all__ = [
    "api",
    "convert",
    "federate_stages",
    "apply_stage_anchor_transform",
    "ConversionDefaults",
    "ConversionSettings",
    "FederationDefaults",
    "FederationSettings",
    "CONVERSION_DEFAULTS",
    "FEDERATION_DEFAULTS",
    "DEFAULT_CONVERSION_OPTIONS",
    "AnchorMode",
    "AnchorModeSetting",
    "ConversionOptions",
    "ConversionResult",
    "ConversionManifest",
    "CurveWidthRule",
    "ConversionCancelledError",
    "FederationTask",
    "AnchorInfo",
    "build_federated_stage",
    "validate_federation",
]
