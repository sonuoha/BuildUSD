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
    MissingAnchorPolicy,
    FederationSettings,
    apply_stage_anchor_transform,
    convert,
    federate_into_stage,
    federate_stages,
    EnrichmentTarget,
    TargetedEnrichmentPlan,
    BuildUSDJobResult,
    build_targeted_enrichment_plan,
    load_semantic_graph,
    read_job,
    run_job,
    TargetedDetailJob,
    TargetedDetailTarget,
    write_job,
    CONVERSION_DEFAULTS,
    FEDERATION_DEFAULTS,
    DEFAULT_CONVERSION_OPTIONS,
)
from .config.manifest import ConversionManifest
from .federation_orchestrator import FederationTask
from .federation_builder import AnchorInfo, build_federated_stage, validate_federation
from .main import ConversionCancelledError, ConversionOptions, ConversionResult
from .process_ifc import CurveWidthRule
from .forma import (
    FormaAuthenticationError,
    FormaClient,
    FormaIntegrationError,
    FormaProjectFile,
    FormaProjectLocator,
    FormaSource,
    FormaSourceError,
    FormaTranslationError,
    parse_acc_project_url,
)

__all__ = [
    "api",
    "convert",
    "federate_stages",
    "federate_into_stage",
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
    "MissingAnchorPolicy",
    "ConversionOptions",
    "ConversionResult",
    "ConversionManifest",
    "CurveWidthRule",
    "ConversionCancelledError",
    "FederationTask",
    "AnchorInfo",
    "build_federated_stage",
    "validate_federation",
    "EnrichmentTarget",
    "TargetedEnrichmentPlan",
    "BuildUSDJobResult",
    "build_targeted_enrichment_plan",
    "load_semantic_graph",
    "read_job",
    "run_job",
    "TargetedDetailJob",
    "TargetedDetailTarget",
    "write_job",
    "FormaAuthenticationError",
    "FormaClient",
    "FormaIntegrationError",
    "FormaProjectFile",
    "FormaProjectLocator",
    "FormaSource",
    "FormaSourceError",
    "FormaTranslationError",
    "parse_acc_project_url",
]
