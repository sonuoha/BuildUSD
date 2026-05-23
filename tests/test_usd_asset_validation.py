from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from usd_validation import assert_valid_usd_asset, require_usd_validator


pytestmark = [pytest.mark.slow, pytest.mark.usd_validation]


TEST_IFC_DIR = Path(__file__).parent / "data" / "ifc" / "buildingSMART"
FEDERATION_ANCHOR = {
    "easting": 320447.054,
    "northing": 5812195.56,
    "height": 0.0,
    "unit": "m",
    "epsg": "EPSG:7855",
}


def _sample_ifc(name: str) -> Path:
    path = TEST_IFC_DIR / name
    if not path.exists():
        pytest.skip(f"Sample IFC missing: {path}")
    return path


def _require_ifcopenshell_geom() -> None:
    ifc_module = sys.modules.get("ifcopenshell")
    if isinstance(ifc_module, Mock) or (
        ifc_module is not None and getattr(ifc_module, "__file__", None) is None
    ):
        for module_name in list(sys.modules):
            if module_name == "ifcopenshell" or module_name.startswith("ifcopenshell."):
                del sys.modules[module_name]

    ifcopenshell = pytest.importorskip("ifcopenshell")
    try:
        import ifcopenshell.geom as _ifc_geom  # type: ignore

        ifcopenshell.geom = _ifc_geom  # type: ignore[attr-defined]
    except Exception:
        pytest.skip("ifcopenshell.geom unavailable (need OCC-enabled ifcopenshell)")

    for module_name in (
        "buildusd.ifc_visuals",
        "buildusd.process_ifc",
        "buildusd.conversion",
        "buildusd.api",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)


def _require_windows_validator_before_conversion() -> None:
    if sys.platform.startswith("win"):
        require_usd_validator()


def _require_non_windows_validator_after_conversion() -> None:
    if not sys.platform.startswith("win"):
        require_usd_validator()


def _convert_sample_stage(tmp_path: Path) -> Path:
    from buildusd.api import ConversionSettings, convert

    sample = _sample_ifc("column-straight-rectangle-tessellation.ifc")
    out_dir = tmp_path / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = ConversionSettings(
        input_path=sample,
        output_dir=out_dir,
        offline=True,
        usd_format="usda",
        usd_auto_binary_threshold_mb=None,
        map_coordinate_system="EPSG:7855",
    )
    results = convert(settings)
    assert results, "No conversion result produced"

    stage_path = Path(results[0].stage_path)
    assert stage_path.exists(), f"Generated stage not found: {stage_path}"
    return stage_path


def _stamp_federation_anchor(stage_path: Path):
    from buildusd.config.manifest import BasePointConfig
    from buildusd.pxr_utils import Usd
    from buildusd.usd_context import initialize_usd

    initialize_usd(offline=True)
    stage = Usd.Stage.Open(str(stage_path))
    assert stage is not None, f"Could not open generated stage: {stage_path}"

    root_layer = stage.GetRootLayer()
    layer_data = dict(getattr(root_layer, "customLayerData", {}) or {})
    layer_data["projectedCRS"] = FEDERATION_ANCHOR["epsg"]
    layer_data["stageOriginProjected"] = dict(FEDERATION_ANCHOR)
    root_layer.customLayerData = layer_data
    root_layer.Save()

    return BasePointConfig(**FEDERATION_ANCHOR)


def test_generated_offline_stage_passes_nvidia_asset_validation(tmp_path: Path):
    _require_ifcopenshell_geom()
    _require_windows_validator_before_conversion()

    stage_path = _convert_sample_stage(tmp_path)

    _require_non_windows_validator_after_conversion()

    assert_valid_usd_asset(stage_path)


def test_federated_stage_passes_nvidia_asset_validation(tmp_path: Path):
    _require_ifcopenshell_geom()
    _require_windows_validator_before_conversion()

    stage_path = _convert_sample_stage(tmp_path)
    federation_origin = _stamp_federation_anchor(stage_path)

    from buildusd.api import federate_into_stage

    federated_stage = tmp_path / "masters" / "federated.usda"
    result = federate_into_stage(
        [stage_path],
        out_stage_path=federated_stage,
        fallback_shared_site_base_point=federation_origin,
        offline=True,
        rebuild=True,
    )

    assert result is not None
    assert federated_stage.exists(), f"Federated stage not found: {federated_stage}"
    assert "payload" in federated_stage.read_text(encoding="utf-8")

    _require_non_windows_validator_after_conversion()

    assert_valid_usd_asset(federated_stage)
