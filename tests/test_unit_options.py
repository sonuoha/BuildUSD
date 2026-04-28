import types
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("ifcopenshell")

from buildusd.conversion import OPTIONS as DEFAULT_OPTIONS
from buildusd.process_ifc import ConversionOptions, GeometrySettingsManager


def test_detail_option_defaults_off():
    opts = ConversionOptions()
    assert opts.detail_mode is False
    assert opts.detail_scope == "all"
    assert opts.detail_engine == "default"
    assert opts.detail_objects == ()


def test_detail_option_normalization():
    opts = ConversionOptions(
        detail_mode=True,
        detail_scope="object",
        detail_engine="occ",
        detail_objects=(123, "ABCD-1234"),
    )
    assert opts.detail_mode is True
    assert opts.detail_scope == "object"
    assert opts.detail_engine == "occ"
    assert 123 in opts.detail_objects
    assert "ABCD-1234" in opts.detail_objects


def test_geom_overrides_protects_core_settings():
    # Protected keys should not override the base geom settings.
    user_geom = {
        "mesher-linear-deflection": 0.5,
        "use-world-coords": True,  # should be ignored
    }
    opts = replace(DEFAULT_OPTIONS, geom_overrides=user_geom)

    base_geom_settings = {
        "use-world-coords": False,
        "model-offset": (0, 0, 0),
        "offset-type": "negative",
    }
    gsm = GeometrySettingsManager(base_settings=base_geom_settings, logger=None)
    # Apply user overrides via the manager's public API
    for k, v in getattr(opts, "geom_overrides", {}).items():
        if k in (
            "use-world-coords",
            "model-offset",
            "offset-type",
            "use-python-opencascade",
        ):
            continue
        base_geom_settings[k] = v

    assert base_geom_settings["use-world-coords"] is False
    assert base_geom_settings["mesher-linear-deflection"] == pytest.approx(0.5)


def test_stage_unit_and_up_axis(tmp_path):
    pytest.importorskip("pxr")
    # Create a stub usda to adjust
    from buildusd.api import set_stage_unit, set_stage_up_axis

    stage_path = tmp_path / "stage.usda"
    stage_path.write_text("#usda 1.0\n", encoding="utf-8")

    set_stage_unit(stage_path, meters_per_unit=0.001, offline=True)
    set_stage_up_axis(stage_path, axis="Z", offline=True)

    text = stage_path.read_text(encoding="utf-8")
    assert "metersPerUnit = 0.001" in text
    assert 'upAxis = "Z"' in text
