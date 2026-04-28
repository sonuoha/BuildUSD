from pathlib import Path

import pytest

pytest.importorskip("ifcopenshell")

from buildusd.api import FederationSettings, federate_stages
from buildusd.config.manifest import ConversionManifest


TEST_USD_DIR = Path(__file__).parent / "data" / "usd"


def _existing_stage_paths() -> list[Path]:
    if not TEST_USD_DIR.exists():
        return []
    return [p for p in TEST_USD_DIR.rglob("*.usd*") if p.is_file()]


def test_federation_runs_with_sample_manifest(tmp_path):
    pytest.importorskip("pxr")  # federate depends on USD bindings being present
    stages = _existing_stage_paths()
    if not stages:
        pytest.skip(
            "No USD stages found under tests/data/usd; run integration conversion tests first."
        )

    # Minimal manifest stub: use defaults from the sample manifest in config
    sample_manifest_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "buildusd"
        / "config"
        / "sample_manifest.json"
    )
    if not sample_manifest_path.exists():
        pytest.skip("Sample manifest not found.")

    manifest = ConversionManifest.from_file(sample_manifest_path)
    settings = FederationSettings(
        stage_paths=stages,
        manifest=manifest,
        masters_root=tmp_path / "masters",
        offline=True,
    )
    tasks = federate_stages(settings)
    # We only assert that federation returns tasks (may be empty if rules don't match names)
    assert tasks is not None
