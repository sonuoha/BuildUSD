from pathlib import Path

import pytest

try:
    import ifcopenshell  # type: ignore
except ImportError:
    pytest.skip(
        "ifcopenshell not available in this environment", allow_module_level=True
    )

# Explicitly import geom; skip gracefully if OCC support is missing.
try:
    import ifcopenshell.geom as _ifc_geom  # type: ignore

    ifcopenshell.geom = _ifc_geom  # type: ignore[attr-defined]
except Exception:
    pytest.skip(
        "ifcopenshell.geom unavailable (need OCC-enabled ifcopenshell)",
        allow_module_level=True,
    )

from buildusd.api import ConversionSettings, convert
from buildusd.process_ifc import ConversionOptions
from buildusd.occ_detail import is_available as occ_available

pytestmark = pytest.mark.slow


TEST_IFC_DIR = Path(__file__).parent / "data" / "ifc"
TEST_USD_DIR = Path(__file__).parent / "data" / "usd"
# Golden text USD snapshots live alongside the IFC samples.
GOLDEN_USD = {
    "Ifc2x3_Duplex_Architecture.ifc": TEST_USD_DIR / "Ifc2x3_Duplex_Architecture.usda",
    "Ifc4_Revit_ARC.ifc": TEST_USD_DIR / "Ifc4_Revit_ARC.usda",
    "2022020320211122Wellness center Sama.ifc": TEST_USD_DIR
    / "2022020320211122Wellness center Sama.usda",
}


def _sample_ifc(name: str) -> Path:
    path = TEST_IFC_DIR / name
    if not path.exists():
        pytest.skip(f"Sample IFC missing: {path}")
    return path


def test_convert_offline_base(tmp_path):
    sample = _sample_ifc("Ifc4_CubeAdvancedBrep.ifc")
    out_dir = TEST_USD_DIR / "base"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = ConversionSettings(
        input_path=sample,
        output_dir=out_dir,
        offline=True,
        map_coordinate_system="EPSG:7855",
    )
    results = convert(settings)
    assert results, "No conversion result produced"
    stage_path = Path(results[0].stage_path)
    assert stage_path.exists()


def test_convert_detail_semantic(tmp_path):
    sample = _sample_ifc("Ifc4_CubeAdvancedBrep.ifc")
    out_dir = TEST_USD_DIR / "semantic"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = ConversionSettings(
        input_path=sample,
        output_dir=out_dir,
        offline=True,
        map_coordinate_system="EPSG:7855",
    )
    options = ConversionOptions(
        detail_mode=True,
        detail_scope="all",
        detail_engine="semantic",
    )
    results = convert(settings, options=options)
    assert results, "No conversion result produced"
    stage_path = Path(results[0].stage_path)
    assert stage_path.exists()


def _normalize_usda_text(text: str) -> str:
    """Drop volatile lines and whitespace for stable comparisons."""
    keep: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        keep.append(stripped)
    return "\n".join(keep)


def _assert_matches_golden(generated: Path, golden: Path):
    assert golden.exists(), f"Golden USD missing: {golden}"
    gen_text = generated.read_text(encoding="utf-8", errors="ignore")
    gold_text = golden.read_text(encoding="utf-8", errors="ignore")
    assert _normalize_usda_text(gen_text) == _normalize_usda_text(
        gold_text
    ), f"Generated USD differs from golden: {generated} vs {golden}"


@pytest.mark.slow
@pytest.mark.parametrize("ifc_name", list(GOLDEN_USD.keys()))
def test_golden_usda_matches_snapshot(tmp_path: Path, ifc_name: str):
    """Run conversion to USD(A) and compare against committed golden snapshots."""
    sample = _sample_ifc(ifc_name)
    golden = GOLDEN_USD[ifc_name]
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = ConversionSettings(
        input_path=sample,
        output_dir=out_dir,
        offline=True,
        usd_format="usda",
        map_coordinate_system="EPSG:7855",
    )
    results = convert(settings)
    assert results, "No conversion result produced"
    stage_path = Path(results[0].stage_path)
    assert stage_path.exists(), f"Generated stage not found: {stage_path}"
    _assert_matches_golden(stage_path, golden)


@pytest.mark.skipif(not occ_available(), reason="OCC detail not available")
def test_convert_detail_occ(tmp_path):
    sample = _sample_ifc("Ifc4_CubeAdvancedBrep.ifc")
    out_dir = TEST_USD_DIR / "occ"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = ConversionSettings(
        input_path=sample,
        output_dir=out_dir,
        offline=True,
        map_coordinate_system="EPSG:7855",
    )
    options = ConversionOptions(
        detail_mode=True,
        detail_scope="all",
        detail_engine="occ",
    )
    results = convert(settings, options=options)
    assert results, "No conversion result produced"
    stage_path = Path(results[0].stage_path)
    assert stage_path.exists()
