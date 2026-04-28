# Tests

How to run the suite and avoid common environment pitfalls.

## Environment
- Use Python 3.11 with an OCC-enabled IfcOpenShell (geom present). Example:
  - `conda create -n buildusd-occ python=3.11 -y`
  - `conda activate buildusd-occ`
  - `python -m pip install --extra-index-url https://ifcopenshell.github.io/ifcopenshell.github.io/whl/ ifcopenshell==0.8.3.post2`
  - (Optional) `conda install -c conda-forge pythonocc-core=7.6.3`
- Install project + test deps from repo root:
  - `python -m pip install -e .[test]`

## Running
- Always drive pytest with the env’s python so geom is visible:
  - `python -m pytest`
- Sample IFCs live under `tests/data/ifc`; USD outputs land in `tests/data/usd`.
- Integration tests skip automatically if `ifcopenshell.geom` is unavailable.
- Windows tip: if you hit Temp cleanup permission errors, run
  - `python -m pytest --basetemp .\.pytest_tmp`
- Slow tests: Golden snapshot conversions are marked `@pytest.mark.slow` and are skipped by default via `pytest.ini` (`addopts = -m "not slow"`). In CI (or when you want full coverage) run:
  - `python -m pytest -m "slow or not slow"`
  - or run only the heavy set: `python -m pytest -m slow`

## What to expect
- Offline integration tests convert the sample IFCs; failures usually mean geom was missing or the wrong interpreter was used.
- Federation and option-normalization tests rely only on the local src code and sample data.
