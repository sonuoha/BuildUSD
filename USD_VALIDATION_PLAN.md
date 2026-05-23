# USD Validation Integration Plan

Trackable plan for integrating NVIDIA OpenUSD Exchange validation into the BuildUSD pipeline.

## Recommendation

Integrate the NVIDIA Asset Validator and `usdex.test` as an optional validation layer for generated USD assets. Do not make the full USD Exchange SDK a required runtime dependency for BuildUSD or for the default fast test matrix.

## Review Decisions

- [ ] Decide whether validation should start as test-only, or also ship a user-facing CLI command immediately.
- [ ] Decide whether CI should initially validate one small sample, or all committed golden samples.
- [ ] Decide whether validation failures should block PRs immediately, or run non-blocking while the rule noise profile is established.

Recommended starting position:

- [x] Test-only first.
- [x] Validate one small generated sample plus one federation sample.
- [x] Wire the validation job into the blocking aggregate CI check.

## Phase 1: Optional Dependency

- [x] Add a dedicated optional dependency group in `pyproject.toml`, for example `usd-validation`.
- [x] Include `usd-exchange[test]` in that optional group.
- [x] Keep the existing `test` and `dev` extras unchanged until compatibility is proven.
- [x] Confirm install behavior with a uv-managed validation environment.

## Phase 2: Validation Helper

- [x] Add a small helper module for tests, likely `tests/usd_validation.py`.
- [x] Detect whether `usdex.test` is importable.
- [x] Detect whether `omni.asset_validator` is importable.
- [x] Skip validation tests cleanly when optional validation dependencies are unavailable.
- [x] Prefer `usdex.test.TestCase.assertIsValidUsd()` when practical.
- [x] Fall back to `omni.asset_validator.ValidationEngine().validate()` if that produces cleaner pytest integration.
- [x] Centralize any intentional validator issue suppressions.
- [x] Make validation failure output readable in CI logs.

## Phase 3: Targeted Pytest Coverage

- [x] Add `tests/test_usd_asset_validation.py`.
- [x] Generate a small offline USD sample from `column-straight-rectangle-tessellation.ifc`.
- [x] Replace the missing local-only IFC sample with a small CC BY 4.0 buildingSMART fixture.
- [x] Validate the generated composed root stage.
- [x] Treat composed root stages as the primary validator target.
- [x] Do not validate isolated generated sublayers by default; they can contain expected cross-layer references that are valid only when composed through the root stage.
- [x] Add a federation validation test once the single-stage validation is stable.
- [x] Mark these tests with `pytest.mark.usd_validation`.
- [x] Keep these tests marked `pytest.mark.slow`.
- [x] Add the `usd_validation` marker to `pytest.ini`.
- [x] Add the `usd_validation` marker to `pyproject.toml`.

## Phase 4: CI Integration

- [x] Add a separate GitHub Actions job named `USD validation / ubuntu / Python 3.11`.
- [x] Install with `python -m pip install -e ".[usd-validation]"`.
- [x] Run only validator tests with `python -m pytest -m "usd_validation" --basetemp .pytest_tmp_usd_validation`.
- [x] Keep the existing OS/Python matrix unchanged.
- [x] Do not add Windows validation until dependency support is verified.
- [x] Do not add Python 3.12 validation until dependency support is verified.

## Phase 5: Documentation

- [x] Update `tests/README.md` with local install instructions for validation dependencies.
- [x] Document how to run validator tests locally.
- [x] Document what common validator failures usually mean.
- [x] Document how intentional validator suppressions should be added and justified.
- [x] Add a short USD validation section to `README.md`.

## Phase 6: Optional CLI Gate

- [ ] Decide whether BuildUSD should expose validation as a separate command, for example `python -m buildusd.validate path/to/stage.usda`.
- [ ] Decide whether conversion should support an inline validation flag, for example `--validate-usd`.
- [ ] Keep any CLI validation dependency optional and fail with a clear install message when missing.
- [ ] Reuse the same validation helper logic as the tests where practical.

## Phase 7: Expansion Criteria

- [ ] Expand from one sample to all golden snapshot outputs after the validator job is stable.
- [x] Add federated master validation after single-stage validation is stable.
- [ ] Revisit isolated sublayer validation only if we add composition-aware validation or rule-specific suppressions with justification.
- [ ] Add Windows CI validation only after local and CI package compatibility is confirmed.
- [ ] Add Python 3.12 CI validation only after package compatibility is confirmed.
- [ ] Require comments and tests for any validator rule suppressions.

## Useful References

- NVIDIA OpenUSD Exchange SDK: https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/index.html
- Asset Validator: https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/docs/devtools.html#asset-validator
- Testing and Debugging: https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/docs/testing-debugging.html
- `usdex.test`: https://docs.omniverse.nvidia.com/usd/code-docs/usd-exchange-sdk/latest/docs/python-usdex-test.html
