from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def assert_valid_usd_asset(asset: str | Path) -> None:
    """Validate a USD asset with NVIDIA's optional USD Exchange test tooling."""
    asset_path = str(asset)
    validator = _load_validator()
    if validator.kind == "usdex.test":
        _assert_valid_with_usdex_test(validator.module, asset_path)
        return
    _assert_valid_with_asset_validator(validator.module, asset_path)


def require_usd_validator() -> None:
    """Skip unless NVIDIA's optional USD validation tooling can be imported."""
    _load_validator()


class _Validator:
    def __init__(self, kind: str, module: Any) -> None:
        self.kind = kind
        self.module = module


def _load_validator() -> _Validator:
    import_errors: list[str] = []

    try:
        import omni.asset_validator as asset_validator  # type: ignore[import-not-found]
    except Exception as exc:
        import_errors.append(f"omni.asset_validator: {exc}")
    else:
        return _Validator("omni.asset_validator", asset_validator)

    try:
        import usdex.test as usdex_test  # type: ignore[import-not-found]
    except Exception as exc:
        import_errors.append(f"usdex.test: {exc}")
    else:
        return _Validator("usdex.test", usdex_test)

    pytest.skip(
        "NVIDIA USD validation dependencies are unavailable. "
        "Install with `python -m pip install -e \".[usd-validation]\"`. "
        + "; ".join(import_errors)
    )


def _assert_valid_with_usdex_test(usdex_test: Any, asset_path: str) -> None:
    case = usdex_test.TestCase(methodName="runTest")
    set_up = getattr(case, "setUp", None)
    tear_down = getattr(case, "tearDown", None)
    if set_up:
        set_up()
    try:
        case.assertIsValidUsd(asset_path)
    finally:
        if tear_down:
            tear_down()


def _assert_valid_with_asset_validator(asset_validator: Any, asset_path: str) -> None:
    engine = asset_validator.ValidationEngine()
    results = engine.validate(asset_path)
    issues = list(results.issues())
    actionable = [issue for issue in issues if not _is_success_issue(issue)]
    assert not actionable, _format_issues(asset_path, actionable)


def _is_success_issue(issue: Any) -> bool:
    severity = getattr(issue, "severity", None)
    if severity is None:
        return False
    severity_name = str(severity).lower()
    return "none" in severity_name or "success" in severity_name


def _format_issues(asset_path: str, issues: list[Any]) -> str:
    lines = [f"NVIDIA USD validation reported {len(issues)} issue(s) for {asset_path}:"]
    for index, issue in enumerate(issues, start=1):
        severity = getattr(issue, "severity", "unknown")
        message = getattr(issue, "message", str(issue))
        rule = getattr(issue, "rule", None) or getattr(issue, "rule_id", None)
        location = getattr(issue, "at", None) or getattr(issue, "locations", None)
        details = f"{index}. [{severity}] {message}"
        if rule:
            details += f" (rule: {rule})"
        if location:
            details += f" at {location}"
        lines.append(details)
    return "\n".join(lines)
