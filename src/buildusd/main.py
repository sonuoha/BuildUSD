"""Compatibility shim for buildusd.main.

Legacy callers import from buildusd.main; the implementation now lives in
buildusd.conversion so this module simply re-exports the public API.
"""

from __future__ import annotations

from .conversion import (
    ConversionCancelledError,
    ConversionManifest,
    ConversionOptions,
    ConversionResult,
    CurveWidthRule,
    convert,
    main as _conversion_main,
    parse_args,
    set_stage_unit,
    set_stage_up_axis,
)

__all__ = [
    "ConversionCancelledError",
    "ConversionResult",
    "convert",
    "main",
    "parse_args",
    "ConversionOptions",
    "CurveWidthRule",
    "ConversionManifest",
    "set_stage_unit",
    "set_stage_up_axis",
]


def main(argv=None):
    return _conversion_main(argv)
