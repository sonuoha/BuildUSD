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
    parse_args,
    set_stage_unit,
    set_stage_up_axis,
)

__all__ = [
    "ConversionCancelledError",
    "ConversionResult",
    "convert",
    "parse_args",
    "ConversionOptions",
    "CurveWidthRule",
    "ConversionManifest",
    "set_stage_unit",
    "set_stage_up_axis",
]
