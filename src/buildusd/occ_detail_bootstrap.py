from __future__ import annotations

import logging
import sys
import threading

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_BOOTSTRAPPED = False
_SYMBOLS: dict[str, object] = {}
_LAST_FAILURE: str | None = None

_REQUIRED = [
    ("OCC.Core.BRepMesh", "BRepMesh_IncrementalMesh"),
    ("OCC.Core.IMeshTools", "IMeshTools_Parameters"),
    ("OCC.Core.BRepCheck", "BRepCheck_Analyzer"),
    ("OCC.Core.ShapeFix", "ShapeFix_Shape"),
    ("OCC.Core.BRepBuilderAPI", "BRepBuilderAPI_Sewing"),
    ("OCC.Core.TopExp", "TopExp_Explorer"),
    ("OCC.Core.TopAbs", "TopAbs_FACE"),
    ("OCC.Core.TopAbs", "TopAbs_SHELL"),
    ("OCC.Core.TopAbs", "TopAbs_SOLID"),
    ("OCC.Core.TopAbs", "TopAbs_COMPOUND"),
    ("OCC.Core.TopAbs", "TopAbs_COMPSOLID"),
    ("OCC.Core.TopoDS", "TopoDS_Shape"),
    ("OCC.Core.TopLoc", "TopLoc_Location"),
    ("OCC.Core.BRep", "BRep_Tool"),
    ("OCC.Core.Bnd", "Bnd_Box"),
    ("OCC.Core.BRepBndLib", "brepbndlib_Add"),
]

_OPTIONAL: list[tuple[str, str]] = []


def _import_symbol(module_name: str, symbol: str):
    module = __import__(module_name, fromlist=[symbol])
    return getattr(module, symbol)


def bootstrap_occ(strict: bool = True) -> None:
    global _BOOTSTRAPPED, _SYMBOLS, _LAST_FAILURE
    if _BOOTSTRAPPED:
        return
    with _LOCK:
        if _BOOTSTRAPPED:
            return
        symbols: dict[str, object] = {}
        missing: list[str] = []
        for module_name, symbol_name in _REQUIRED:
            try:
                symbols[symbol_name] = _import_symbol(module_name, symbol_name)
            except Exception as exc:
                missing.append(f"{module_name}.{symbol_name}: {exc}")
        try:
            occ_utils = __import__("ifcopenshell.geom", fromlist=["occ_utils"]).occ_utils
            symbols["occ_utils"] = occ_utils
        except Exception as exc:
            missing.append(f"ifcopenshell.geom.occ_utils: {exc}")

        optional_messages: list[str] = []
        for module_name, symbol_name in _OPTIONAL:
            try:
                symbols[symbol_name] = _import_symbol(module_name, symbol_name)
            except Exception as exc:
                optional_messages.append(f"{module_name}.{symbol_name}: {exc}")

        if missing:
            detail = "\n  ".join(missing)
            _LAST_FAILURE = detail
            if strict:
                raise ImportError("OCC bootstrap failed:\n  " + detail)
            _LOG.debug("OCC bootstrap missing symbols: %s", missing)
            return
        _SYMBOLS = symbols
        _BOOTSTRAPPED = True
        _LAST_FAILURE = None
        occ_module = None
        try:
            import OCC  # type: ignore

            occ_module = getattr(OCC, "__file__", None)
        except Exception:  # pragma: no cover - diagnostics only
            occ_module = None
        _LOG.info(
            "OCC bootstrap complete (python=%s, occ=%s)",
            sys.executable,
            occ_module or "<unknown>",
        )
        if optional_messages:
            _LOG.debug("OCC optional symbols unavailable: %s", optional_messages)


def is_available() -> bool:
    try:
        bootstrap_occ(strict=False)
        return _BOOTSTRAPPED
    except Exception:
        return False


def require_occ() -> None:
    bootstrap_occ(strict=True)


def sym_optional(name: str):
    require_occ()
    return _SYMBOLS.get(name)


def last_failure_reason() -> str | None:
    return _LAST_FAILURE


def sym(name: str):
    require_occ()
    return _SYMBOLS[name]
