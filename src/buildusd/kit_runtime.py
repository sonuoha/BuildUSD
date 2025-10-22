from __future__ import annotations

from typing import Iterable, Sequence

_KIT_APP = None
_ENABLED_EXTENSIONS: set[str] = set()

_BASE_ARGS: list[str] = ["--no-window"]
_DEFAULT_EXTENSIONS: tuple[str, ...] = ("omni.client", "omni.usd")


def _normalize_extensions(extensions: Iterable[str] | None) -> Sequence[str]:
    if not extensions:
        return _DEFAULT_EXTENSIONS
    return tuple(dict.fromkeys([ext for ext in extensions if ext]))


def ensure_kit(extensions: Iterable[str] | None = None):
    """Start Kit (if needed) with the requested extensions enabled.

    Returns the shared KitApp instance so callers can retrieve omni.client, pxr, etc.
    """

    global _KIT_APP, _ENABLED_EXTENSIONS

    required = _normalize_extensions(extensions)

    try:
        from omni.kit_app import KitApp  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "omni.kit_app is unavailable. Install Omniverse Kit (pip install --extra-index-url "
            "https://pypi.nvidia.com omniverse-kit) or run within an Omniverse Python runtime."
        ) from exc

    if _KIT_APP is None:
        _KIT_APP = KitApp()
        args: list[str] = list(_BASE_ARGS)
        all_exts = _normalize_extensions(tuple(dict.fromkeys(list(required) + list(_DEFAULT_EXTENSIONS))))
        for ext in all_exts:
            if ext and ext not in _ENABLED_EXTENSIONS:
                args.extend(["--enable", ext])
                _ENABLED_EXTENSIONS.add(ext)
        _KIT_APP.startup(args)
    return _KIT_APP


def shutdown_kit():
    """Shutdown the shared Kit application if it was started via ensure_kit."""

    global _KIT_APP, _ENABLED_EXTENSIONS
    if _KIT_APP is None:
        return
    try:
        _KIT_APP.shutdown()
    except Exception:
        pass
    finally:
        _KIT_APP = None
        _ENABLED_EXTENSIONS.clear()
