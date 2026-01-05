from __future__ import annotations

import os
import threading
import time
from typing import Iterable, Sequence

_KIT_APP = None
_KIT_APP_OWNED = False
_ENABLED_EXTENSIONS: set[str] = set()
_KIT_THREAD = None
_KIT_THREAD_STOP = None

_BASE_ARGS: list[str] = ["--no-window"]
_DEFAULT_EXTENSIONS: tuple[str, ...] = ("omni.client", "omni.usd")
_DEFAULT_EXT_FOLDERS: tuple[str, ...] = (r"C:\Users\samue\_dev\kit-sdk-106\exts",)
_CESIUM_EXT_ENV = "BUILDUSD_CESIUM_EXT_FOLDERS"
_CESIUM_DEFAULT_FOLDERS = (
    r"C:\Users\samue\OneDrive\Documents\Kit\shared\exts",
    r"C:\Users\samue\OneDrive\Documents\Kit\shared\exts\cesium.omniverse",
    r"C:\Users\samue\OneDrive\Documents\Kit\shared\exts\cesium.usd.plugins",
)
os.environ.setdefault(_CESIUM_EXT_ENV, os.pathsep.join(_CESIUM_DEFAULT_FOLDERS))


def _normalize_extensions(extensions: Iterable[str] | None) -> Sequence[str]:
    if not extensions:
        return _DEFAULT_EXTENSIONS
    return tuple(dict.fromkeys([ext for ext in extensions if ext]))


def _collect_ext_folders() -> list[str]:
    folders: list[str] = []
    env_paths = os.environ.get("BUILDUSD_KIT_EXT_FOLDERS") or ""
    for raw in env_paths.split(os.pathsep):
        path = raw.strip()
        if path:
            folders.append(path)
    cesium_paths = os.environ.get(_CESIUM_EXT_ENV) or ""
    for raw in cesium_paths.split(os.pathsep):
        path = raw.strip()
        if path:
            folders.append(path)
            # If a specific extension folder is provided, also add its parent.
            parent = os.path.dirname(path)
            if parent:
                folders.append(parent)
    for default in _DEFAULT_EXT_FOLDERS:
        if default:
            folders.append(default)
    unique: list[str] = []
    for path in folders:
        norm = os.path.normpath(path)
        if norm not in unique and os.path.isdir(norm):
            unique.append(norm)
    return unique


def _start_kit_loop() -> None:
    global _KIT_THREAD, _KIT_THREAD_STOP
    if os.environ.get("BUILDUSD_KIT_DISABLE_LOOP", "").strip():
        return
    if _KIT_THREAD is not None and _KIT_THREAD.is_alive():
        return
    if _KIT_APP is None:
        return
    _KIT_THREAD_STOP = threading.Event()

    def _run() -> None:
        app = _KIT_APP
        if app is None:
            return
        update = getattr(app, "update", None)
        is_running = getattr(app, "is_running", None)
        if not callable(update):
            return
        while not _KIT_THREAD_STOP.is_set():
            try:
                if callable(is_running) and not is_running():
                    break
            except Exception:
                break
            try:
                update()
            except Exception:
                break
            time.sleep(0.01)

    _KIT_THREAD = threading.Thread(target=_run, name="buildusd_kit_loop", daemon=True)
    _KIT_THREAD.start()


def _get_running_kit_app():
    """Return an already-running Kit app if executing inside Kit."""
    try:
        from omni.kit.app import get_app  # type: ignore
    except ModuleNotFoundError:
        return None

    try:
        existing = get_app()
    except Exception:
        return None

    return existing


def ensure_kit(extensions: Iterable[str] | None = None):
    """Start Kit (if needed) with the requested extensions enabled.

    Returns the shared KitApp instance so callers can retrieve omni.client, pxr, etc.
    """

    global _KIT_APP, _KIT_APP_OWNED, _ENABLED_EXTENSIONS

    required = _normalize_extensions(extensions)

    if _KIT_APP is not None:
        is_running = getattr(_KIT_APP, "is_running", None)
        if callable(is_running):
            try:
                if not is_running():
                    _KIT_APP = None
                    _KIT_APP_OWNED = False
                    _ENABLED_EXTENSIONS.clear()
            except Exception:
                pass

    if _KIT_APP is None:
        # Detect whether we're already inside a Kit runtime (e.g., extension template).
        existing = _get_running_kit_app()
        if existing is not None:
            _KIT_APP = existing
            _KIT_APP_OWNED = False
            return _KIT_APP

        try:
            from omni.kit_app import KitApp  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "omni.kit_app is unavailable. Install Omniverse Kit (pip install --extra-index-url "
                "https://pypi.nvidia.com omniverse-kit) or run within an Omniverse Python runtime."
            ) from exc

        _KIT_APP = KitApp()
        _KIT_APP_OWNED = True
        args: list[str] = list(_BASE_ARGS)
        quit_after = os.environ.get("BUILDUSD_KIT_QUIT_AFTER")
        if quit_after is None:
            args.append("--/app/quitAfter=-1")
        else:
            value = quit_after.strip()
            if value:
                args.append(f"--/app/quitAfter={value}")
        all_exts = _normalize_extensions(
            tuple(dict.fromkeys(list(required) + list(_DEFAULT_EXTENSIONS)))
        )
        for ext in all_exts:
            if ext and ext not in _ENABLED_EXTENSIONS:
                args.extend(["--enable", ext])
                _ENABLED_EXTENSIONS.add(ext)
        for folder in _collect_ext_folders():
            args.extend(["--ext-folder", folder])
        _KIT_APP.startup(args)
    _start_kit_loop()
    return _KIT_APP


def shutdown_kit():
    """Shutdown the shared Kit application if it was started via ensure_kit."""

    global _KIT_APP, _KIT_APP_OWNED, _ENABLED_EXTENSIONS, _KIT_THREAD, _KIT_THREAD_STOP
    if _KIT_APP is None:
        return

    if _KIT_THREAD_STOP is not None:
        _KIT_THREAD_STOP.set()
    if _KIT_THREAD is not None:
        _KIT_THREAD.join(timeout=2)
    _KIT_THREAD = None
    _KIT_THREAD_STOP = None

    if _KIT_APP_OWNED:
        try:
            _KIT_APP.shutdown()
        except Exception:
            pass

    _KIT_APP = None
    _KIT_APP_OWNED = False
    _ENABLED_EXTENSIONS.clear()
