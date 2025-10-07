from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Sequence, Union

try:
    import omni.client as _omni_client
except Exception:  # pragma: no cover - optional dependency
    _omni_client = None

PathLike = Union[str, Path]

_OMNI_PREFIXES = ("omniverse://",)


def _require_omni_client():
    if _omni_client is None:
        raise RuntimeError(
            "omni.client is required for omniverse:// paths. Install omniverse-kit and ensure it is available."
        )
    return _omni_client


def is_omniverse_path(value: PathLike) -> bool:
    s = _as_string(value).lower()
    return any(s.startswith(prefix) for prefix in _OMNI_PREFIXES)


def _as_string(value: PathLike) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def ensure_directory(path: PathLike) -> None:
    if is_omniverse_path(path):
        client = _require_omni_client()
        uri = _normalize_nucleus_dir(_as_string(path))
        _ensure_nucleus_directory(client, uri)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def ensure_parent_directory(path: PathLike) -> None:
    if is_omniverse_path(path):
        parent_uri = parent_path(path)
        if parent_uri:
            ensure_directory(parent_uri)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def parent_path(path: PathLike) -> PathLike:
    if is_omniverse_path(path):
        uri = _normalize_nucleus_path(_as_string(path))
        if "/" not in uri[len("omniverse://") :]:
            return uri
        return uri.rsplit("/", 1)[0]
    return Path(path).parent


def join_path(base: PathLike, *parts: str) -> PathLike:
    if is_omniverse_path(base):
        uri = _normalize_nucleus_path(_as_string(base))
        segments = [seg.strip("/") for seg in parts if seg]
        if not segments:
            return uri
        return "/".join([uri.rstrip("/")] + segments)
    path = Path(base)
    for part in parts:
        if part:
            path /= part
    return path


def path_suffix(path: PathLike) -> str:
    name = path_name(path)
    idx = name.rfind(".")
    return "" if idx <= 0 else name[idx:].lower()


def path_stem(path: PathLike) -> str:
    name = path_name(path)
    suffix = path_suffix(path)
    return name[: -len(suffix)] if suffix else name


def path_name(path: PathLike) -> str:
    if is_omniverse_path(path):
        uri = _normalize_nucleus_path(_as_string(path)).rstrip("/")
        return uri.rsplit("/", 1)[-1]
    return Path(path).name


def read_text(path: PathLike, encoding: str = "utf-8") -> str:
    if is_omniverse_path(path):
        client = _require_omni_client()
        result, content = client.read_file(_normalize_nucleus_path(_as_string(path)))
        if result != client.Result.OK:
            raise RuntimeError(f"Failed to read {path}: {result}")
        return bytes(content).decode(encoding)
    return Path(path).read_text(encoding=encoding)


def read_bytes(path: PathLike) -> bytes:
    if is_omniverse_path(path):
        client = _require_omni_client()
        result, content = client.read_file(_normalize_nucleus_path(_as_string(path)))
        if result != client.Result.OK:
            raise RuntimeError(f"Failed to read {path}: {result}")
        return bytes(content)
    return Path(path).read_bytes()


def write_text(path: PathLike, data: str, encoding: str = "utf-8") -> None:
    if is_omniverse_path(path):
        client = _require_omni_client()
        ensure_parent_directory(path)
        result = client.write_file(
            _normalize_nucleus_path(_as_string(path)),
            data.encode(encoding),
        )
        if result != client.Result.OK:
            raise RuntimeError(f"Failed to write {path}: {result}")
    else:
        ensure_parent_directory(path)
        Path(path).write_text(data, encoding=encoding)


def write_bytes(path: PathLike, data: bytes) -> None:
    if is_omniverse_path(path):
        client = _require_omni_client()
        ensure_parent_directory(path)
        result = client.write_file(
            _normalize_nucleus_path(_as_string(path)),
            data,
        )
        if result != client.Result.OK:
            raise RuntimeError(f"Failed to write {path}: {result}")
    else:
        ensure_parent_directory(path)
        Path(path).write_bytes(data)


def create_checkpoint(path: PathLike, *, note: Optional[str] = None, tags: Optional[Sequence[str]] = None) -> bool:
    """Create a Nucleus checkpoint for the given path when possible.

    Returns:
        True when a checkpoint was created, False when the path is not an omniverse URI.
    """
    if not is_omniverse_path(path):
        return False
    client = _require_omni_client()
    tag_list: List[str] = []
    if tags:
        tag_list = [str(tag).strip() for tag in tags if str(tag).strip()]
    result = client.create_checkpoint(
        _normalize_nucleus_path(_as_string(path)),
        note or "",
        tag_list,
    )
    if result != client.Result.OK:
        raise RuntimeError(f"Failed to checkpoint {path}: {result}")
    return True


def list_directory(path: PathLike):
    if is_omniverse_path(path):
        client = _require_omni_client()
        uri = _normalize_nucleus_dir(_as_string(path))
        result, entries = client.list(uri)
        if result != client.Result.OK:
            raise RuntimeError(f"Failed to list {path}: {result}")
        return entries
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return list(p.iterdir())


def stat_entry(path: PathLike):
    if is_omniverse_path(path):
        client = _require_omni_client()
        result, entry = client.stat(_normalize_nucleus_path(_as_string(path)))
        if result != client.Result.OK:
            return None
        return entry
    p = Path(path)
    if not p.exists():
        return None
    return p


def normalize_exclusions(exclusions: Sequence[str] | None) -> set[str]:
    if not exclusions:
        return set()
    normalized = set()
    for item in exclusions:
        name = (item or "").strip()
        if not name:
            continue
        if name.lower().endswith(".ifc"):
            name = name[:-4]
        normalized.add(name.lower())
    return normalized


def _normalize_nucleus_dir(uri: str) -> str:
    if not uri.endswith("/"):
        uri = uri + "/"
    return _normalize_nucleus_path(uri)


def _normalize_nucleus_path(uri: str) -> str:
    if uri.startswith("omniverse:/") and not uri.startswith("omniverse://"):
        uri = uri.replace("omniverse:/", "omniverse://", 1)
    return uri


def _ensure_nucleus_directory(client, uri: str) -> None:
    uri = _normalize_nucleus_dir(uri)
    result, entry = client.stat(uri)
    if result == client.Result.OK:
        return
    # recurse to parent
    parent = uri.rstrip("/")
    if "/" in parent[len("omniverse://") :]:
        _ensure_nucleus_directory(client, parent.rsplit("/", 1)[0])
    create_result = client.create_folder(uri.rstrip("/"))
    if create_result not in (client.Result.OK, getattr(client.Result, "ALREADY_EXISTS", client.Result.OK)):
        raise RuntimeError(f"Failed to create omniverse directory {uri}: {create_result}")
