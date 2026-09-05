"""Read-only Autodesk Forma/ACC source acquisition for BuildUSD.

The adapter resolves an immutable Data Management version. IFC versions are
downloaded directly; RVT versions are translated to IFC by Model Derivative.
The resulting local IFC can then enter BuildUSD's existing conversion pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

APS_API_BASE = "https://developer.api.autodesk.com"
APS_TOKEN_URL = f"{APS_API_BASE}/authentication/v2/token"
DEFAULT_SCOPES = ("data:read", "data:write", "bucket:read")

LOG = logging.getLogger(__name__)


class FormaIntegrationError(RuntimeError):
    """Base error for Autodesk Forma source acquisition."""


class FormaAuthenticationError(FormaIntegrationError):
    """Authentication configuration or token acquisition failed."""


class FormaSourceError(FormaIntegrationError):
    """The selected Forma version cannot be used as a BuildUSD source."""


class FormaTranslationError(FormaIntegrationError):
    """Model Derivative could not produce an IFC derivative."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()

    def json(self) -> Any:
        if not self.body:
            return {}
        return json.loads(self.body.decode("utf-8"))

    def header_values(self, name: str) -> list[str]:
        lowered = name.lower()
        return [value for key, value in self.headers if key.lower() == lowered]


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> HttpResponse: ...

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None: ...


class UrlLibTransport:
    """Small standard-library HTTP transport with streaming downloads."""

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> HttpResponse:
        request = Request(url, data=body, headers=dict(headers or {}), method=method)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return HttpResponse(
                    status=int(response.status),
                    body=response.read(),
                    headers=tuple(response.headers.items()),
                )
        except HTTPError as exc:
            response_body = exc.read()
            detail = response_body.decode("utf-8", errors="replace")[:1000]
            raise FormaIntegrationError(
                f"APS request failed ({exc.code}) {method} {_redact_url(url)}: {detail}"
            ) from exc
        except URLError as exc:
            raise FormaIntegrationError(
                f"APS request failed {method} {_redact_url(url)}: {exc.reason}"
            ) from exc

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers=dict(headers or {}), method="GET")
        temp_path: Optional[Path] = None
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, delete=False
                ) as handle:
                    temp_path = Path(handle.name)
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            os.replace(temp_path, destination)
            temp_path = None
        except (HTTPError, URLError) as exc:
            raise FormaIntegrationError(
                f"Failed to download {_redact_url(url)}: {exc}"
            ) from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


class AccessTokenProvider(Protocol):
    def get_token(self) -> str: ...


@dataclass(frozen=True, slots=True)
class StaticAccessToken:
    token: str

    def get_token(self) -> str:
        if not self.token.strip():
            raise FormaAuthenticationError("The APS access token is empty.")
        return self.token.strip()


class ClientCredentialsTokenProvider:
    """APS OAuth client-credentials provider for provisioned server apps."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        transport: HttpTransport,
        *,
        scopes: Sequence[str] = DEFAULT_SCOPES,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.transport = transport
        self.scopes = tuple(scopes)
        self._token: Optional[str] = None
        self._expires_at = 0.0

    def get_token(self) -> str:
        now = time.monotonic()
        if self._token and now < self._expires_at - 60.0:
            return self._token
        if not self.client_id or not self.client_secret:
            raise FormaAuthenticationError(
                "APS_CLIENT_ID and APS_CLIENT_SECRET must both be configured."
            )
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("ascii")
        response = self.transport.request(
            "POST",
            APS_TOKEN_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body=urlencode(
                {"grant_type": "client_credentials", "scope": " ".join(self.scopes)}
            ).encode("ascii"),
        )
        payload = response.json()
        token = str(payload.get("access_token") or "").strip()
        if not token:
            raise FormaAuthenticationError(
                "APS authentication response did not contain an access_token."
            )
        try:
            expires_in = max(60.0, float(payload.get("expires_in", 3600)))
        except (TypeError, ValueError):
            expires_in = 3600.0
        self._token = token
        self._expires_at = now + expires_in
        return token


def token_provider_from_environment(transport: HttpTransport) -> AccessTokenProvider:
    """Build a token provider without placing secrets in command-line arguments."""

    access_token = os.environ.get("APS_ACCESS_TOKEN", "").strip()
    if access_token:
        return StaticAccessToken(access_token)
    client_id = os.environ.get("APS_CLIENT_ID", "").strip()
    client_secret = os.environ.get("APS_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return ClientCredentialsTokenProvider(client_id, client_secret, transport)
    raise FormaAuthenticationError(
        "Configure APS_ACCESS_TOKEN, or both APS_CLIENT_ID and APS_CLIENT_SECRET. "
        "The APS application must also be provisioned in the Autodesk Forma hub."
    )


@dataclass(frozen=True, slots=True)
class FormaSource:
    """An immutable Autodesk Forma Data Management file version."""

    project_id: str
    version_id: str
    cache_dir: Optional[Path] = None
    local_name: Optional[str] = None
    ifc_export_setting: Optional[str] = "IFC4 Reference View"
    force_translation: bool = False
    translation_timeout_seconds: float = 1800.0
    poll_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise ValueError("FormaSource.project_id is required.")
        if not self.version_id.strip():
            raise ValueError("FormaSource.version_id is required.")
        if self.local_name is not None and not self.local_name.strip():
            raise ValueError("FormaSource.local_name cannot be empty.")
        if self.translation_timeout_seconds <= 0:
            raise ValueError("translation_timeout_seconds must be greater than zero.")
        if self.poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds cannot be negative.")


@dataclass(frozen=True, slots=True)
class FormaVersion:
    project_id: str
    version_id: str
    display_name: str
    storage_urn: Optional[str]

    @property
    def suffix(self) -> str:
        return Path(self.display_name).suffix.lower()


@dataclass(frozen=True, slots=True)
class FormaProjectLocator:
    """Project, folder, and item identifiers decoded from an ACC Docs URL."""

    project_id: str
    folder_id: Optional[str] = None
    item_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class FormaProjectFile:
    """One discoverable RVT or IFC item pinned to its current tip version."""

    project_id: str
    item_id: str
    version_id: str
    display_name: str
    folder_path: str
    folder_id: Optional[str]
    web_url: str

    @property
    def relative_path(self) -> str:
        if not self.folder_path:
            return self.display_name
        return f"{self.folder_path.rstrip('/')}/{self.display_name}"

    @property
    def suffix(self) -> str:
        return Path(self.display_name).suffix.lower()

    @property
    def version_number(self) -> Optional[int]:
        values = parse_qs(urlsplit(self.version_id).query).get("version") or []
        try:
            return int(values[0]) if values else None
        except (TypeError, ValueError):
            return None

    def as_source(
        self,
        *,
        cache_dir: Optional[Path] = None,
        local_name: Optional[str] = None,
        ifc_export_setting: Optional[str] = "IFC4 Reference View",
        force_translation: bool = False,
        translation_timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 5.0,
    ) -> FormaSource:
        return FormaSource(
            project_id=self.project_id,
            version_id=self.version_id,
            cache_dir=cache_dir,
            local_name=local_name,
            ifc_export_setting=ifc_export_setting,
            force_translation=force_translation,
            translation_timeout_seconds=translation_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )


def parse_acc_project_url(url: str) -> FormaProjectLocator:
    """Decode project, folder, and item identifiers from an ACC Docs URL."""

    split = urlsplit(url.strip())
    if split.scheme not in {"http", "https"} or split.netloc.lower() not in {
        "acc.autodesk.com",
        "docs.b360.autodesk.com",
    }:
        raise FormaSourceError("Expected an Autodesk ACC or BIM 360 Docs URL.")
    match = re.search(r"/projects/([^/?#]+)", split.path, flags=re.IGNORECASE)
    if not match:
        raise FormaSourceError("The Autodesk URL does not contain a project ID.")
    raw_project_id = match.group(1)
    project_id = (
        raw_project_id if raw_project_id.startswith("b.") else f"b.{raw_project_id}"
    )
    query = parse_qs(split.query)
    folder_id = (query.get("folderUrn") or [None])[0]
    item_id = (query.get("entityId") or [None])[0]
    return FormaProjectLocator(
        project_id=project_id,
        folder_id=str(folder_id) if folder_id else None,
        item_id=str(item_id) if item_id else None,
    )


def default_forma_cache_dir() -> Path:
    configured = os.environ.get("BUILDUSD_FORMA_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "BuildUSD" / "cache" / "forma"
    cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "buildusd" / "forma"


def encode_derivative_urn(version_urn: str) -> str:
    encoded = base64.urlsafe_b64encode(version_urn.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def parse_storage_urn(storage_urn: str) -> tuple[str, str]:
    prefix = "urn:adsk.objects:os.object:"
    if not storage_urn.startswith(prefix):
        raise FormaSourceError(f"Unsupported APS storage URN: {storage_urn}")
    location = storage_urn[len(prefix) :]
    bucket_key, separator, object_key = location.partition("/")
    if not separator or not bucket_key or not object_key:
        raise FormaSourceError(f"Malformed APS storage URN: {storage_urn}")
    return bucket_key, object_key


class FormaClient:
    """Client for the Forma Data Management and Model Derivative workflow."""

    def __init__(
        self,
        token_provider: AccessTokenProvider,
        *,
        transport: Optional[HttpTransport] = None,
        api_base: str = APS_API_BASE,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.token_provider = token_provider
        self.transport = transport or UrlLibTransport()
        self.api_base = api_base.rstrip("/")
        self.log = logger or LOG

    @classmethod
    def from_environment(
        cls,
        *,
        transport: Optional[HttpTransport] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "FormaClient":
        resolved_transport = transport or UrlLibTransport()
        return cls(
            token_provider_from_environment(resolved_transport),
            transport=resolved_transport,
            logger=logger,
        )

    def _headers(self, *, json_content: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.token_provider.get_token()}",
            "Accept": "application/json",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def get_version(self, source: FormaSource) -> FormaVersion:
        project_id = quote(source.project_id, safe="")
        version_id = quote(source.version_id, safe="")
        response = self.transport.request(
            "GET",
            f"{self.api_base}/data/v1/projects/{project_id}/versions/{version_id}",
            headers=self._headers(),
        )
        payload = response.json()
        data = payload.get("data") or {}
        attributes = data.get("attributes") or {}
        extension_data = (attributes.get("extension") or {}).get("data") or {}
        display_name = str(
            extension_data.get("sourceFileName")
            or attributes.get("displayName")
            or attributes.get("name")
            or "forma-source"
        )
        storage_urn = (
            ((data.get("relationships") or {}).get("storage") or {}).get("data") or {}
        ).get("id")
        immutable_id = str(data.get("id") or source.version_id)
        return FormaVersion(
            project_id=source.project_id,
            version_id=immutable_id,
            display_name=display_name,
            storage_urn=str(storage_urn) if storage_urn else None,
        )

    def list_project_files(
        self,
        project_id: str,
        *,
        start_folder_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        extensions: Sequence[str] = (".rvt", ".ifc"),
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> list[FormaProjectFile]:
        """Recursively list supported files in a project or selected folder tree."""

        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth cannot be negative.")
        normalized_project_id = _normalize_project_id(project_id)
        check_cancel = cancel_check or (lambda: None)
        normalized_extensions = {
            suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
            for suffix in extensions
        }
        folders: list[tuple[str, tuple[str, ...], int]] = []
        if start_folder_id:
            folder = self._get_folder(normalized_project_id, start_folder_id)
            folder_name = _display_name(folder, fallback="Selected Folder")
            folders.append((start_folder_id, (folder_name,), 0))
        else:
            hub_id = self._find_project_hub(normalized_project_id, check_cancel)
            top_folders_url = (
                f"{self.api_base}/project/v1/hubs/{quote(hub_id, safe='')}"
                f"/projects/{quote(normalized_project_id, safe='')}/topFolders"
            )
            for folder in self._iter_collection(top_folders_url, check_cancel):
                folder_id = str(folder.get("id") or "")
                if not folder_id:
                    continue
                folder_name = _display_name(folder, fallback="Project Files")
                folders.append((folder_id, (folder_name,), 0))

        discovered: list[FormaProjectFile] = []
        visited: set[str] = set()
        while folders:
            check_cancel()
            folder_id, path_parts, folder_depth = folders.pop()
            if folder_id in visited:
                continue
            visited.add(folder_id)
            contents_url = (
                f"{self.api_base}/data/v1/projects/"
                f"{quote(normalized_project_id, safe='')}/folders/"
                f"{quote(folder_id, safe='')}/contents"
            )
            for entry in self._iter_collection(contents_url, check_cancel):
                entry_type = str(entry.get("type") or "").casefold()
                if entry_type == "folders":
                    child_id = str(entry.get("id") or "")
                    if child_id and (max_depth is None or folder_depth < max_depth):
                        folders.append(
                            (
                                child_id,
                                (*path_parts, _display_name(entry, fallback="Folder")),
                                folder_depth + 1,
                            )
                        )
                    continue
                if entry_type != "items":
                    continue
                display_name = _display_name(entry, fallback="Forma file")
                if Path(display_name).suffix.casefold() not in normalized_extensions:
                    continue
                try:
                    discovered.append(
                        self._project_file_from_item(
                            normalized_project_id,
                            entry,
                            folder_id=folder_id,
                            folder_path="/".join(path_parts),
                        )
                    )
                except FormaSourceError as exc:
                    self.log.warning("Skipping Forma item %s: %s", display_name, exc)

        return sorted(
            discovered,
            key=lambda entry: (entry.relative_path.casefold(), entry.item_id),
        )

    def _get_folder(self, project_id: str, folder_id: str) -> Mapping[str, Any]:
        endpoint = (
            f"{self.api_base}/data/v1/projects/{quote(project_id, safe='')}"
            f"/folders/{quote(folder_id, safe='')}"
        )
        response = self.transport.request("GET", endpoint, headers=self._headers())
        payload = response.json()
        data = payload.get("data") or {}
        if not isinstance(data, Mapping):
            raise FormaSourceError("APS folder response did not contain folder data.")
        return data

    def get_item_tip(
        self,
        project_id: str,
        item_id: str,
        *,
        folder_id: Optional[str] = None,
        folder_path: str = "",
    ) -> FormaProjectFile:
        """Resolve one Data Management item/lineage URN to its current tip."""

        normalized_project_id = _normalize_project_id(project_id)
        endpoint = (
            f"{self.api_base}/data/v1/projects/"
            f"{quote(normalized_project_id, safe='')}/items/{quote(item_id, safe='')}"
        )
        response = self.transport.request("GET", endpoint, headers=self._headers())
        payload = response.json()
        data = payload.get("data") or {}
        if not isinstance(data, Mapping):
            raise FormaSourceError("APS item response did not contain item data.")
        return self._project_file_from_item(
            normalized_project_id,
            data,
            folder_id=folder_id,
            folder_path=folder_path,
        )

    def _find_project_hub(
        self, project_id: str, cancel_check: Callable[[], None]
    ) -> str:
        hubs_url = f"{self.api_base}/project/v1/hubs"
        accessible_hubs = 0
        for hub in self._iter_collection(hubs_url, cancel_check):
            hub_id = str(hub.get("id") or "")
            if not hub_id:
                continue
            accessible_hubs += 1
            projects_url = (
                f"{self.api_base}/project/v1/hubs/{quote(hub_id, safe='')}/projects"
            )
            for project in self._iter_collection(projects_url, cancel_check):
                if str(project.get("id") or "") == project_id:
                    return hub_id
        raise FormaSourceError(
            f"Project '{project_id}' was not found in {accessible_hubs} accessible "
            "APS hub(s). Check Forma Custom Integrations and project access."
        )

    def _iter_collection(
        self, url: str, cancel_check: Callable[[], None]
    ) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        current_url: Optional[str] = url
        seen_urls: set[str] = set()
        api_host = urlsplit(self.api_base).netloc.casefold()
        while current_url:
            cancel_check()
            if current_url in seen_urls:
                raise FormaSourceError("APS pagination returned a repeated next link.")
            seen_urls.add(current_url)
            response = self.transport.request(
                "GET", current_url, headers=self._headers()
            )
            payload = response.json()
            data = payload.get("data") or []
            if not isinstance(data, list):
                raise FormaSourceError(
                    "APS collection response did not contain a list."
                )
            entries.extend(entry for entry in data if isinstance(entry, Mapping))
            next_link = (payload.get("links") or {}).get("next")
            if isinstance(next_link, Mapping):
                next_link = next_link.get("href")
            if not next_link:
                current_url = None
                continue
            candidate = urljoin(current_url, str(next_link))
            if urlsplit(candidate).netloc.casefold() != api_host:
                raise FormaSourceError("APS pagination returned an unexpected host.")
            current_url = candidate
        return entries

    @staticmethod
    def _project_file_from_item(
        project_id: str,
        item: Mapping[str, Any],
        *,
        folder_id: Optional[str],
        folder_path: str,
    ) -> FormaProjectFile:
        item_id = str(item.get("id") or "")
        relationships = item.get("relationships") or {}
        tip_data = (relationships.get("tip") or {}).get("data") or {}
        version_id = str(tip_data.get("id") or "")
        if not item_id or not version_id:
            raise FormaSourceError(
                "An Autodesk file item did not include its item or tip-version ID."
            )
        display_name = _display_name(item, fallback="Forma file")
        links = item.get("links") or {}
        web_url = _link_href(links.get("webView"))
        if not web_url:
            web_url = _build_acc_file_url(project_id, folder_id, item_id)
        return FormaProjectFile(
            project_id=project_id,
            item_id=item_id,
            version_id=version_id,
            display_name=display_name,
            folder_path=folder_path,
            folder_id=folder_id,
            web_url=web_url,
        )

    def materialize_ifc(
        self,
        source: FormaSource,
        *,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Path:
        check_cancel = cancel_check or (lambda: None)
        check_cancel()
        version = self.get_version(source)
        if version.suffix not in {".ifc", ".rvt"}:
            raise FormaSourceError(
                f"Forma version '{version.display_name}' is not IFC or RVT."
            )

        cache_root = (source.cache_dir or default_forma_cache_dir()).expanduser()
        cache_key = self._cache_key(version, source)
        cache_entry = cache_root / cache_key
        local_name = source.local_name or version.display_name
        output_name = _safe_filename(Path(local_name).stem) + ".ifc"
        output_path = cache_entry / output_name
        if output_path.exists() and output_path.stat().st_size > 0:
            self.log.info("Using cached Forma source %s", output_path)
            return output_path

        cache_entry.mkdir(parents=True, exist_ok=True)
        check_cancel()
        if version.suffix == ".ifc":
            if not version.storage_urn:
                raise FormaSourceError(
                    f"Forma IFC version '{version.display_name}' has no storage relationship."
                )
            self.log.info("Downloading hosted IFC: %s", version.display_name)
            self._download_storage(version.storage_urn, output_path)
        else:
            derivative_urn = encode_derivative_urn(version.version_id)
            self.log.info(
                "Submitting APS RVT-to-IFC translation: %s", version.display_name
            )
            self._submit_ifc_translation(derivative_urn, source)
            self.log.info(
                "APS translation accepted; waiting for %s (timeout %.0f seconds)",
                version.display_name,
                source.translation_timeout_seconds,
            )
            manifest = self._wait_for_translation(
                derivative_urn,
                source,
                display_name=version.display_name,
                cancel_check=check_cancel,
            )
            derivative = self._find_ifc_derivative(manifest)
            self.log.info(
                "APS translation complete; downloading IFC derivative for %s",
                version.display_name,
            )
            self._download_derivative(derivative_urn, derivative, output_path)

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise FormaSourceError(
                f"Forma acquisition produced an empty IFC file: {output_path}"
            )
        self._write_provenance(cache_entry, version, source)
        return output_path

    @staticmethod
    def _cache_key(version: FormaVersion, source: FormaSource) -> str:
        payload = {
            "project_id": version.project_id,
            "version_id": version.version_id,
            "ifc_export_setting": source.ifc_export_setting,
            "local_name": source.local_name,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _download_storage(self, storage_urn: str, destination: Path) -> None:
        bucket_key, object_key = parse_storage_urn(storage_urn)
        endpoint = (
            f"{self.api_base}/oss/v2/buckets/{quote(bucket_key, safe='')}"
            f"/objects/{quote(object_key, safe='')}/signeds3download"
        )
        response = self.transport.request("GET", endpoint, headers=self._headers())
        payload = response.json()
        signed_url = payload.get("url")
        if not signed_url and payload.get("urls"):
            signed_url = payload["urls"][0]
        if not signed_url:
            raise FormaSourceError(
                "APS signed download response did not include a URL."
            )
        self.transport.download(str(signed_url), destination)

    def _submit_ifc_translation(self, derivative_urn: str, source: FormaSource) -> None:
        output: dict[str, Any] = {"type": "ifc"}
        if source.ifc_export_setting:
            output["advanced"] = {"exportSettingName": source.ifc_export_setting}
        headers = self._headers(json_content=True)
        if source.force_translation:
            headers["x-ads-force"] = "true"
        body = json.dumps(
            {"input": {"urn": derivative_urn}, "output": {"formats": [output]}}
        ).encode("utf-8")
        self.transport.request(
            "POST",
            f"{self.api_base}/modelderivative/v2/designdata/job",
            headers=headers,
            body=body,
        )

    def _wait_for_translation(
        self,
        derivative_urn: str,
        source: FormaSource,
        *,
        display_name: str,
        cancel_check: Callable[[], None],
    ) -> dict[str, Any]:
        deadline = time.monotonic() + source.translation_timeout_seconds
        next_heartbeat = 0.0
        last_state: tuple[str, str] | None = None
        endpoint = (
            f"{self.api_base}/modelderivative/v2/designdata/"
            f"{quote(derivative_urn, safe='')}/manifest"
        )
        while True:
            cancel_check()
            response = self.transport.request("GET", endpoint, headers=self._headers())
            manifest = response.json()
            status = str(manifest.get("status") or "").lower()
            progress = str(manifest.get("progress") or "").strip()
            state = (status, progress)
            now = time.monotonic()
            if state != last_state or now >= next_heartbeat:
                progress_suffix = f" ({progress})" if progress else ""
                self.log.info(
                    "APS translation status for %s: %s%s",
                    display_name,
                    status or "pending",
                    progress_suffix,
                )
                last_state = state
                next_heartbeat = now + 60.0
            if status == "success":
                return manifest
            if status in {"failed", "timeout"}:
                raise FormaTranslationError(
                    f"RVT to IFC translation ended with status '{status}'."
                )
            if now >= deadline:
                raise FormaTranslationError(
                    "Timed out waiting for the RVT to IFC derivative."
                )
            time.sleep(source.poll_interval_seconds)

    @staticmethod
    def _find_ifc_derivative(manifest: Mapping[str, Any]) -> str:
        candidates: list[Mapping[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                candidates.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(manifest.get("derivatives") or [])
        for candidate in candidates:
            urn = str(candidate.get("urn") or "")
            mime = str(candidate.get("mime") or "").lower()
            output_type = str(candidate.get("outputType") or "").lower()
            if urn and (
                output_type == "ifc"
                or "ifc" in mime
                or urlsplit(urn).path.lower().endswith(".ifc")
            ):
                return urn
        raise FormaTranslationError(
            "Model Derivative succeeded but its manifest contained no IFC derivative."
        )

    def _download_derivative(
        self, source_urn: str, derivative_urn: str, destination: Path
    ) -> None:
        endpoint = (
            f"{self.api_base}/modelderivative/v2/designdata/"
            f"{quote(source_urn, safe='')}/manifest/"
            f"{quote(derivative_urn, safe='')}/signedcookies?useCdn=true"
        )
        response = self.transport.request("GET", endpoint, headers=self._headers())
        payload = response.json()
        download_url = str(payload.get("url") or "")
        if not download_url:
            raise FormaTranslationError(
                "APS derivative download response did not include a URL."
            )
        signed_url = _apply_cloudfront_cookies(
            download_url, response.header_values("Set-Cookie")
        )
        self.transport.download(signed_url, destination)

    @staticmethod
    def _write_provenance(
        cache_entry: Path, version: FormaVersion, source: FormaSource
    ) -> None:
        payload = {
            "schema": "buildusd.forma_source.v1",
            "project_id": version.project_id,
            "version_id": version.version_id,
            "display_name": version.display_name,
            "storage_urn": version.storage_urn,
            "ifc_export_setting": source.ifc_export_setting,
            "local_name": source.local_name,
        }
        path = cache_entry / "source.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_entry,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temp_path, path)


def _normalize_project_id(project_id: str) -> str:
    value = project_id.strip()
    if not value:
        raise FormaSourceError("An Autodesk project ID is required.")
    return value if value.startswith("b.") else f"b.{value}"


def _display_name(entry: Mapping[str, Any], *, fallback: str) -> str:
    attributes = entry.get("attributes") or {}
    extension_data = (attributes.get("extension") or {}).get("data") or {}
    return str(
        extension_data.get("sourceFileName")
        or attributes.get("displayName")
        or attributes.get("name")
        or fallback
    )


def _link_href(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("href") or "")
    return str(value or "")


def _build_acc_file_url(project_id: str, folder_id: Optional[str], item_id: str) -> str:
    web_project_id = project_id[2:] if project_id.startswith("b.") else project_id
    parameters = {
        "entityId": item_id,
        "viewModel": "detail",
        "moduleId": "folders",
    }
    if folder_id:
        parameters["folderUrn"] = folder_id
    return (
        f"https://acc.autodesk.com/docs/files/projects/{web_project_id}?"
        f"{urlencode(parameters)}"
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "forma-source"


def _redact_url(url: str) -> str:
    """Remove signed query parameters before a URL reaches an exception message."""

    split = urlsplit(url)
    return urlunsplit((split.scheme, split.netloc, split.path, "", ""))


def _apply_cloudfront_cookies(url: str, set_cookie_headers: Sequence[str]) -> str:
    cookie = SimpleCookie()
    for header in set_cookie_headers:
        cookie.load(header)
    parameter_names = {
        "CloudFront-Policy": "Policy",
        "CloudFront-Key-Pair-Id": "Key-Pair-Id",
        "CloudFront-Signature": "Signature",
    }
    parameters = {
        query_name: cookie[cookie_name].value
        for cookie_name, query_name in parameter_names.items()
        if cookie_name in cookie
    }
    if not parameters:
        return url
    split = urlsplit(url)
    query = split.query
    signed_query = urlencode(parameters)
    combined_query = f"{query}&{signed_query}" if query else signed_query
    return urlunsplit(
        (split.scheme, split.netloc, split.path, combined_query, split.fragment)
    )


__all__ = [
    "AccessTokenProvider",
    "ClientCredentialsTokenProvider",
    "FormaAuthenticationError",
    "FormaClient",
    "FormaIntegrationError",
    "FormaProjectFile",
    "FormaProjectLocator",
    "FormaSource",
    "FormaSourceError",
    "FormaTranslationError",
    "FormaVersion",
    "HttpResponse",
    "HttpTransport",
    "StaticAccessToken",
    "UrlLibTransport",
    "default_forma_cache_dir",
    "encode_derivative_urn",
    "parse_storage_urn",
    "parse_acc_project_url",
    "token_provider_from_environment",
]
