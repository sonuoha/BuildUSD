from __future__ import annotations

import json
import importlib
import logging
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

import pytest

from buildusd import api
from buildusd.conversion import (
    _parse_forma_selection,
    _resolve_forma_cli_sources,
    _select_forma_files,
    parse_args,
)
from buildusd.forma import (
    APS_TOKEN_URL,
    ClientCredentialsTokenProvider,
    FormaClient,
    FormaProjectFile,
    FormaSource,
    FormaTranslationError,
    HttpResponse,
    StaticAccessToken,
    encode_derivative_urn,
    parse_acc_project_url,
    parse_storage_urn,
)


class FakeTransport:
    def __init__(self) -> None:
        self.responses: dict[tuple[str, str], deque[HttpResponse]] = defaultdict(deque)
        self.downloads: dict[str, bytes] = {}
        self.requests: list[tuple[str, str, Mapping[str, str], bytes | None]] = []
        self.download_requests: list[tuple[str, Path]] = []

    def add_json(
        self,
        method: str,
        url: str,
        payload: object,
        *,
        headers: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.responses[(method, url)].append(
            HttpResponse(200, json.dumps(payload).encode("utf-8"), headers)
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
    ) -> HttpResponse:
        self.requests.append((method, url, dict(headers or {}), body))
        queued = self.responses[(method, url)]
        if not queued:
            raise AssertionError(f"Unexpected request: {method} {url}")
        return queued.popleft()

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        del headers
        self.download_requests.append((url, destination))
        if url not in self.downloads:
            raise AssertionError(f"Unexpected download: {url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.downloads[url])


def _version_payload(version_id: str, name: str, storage_urn: str | None) -> dict:
    relationships = {}
    if storage_urn:
        relationships = {"storage": {"data": {"id": storage_urn}}}
    return {
        "data": {
            "id": version_id,
            "attributes": {"displayName": name},
            "relationships": relationships,
        }
    }


def test_direct_ifc_download_is_cached_by_immutable_version(tmp_path: Path) -> None:
    transport = FakeTransport()
    project_id = "b.project"
    requested_version = "urn:adsk.wipprod:fs.file:vf.item?version=2"
    immutable_version = "urn:adsk.wipprod:fs.file:vf.item?version=7"
    storage_urn = "urn:adsk.objects:os.object:wip.dm.prod/folder/model.ifc"
    api = "https://developer.api.autodesk.com"
    version_url = (
        f"{api}/data/v1/projects/b.project/versions/"
        "urn%3Aadsk.wipprod%3Afs.file%3Avf.item%3Fversion%3D2"
    )
    signed_endpoint = (
        f"{api}/oss/v2/buckets/wip.dm.prod/objects/folder%2Fmodel.ifc/signeds3download"
    )
    signed_url = "https://signed.example/model.ifc?secret=hidden"
    version_payload = _version_payload(immutable_version, "Tower.ifc", storage_urn)
    transport.add_json("GET", version_url, version_payload)
    transport.add_json("GET", signed_endpoint, {"url": signed_url})
    transport.downloads[signed_url] = b"ISO-10303-21;"

    client = FormaClient(StaticAccessToken("test-token"), transport=transport)
    source = FormaSource(project_id, requested_version, cache_dir=tmp_path)
    output = client.materialize_ifc(source)

    assert output.name == "Tower.ifc"
    assert output.read_bytes() == b"ISO-10303-21;"
    assert transport.requests[0][2]["Authorization"] == "Bearer test-token"
    provenance = json.loads((output.parent / "source.json").read_text("utf-8"))
    assert provenance["version_id"] == immutable_version

    transport.add_json("GET", version_url, version_payload)
    assert client.materialize_ifc(source) == output
    assert len(transport.download_requests) == 1


def test_rvt_translation_downloads_ifc_derivative_with_signed_cookies(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    transport = FakeTransport()
    project_id = "b.project"
    version_id = "urn:adsk.wipprod:fs.file:vf.rvt?version=3"
    encoded_urn = encode_derivative_urn(version_id)
    api = "https://developer.api.autodesk.com"
    version_url = (
        f"{api}/data/v1/projects/b.project/versions/"
        "urn%3Aadsk.wipprod%3Afs.file%3Avf.rvt%3Fversion%3D3"
    )
    job_url = f"{api}/modelderivative/v2/designdata/job"
    manifest_url = f"{api}/modelderivative/v2/designdata/{encoded_urn}/manifest"
    derivative_urn = "urn:adsk.viewing:fs.file:derivative/output/model.ifc"
    cookies_url = (
        f"{manifest_url}/{derivative_urn.replace(':', '%3A').replace('/', '%2F')}"
        "/signedcookies?useCdn=true"
    )
    cdn_url = "https://cdn.example/output/model.ifc"
    transport.add_json(
        "GET", version_url, _version_payload(version_id, "Campus.rvt", None)
    )
    transport.add_json("POST", job_url, {"result": "created"})
    transport.add_json("GET", manifest_url, {"status": "inprogress"})
    transport.add_json(
        "GET",
        manifest_url,
        {
            "status": "success",
            "derivatives": [
                {"children": [{"outputType": "ifc", "urn": derivative_urn}]}
            ],
        },
    )
    transport.add_json(
        "GET",
        cookies_url,
        {"url": cdn_url},
        headers=(
            ("Set-Cookie", "CloudFront-Policy=policy-value; Path=/; Secure"),
            ("Set-Cookie", "CloudFront-Key-Pair-Id=key-id; Path=/; Secure"),
            ("Set-Cookie", "CloudFront-Signature=signature-value; Path=/; Secure"),
        ),
    )
    expected_signed_url = (
        f"{cdn_url}?Policy=policy-value&Key-Pair-Id=key-id&Signature=signature-value"
    )
    transport.downloads[expected_signed_url] = b"translated-ifc"

    client = FormaClient(StaticAccessToken("test-token"), transport=transport)
    source = FormaSource(
        project_id,
        version_id,
        cache_dir=tmp_path,
        poll_interval_seconds=0,
    )
    output = client.materialize_ifc(source)

    assert output.name == "Campus.ifc"
    assert output.read_bytes() == b"translated-ifc"
    job_request = next(call for call in transport.requests if call[0] == "POST")
    job_payload = json.loads((job_request[3] or b"").decode("utf-8"))
    assert job_payload["input"]["urn"] == encoded_urn
    assert job_payload["output"]["formats"][0]["advanced"] == {
        "exportSettingName": "IFC4 Reference View"
    }
    assert parse_qs(urlsplit(transport.download_requests[0][0]).query) == {
        "Policy": ["policy-value"],
        "Key-Pair-Id": ["key-id"],
        "Signature": ["signature-value"],
    }
    assert "Submitting APS RVT-to-IFC translation: Campus.rvt" in caplog.text
    assert "APS translation status for Campus.rvt: inprogress" in caplog.text
    assert "APS translation complete; downloading IFC derivative" in caplog.text


def test_failed_rvt_translation_raises_clear_error(tmp_path: Path) -> None:
    transport = FakeTransport()
    project_id = "b.project"
    version_id = "urn:adsk.wipprod:fs.file:vf.rvt?version=3"
    encoded_urn = encode_derivative_urn(version_id)
    api = "https://developer.api.autodesk.com"
    version_url = (
        f"{api}/data/v1/projects/b.project/versions/"
        "urn%3Aadsk.wipprod%3Afs.file%3Avf.rvt%3Fversion%3D3"
    )
    transport.add_json(
        "GET", version_url, _version_payload(version_id, "Campus.rvt", None)
    )
    transport.add_json(
        "POST", f"{api}/modelderivative/v2/designdata/job", {"result": "created"}
    )
    transport.add_json(
        "GET",
        f"{api}/modelderivative/v2/designdata/{encoded_urn}/manifest",
        {"status": "failed"},
    )
    client = FormaClient(StaticAccessToken("test-token"), transport=transport)

    with pytest.raises(FormaTranslationError, match="status 'failed'"):
        client.materialize_ifc(FormaSource(project_id, version_id, cache_dir=tmp_path))


def test_client_credentials_token_is_cached() -> None:
    transport = FakeTransport()
    transport.add_json(
        "POST", APS_TOKEN_URL, {"access_token": "access", "expires_in": 3600}
    )
    provider = ClientCredentialsTokenProvider("client", "secret", transport)

    assert provider.get_token() == "access"
    assert provider.get_token() == "access"
    assert len(transport.requests) == 1
    method, url, headers, body = transport.requests[0]
    assert (method, url) == ("POST", APS_TOKEN_URL)
    assert headers["Authorization"].startswith("Basic ")
    assert body == (
        b"grant_type=client_credentials&scope=data%3Aread+data%3Awrite+bucket%3Aread"
    )


def test_helpers_and_cli_source_options() -> None:
    assert encode_derivative_urn("urn:test") == "dXJuOnRlc3Q"
    assert parse_storage_urn("urn:adsk.objects:os.object:bucket/folder/model.ifc") == (
        "bucket",
        "folder/model.ifc",
    )
    args = parse_args(
        [
            "--forma-project-id",
            "b.project",
            "--forma-version-id",
            "urn:version",
            "--forma-cache-dir",
            "cache",
        ]
    )
    assert args.forma_project_id == "b.project"
    assert args.forma_folder_id is None
    assert args.forma_version_id == "urn:version"
    assert args.forma_cache_dir == "cache"

    scoped_args = parse_args(
        [
            "--base-url",
            "https://acc.autodesk.com/docs/files/projects/project?folderUrn=folder",
            "--subfolder",
            "--depth",
            "1",
            "--all",
        ]
    )
    assert scoped_args.forma_project_url.startswith("https://acc.autodesk.com/")
    assert scoped_args.forma_subfolders is True
    assert scoped_args.forma_depth == 1
    assert scoped_args.process_all is True
    with pytest.raises(SystemExit):
        parse_args(["--depth", "-1"])


def test_acc_url_decodes_project_folder_and_item() -> None:
    locator = parse_acc_project_url(
        "https://acc.autodesk.com/docs/files/projects/"
        "ff88d3b0-04d4-4634-b365-b72da6964f55"
        "?folderUrn=urn%3Aadsk.wipprod%3Afs.folder%3Aco.S7o9"
        "&entityId=urn%3Aadsk.wipprod%3Adm.lineage%3AnStSr3"
    )

    assert locator.project_id == "b.ff88d3b0-04d4-4634-b365-b72da6964f55"
    assert locator.folder_id == "urn:adsk.wipprod:fs.folder:co.S7o9"
    assert locator.item_id == "urn:adsk.wipprod:dm.lineage:nStSr3"


def test_project_crawl_recurses_and_returns_tip_versions() -> None:
    transport = FakeTransport()
    api = "https://developer.api.autodesk.com"
    project_id = "b.project"
    hub_id = "b.hub"
    root_id = "urn:adsk.wipprod:fs.folder:co.root"
    architecture_id = "urn:adsk.wipprod:fs.folder:co.arch"
    transport.add_json(
        "GET",
        f"{api}/project/v1/hubs",
        {"data": [{"type": "hubs", "id": hub_id}]},
    )
    transport.add_json(
        "GET",
        f"{api}/project/v1/hubs/b.hub/projects",
        {"data": [{"type": "projects", "id": project_id}]},
    )
    transport.add_json(
        "GET",
        f"{api}/project/v1/hubs/b.hub/projects/b.project/topFolders",
        {
            "data": [
                {
                    "type": "folders",
                    "id": root_id,
                    "attributes": {"displayName": "Project Files"},
                }
            ]
        },
    )
    transport.add_json(
        "GET",
        f"{api}/data/v1/projects/b.project/folders/"
        "urn%3Aadsk.wipprod%3Afs.folder%3Aco.root/contents",
        {
            "data": [
                {
                    "type": "folders",
                    "id": architecture_id,
                    "attributes": {"displayName": "Architecture"},
                },
                {
                    "type": "items",
                    "id": "urn:adsk.wipprod:dm.lineage:notes",
                    "attributes": {"displayName": "notes.txt"},
                },
            ]
        },
    )
    transport.add_json(
        "GET",
        f"{api}/data/v1/projects/b.project/folders/"
        "urn%3Aadsk.wipprod%3Afs.folder%3Aco.arch/contents",
        {
            "data": [
                {
                    "type": "items",
                    "id": "urn:adsk.wipprod:dm.lineage:model",
                    "attributes": {"displayName": "Tower.rvt"},
                    "relationships": {
                        "tip": {
                            "data": {
                                "id": "urn:adsk.wipprod:fs.file:vf.model?version=4"
                            }
                        }
                    },
                    "links": {"webView": {"href": "https://acc.example/tower"}},
                },
                {
                    "type": "items",
                    "id": "urn:adsk.wipprod:dm.lineage:site",
                    "attributes": {
                        "displayName": "Site",
                        "extension": {"data": {"sourceFileName": "Site.ifc"}},
                    },
                    "relationships": {
                        "tip": {
                            "data": {"id": "urn:adsk.wipprod:fs.file:vf.site?version=2"}
                        }
                    },
                },
            ]
        },
    )

    client = FormaClient(StaticAccessToken("test-token"), transport=transport)
    files = client.list_project_files(project_id)

    assert [entry.relative_path for entry in files] == [
        "Project Files/Architecture/Site.ifc",
        "Project Files/Architecture/Tower.rvt",
    ]
    assert files[0].version_number == 2
    assert files[0].web_url.startswith("https://acc.autodesk.com/docs/files/projects/")
    assert files[1].web_url == "https://acc.example/tower"


def test_folder_scoped_crawl_only_walks_selected_subtree() -> None:
    transport = FakeTransport()
    api = "https://developer.api.autodesk.com"
    project_id = "b.project"
    architecture_id = "urn:adsk.wipprod:fs.folder:co.arch"
    interiors_id = "urn:adsk.wipprod:fs.folder:co.interiors"
    architecture_url = (
        f"{api}/data/v1/projects/b.project/folders/"
        "urn%3Aadsk.wipprod%3Afs.folder%3Aco.arch"
    )
    transport.add_json(
        "GET",
        architecture_url,
        {
            "data": {
                "type": "folders",
                "id": architecture_id,
                "attributes": {"displayName": "Architecture"},
            }
        },
    )
    transport.add_json(
        "GET",
        f"{architecture_url}/contents",
        {
            "data": [
                {
                    "type": "folders",
                    "id": interiors_id,
                    "attributes": {"displayName": "Interiors"},
                },
                {
                    "type": "items",
                    "id": "urn:adsk.wipprod:dm.lineage:shell",
                    "attributes": {"displayName": "Shell.rvt"},
                    "relationships": {
                        "tip": {
                            "data": {
                                "id": "urn:adsk.wipprod:fs.file:vf.shell?version=3"
                            }
                        }
                    },
                },
            ]
        },
    )
    interiors_url = (
        f"{api}/data/v1/projects/b.project/folders/"
        "urn%3Aadsk.wipprod%3Afs.folder%3Aco.interiors/contents"
    )
    transport.add_json(
        "GET",
        interiors_url,
        {
            "data": [
                {
                    "type": "folders",
                    "id": "urn:adsk.wipprod:fs.folder:co.details",
                    "attributes": {"displayName": "Details"},
                },
                {
                    "type": "items",
                    "id": "urn:adsk.wipprod:dm.lineage:fitout",
                    "attributes": {"displayName": "Fitout.ifc"},
                    "relationships": {
                        "tip": {
                            "data": {
                                "id": "urn:adsk.wipprod:fs.file:vf.fitout?version=6"
                            }
                        }
                    },
                },
            ]
        },
    )

    client = FormaClient(StaticAccessToken("test-token"), transport=transport)
    files = client.list_project_files(
        project_id,
        start_folder_id=architecture_id,
        max_depth=1,
    )

    assert [entry.relative_path for entry in files] == [
        "Architecture/Interiors/Fitout.ifc",
        "Architecture/Shell.rvt",
    ]
    requested_urls = [request[1] for request in transport.requests]
    assert requested_urls == [
        architecture_url,
        f"{architecture_url}/contents",
        interiors_url,
    ]


def test_selection_supports_ranges_all_and_case_insensitive_globs() -> None:
    files = [
        FormaProjectFile(
            project_id="b.project",
            item_id=f"item-{index}",
            version_id=f"urn:version:{index}?version={index}",
            display_name=name,
            folder_path=folder,
            folder_id="folder",
            web_url="https://acc.example/file",
        )
        for index, (folder, name) in enumerate(
            [
                ("Project Files/Architecture", "Tower.rvt"),
                ("Project Files/Civil", "Site.ifc"),
                ("Project Files/Archive", "Old.rvt"),
            ],
            start=1,
        )
    ]

    assert _parse_forma_selection("1,3-2", 3) == [0, 1, 2]
    assert _parse_forma_selection("all", 3) == [0, 1, 2]
    args = SimpleNamespace(forma_files=["*/CIVIL/*.IFC"], process_all=False)
    assert _select_forma_files(files, args) == [files[1]]


def test_file_url_resolves_tip_without_project_crawl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = FormaProjectFile(
        project_id="b.project",
        item_id="urn:adsk.wipprod:dm.lineage:model",
        version_id="urn:adsk.wipprod:fs.file:vf.model?version=8",
        display_name="Tower.rvt",
        folder_path="",
        folder_id="urn:folder",
        web_url="https://acc.example/tower",
    )

    class StubClient:
        def get_item_tip(
            self,
            project_id: str,
            item_id: str,
            *,
            folder_id: str | None = None,
        ) -> FormaProjectFile:
            assert project_id == "b.project"
            assert item_id == "urn:adsk.wipprod:dm.lineage:model"
            assert folder_id == "urn:folder"
            return entry

    client = StubClient()
    conversion_module = importlib.import_module("buildusd.conversion")
    monkeypatch.setattr(
        conversion_module,
        "FormaClient",
        SimpleNamespace(from_environment=lambda **kwargs: client),
    )
    args = SimpleNamespace(
        forma_project_id=None,
        forma_project_url=(
            "https://acc.autodesk.com/docs/files/projects/project"
            "?folderUrn=urn%3Afolder"
            "&entityId=urn%3Aadsk.wipprod%3Adm.lineage%3Amodel"
        ),
        forma_folder_id=None,
        forma_version_id=None,
        forma_browse=False,
        forma_files=[],
        process_all=False,
        forma_cache_dir=None,
        forma_ifc_export_setting="IFC4 Reference View",
        forma_force_translation=False,
        forma_translation_timeout_seconds=1800.0,
        forma_poll_interval_seconds=5.0,
        input_path="unused",
    )

    sources, resolved_client = _resolve_forma_cli_sources(args)

    assert resolved_client is client
    assert len(sources) == 1
    assert isinstance(sources[0], FormaSource)
    assert sources[0].version_id == entry.version_id


@pytest.mark.parametrize(
    ("include_subfolders", "depth", "expected_max_depth"),
    [
        (False, None, 0),
        (True, None, None),
        (False, 1, 1),
    ],
)
def test_folder_url_scopes_cli_discovery(
    monkeypatch: pytest.MonkeyPatch,
    include_subfolders: bool,
    depth: int | None,
    expected_max_depth: int | None,
) -> None:
    folder_id = "urn:adsk.wipprod:fs.folder:co.arch"
    entry = FormaProjectFile(
        project_id="b.project",
        item_id="urn:adsk.wipprod:dm.lineage:model",
        version_id="urn:adsk.wipprod:fs.file:vf.model?version=8",
        display_name="Tower.rvt",
        folder_path="Architecture",
        folder_id=folder_id,
        web_url="https://acc.example/tower",
    )

    class StubClient:
        def list_project_files(
            self,
            project_id: str,
            *,
            start_folder_id: str | None = None,
            max_depth: int | None = None,
        ) -> list[FormaProjectFile]:
            assert project_id == "b.project"
            assert start_folder_id == folder_id
            assert max_depth == expected_max_depth
            return [entry]

    client = StubClient()
    conversion_module = importlib.import_module("buildusd.conversion")
    monkeypatch.setattr(
        conversion_module,
        "FormaClient",
        SimpleNamespace(from_environment=lambda **kwargs: client),
    )
    args = SimpleNamespace(
        forma_project_id=None,
        forma_project_url=(
            "https://acc.autodesk.com/docs/files/projects/project"
            "?folderUrn=urn%3Aadsk.wipprod%3Afs.folder%3Aco.arch"
        ),
        forma_folder_id=None,
        forma_version_id=None,
        forma_browse=True,
        forma_subfolders=include_subfolders,
        forma_depth=depth,
        forma_files=[],
        process_all=True,
        forma_cache_dir=None,
        forma_ifc_export_setting="IFC4 Reference View",
        forma_force_translation=False,
        forma_translation_timeout_seconds=1800.0,
        forma_poll_interval_seconds=5.0,
        input_path="unused",
    )

    sources, resolved_client = _resolve_forma_cli_sources(args)

    assert resolved_client is client
    assert len(sources) == 1
    assert isinstance(sources[0], FormaSource)
    assert sources[0].version_id == entry.version_id


def test_public_api_forwards_forma_client(monkeypatch: pytest.MonkeyPatch) -> None:
    source = FormaSource("b.project", "urn:version")
    client = object()
    captured: dict[str, object] = {}

    def fake_convert(input_path: object, **kwargs: object) -> list:
        captured["input_path"] = input_path
        captured.update(kwargs)
        return []

    monkeypatch.setattr(api, "_convert", fake_convert)

    assert api.convert(api.ConversionSettings(source), forma_client=client) == []
    assert captured["input_path"] is source
    assert captured["forma_client"] is client
