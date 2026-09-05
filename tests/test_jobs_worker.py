from __future__ import annotations

import json
from pathlib import Path

from buildusd.conversion import ConversionResult
from buildusd.jobs import (
    TargetedDetailJob,
    TargetedDetailTarget,
    read_job,
    run_job,
    write_job,
)
from buildusd.worker import run_worker_once


def _fake_result(tmp_path: Path) -> ConversionResult:
    stage = tmp_path / "detail.usda"
    graph = tmp_path / "graphs" / "detail.semantic_graph.json"
    return ConversionResult(
        ifc_path=tmp_path / "model.ifc",
        stage_path=stage,
        master_stage_path=stage,
        layers={
            "stage": str(stage),
            "semantic_graph": str(graph),
            "instances": str(tmp_path / "instances.usda"),
        },
        projected_crs="EPSG:7855",
        geodetic_crs="EPSG:4326",
        geodetic_coordinates=None,
        geospatial_mode="auto",
        counts={"instances": 1},
        plan=None,
        revision="test",
    )


def test_targeted_detail_job_builds_object_scope_conversion_options(
    tmp_path: Path,
) -> None:
    source_ifc = tmp_path / "model.ifc"
    source_ifc.write_text("IFC content", encoding="utf-8")
    job = TargetedDetailJob.from_payload(
        {
            "schema": "buildusd.job.v1",
            "job_type": "targeted_detail",
            "job_id": "job-001",
            "source_ifc": str(source_ifc),
            "output_dir": str(tmp_path / "out"),
            "detail_engine": "semantic",
            "targets": [
                {"guid": "abc123", "source_usd_prim": "/World/A"},
                {"step_id": 42},
                "abc123",
            ],
        }
    )

    options = job.to_conversion_options()

    assert job.detail_objects == ("abc123", 42)
    assert options.detail_mode is True
    assert options.detail_scope == "object"
    assert options.detail_objects == ("abc123", 42)
    assert options.detail_engine == "semantic"
    assert len(job.cache_key) == 64
    assert (
        job.cache_key
        == TargetedDetailJob.from_payload(
            {**job.to_payload(), "job_id": "different"}
        ).cache_key
    )


def test_run_job_writes_targeted_detail_manifest(tmp_path: Path) -> None:
    captured = {}
    job = TargetedDetailJob(
        job_id="job-002",
        source_ifc=tmp_path / "model.ifc",
        output_dir=tmp_path / "out",
        targets=(TargetedDetailTarget(guid="guid-1"),),
    )

    def fake_convert(settings, *, options):
        captured["settings"] = settings
        captured["options"] = options
        return [_fake_result(tmp_path)]

    result = run_job(job, converter=fake_convert)

    assert result.status == "completed"
    assert captured["settings"].input_path == tmp_path / "model.ifc"
    assert captured["options"].detail_scope == "object"
    manifest_path = Path(str(result.manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "buildusd.targeted_detail_manifest.v1"
    assert manifest["cache_key"] == result.cache_key
    assert manifest["contract"]["source_layers_are_not_mutated"] is True
    assert manifest["targets"][0]["guid"] == "guid-1"
    assert manifest["conversion_results"][0]["stage_path"].endswith("detail.usda")
    assert result.cache_hit is False


def test_run_job_reuses_completed_cache(tmp_path: Path) -> None:
    calls = {"count": 0}
    source_ifc = tmp_path / "model.ifc"
    source_ifc.write_text("IFC content", encoding="utf-8")
    job = TargetedDetailJob(
        job_id="first-job",
        source_ifc=source_ifc,
        output_dir=tmp_path / "out",
        targets=(TargetedDetailTarget(guid="guid-1"),),
    )

    def fake_convert(settings, *, options):
        calls["count"] += 1
        return [_fake_result(tmp_path)]

    first = run_job(job, converter=fake_convert)
    second = run_job(
        TargetedDetailJob(
            job_id="second-job",
            source_ifc=source_ifc,
            output_dir=tmp_path / "out",
            targets=(TargetedDetailTarget(guid="guid-1"),),
        ),
        converter=fake_convert,
    )

    assert calls["count"] == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.cache_key == first.cache_key
    assert Path(str(second.manifest_path)) == Path(str(first.manifest_path))
    assert second.artifacts["cache_record"].endswith("cache.json")


def test_file_backed_worker_claims_jobs_and_writes_results(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    pending = queue / "pending"
    pending.mkdir(parents=True)
    job = TargetedDetailJob(
        job_id="job-003",
        source_ifc=tmp_path / "model.ifc",
        output_dir=tmp_path / "out",
        targets=(TargetedDetailTarget(step_id=123),),
    )
    write_job(pending / "job-003.json", job)

    def fake_convert(settings, *, options):
        return [_fake_result(tmp_path)]

    results = run_worker_once(queue, converter=fake_convert)

    assert len(results) == 1
    assert results[0].status == "completed"
    assert not (pending / "job-003.json").exists()
    assert (queue / "completed" / "job-003.json").exists()
    result_path = queue / "results" / "job-003.result.json"
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload["schema"] == "buildusd.job_result.v1"
    assert result_payload["status"] == "completed"
    assert result_payload["cache_hit"] is False
    assert result_payload["cache_key"]
    assert Path(result_payload["manifest_path"]).exists()
    assert read_job(queue / "completed" / "job-003.json").job_id == "job-003"
