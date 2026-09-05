"""File-backed BuildUSD worker CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Sequence

from .jobs import BuildUSDJobResult, read_job, run_job


LOG = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="buildusd.worker",
        description="Run file-backed BuildUSD enrichment jobs.",
    )
    parser.add_argument(
        "--jobs",
        required=True,
        help="Queue directory. If a 'pending' child exists it is used; otherwise JSON files in this directory are treated as pending jobs.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Directory for result JSON files. Defaults to '<jobs>/results'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of pending jobs to run in this invocation.",
    )
    parser.add_argument(
        "--fail-on-job-error",
        action="store_true",
        help="Return a non-zero process code if any processed job fails.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args(argv)


def run_worker_once(
    jobs_dir: str | Path,
    *,
    results_dir: str | Path | None = None,
    limit: int = 1,
    converter: Callable | None = None,
) -> list[BuildUSDJobResult]:
    """Claim and run pending jobs from a local file-backed queue."""

    root = Path(jobs_dir)
    pending_dir = root / "pending" if (root / "pending").is_dir() else root
    running_dir = root / "running"
    completed_dir = root / "completed"
    failed_dir = root / "failed"
    results_root = Path(results_dir) if results_dir is not None else root / "results"
    for directory in (running_dir, completed_dir, failed_dir, results_root):
        directory.mkdir(parents=True, exist_ok=True)

    outcomes: list[BuildUSDJobResult] = []
    for job_path in _pending_job_files(pending_dir):
        if len(outcomes) >= max(0, limit):
            break
        running_path = _claim_job(job_path, running_dir)
        if running_path is None:
            continue
        try:
            job = read_job(running_path)
            result = run_job(job, converter=converter)
        except Exception as exc:
            LOG.exception("Failed to read or run job %s", running_path)
            result = BuildUSDJobResult(
                job_id=running_path.stem.replace(".running", ""),
                job_type="targeted_detail",
                status="failed",
                error=str(exc),
            )
        _write_result(result, results_root)
        _archive_job(
            running_path, completed_dir if result.status == "completed" else failed_dir
        )
        outcomes.append(result)
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    results = run_worker_once(args.jobs, results_dir=args.results, limit=args.limit)
    if not results:
        LOG.info("No pending BuildUSD jobs found in %s.", args.jobs)
        return 0
    failed = [result for result in results if result.status == "failed"]
    LOG.info("Processed %d BuildUSD job(s), %d failed.", len(results), len(failed))
    return 1 if failed and args.fail_on_job_error else 0


def _pending_job_files(pending_dir: Path) -> list[Path]:
    if not pending_dir.exists():
        return []
    files = []
    for path in sorted(pending_dir.glob("*.json")):
        name = path.name.lower()
        if name.endswith(".result.json") or name.endswith(".running.json"):
            continue
        files.append(path)
    return files


def _claim_job(path: Path, running_dir: Path) -> Path | None:
    running_dir.mkdir(parents=True, exist_ok=True)
    running_path = running_dir / f"{path.stem}.running.json"
    try:
        path.replace(running_path)
    except FileNotFoundError:
        return None
    return running_path


def _write_result(result: BuildUSDJobResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.job_id}.result.json"
    path.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _archive_job(path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / path.name.replace(".running.json", ".json")
    path.replace(archive_path)


if __name__ == "__main__":
    raise SystemExit(main())
