# Contributing to BuildUSD

Thanks for your interest in improving BuildUSD! This document outlines the basics to get started.

## Ground rules
- Be respectful and collaborative. See CODE_OF_CONDUCT.md.
- Prefer small, focused PRs with clear scope.
- Follow SemVer: breaking changes must bump MAJOR.

## How to contribute
1. Fork and create a topic branch off `main`.
2. Keep changes minimal per PR; include tests where possible.
3. Run the lint/tests before submitting (see below).
4. Open a PR with a clear description and checklist of what changed and how it was tested.

## Development setup
- Python 3.11+ recommended.
- Create a virtual environment and install dev deps:
  - `pip install -e .[dev]` (or `uv sync` if you use uv).
- Optional runtimes: pxr USD (offline) or Omniverse Kit; ensure one runtime is available for integration tests.

## Tests and quality
- Run lint/format/type checks: `ruff check .`, `black --check .`, `mypy .` (when configured).
- Run unit/integration tests: `pytest` (when present).
- For IFC → USD integration, use the sample IFC corpus and compare against golden USD outputs if available.

## Commit style
- Use clear commit messages; reference issues when applicable.
- Avoid committing generated artifacts, large binaries, or local logs.

## Opening issues
- Include reproduction steps, logs, environment (OS, Python, USD runtime), and minimal IFC samples when possible.

## Security
- Report security issues privately (see SECURITY.md) before opening a public issue.
