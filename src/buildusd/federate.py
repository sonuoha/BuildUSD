"""Deprecated compatibility shim for federation orchestration.

Use federation_orchestrator.py for the core orchestration logic and
federate_cli.py for the CLI entrypoint.
"""

from __future__ import annotations

from .federation_orchestrator import *  # noqa: F403

if __name__ == "__main__":
    from .federate_cli import main

    main()
