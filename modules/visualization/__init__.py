"""Visualization stage public API.

Two halves are exposed independently so the JSON exporter can be called
from a future API endpoint without re-running the DB stage:

  - run                          → pipeline stage (DB write + JSON export)
  - run_visualization            → DB write only (fingerprint-gated)
  - export_visualization_json    → JSON write only (reads DB)
"""

from core.config import Secrets, Settings
from modules.visualization.export import export_visualization_json
from modules.visualization.pipeline import run_visualization


def run(settings: Settings, secrets: Secrets, cases: tuple[str, ...]) -> None:
    """Per-case fingerprint-gated DB build, followed by an unconditional
    JSON export of the cases marked exposed to the viewer."""
    run_visualization(cases=cases)
    export_visualization_json(settings=settings, cases=cases)


__all__ = [
    "export_visualization_json",
    "run",
    "run_visualization",
]
