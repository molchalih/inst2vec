"""Visualization stage public API.

Two halves are exposed independently so the JSON exporter can be called
from a future API endpoint without re-running the DB stage:

  - run                          → pipeline stage (DB write + JSON export)
  - run_visualization            → DB write only (fingerprint-gated)
  - export_visualization_json    → JSON write only (reads DB)
"""

from core.config import Secrets, Settings
from core.log import stage
from modules.visualization.export import export_visualization_json
from modules.visualization.pipeline import run_visualization


@stage("visualization")
def run(settings: Settings, secrets: Secrets, cases: tuple[str, ...]) -> None:
    """Per-case fingerprint-gated DB build of the viewer layouts.

    The static-JSON export is intentionally OFF: the frontend is served from
    the serving DB via the atlas API (offload reads ``build_case_payloads``
    straight from this DB), so the pipeline no longer writes the on-disk JSON
    tree. ``export_visualization_json`` stays exposed for manual/local use —
    re-add the call here to restore in-pipeline JSON output.
    """
    run_visualization(cases=cases)


__all__ = [
    "export_visualization_json",
    "run",
    "run_visualization",
]
