"""Helpers for rendering project outputs inside Quarto documents."""
from __future__ import annotations

import sys
from pathlib import Path

from IPython.display import Markdown


def _ensure_project_root_on_path() -> None:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def render_clustering_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.cluster_results_all import summarize_all_to_markdown
    from modules.cluster_results import DEFAULT_CASES

    if eng is None:
        from modules.database import engine as default_engine

        eng = default_engine

    return Markdown(
        summarize_all_to_markdown(
            eng,
            cases=DEFAULT_CASES,
        )
    )

def render_clustering_summary_by_case(case: str, *, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.cluster_results_all import summarize_to_markdown

    if eng is None:
        from modules.database import engine as default_engine

        eng = default_engine

    return Markdown(summarize_to_markdown(eng, case))


def render_best_cluster_run(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.cluster_results_best import best_runs_all_to_markdown
    from modules.cluster_results import DEFAULT_CASES

    if eng is None:
        from modules.database import engine as default_engine

        eng = default_engine

    return Markdown(best_runs_all_to_markdown(eng, cases=DEFAULT_CASES))