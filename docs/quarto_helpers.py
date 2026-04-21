"""Helpers for rendering project outputs inside Quarto documents."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from IPython.display import Markdown
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url


def _ensure_project_root_on_path() -> None:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_quarto_database_url(raw_url: str) -> str:
    url = make_url(raw_url)
    if url.get_backend_name() != "sqlite":
        return raw_url

    database = url.database
    if not database or database == ":memory:" or database.startswith("file:"):
        return raw_url

    if Path(database).is_absolute():
        return raw_url

    return str(url.set(database=str((_project_root() / database).resolve())))


@lru_cache(maxsize=1)
def _get_default_engine() -> Engine:
    load_dotenv(_project_root() / ".env")
    return create_engine(_resolve_quarto_database_url(os.environ["DATABASE_URL"]))


def render_clustering_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.cluster_results_all import summarize_all_to_markdown
    from modules.cluster_results import DEFAULT_CASES

    if eng is None:
        eng = _get_default_engine()

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
        eng = _get_default_engine()

    return Markdown(summarize_to_markdown(eng, case))


def render_best_cluster_run(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.cluster_results_best import best_runs_all_to_markdown
    from modules.cluster_results import DEFAULT_CASES

    if eng is None:
        eng = _get_default_engine()

    return Markdown(best_runs_all_to_markdown(eng, cases=DEFAULT_CASES))