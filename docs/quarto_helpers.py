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


# ── CLUSTERING-related ───────────────────────────────────────────────────────────


def render_clustering_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.cluster_results_all import summarize_all_to_markdown
    from modules.clustering import DEFAULT_CASES

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
    from modules.clustering import DEFAULT_CASES

    if eng is None:
        eng = _get_default_engine()

    return Markdown(best_runs_all_to_markdown(eng, cases=DEFAULT_CASES))


# ── USER-related ─────────────────────────────────────────────────────────────────


def render_users_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.dataset_summary_users import users_summary_to_markdown

    if eng is None:
        eng = _get_default_engine()

    return Markdown(users_summary_to_markdown(eng))


def get_users_summary_cells(*, eng=None) -> dict[str, str]:
    _ensure_project_root_on_path()

    from generators.dataset_summary_users import get_users_summary_cells

    if eng is None:
        eng = _get_default_engine()

    return get_users_summary_cells(eng)


# ── CLIP-related ──────────────────────────────────────────────────────────────────


def render_clips_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from generators.dataset_summary_clips import clips_summary_to_markdown

    if eng is None:
        eng = _get_default_engine()

    return Markdown(clips_summary_to_markdown(eng))


def get_clips_summary_cells(*, eng=None) -> dict[str, str]:
    _ensure_project_root_on_path()

    from generators.dataset_summary_clips import get_clips_summary_cells

    if eng is None:
        eng = _get_default_engine()

    return get_clips_summary_cells(eng)


def render_spotify_feature_metrics() -> Markdown:
    _ensure_project_root_on_path()

    from generators.spotify_feature_metrics import spotify_feature_metrics_to_markdown

    return Markdown(spotify_feature_metrics_to_markdown())


def render_cluster_plot_by_case(
    case: str,
    *,
    eng=None,
    title_label: str | None = None,
):
    _ensure_project_root_on_path()

    from generators.cluster_case_plots import cluster_plot_figure_for_case

    if eng is None:
        eng = _get_default_engine()

    return cluster_plot_figure_for_case(eng, case, title_label=title_label)


def render_audio_cluster_plot(*, eng=None):
    return render_cluster_plot_by_case("audio", eng=eng)


def render_video_cluster_plot(*, eng=None):
    return render_cluster_plot_by_case("video", eng=eng)


def render_multimodal_cluster_plot(*, eng=None):
    return render_cluster_plot_by_case(
        "sandwich",
        eng=eng,
        title_label="multimodal",
    )
