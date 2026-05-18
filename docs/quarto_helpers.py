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

    from modules.visualization.tables.cluster_results_all import (
        summarize_all_to_markdown,
    )

    if eng is None:
        eng = _get_default_engine()

    return Markdown(
        summarize_all_to_markdown(
            eng,
            cases=_quarto_cases(),
        )
    )


def render_clustering_summary_by_case(case: str, *, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from modules.visualization.tables.cluster_results_all import summarize_to_markdown

    if eng is None:
        eng = _get_default_engine()

    return Markdown(summarize_to_markdown(eng, case))


def _load_config_toml() -> dict:
    import tomllib

    config_path = _project_root() / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _load_plateau_drop_threshold() -> float:
    """Read validation.plateau_drop_threshold from config.toml.

    Avoids load_runtime_config() so docs/notebooks can render without
    requiring the full set of runtime secrets in the environment.
    """
    return float(_load_config_toml()["validation"]["plateau_drop_threshold"])


def _quarto_cases() -> tuple[str, ...]:
    """Embedding cases to render in Quarto, matching pipeline default_cases.

    Mirrors modules.embeddings.cases.default_cases without importing it, to
    keep this helper free of pipeline runtime imports during notebook render.
    """
    cases = ("video", "sandwich", "audio")
    if _load_config_toml().get("embeddings", {}).get("gemini_enabled", False):
        return (*cases, "gemini_mm")
    return cases


def render_best_cluster_run(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from modules.visualization.tables.cluster_results_best import (
        best_runs_all_to_markdown,
    )

    if eng is None:
        eng = _get_default_engine()

    threshold = _load_plateau_drop_threshold()

    return Markdown(
        best_runs_all_to_markdown(eng, threshold=threshold, cases=_quarto_cases())
    )


# ── USER-related ─────────────────────────────────────────────────────────────────


def render_users_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

    if eng is None:
        eng = _get_default_engine()

    return Markdown(users_summary_to_markdown(eng))


def get_users_summary_cells(*, eng=None) -> dict[str, str]:
    _ensure_project_root_on_path()

    from modules.visualization.tables.dataset_summary_users import (
        get_users_summary_cells,
    )

    if eng is None:
        eng = _get_default_engine()

    return get_users_summary_cells(eng)


# ── CLIP-related ──────────────────────────────────────────────────────────────────


def render_clips_summary(*, eng=None) -> Markdown:
    _ensure_project_root_on_path()

    from modules.visualization.tables.dataset_summary_clips import (
        clips_summary_to_markdown,
    )

    if eng is None:
        eng = _get_default_engine()

    return Markdown(clips_summary_to_markdown(eng))


def get_clips_summary_cells(*, eng=None) -> dict[str, str]:
    _ensure_project_root_on_path()

    from modules.visualization.tables.dataset_summary_clips import (
        get_clips_summary_cells,
    )

    if eng is None:
        eng = _get_default_engine()

    return get_clips_summary_cells(eng)


def render_spotify_feature_metrics() -> Markdown:
    _ensure_project_root_on_path()

    from modules.visualization.tables.spotify_feature_metrics import (
        spotify_feature_metrics_to_markdown,
    )

    return Markdown(spotify_feature_metrics_to_markdown())


def render_cluster_plot_by_case(
    case: str,
    *,
    eng=None,
    title_label: str | None = None,
):
    _ensure_project_root_on_path()

    from modules.visualization.plots.cluster_case_plots import (
        cluster_plot_figure_for_case,
    )

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
