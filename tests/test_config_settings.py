"""Defaults + validation for the small per-section Pydantic Settings models.

The TOML loader + composite Settings shape is covered by
``test_config_loader.py``; path helpers by ``test_config_paths.py``. This
file covers the narrow per-section knobs (embeddings batch size, search
settings) so we don't pay for one tiny file per model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import EmbeddingsSettings, SearchSettings


def _embeddings(**overrides) -> EmbeddingsSettings:
    base = dict(
        exclude_disqualified_users=True,
        embed_max_length=32768,
        adaptive_max_frames=96,
        adaptive_default_fps=2.0,
    )
    base.update(overrides)
    return EmbeddingsSettings(**base)


def test_embeddings_batch_size_default():
    assert _embeddings().embed_batch_size == 1


def test_embeddings_batch_size_must_be_positive():
    with pytest.raises(ValidationError):
        _embeddings(embed_batch_size=0)


def _search(**overrides) -> SearchSettings:
    base = dict(
        umap_n_components=[20],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[15],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    base.update(overrides)
    return SearchSettings(**base)


def test_search_settings_accepts_hdbscan_min_samples():
    s = _search(hdbscan_min_samples=[5, 10, 15])
    assert s.hdbscan_min_samples == [5, 10, 15]


def test_search_settings_hdbscan_min_samples_defaults_to_empty():
    assert _search().hdbscan_min_samples == []
