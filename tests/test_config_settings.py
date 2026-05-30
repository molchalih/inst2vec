"""Defaults + validation for the small per-section Pydantic Settings models.

The TOML loader + composite Settings shape is covered by
``test_config_loader.py``; path helpers by ``test_config_paths.py``. This
file covers the narrow per-section knobs (distributed embeddings,
search settings, runpod) so we don't pay for one tiny file per model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import EmbeddingsSettings, RunpodSettings, SearchSettings

# ── Embeddings: distributed/runtime knobs ────────────────────────────────


def _embeddings(**overrides) -> EmbeddingsSettings:
    base = dict(
        exclude_disqualified_users=True,
        embed_max_length=32768,
        adaptive_max_frames=96,
        adaptive_default_fps=2.0,
    )
    base.update(overrides)
    return EmbeddingsSettings(**base)


def test_embeddings_distributed_defaults():
    s = _embeddings()
    assert s.coordinator_bind_host == "0.0.0.0"
    assert s.coordinator_bind_port > 0
    assert s.lease_ttl_s == 600
    assert s.max_attempts == 3
    assert s.worker_request_timeout_s == 120
    assert s.worker_max_retries == 3
    assert s.pod_connect_timeout_s == 600


def test_embeddings_lease_ttl_must_be_positive():
    with pytest.raises(ValidationError):
        _embeddings(lease_ttl_s=0)


def test_embeddings_pod_idle_ttl_default():
    assert _embeddings().pod_idle_ttl_s == 300


# ── RunpodSettings: GPU + volume defaults ────────────────────────────────


def test_runpod_settings_defaults():
    rp = RunpodSettings()
    assert rp.volume_mount_path == "/runpod-volume"
    assert rp.reconcile_path == ".runpod_fleet.json"
    assert rp.pod_video_root == "/runpod-volume/videos"
    assert rp.pod_model_path == "/runpod-volume/models/Qwen3-VL-Embedding-8B"
    assert rp.template_id == ""  # empty -> deploy from [runpod].image instead
    assert rp.gpu_type_id == ""  # empty -> auto-fetch GPUs in the volume's DC
    assert rp.gpu_max_price_hr == 0.80
    assert rp.gpu_min_vram_gb == 24
    assert rp.gpu_min_ram_gb == 30


# ── SearchSettings: hdbscan_min_samples axis ─────────────────────────────


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
