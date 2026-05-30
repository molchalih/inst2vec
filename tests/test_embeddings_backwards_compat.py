"""Frozen factory-name guard for sealed ClipEmbedding rows.

If ``CASE_REGISTRY`` provider factories are renamed, every existing sealed
case re-hashes its ``case_config_identity`` and the ~500 stored embeddings
get wiped and recomputed on the next run. End-to-end skip-on-rerun is
covered by ``test_clip_embeddings_idempotence.test_rerun_identical_inputs_is_noop``.
"""

from __future__ import annotations

from modules.embeddings.cases import CASE_REGISTRY


def test_frozen_factory_names():
    assert getattr(CASE_REGISTRY["video"].provider_factory, "__name__", None) == (
        "qwen_provider_video"
    )
    assert getattr(CASE_REGISTRY["audio"].provider_factory, "__name__", None) == (
        "qwen_provider_text"
    )
