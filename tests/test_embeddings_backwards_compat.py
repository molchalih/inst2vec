"""Existing sealed ClipEmbedding rows survive a re-run: no wipe, no re-embed.

Guards the invariant that retiring the remote provider did not change
``case_config_identity`` (frozen factory ``__name__`` + recipe_version).
If either drifted, every sealed case would re-hash and the ~500 existing
embeddings would be wiped/recomputed on the next run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from core import fingerprint as fp
from core.database import Clip, ClipEmbedding, User, get_session
from core.pipeline import Stage
from modules.embeddings import runner
from modules.embeddings.cases import CASE_REGISTRY, case_config_identity
from modules.embeddings.state import per_clip_source_hashes_and_aggregate


def _settings():
    # load_runtime_config() reads the non-secret half from config.toml and the
    # secret half from env. We only need the Settings half (case_config_identity
    # + paths), so supply placeholder credentials for the secret reads. The
    # values are never used — we never construct a real provider.
    import os

    from core.config import load_runtime_config

    os.environ.setdefault("HIKER_API_KEY", "test")
    os.environ.setdefault("HUGGINGFACE_TOKEN", "test")
    s, _ = load_runtime_config()
    return s


def test_frozen_factory_names():
    # Hard guard: renaming these wipes every existing embedding.
    assert getattr(CASE_REGISTRY["video"].provider_factory, "__name__", None) == (
        "qwen_provider_video"
    )
    assert getattr(CASE_REGISTRY["audio"].provider_factory, "__name__", None) == (
        "qwen_provider_text"
    )


def test_existing_sealed_case_is_skipped():
    settings = _settings()
    session = get_session()
    # Seed one eligible user + clip that is selected+downloaded so it is a
    # candidate for clip_used_in_analysis() (is_selected + is_downloaded).
    user = User(id=9001, is_eligible=True)
    clip = Clip(
        id=9001,
        user_id=9001,
        is_selected=True,
        is_downloaded=True,
    )
    session.merge(user)
    session.merge(clip)
    session.commit()

    spec = CASE_REGISTRY["audio"]  # text-only: no media file needed to seal

    # Mirror runner._compute_fingerprint_and_per_clip EXACTLY so the sealed
    # fingerprint matches what _run_case recomputes (otherwise is_stale -> True
    # and the test would re-embed for the wrong reason).
    candidate_ids = sorted([clip.id])
    per_clip, dep_agg = per_clip_source_hashes_and_aggregate(
        session, spec.name, candidate_ids, settings=settings
    )
    current = fp.Fingerprint(
        data=fp.hash_rows((cid,) for cid in candidate_ids),
        config=fp.hash_text(case_config_identity(spec, settings)),
        dependency=dep_agg,
    )

    blob = np.zeros(4, dtype=np.float32).tobytes()
    session.merge(
        ClipEmbedding(
            clip_id=clip.id,
            embedding_case=spec.name,
            embedding=blob,
            source_hash=per_clip[clip.id],
        )
    )
    fp.mark_complete(session, Stage.CLIP_EMBEDDINGS, spec.name, current)
    session.commit()

    # If the provider is ever constructed, the stage tried to re-embed ->
    # the config hash drifted. Make that fail loudly via a boom factory.
    # dataclasses.replace preserves name/recipe_version, and we copy the
    # factory __name__ so case_config_identity is byte-for-byte unchanged
    # (it hashes provider_factory.__name__) — only the body raises.
    def _boom(*a, **k):
        raise AssertionError("re-embedded!")

    _boom.__name__ = getattr(spec.provider_factory, "__name__", "qwen_provider_text")
    boom_spec = dataclasses.replace(spec, provider_factory=_boom)

    runner._run_case(settings, None, boom_spec)

    rows = (
        session.query(ClipEmbedding)
        .filter_by(clip_id=clip.id, embedding_case=spec.name)
        .all()
    )
    assert len(rows) == 1
    assert rows[0].source_hash == per_clip[clip.id]
    session.close()
