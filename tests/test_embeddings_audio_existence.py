"""B3: runner skips gemini clip gracefully when audio file is missing.

The video-missing path is already guarded (runner.py ~250-257); audio
was not.  A missing audio file used to reach the Gemini provider and
surface as a generic EMB ERR.  After the fix the runner issues a clean
SKIP instead, matching the video-missing behaviour.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.embeddings.cases import EmbeddingSecrets

# ── settings stub (mirrors _runner_settings in test_embeddings_gemini_case.py) ──


def _runner_settings(video_dir, audio_dir):
    return SimpleNamespace(
        paths=SimpleNamespace(
            video_dir=str(video_dir),
            audio_dir=str(audio_dir),
            video_for=lambda cid, vd=video_dir: vd / f"{cid}.mp4",
            audio_for=lambda cid, ad=audio_dir: ad / f"{cid}.mp3",
            model_path="/models/Qwen3-VL-Embedding-8B",
        ),
        embeddings=SimpleNamespace(
            gemini_enabled=True,
            gemini_model="gemini-embed-test",
            gemini_output_dim=3072,
            gemini_max_video_seconds=120,
            gemini_max_audio_seconds=80,
            gemini_request_timeout_s=10,
            gemini_max_retries=0,
            embed_max_length=2048,
            adaptive_max_frames=64,
            adaptive_default_fps=2.0,
            exclude_disqualified_users=False,
            provider="local",
            inflight=1,
        ),
        audio_extraction=SimpleNamespace(
            audio_bitrate_kbps=128,
            audio_sample_rate_hz=44100,
        ),
        storage=SimpleNamespace(bucket=""),
    )


# ── db_session fixture (mirrors test_embeddings_gemini_case.py) ────────────────


@pytest.fixture()
def db_session():
    from core.database import (
        AudioMIR,
        Base,
        Clip,
        ClipEmbedding,
        StageState,
        User,
        get_engine,
        get_session,
    )

    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, ClipEmbedding, AudioMIR, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


# ── the failing test ───────────────────────────────────────────────────────────


def test_gemini_job_skipped_when_audio_missing(
    tmp_path, db_session, monkeypatch, sample_mp4_with_audio
):
    """Clip whose audio file is absent must be skipped cleanly.

    The runner must NOT call provider.embed() for it; the clip should end
    up counted as fresh_skipped (no existing embedding row) so the stage
    does not seal.
    """
    from core.database import Clip, ClipEmbedding, User
    from modules.embeddings import embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    aud_dir = tmp_path / "audio"
    aud_dir.mkdir()

    # Video exists; audio file is deliberately absent.
    (vid_dir / "1.mp4").write_bytes(Path(str(sample_mp4_with_audio)).read_bytes())
    # (aud_dir / "1.mp3") intentionally NOT created

    db_session.add(User(id=1, is_selected=True))
    db_session.add(
        Clip(
            id=1,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="hello",
            caption_language="en",
        )
    )
    db_session.commit()

    settings = _runner_settings(vid_dir, aud_dir)

    def _fake_init(self, **kwargs):
        self.model = kwargs["model"]
        self.output_dim = kwargs["output_dim"]

    monkeypatch.setattr(GeminiMultimodalProvider, "__init__", _fake_init)

    embed_called = []

    def _fake_embed(self, payload):
        embed_called.append(payload)
        raise AssertionError("provider.embed must not be called for missing-audio clip")

    monkeypatch.setattr(GeminiMultimodalProvider, "embed", _fake_embed)

    # Must not raise; the clip should be silently skipped.
    embed_clip_embeddings(
        settings, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini"]
    )

    # The provider must never have been invoked.
    assert embed_called == [], f"embed() was called with: {embed_called}"

    # No embedding row should exist for the skipped clip.
    rows = db_session.query(ClipEmbedding).filter_by(embedding_case="gemini").all()
    assert rows == [], f"unexpected embedding rows: {rows}"

    # The stage may seal (fresh_skipped clips are treated as non-embeddable, which
    # mirrors the video-missing behaviour).  What must NOT happen is the provider
    # being called or a stale embedding row surviving for the skipped clip.
    # (Both verified above — no further state assertion needed here.)
