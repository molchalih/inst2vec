"""Tests for gemini_mm case gating and explicit-request rejection."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    User,
    get_engine,
    get_session,
)
from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
from modules.embeddings.cases import (
    EmbeddingCaseSpec,
    default_cases,
)


def _stub_settings(gemini_enabled: bool):
    """Create a minimal settings stub for gating tests."""
    return SimpleNamespace(embeddings=SimpleNamespace(gemini_enabled=gemini_enabled))


def test_default_cases_excludes_gemini_when_disabled():
    """default_cases should not include gemini_mm when gemini_enabled=False."""
    assert "gemini_mm" not in default_cases(_stub_settings(gemini_enabled=False))


def test_default_cases_includes_gemini_when_enabled():
    """default_cases should include gemini_mm when gemini_enabled=True."""
    assert "gemini_mm" in default_cases(_stub_settings(gemini_enabled=True))


def test_explicit_gemini_request_raises_when_disabled():
    """Requesting gemini_mm explicitly should raise when gemini_enabled=False."""
    settings = _stub_settings(gemini_enabled=False)
    with pytest.raises(RuntimeError, match="gemini_enabled"):
        embed_clip_embeddings(settings, cases=["gemini_mm"])


# ── test setup for config drift test ──────────────────────────────────────────


class _TorchLikeArray:
    """Minimal duck-type for a torch tensor, good enough for to_bytes()."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr.astype(np.float32)

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._arr

    def __getitem__(self, idx):
        return _TorchLikeArray(self._arr[idx])


@dataclass
class _FakeGeminiProvider:
    """Fake Gemini provider that respects output_dim."""

    output_dim: int

    def embed(self, payload: dict) -> _TorchLikeArray:
        """Return a fake embedding with the configured output_dim.

        Use output_dim as seed so different dims produce different vectors.
        """
        rng = np.random.default_rng(self.output_dim)
        arr = rng.standard_normal((1, self.output_dim)).astype(np.float32)
        return _TorchLikeArray(arr)


def _fake_gemini_factory(settings, _secrets=None):
    """Factory for fake Gemini provider that reads output_dim from settings."""
    output_dim = getattr(settings.embeddings, "gemini_output_dim", 3072)
    return _FakeGeminiProvider(output_dim=output_dim)


@dataclass
class _PathsStub:
    video_dir: str
    audio_dir: str
    model_path: str = "/fake/qwen"


@dataclass
class _EmbeddingsStub:
    exclude_disqualified_users: bool = True
    embed_max_length: int = 1024
    adaptive_max_frames: int = 8
    adaptive_default_fps: float = 1.0
    gemini_enabled: bool = True
    gemini_output_dim: int = 3072
    inflight: int = 1
    provider: str = "local"


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub


def _runner_settings(tmp_path, vid_dir, aud_dir) -> _SettingsStub:
    """Create a fresh settings stub with independent paths."""
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(vid_dir), audio_dir=str(aud_dir)),
        embeddings=_EmbeddingsStub(),
    )


def test_config_drift_wipes_case(tmp_path, monkeypatch):
    """When gemini_output_dim drifts, prior row is wiped and re-embedded.

    Phase (a): baseline run with output_dim=3072 → produces 1 row, 1 embed call.
    Phase (b): change output_dim to 768 → triggers config_hash drift → wipes row →
               re-embeds → produces 1 new row, 1 embed call.
    """
    # Setup: create DB
    Base.metadata.create_all(get_engine())
    db_session = get_session()
    try:
        # Create dirs and files
        vid_dir = tmp_path / "videos"
        aud_dir = tmp_path / "audio"
        vid_dir.mkdir()
        aud_dir.mkdir()
        (vid_dir / "1.mp4").write_bytes(b"\x00")
        (aud_dir / "1.mp3").write_bytes(b"\x00")

        # Seed DB with user + clip
        db_session.add(User(id=1, is_selected=True, is_eligible=True))
        db_session.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_text="hi",
                caption_language="en",
            )
        )
        db_session.commit()

        # Register a fake gemini_mm case in CASE_REGISTRY
        def _gemini_text_builder(clip, music_map):
            return "fake gemini text"

        def _gemini_payload(clip, text, video_path, fps, max_frames) -> dict:
            return {
                "video_path": video_path,
                "text": text,
            }

        gemini_spec = EmbeddingCaseSpec(
            name="gemini_mm",
            text_builder=_gemini_text_builder,
            requires_video=True,
            provider_factory=_fake_gemini_factory,
            payload_builder=_gemini_payload,
            apply_video_token_fallback=False,
        )

        # Patch CASE_REGISTRY in both modules (runner imports it at module level)
        from modules.embeddings import cases as cases_mod
        from modules.embeddings import runner as runner_mod

        new_registry = dict(cases_mod.CASE_REGISTRY)
        new_registry["gemini_mm"] = gemini_spec
        monkeypatch.setattr(cases_mod, "CASE_REGISTRY", new_registry)
        monkeypatch.setattr(runner_mod, "CASE_REGISTRY", new_registry)

        # Track embed calls
        captured_payloads: list[dict] = []

        original_embed = _FakeGeminiProvider.embed

        def _tracked_embed(self, payload: dict):
            captured_payloads.append(payload)
            return original_embed(self, payload)

        monkeypatch.setattr(_FakeGeminiProvider, "embed", _tracked_embed)

        # Phase (a): baseline run with output_dim=3072
        s1 = _runner_settings(tmp_path, vid_dir, aud_dir)

        embed_clip_embeddings(
            s1, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"]
        )

        assert len(captured_payloads) == 1, (
            f"Expected 1 embed in phase (a), got {len(captured_payloads)}"
        )
        captured_payloads.clear()

        first_rows = (
            db_session.query(ClipEmbedding).filter_by(embedding_case="gemini_mm").all()
        )
        assert len(first_rows) == 1, (
            f"Expected 1 row after phase (a), got {len(first_rows)}"
        )
        first_embedding = first_rows[0].embedding
        first_embedding_len = len(first_embedding)

        # Phase (b): drift config by changing output_dim
        s2 = _runner_settings(tmp_path, vid_dir, aud_dir)
        s2.embeddings.gemini_output_dim = 768  # drift the config

        embed_clip_embeddings(
            s2, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"]
        )

        # Verify: exactly 1 re-embed after wipe + re-insert
        assert len(captured_payloads) == 1, (
            f"Expected 1 re-embed in phase (b), got {len(captured_payloads)}"
        )

        # Verify: row count stays at 1 (wipe + re-insert)
        db_session.expire_all()  # refresh from DB
        final_rows = (
            db_session.query(ClipEmbedding).filter_by(embedding_case="gemini_mm").all()
        )
        assert len(final_rows) == 1, (
            f"Expected 1 row after phase (b), got {len(final_rows)}"
        )

        # Verify: new row has different embedding (different dim)
        final_embedding = final_rows[0].embedding
        final_embedding_len = len(final_embedding)
        # 3072 floats * 4 bytes = 12288; 768 floats * 4 bytes = 3072
        assert first_embedding_len == 12288, (
            f"Expected 12288 bytes for 3072-dim, got {first_embedding_len}"
        )
        assert final_embedding_len == 3072, (
            f"Expected 3072 bytes for 768-dim, got {final_embedding_len}"
        )

    finally:
        db_session.close()
