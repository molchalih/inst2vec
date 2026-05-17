import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.embeddings import embed_clip_embeddings
from modules.embeddings.cases import default_cases
from modules.embeddings.gemini import (
    GeminiClipTooLongError,
    GeminiMultimodalProvider,
    GeminiOutputDimMismatch,
)
from modules.embeddings.text import build_gemini_text


def _make_provider(monkeypatch, video_seconds, audio_seconds):
    """Build a provider with monkeypatched ffprobe + injected mock client."""
    fake_client = MagicMock()
    durations = {"v.mp4": video_seconds, "a.mp3": audio_seconds}
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda path: durations[os.path.basename(path)],
    )
    return GeminiMultimodalProvider(
        api_key="x",
        model="m",
        output_dim=3072,
        max_video_seconds=120,
        max_audio_seconds=80,
        request_timeout_s=10,
        max_retries=0,
        client=fake_client,
    )


def test_provider_skips_oversize_video(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    a = tmp_path / "a.mp3"
    a.write_bytes(b"x")
    provider = _make_provider(monkeypatch, video_seconds=150, audio_seconds=10)
    with pytest.raises(GeminiClipTooLongError):
        provider.embed({"video_path": str(v), "audio_path": str(a), "text": "t"})
    # No upload attempted.
    assert not provider._client.files.upload.called


def test_provider_skips_oversize_audio(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"x")
    a = tmp_path / "a.mp3"
    a.write_bytes(b"x")
    provider = _make_provider(monkeypatch, video_seconds=10, audio_seconds=90)
    with pytest.raises(GeminiClipTooLongError):
        provider.embed({"video_path": str(v), "audio_path": str(a), "text": "t"})


def test_embed_uploads_and_returns_vector(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"v")
    a = tmp_path / "a.mp3"
    a.write_bytes(b"a")
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda p: 10.0,
    )

    fake_client = MagicMock()
    upload_video = MagicMock(uri="files/abc", mime_type="video/mp4")
    upload_audio = MagicMock(uri="files/def", mime_type="audio/mpeg")
    fake_client.files.upload.side_effect = [upload_video, upload_audio]
    fake_embed = MagicMock()
    fake_embed.values = [0.1] * 3072
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embed]
    fake_client.models.embed_content.return_value = fake_response

    p = GeminiMultimodalProvider(
        api_key="x",
        model="m",
        output_dim=3072,
        max_video_seconds=120,
        max_audio_seconds=80,
        request_timeout_s=10,
        max_retries=0,
        client=fake_client,
    )
    # Bypass the real google.genai types call — the test must not require
    # the optional dependency to be installed.
    monkeypatch.setattr(
        GeminiMultimodalProvider,
        "_build_request",
        lambda self, t, v_file, a_file: ([t, "vp", "ap"], {"dim": self.output_dim}),
    )

    out = p.embed({"video_path": str(v), "audio_path": str(a), "text": "hello"})
    assert len(out) == 1
    assert len(out[0]) == 3072
    assert fake_client.files.upload.call_count == 2
    fake_client.models.embed_content.assert_called_once()


def test_embed_raises_on_dim_mismatch(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"v")
    a = tmp_path / "a.mp3"
    a.write_bytes(b"a")
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda p: 10.0,
    )
    fake_client = MagicMock()
    fake_client.files.upload.side_effect = [
        MagicMock(uri="files/abc", mime_type="video/mp4"),
        MagicMock(uri="files/def", mime_type="audio/mpeg"),
    ]
    fake_embed = MagicMock()
    fake_embed.values = [0.0] * 768  # wrong length
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embed]
    fake_client.models.embed_content.return_value = fake_response

    p = GeminiMultimodalProvider(
        api_key="x",
        model="m",
        output_dim=3072,
        max_video_seconds=120,
        max_audio_seconds=80,
        request_timeout_s=10,
        max_retries=0,
        client=fake_client,
    )
    monkeypatch.setattr(
        GeminiMultimodalProvider,
        "_build_request",
        lambda self, t, v_file, a_file: ([t, "vp", "ap"], {"dim": self.output_dim}),
    )
    with pytest.raises(GeminiOutputDimMismatch):
        p.embed({"video_path": str(v), "audio_path": str(a), "text": "x"})


def test_embed_retries_on_5xx(monkeypatch, tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"v")
    a = tmp_path / "a.mp3"
    a.write_bytes(b"a")
    monkeypatch.setattr(
        "modules.embeddings.gemini._probe_duration_seconds",
        lambda p: 10.0,
    )

    fake_client = MagicMock()
    fake_client.files.upload.side_effect = [
        MagicMock(uri="files/abc", mime_type="video/mp4"),
        MagicMock(uri="files/def", mime_type="audio/mpeg"),
    ]
    fake_embed = MagicMock()
    fake_embed.values = [0.0] * 3072
    fake_response = MagicMock()
    fake_response.embeddings = [fake_embed]

    class _Transient(Exception):
        status_code = 503

    fake_client.models.embed_content.side_effect = [
        _Transient("upstream busy"),
        _Transient("upstream busy"),
        fake_response,
    ]

    p = GeminiMultimodalProvider(
        api_key="x",
        model="m",
        output_dim=3072,
        max_video_seconds=120,
        max_audio_seconds=80,
        request_timeout_s=10,
        max_retries=5,
        client=fake_client,
    )
    monkeypatch.setattr(
        GeminiMultimodalProvider,
        "_build_request",
        lambda self, t, v, a: ([t, "vp", "ap"], {"dim": 3072}),
    )
    # Eliminate sleep delay between retries to keep the test fast.
    monkeypatch.setattr("modules.embeddings.gemini.time.sleep", lambda *_: None)

    out = p.embed({"video_path": str(v), "audio_path": str(a), "text": "x"})
    assert len(out[0]) == 3072
    assert fake_client.models.embed_content.call_count == 3


def test_gemini_text_joins_caption_and_transcript():
    clip = SimpleNamespace(
        caption_text="cap",
        caption_clean="cap clean",
        caption_language="es",
        caption_translation="cap en",
        speech_transcription="hi",
        speech_language="es",
        speech_translation="hello",
    )
    text = build_gemini_text(clip, {})
    assert "cap en" in text
    assert "hello" in text
    assert "---" in text  # separator marker


def test_gemini_text_returns_none_when_empty():
    clip = SimpleNamespace(
        caption_text="",
        caption_clean=None,
        caption_language=None,
        caption_translation=None,
        speech_transcription=None,
        speech_language=None,
        speech_translation=None,
    )
    assert build_gemini_text(clip, {}) is None


@pytest.fixture
def db_session():
    from core.database import (
        Base,
        Clip,
        ClipEmbedding,
        Music,
        StageState,
        User,
        get_engine,
        get_session,
    )

    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def test_dependency_rows_gemini_mm_includes_file_stats(
    db_session, monkeypatch, tmp_path
):
    from core.database import Clip, User
    from modules.embeddings.state import dependency_rows_for_case

    db_session.add(User(id=1, is_selected=True))
    db_session.add(
        Clip(
            id=1,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="cap",
            caption_language="en",
            speech_transcription="hi",
            speech_language="en",
        )
    )
    db_session.commit()

    fake_settings = SimpleNamespace(
        paths=SimpleNamespace(
            video_dir=str(tmp_path / "video"),
            audio_dir=str(tmp_path / "audio"),
        )
    )
    # Patch the stat helpers so the test does not need real files.
    monkeypatch.setattr(
        "modules.embeddings.state._video_file_stat", lambda video_dir, cid: (1234, 1000)
    )
    monkeypatch.setattr(
        "modules.embeddings.state._audio_file_stat", lambda audio_dir, cid: (567, 2000)
    )

    rows = dependency_rows_for_case(
        db_session, "gemini_mm", [1], settings=fake_settings
    )
    assert len(rows) == 1
    row = rows[0]
    assert (1234, 1000) in row
    assert (567, 2000) in row
    assert row[0] == 1  # clip id is first


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


def _runner_settings(video_dir, audio_dir):
    """Settings stub carrying every field the runner + gemini case read."""
    return SimpleNamespace(
        paths=SimpleNamespace(
            video_dir=str(video_dir),
            audio_dir=str(audio_dir),
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
            audio_bitrate_kbps=128,
            audio_sample_rate_hz=44100,
            embed_max_length=2048,
            adaptive_max_frames=64,
            adaptive_default_fps=2.0,
            exclude_disqualified_users=False,
            provider="local",
            inflight=1,
        ),
    )


def test_runner_seals_when_all_clips_embed(
    tmp_path, db_session, monkeypatch, sample_mp4_with_audio
):
    from pathlib import Path

    from core.database import Clip, ClipEmbedding, StageState, User
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    aud_dir = tmp_path / "audio"
    aud_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(str(sample_mp4_with_audio)).read_bytes())
    (aud_dir / "1.mp3").write_bytes(b"fake_mp3")

    db_session.add(User(id=1, is_selected=True))
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

    settings = _runner_settings(vid_dir, aud_dir)

    # Stub provider __init__ to avoid importing google.genai or doing any I/O.
    def _fake_init(self, **kwargs):
        self.model = kwargs["model"]
        self.output_dim = kwargs["output_dim"]

    monkeypatch.setattr(GeminiMultimodalProvider, "__init__", _fake_init)

    # Stub embed to return a fixed 3072-d vector — no upload, no network.
    # The runner serializes via vectors.to_bytes(out[0]), which expects a
    # torch-like tensor (.cpu().float().numpy()), so return a torch tensor.
    import torch

    def _fake_embed(self, payload):
        return [torch.full((3072,), 0.1, dtype=torch.float32)]

    monkeypatch.setattr(GeminiMultimodalProvider, "embed", _fake_embed)

    embed_clip_embeddings(
        settings, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"]
    )

    rows = db_session.query(ClipEmbedding).filter_by(embedding_case="gemini_mm").all()
    assert len(rows) == 1
    assert rows[0].clip_id == 1
    state = db_session.get(StageState, ("clip_embeddings", "gemini_mm"))
    assert state is not None


def test_runner_does_not_seal_on_failure(
    tmp_path, db_session, monkeypatch, sample_mp4_with_audio
):
    from pathlib import Path

    from core.database import Clip, ClipEmbedding, StageState, User
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    aud_dir = tmp_path / "audio"
    aud_dir.mkdir()
    fixture_bytes = Path(str(sample_mp4_with_audio)).read_bytes()
    for cid in (1, 2):
        (vid_dir / f"{cid}.mp4").write_bytes(fixture_bytes)
        (aud_dir / f"{cid}.mp3").write_bytes(b"fake_mp3")

    db_session.add(User(id=1, is_selected=True))
    for cid in (1, 2):
        db_session.add(
            Clip(
                id=cid,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_text="hi",
                caption_language="en",
            )
        )
    db_session.commit()

    settings = _runner_settings(vid_dir, aud_dir)

    def _fake_init(self, **kwargs):
        self.model = kwargs["model"]
        self.output_dim = kwargs["output_dim"]

    monkeypatch.setattr(GeminiMultimodalProvider, "__init__", _fake_init)

    import torch

    def _fake_embed(self, payload):
        if payload["video_path"].endswith("1.mp4"):
            raise RuntimeError("boom")
        return [torch.full((3072,), 0.2, dtype=torch.float32)]

    monkeypatch.setattr(GeminiMultimodalProvider, "embed", _fake_embed)

    embed_clip_embeddings(
        settings, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"]
    )

    rows = db_session.query(ClipEmbedding).filter_by(embedding_case="gemini_mm").all()
    assert {r.clip_id for r in rows} == {2}
    assert db_session.get(StageState, ("clip_embeddings", "gemini_mm")) is None


def test_config_drift_wipes_case(
    tmp_path, db_session, monkeypatch, sample_mp4_with_audio
):
    from pathlib import Path

    from core.database import Clip, ClipEmbedding, User
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    aud_dir = tmp_path / "audio"
    aud_dir.mkdir()
    (vid_dir / "1.mp4").write_bytes(Path(str(sample_mp4_with_audio)).read_bytes())
    (aud_dir / "1.mp3").write_bytes(b"fake_mp3")

    db_session.add(User(id=1, is_selected=True))
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

    captured: list[str] = []

    def _fake_init(self, **kwargs):
        self.model = kwargs["model"]
        self.output_dim = kwargs["output_dim"]

    monkeypatch.setattr(GeminiMultimodalProvider, "__init__", _fake_init)

    import torch

    def _fake_embed(self, payload):
        captured.append((payload["video_path"], self.output_dim))
        return [torch.full((self.output_dim,), 0.0, dtype=torch.float32)]

    monkeypatch.setattr(GeminiMultimodalProvider, "embed", _fake_embed)

    # Phase (a): baseline run with output_dim=3072.
    s1 = _runner_settings(vid_dir, aud_dir)
    embed_clip_embeddings(s1, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"])
    assert len(captured) == 1
    first_rows = (
        db_session.query(ClipEmbedding).filter_by(embedding_case="gemini_mm").all()
    )
    assert len(first_rows) == 1
    first_blob_size = len(first_rows[0].embedding)
    captured.clear()

    # Phase (b): drift gemini_output_dim → config_hash changes → wipe + re-embed.
    s2 = _runner_settings(vid_dir, aud_dir)
    s2.embeddings.gemini_output_dim = 768
    embed_clip_embeddings(s2, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"])
    assert len(captured) == 1, f"expected exactly one re-embed, got {len(captured)}"

    # The runner uses its own session; expire ours so we see the wipe + insert.
    db_session.expire_all()
    rows = db_session.query(ClipEmbedding).filter_by(embedding_case="gemini_mm").all()
    assert len(rows) == 1
    # Row was wiped and replaced — new blob size reflects the new dim.
    assert len(rows[0].embedding) != first_blob_size
    assert len(rows[0].embedding) == 768 * 4  # float32 = 4 bytes/element


def test_per_clip_diff_re_embeds_only_touched_clip(
    tmp_path, db_session, monkeypatch, sample_mp4_with_audio
):
    from pathlib import Path

    from core.database import Clip, User
    from modules.embeddings import EmbeddingSecrets, embed_clip_embeddings
    from modules.embeddings.gemini import GeminiMultimodalProvider

    vid_dir = tmp_path / "video"
    vid_dir.mkdir()
    aud_dir = tmp_path / "audio"
    aud_dir.mkdir()
    fixture_bytes = Path(str(sample_mp4_with_audio)).read_bytes()
    for cid in (1, 2):
        (vid_dir / f"{cid}.mp4").write_bytes(fixture_bytes)
        (aud_dir / f"{cid}.mp3").write_bytes(b"fake_mp3")

    db_session.add(User(id=1, is_selected=True))
    for cid in (1, 2):
        db_session.add(
            Clip(
                id=cid,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_text="hi",
                caption_language="en",
            )
        )
    db_session.commit()

    settings = _runner_settings(vid_dir, aud_dir)

    def _fake_init(self, **kwargs):
        self.model = kwargs["model"]
        self.output_dim = kwargs["output_dim"]

    monkeypatch.setattr(GeminiMultimodalProvider, "__init__", _fake_init)

    import torch

    seen: list[str] = []

    def _fake_embed(self, payload):
        seen.append(payload["video_path"])
        return [torch.full((3072,), 0.0, dtype=torch.float32)]

    monkeypatch.setattr(GeminiMultimodalProvider, "embed", _fake_embed)

    # Phase 1: both clips embedded.
    embed_clip_embeddings(
        settings, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"]
    )
    assert len(seen) == 2
    seen.clear()

    # Mutate only clip 1's caption — its per-clip source hash changes.
    clip1 = db_session.get(Clip, 1)
    clip1.caption_text = "new caption"
    db_session.commit()

    # Phase 2: only clip 1 should be re-embedded; clip 2's hash is unchanged.
    embed_clip_embeddings(
        settings, EmbeddingSecrets(gemini_api_key="x"), cases=["gemini_mm"]
    )
    assert seen == [str(vid_dir / "1.mp4")]
