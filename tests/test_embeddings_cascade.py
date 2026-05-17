"""End-to-end cascade: change one audio knob, only audio rows are wiped.

Deviation notes
---------------
1. **Frozen-dataclass monkeypatch** — ``EmbeddingCaseSpec`` is
   ``@dataclass(frozen=True)``, so ``monkeypatch.setattr(spec,
   "provider_factory", ...)`` raises ``FrozenInstanceError``.
   Fixed via ``dataclasses.replace`` to build a new registry dict; both
   ``cases_mod.CASE_REGISTRY`` and ``runner_mod.CASE_REGISTRY`` are
   patched (runner imports the name at module level).

2. **_FakeProvider.embed return type** — ``to_bytes()`` calls
   ``.cpu().float().numpy().tobytes()`` (torch tensor API).  A plain
   ``np.ndarray`` raises ``AttributeError``.  Resolved with the same
   ``_TorchLikeArray`` duck-type wrapper used in
   ``test_clip_embeddings_idempotence.py`` (re-defined here so the file is
   self-contained).

3. **SQLite second-precision timestamps** — ``ClipEmbedding.updated_at``
   uses ``server_default=func.now()`` (second precision in SQLite).  When
   clip_embeddings deletes + re-inserts an audio row within the same second,
   the new row's ``updated_at`` is identical to the old row's, so
   ``user_embeddings`` fingerprint sees no dependency change and skips.

   Fix: after the second ``embed_clip_embeddings`` call the test manually
   bumps the audio ``ClipEmbedding.updated_at`` to a timestamp one second
   in the future via a direct SQL ``UPDATE``.  This correctly exercises the
   cascade logic (user_embeddings detects the dependency change and
   re-runs) without requiring a wall-clock ``sleep(1)``.

   The cascade test therefore has three explicit phases:
   (a) baseline run (both stages, all cases);
   (b) mutate AUDIO_INSTRUCTION → re-run clip_embeddings → bump updated_at
       on audio ClipEmbedding → re-run user_embeddings;
   (c) assert video/sandwich sealed at both layers; audio re-ran at both.

   Assertions for "sealed" cases use ``updated_at`` (genuinely unchanged).
   Assertions for "re-ran" audio case use ``config_hash`` at clip layer
   (changes because AUDIO_INSTRUCTION is part of the identity string) and
   ``dependency_hash`` at user layer (changes because ClipEmbedding
   updated_at changed after the manual bump).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import text

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    StageState,
    User,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.embeddings import cases as cases_mod
from modules.embeddings.runner import embed_clip_embeddings
from modules.embeddings.users import embed_user_embeddings

# ── fake provider ─────────────────────────────────────────────────────────────


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
class _FakeProvider:
    def embed(self, payload: dict) -> _TorchLikeArray:
        seed = abs(hash(repr(sorted(payload.items())))) % (2**32)
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal((1, 4)).astype(np.float32)
        return _TorchLikeArray(arr)


_FAKE_SECRETS = object()  # sentinel; never inspected by the fake factory


def _fake_factory(_settings, _secrets):
    return _FakeProvider()


# ── settings stub ─────────────────────────────────────────────────────────────


@dataclass
class _PathsStub:
    video_dir: str
    model_path: str = "/fake/qwen"


@dataclass
class _EmbeddingsStub:
    exclude_disqualified_users: bool = True
    embed_max_length: int = 1024
    adaptive_max_frames: int = 8
    adaptive_default_fps: float = 1.0
    inflight: int = 1


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    _cleanup(session)
    yield session
    _cleanup(session)
    session.close()


def _cleanup(session) -> None:
    for model in (
        StageState,
        UserEmbedding,
        ClipEmbedding,
        Clip,
        Music,
        User,
    ):
        session.query(model).delete()
    session.commit()


@pytest.fixture
def stub_providers(monkeypatch):
    """Replace every CASE_REGISTRY spec with a fake-factory variant.

    Because EmbeddingCaseSpec is frozen we use dataclasses.replace() to
    build fresh instances, then monkeypatch both the cases module dict and
    runner module dict (runner imports CASE_REGISTRY at module level).
    """
    new_registry = {
        name: dataclasses.replace(spec, provider_factory=_fake_factory)
        for name, spec in cases_mod.CASE_REGISTRY.items()
    }
    monkeypatch.setattr(cases_mod, "CASE_REGISTRY", new_registry)

    from modules.embeddings import runner as runner_mod
    from modules.embeddings import sampling as sampling_mod

    monkeypatch.setattr(runner_mod, "CASE_REGISTRY", new_registry)
    monkeypatch.setattr(
        sampling_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )
    monkeypatch.setattr(
        runner_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _settings(tmp_path) -> _SettingsStub:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir)),
        embeddings=_EmbeddingsStub(),
    )


def _seed(session, settings: _SettingsStub) -> None:
    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            speech_transcription="hi",
        )
    )
    import os

    with open(os.path.join(settings.paths.video_dir, "10.mp4"), "wb") as f:
        f.write(b"\x00")
    session.commit()


def _bump_audio_clip_embedding_ts(session) -> None:
    """Advance updated_at on the audio ClipEmbedding row by one second.

    SQLite stores timestamps at second precision via ``func.now()``, so two
    runs within the same wall-clock second produce identical timestamps.
    Bumping the timestamp directly exercises the cascade logic
    (user_embeddings detects the dependency change) without a wall-clock
    sleep.
    """
    future = (datetime.now(UTC) + timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S")
    session.execute(
        text(
            "UPDATE clip_embeddings SET updated_at = :ts WHERE embedding_case = 'audio'"
        ),
        {"ts": future},
    )
    session.commit()


# ── test ──────────────────────────────────────────────────────────────────────


def test_audio_instruction_change_cascades_only_to_audio(
    db_session, stub_providers, tmp_path, monkeypatch
):
    """Mutating AUDIO_INSTRUCTION wipes audio rows at both layers; video +
    sandwich remain sealed.

    Phase (a): baseline run — all six stage-state rows written.
    Phase (b): mutate AUDIO_INSTRUCTION, re-run clip_embeddings (audio
               re-runs; video/sandwich skip), bump audio ClipEmbedding
               updated_at to make the cascade detectable, re-run
               user_embeddings (audio re-runs; video/sandwich skip).
    Phase (c): assert the correct rows changed / stayed sealed.
    """
    settings = _settings(tmp_path)
    _seed(db_session, settings)

    # Phase (a): baseline
    embed_clip_embeddings(settings, _FAKE_SECRETS)
    embed_user_embeddings(settings)
    db_session.expire_all()

    def _clip_state(case: str) -> StageState:
        return db_session.get(StageState, ("clip_embeddings", case))

    def _user_state(case: str) -> StageState:
        return db_session.get(StageState, ("user_embeddings", case))

    # Capture "sealed" timestamps for video + sandwich (both layers).
    before_clip_video_ts = _clip_state("video").updated_at
    before_clip_sandwich_ts = _clip_state("sandwich").updated_at
    before_user_video_ts = _user_state("video").updated_at
    before_user_sandwich_ts = _user_state("sandwich").updated_at

    # Capture config_hash / dependency_hash for the audio case.
    # config_hash changes when AUDIO_INSTRUCTION changes (clip layer).
    # dependency_hash changes when ClipEmbedding updated_at changes (user layer).
    before_clip_audio_cfg = _clip_state("audio").config_hash
    before_user_audio_dep = _user_state("audio").dependency_hash

    # Phase (b): mutate audio knob → re-run both stages
    monkeypatch.setattr(cases_mod, "AUDIO_INSTRUCTION", "DIFFERENT INSTRUCTION")

    embed_clip_embeddings(settings, _FAKE_SECRETS)  # audio re-runs; video/sandwich skip
    # Bump updated_at on the audio ClipEmbedding row so user_embeddings
    # can detect the upstream change (sub-second cascade workaround).
    _bump_audio_clip_embedding_ts(db_session)
    embed_user_embeddings(settings)  # audio re-runs; video/sandwich skip
    db_session.expire_all()

    # Phase (c): assertions

    # Video + sandwich sealed at the clip layer.
    assert _clip_state("video").updated_at == before_clip_video_ts
    assert _clip_state("sandwich").updated_at == before_clip_sandwich_ts

    # Audio re-ran at the clip layer (config_hash changed).
    assert _clip_state("audio").config_hash != before_clip_audio_cfg

    # Video + sandwich sealed at the user layer.
    assert _user_state("video").updated_at == before_user_video_ts
    assert _user_state("sandwich").updated_at == before_user_sandwich_ts

    # Audio re-ran at the user layer (dependency_hash changed).
    assert _user_state("audio").dependency_hash != before_user_audio_dep
