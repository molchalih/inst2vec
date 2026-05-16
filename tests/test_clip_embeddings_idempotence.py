"""Idempotence tests for embed_clip_embeddings.

Uses a fake provider injected via spec.provider_factory monkey-patch so
no Qwen model is loaded.

Frozen-dataclass deviation note
--------------------------------
``EmbeddingCaseSpec`` is ``@dataclass(frozen=True)``, so
``monkeypatch.setattr(spec, "provider_factory", _fake_factory)`` raises
``FrozenInstanceError``.  The spec plan used that pattern, but it cannot
work as written.

Fix: instead of patching individual spec instances we replace the entire
``CASE_REGISTRY`` dict with freshly-constructed (non-frozen, via
``object.__setattr__``) copies that have ``provider_factory`` swapped
out.  The test semantics are identical — every spec that the runner looks
up via ``CASE_REGISTRY[name]`` gets the fake factory — with no change to
production code.

Concretely, ``stub_providers`` uses ``dataclasses.replace`` to build a
new ``EmbeddingCaseSpec`` (which is valid even for frozen dataclasses —
``replace`` constructs a new instance) and then monkeypatches the module-
level ``CASE_REGISTRY`` dict to point at the new instances.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.embeddings import cases as cases_mod
from modules.embeddings.runner import embed_clip_embeddings

# ── fake provider ────────────────────────────────────────────────────────────


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
    salt: str = ""

    def embed(self, payload: dict) -> _TorchLikeArray:
        seed = abs(hash((self.salt, repr(sorted(payload.items()))))) % (2**32)
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal((1, 4)).astype(np.float32)
        return _TorchLikeArray(arr)


def _fake_factory(_settings):
    return _FakeProvider()


@pytest.fixture
def stub_providers(monkeypatch, tmp_path):
    """Patch every spec.provider_factory to a fake; redirect video_dir to tmp.

    Because EmbeddingCaseSpec is frozen we cannot call
    monkeypatch.setattr(spec, "provider_factory", ...) directly.
    Instead we rebuild each spec via dataclasses.replace() — which is
    valid for frozen dataclasses — and monkeypatch the CASE_REGISTRY dict
    to point at the new instances.
    """
    new_registry = {
        name: dataclasses.replace(spec, provider_factory=_fake_factory)
        for name, spec in cases_mod.CASE_REGISTRY.items()
    }
    monkeypatch.setattr(cases_mod, "CASE_REGISTRY", new_registry)

    from modules.embeddings import runner as runner_mod
    from modules.embeddings import sampling as sampling_mod

    # runner.py imports CASE_REGISTRY via a direct ``from ... import``, so
    # patching cases_mod.CASE_REGISTRY alone is not enough — the runner's
    # module-level name also needs to be redirected.
    monkeypatch.setattr(runner_mod, "CASE_REGISTRY", new_registry)

    monkeypatch.setattr(
        sampling_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )
    monkeypatch.setattr(
        runner_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )
    return tmp_path


# ── settings stub ────────────────────────────────────────────────────────────


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


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub


def _settings(tmp_path) -> _SettingsStub:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir)),
        embeddings=_EmbeddingsStub(),
    )


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def _seed_video_file(settings: _SettingsStub, clip_id: int) -> None:
    import os

    path = os.path.join(settings.paths.video_dir, f"{clip_id}.mp4")
    with open(path, "wb") as f:
        f.write(b"\x00")


def _seed(
    session,
    settings: _SettingsStub,
    *,
    clips: list[dict],
    music_rows: list[dict] | None = None,
) -> None:
    music_rows = music_rows or []
    for m in music_rows:
        session.merge(Music(**m))
    user_ids = {c["user_id"] for c in clips}
    for uid in user_ids:
        session.merge(User(id=uid, is_selected=True, is_eligible=True))
    for c in clips:
        defaults = dict(is_selected=True, is_downloaded=True)
        defaults.update(c)
        session.merge(Clip(**defaults))
        if defaults["is_downloaded"]:
            _seed_video_file(settings, defaults["id"])
    session.commit()


# ── tests ────────────────────────────────────────────────────────────────────


def test_first_run_embeds_all_three_cases(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[dict(id=10, user_id=1)],
        music_rows=[dict(id=1, artist="a", track="t")],
    )
    db_session.query(Clip).filter_by(id=10).update({"music_id": 1})
    db_session.commit()

    embed_clip_embeddings(settings)
    db_session.expire_all()

    rows = db_session.query(ClipEmbedding).all()
    cases = {r.embedding_case for r in rows}
    assert cases == {"video", "sandwich", "audio"}

    for case in ("video", "sandwich", "audio"):
        assert db_session.get(StageState, ("clip_embeddings", case)) is not None


def test_rerun_identical_inputs_is_noop(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])

    embed_clip_embeddings(settings, cases=["video"])
    first = db_session.get(StageState, ("clip_embeddings", "video")).updated_at

    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()
    second = db_session.get(StageState, ("clip_embeddings", "video")).updated_at
    assert first == second


def test_new_candidate_triggers_recompute(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])
    embed_clip_embeddings(settings, cases=["video"])

    _seed(db_session, settings, clips=[dict(id=11, user_id=1)])
    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()

    ids = {
        r.clip_id
        for r in db_session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert ids == {10, 11}


def test_audio_speech_change_only_invalidates_audio(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[dict(id=10, user_id=1, speech_transcription="hi")],
    )
    embed_clip_embeddings(settings)
    db_session.expire_all()

    before = {
        case: db_session.get(StageState, ("clip_embeddings", case)).dependency_hash
        for case in ("video", "sandwich", "audio")
    }

    # Mutate speech_transcription — should flip audio + sandwich dep, not video.
    db_session.query(Clip).filter_by(id=10).update(
        {"speech_transcription": "hello there"}
    )
    db_session.commit()

    embed_clip_embeddings(settings)
    db_session.expire_all()

    after = {
        case: db_session.get(StageState, ("clip_embeddings", case)).dependency_hash
        for case in ("video", "sandwich", "audio")
    }
    assert after["video"] == before["video"]  # video untouched
    assert after["sandwich"] != before["sandwich"]  # sandwich dep changed
    assert after["audio"] != before["audio"]  # audio dep changed


def test_audio_instruction_change_only_invalidates_audio(
    db_session, stub_providers, monkeypatch
):
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])
    embed_clip_embeddings(settings)
    db_session.expire_all()
    # Compare config_hash (not updated_at): SQLite server_default/onupdate has
    # second-level precision, so two runs within the same second look identical.
    # config_hash changes iff the case-config identity string changes, which is
    # the invariant we actually care about here.
    before = {
        case: db_session.get(StageState, ("clip_embeddings", case)).config_hash
        for case in ("video", "sandwich", "audio")
    }

    monkeypatch.setattr(cases_mod, "AUDIO_INSTRUCTION", "NEW INSTRUCTION TEXT")

    embed_clip_embeddings(settings)
    db_session.expire_all()
    after = {
        case: db_session.get(StageState, ("clip_embeddings", case)).config_hash
        for case in ("video", "sandwich", "audio")
    }
    assert after["video"] == before["video"]
    assert after["sandwich"] == before["sandwich"]
    assert after["audio"] != before["audio"]


def test_empty_candidates_writes_stage_state(db_session, stub_providers):
    settings = _settings(stub_providers)
    embed_clip_embeddings(settings, cases=["video"])
    state = db_session.get(StageState, ("clip_embeddings", "video"))
    assert state is not None


def test_partial_failure_does_not_seal_stage(db_session, stub_providers, monkeypatch):
    """A clip that fails to embed must not be sealed as complete.

    Otherwise the fingerprint (data/config/dependency) is identical on rerun,
    is_stale returns False, and the missing ClipEmbedding row is never retried.
    """
    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[dict(id=10, user_id=1), dict(id=11, user_id=1)],
    )

    from modules.embeddings import runner as runner_mod

    original = runner_mod._embed_with_token_fallback
    call_log: list[int] = []

    def flaky(provider, spec, clip, *args, **kwargs):
        call_log.append(clip.id)
        if clip.id == 11 and call_log.count(11) == 1:
            return None  # transient failure on first attempt for clip 11
        return original(provider, spec, clip, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "_embed_with_token_fallback", flaky)

    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()

    rows = {
        r.clip_id
        for r in db_session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert rows == {10}, "successful clips should persist"
    assert db_session.get(StageState, ("clip_embeddings", "video")) is None, (
        "stage must remain stale when any clip failed"
    )

    # Rerun: same inputs, but no stage_state → recompute → both clips embedded.
    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()
    rows = {
        r.clip_id
        for r in db_session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert rows == {10, 11}
    assert db_session.get(StageState, ("clip_embeddings", "video")) is not None
