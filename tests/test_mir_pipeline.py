"""End-to-end tests for run_mir with stubbed vendor wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core.config import MirSettings
from core.database import (
    AudioMIR,
    Base,
    Clip,
    StageState,
    User,
    get_engine,
    get_session,
)


@dataclass
class _PathsStub:
    audio_mir_dir: str

    def audio_mir_for(self, clip_id):
        return Path(self.audio_mir_dir) / f"{clip_id}.wav"


@dataclass
class _SettingsStub:
    paths: _PathsStub
    mir: MirSettings


def _make_settings(tmp_path: Path) -> _SettingsStub:
    audio_mir_dir = tmp_path / "audio_mir"
    audio_mir_dir.mkdir(parents=True, exist_ok=True)
    return _SettingsStub(
        paths=_PathsStub(audio_mir_dir=str(audio_mir_dir)),
        mir=MirSettings(commit_every=10, prefetch_queue_size=1),
    )


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, AudioMIR, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def _write_dummy_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + b"\x00" * 200)


@pytest.fixture
def patched_pipeline(monkeypatch):
    from modules.mir import pipeline as pipeline_mod
    from tests.fakes.vendor_mir import FakeEffNet, FakeMAEST

    monkeypatch.setattr(pipeline_mod, "build_maest", lambda mir: FakeMAEST())
    monkeypatch.setattr(pipeline_mod, "build_effnet", lambda mir: FakeEffNet())
    monkeypatch.setattr(pipeline_mod, "ensure_checkpoints", lambda mir: None)
    monkeypatch.setattr(
        pipeline_mod,
        "_load_audio",
        lambda path, sr: np.ones(sr * 5, dtype=np.float32),
    )
    return pipeline_mod


def test_run_mir_writes_one_row_per_selected_clip(
    patched_pipeline, tmp_path, db_session
):
    settings = _make_settings(tmp_path)
    _write_dummy_wav(Path(settings.paths.audio_mir_dir) / "1.wav")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    rows = db_session.query(AudioMIR).all()
    assert len(rows) == 1
    r = rows[0]
    assert r.clip_id == 1
    assert r.is_mir_extracted is True
    assert r.is_happy is True  # FakeEffNet pos=0.9 >= 0.5 default threshold
    assert r.genre_labels  # non-empty CSV
    assert r.danceability is not None
    assert r.moodtheme_labels  # multi-tag CSV
    assert r.instrument_labels


def test_run_mir_is_no_op_on_second_run(patched_pipeline, tmp_path, db_session):
    settings = _make_settings(tmp_path)
    _write_dummy_wav(Path(settings.paths.audio_mir_dir) / "1.wav")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)
    patched_pipeline.run_mir(settings, secrets=None)

    rows = db_session.query(AudioMIR).all()
    assert len(rows) == 1


def test_run_mir_terminal_fails_when_audio_missing(
    patched_pipeline, tmp_path, db_session
):
    settings = _make_settings(tmp_path)
    # Note: no WAV file written

    db_session.add(User(id=1))
    db_session.add(Clip(id=42, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    row = db_session.query(AudioMIR).filter_by(clip_id=42).one()
    assert row.is_mir_extracted is False
    assert row.mir_error == "no_audio_file"


def test_run_mir_skips_unselected_clips(patched_pipeline, tmp_path, db_session):
    settings = _make_settings(tmp_path)
    _write_dummy_wav(Path(settings.paths.audio_mir_dir) / "5.wav")

    db_session.add(User(id=1))
    # is_selected is False → not eligible
    db_session.add(Clip(id=5, user_id=1, is_downloaded=True, is_selected=False))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    assert db_session.query(AudioMIR).count() == 0


def test_run_mir_attributes_maest_failure(
    monkeypatch, patched_pipeline, tmp_path, db_session
):
    from tests.fakes.vendor_mir import FakeMAEST

    def boom(self, audio):
        raise RuntimeError("simulated maest failure")

    monkeypatch.setattr(FakeMAEST, "predict", boom)

    settings = _make_settings(tmp_path)
    _write_dummy_wav(Path(settings.paths.audio_mir_dir) / "1.wav")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    row = db_session.query(AudioMIR).filter_by(clip_id=1).one()
    assert row.is_mir_extracted is False
    assert row.mir_error == "maest"


def test_run_mir_attributes_effnet_failure(
    monkeypatch, patched_pipeline, tmp_path, db_session
):
    from tests.fakes.vendor_mir import FakeEffNet

    def boom(self, audio):
        raise RuntimeError("simulated effnet failure")

    monkeypatch.setattr(FakeEffNet, "embed", boom)

    settings = _make_settings(tmp_path)
    _write_dummy_wav(Path(settings.paths.audio_mir_dir) / "2.wav")

    db_session.add(User(id=1))
    db_session.add(Clip(id=2, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    row = db_session.query(AudioMIR).filter_by(clip_id=2).one()
    assert row.is_mir_extracted is False
    assert row.mir_error == "effnet"


def test_run_mir_resets_when_audio_mir_dependency_drifts(
    patched_pipeline, tmp_path, db_session
):
    """If upstream extract_audio_mir reseals, MIR resets and re-extracts."""
    from core.database import StageState

    settings = _make_settings(tmp_path)
    _write_dummy_wav(Path(settings.paths.audio_mir_dir) / "1.wav")

    db_session.add(User(id=1))
    db_session.add(Clip(id=1, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)
    first = db_session.query(AudioMIR).filter_by(clip_id=1).one()
    assert first.is_mir_extracted is True

    db_session.merge(
        StageState(
            stage_name="audio_extract_mir",
            scope_key="default",
            data_hash="new-data",
            config_hash="new-config",
            dependency_hash="new-dep",
        )
    )
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    row = db_session.query(AudioMIR).filter_by(clip_id=1).one()
    assert row.is_mir_extracted is True
    assert db_session.query(AudioMIR).count() == 1


def test_run_mir_skips_checkpoint_bootstrap_when_no_eligible(
    monkeypatch, tmp_path, db_session
):
    """ensure_checkpoints is not called when no clips need MIR."""
    from modules.mir import pipeline as pipeline_mod
    from tests.fakes.vendor_mir import FakeEffNet, FakeMAEST

    monkeypatch.setattr(pipeline_mod, "build_maest", lambda mir: FakeMAEST())
    monkeypatch.setattr(pipeline_mod, "build_effnet", lambda mir: FakeEffNet())
    monkeypatch.setattr(
        pipeline_mod,
        "_load_audio",
        lambda path, sr: np.ones(sr * 5, dtype=np.float32),
    )
    called = {"n": 0}

    def _boom(mir):
        called["n"] += 1
        raise AssertionError("ensure_checkpoints should not run on no-op")

    monkeypatch.setattr(pipeline_mod, "ensure_checkpoints", _boom)

    settings = _make_settings(tmp_path)
    # No clips at all → eligible is empty
    pipeline_mod.run_mir(settings, secrets=None)
    assert called["n"] == 0


def test_run_mir_processes_multiple_clips_in_order(
    patched_pipeline, tmp_path, db_session
):
    settings = _make_settings(tmp_path)
    for cid in (10, 11, 12):
        _write_dummy_wav(Path(settings.paths.audio_mir_dir) / f"{cid}.wav")

    db_session.add(User(id=1))
    for cid in (10, 11, 12):
        db_session.add(Clip(id=cid, user_id=1, is_downloaded=True, is_selected=True))
    db_session.commit()

    patched_pipeline.run_mir(settings, secrets=None)

    rows = db_session.query(AudioMIR).order_by(AudioMIR.clip_id).all()
    assert [r.clip_id for r in rows] == [10, 11, 12]
    assert all(r.is_mir_extracted is True for r in rows)
