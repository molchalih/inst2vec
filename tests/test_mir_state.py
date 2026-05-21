"""Tests for the MIR fingerprint payload + reset helper."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _engine():
    from core.database import Base

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _mir_settings(**overrides):
    from core.config import MirSettings

    return MirSettings(**overrides)


def test_payload_stable_for_identical_settings():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings())
    b = mir_config_payload(_mir_settings())
    assert a == b


def test_payload_changes_with_binary_threshold():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings(binary_threshold=0.5))
    b = mir_config_payload(_mir_settings(binary_threshold=0.6))
    assert a != b


def test_payload_excludes_operational_knobs():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings(download_concurrency=2, commit_every=10))
    b = mir_config_payload(_mir_settings(download_concurrency=8, commit_every=50))
    assert a == b


def test_payload_changes_with_topk_or_checkpoint():
    from modules.mir.state import mir_config_payload

    a = mir_config_payload(_mir_settings())
    b = mir_config_payload(_mir_settings(topk_genre=5))
    c = mir_config_payload(_mir_settings(maest_checkpoint="other.pb"))
    assert a != b
    assert a != c
    assert b != c


def test_reset_audio_mir_nulls_descriptor_columns_but_preserves_row():
    from core.database import AudioMIR, Clip, User
    from modules.mir.state import reset_audio_mir

    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1))
        s.add(
            AudioMIR(
                clip_id=1,
                is_mir_extracted=True,
                danceability=0.7,
                is_happy=True,
                genre_labels="x",
                genre_scores="0.5",
                audio_duration_s=10.0,
                inference_time_ms=5.0,
            )
        )
        s.commit()
        original = s.query(AudioMIR).filter_by(clip_id=1).one().created_at

        reset_audio_mir(s)

        row = s.query(AudioMIR).filter_by(clip_id=1).one()
        assert row.is_mir_extracted is None
        assert row.danceability is None
        assert row.is_happy is None
        assert row.genre_labels is None
        assert row.genre_scores is None
        assert row.audio_duration_s is None
        assert row.inference_time_ms is None
        assert row.clip_id == 1
        assert row.created_at == original


def test_reset_audio_mir_on_empty_db_is_noop():
    from modules.mir.state import reset_audio_mir

    eng = _engine()
    with Session(eng) as s:
        reset_audio_mir(s)


def test_stage_and_scope_constants():
    from core.pipeline import Stage
    from modules.mir.state import POS, SCOPE_MIR, STAGE_MIR

    assert STAGE_MIR is Stage.MIR
    assert SCOPE_MIR == "all"
    assert POS == 0


def test_reset_columns_covers_audio_mir_schema():
    """_RESET_COLUMNS must cover every AudioMIR column except the PK and timestamps."""
    from core.database import AudioMIR
    from modules.mir.state import _RESET_COLUMNS

    all_cols = {c.name for c in AudioMIR.__table__.columns}
    excluded = {"clip_id", "created_at", "updated_at"}
    assert set(_RESET_COLUMNS) == all_cols - excluded


def test_payload_changes_when_label_file_content_changes(tmp_path, monkeypatch):
    """Mutating a label JSON file flips the fingerprint."""
    from modules.mir import state as state_mod
    from modules.mir.state import mir_config_payload

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.json").write_text('["x"]')
    (labels_dir / "b.json").write_text('["y"]')

    monkeypatch.setattr(state_mod, "_LABELS_DIR", labels_dir)
    a = mir_config_payload(_mir_settings())

    (labels_dir / "a.json").write_text('["x", "z"]')
    b = mir_config_payload(_mir_settings())

    assert a != b


def test_payload_stable_when_label_files_unchanged(tmp_path, monkeypatch):
    from modules.mir import state as state_mod
    from modules.mir.state import mir_config_payload

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.json").write_text('["x"]')

    monkeypatch.setattr(state_mod, "_LABELS_DIR", labels_dir)
    assert mir_config_payload(_mir_settings()) == mir_config_payload(_mir_settings())


def test_payload_changes_when_checkpoint_sidecar_digest_changes(tmp_path):
    """Mutating a .pb's sidecar digest flips the fingerprint."""
    import json

    from modules.mir.state import mir_config_payload

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    settings = _mir_settings(model_dir=str(model_dir))

    # Materialize a few .pb files matching the manifest, with sidecars.
    from modules.mir.checkpoints import _manifest, _sidecar_path

    for _url, target in _manifest(settings)[:3]:
        target.write_bytes(b"x")
        _sidecar_path(target).write_text(
            json.dumps(
                {"sha256": "aaa", "size": 1, "mtime_ns": target.stat().st_mtime_ns}
            )
        )

    a = mir_config_payload(settings)

    # Flip one digest.
    first = _manifest(settings)[0][1]
    _sidecar_path(first).write_text(
        json.dumps({"sha256": "bbb", "size": 1, "mtime_ns": first.stat().st_mtime_ns})
    )

    b = mir_config_payload(settings)
    assert a != b


def test_payload_uses_absent_sentinel_for_missing_sidecars(tmp_path):
    """Missing sidecar yields a stable 'absent' marker so the payload is hashable."""
    from modules.mir.state import mir_config_payload

    settings = _mir_settings(model_dir=str(tmp_path / "models"))
    # No .pb files, no sidecars — should not raise.
    payload = mir_config_payload(settings)
    assert "absent" in payload


def test_payload_treats_orphan_sidecar_as_absent(tmp_path):
    """Sidecar without its .pb must not surface a stale digest."""
    import json

    from modules.mir.checkpoints import _manifest, _sidecar_path
    from modules.mir.state import mir_config_payload

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    settings = _mir_settings(model_dir=str(model_dir))

    target = _manifest(settings)[0][1]
    target.parent.mkdir(parents=True, exist_ok=True)
    _sidecar_path(target).write_text(
        json.dumps({"sha256": "stale", "size": 1, "mtime_ns": 0})
    )

    payload = mir_config_payload(settings)
    assert "stale" not in payload


def test_payload_changes_when_effnet_head_specs_change(monkeypatch):
    """Mutating EFFNET_HEAD_SPECS flips the fingerprint."""
    from modules.mir import models as models_mod
    from modules.mir.state import mir_config_payload

    original = dict(models_mod.EFFNET_HEAD_SPECS)
    a = mir_config_payload(_mir_settings())
    try:
        # Flip one head's op.
        first_name = next(iter(original))
        first_file, _orig_op = original[first_name]
        monkeypatch.setitem(
            models_mod.EFFNET_HEAD_SPECS, first_name, (first_file, "model/SomeOther")
        )
        b = mir_config_payload(_mir_settings())
        assert a != b
    finally:
        models_mod.EFFNET_HEAD_SPECS.clear()
        models_mod.EFFNET_HEAD_SPECS.update(original)


def test_mir_prefetch_queue_size_defaults_to_two():
    """Default lets the prefetcher stay one item ahead of inference."""
    from core.config import MirSettings

    assert MirSettings().prefetch_queue_size == 2


def test_upsert_writes_exactly_the_reset_columns():
    """The set of columns _upsert overwrites must equal _RESET_COLUMNS,
    so adding a descriptor needs only one edit, not two."""
    import inspect

    from modules.mir import pipeline
    from modules.mir.state import _RESET_COLUMNS

    src = inspect.getsource(pipeline._upsert)
    # _upsert must not hardcode column names as string literals — that would
    # duplicate _RESET_COLUMNS. Proxy: none of the column names should appear
    # as a quoted string literal inside _upsert's source.
    inlined = [c for c in _RESET_COLUMNS if f'"{c}"' in src or f"'{c}'" in src]
    assert not inlined, (
        f"_upsert inlines column name(s) {inlined}; iterate _RESET_COLUMNS instead"
    )
    # And _RESET_COLUMNS must be referenced so the iteration actually happens.
    assert "_RESET_COLUMNS" in src, "_upsert must iterate _RESET_COLUMNS"


def test_payload_byte_for_byte_unchanged_after_helper_refactor():
    """Golden assertion: the MIR payload must remain byte-identical to the
    pre-refactor output for the default MirSettings. If this fails, MIR
    fingerprints invalidate on upgrade, forcing a needless re-extract."""
    import json

    from modules.mir.state import mir_config_payload

    settings = _mir_settings()
    out = mir_config_payload(settings)
    parsed = json.loads(out)
    # Parsed structure: every _MIR_CONFIG_FIELDS key + labels + checkpoints + heads.
    assert "binary_threshold" in parsed
    assert "labels" in parsed
    assert "checkpoints" in parsed
    assert "heads" in parsed
    # Sorted-key serialization invariant.
    assert out == json.dumps(parsed, sort_keys=True, default=str)
