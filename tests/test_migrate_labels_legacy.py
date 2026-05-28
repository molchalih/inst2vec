"""Tests for the one-shot labels-stage legacy migration script.

Two responsibilities to cover:

1. ``backfill_label_case_nulls`` rewrites rows where ``label_case`` is NULL
   or empty to ``"video"``, and is idempotent on subsequent runs.
2. ``adopt_legacy_video_fingerprint`` rewrites a stage_state row whose
   ``data_hash``/``dependency_hash`` are ``hash_text("")`` and whose
   ``config_hash`` matches the legacy payload — and is a no-op on any
   other shape, on a missing row, and on a row already in the new shape.

The legacy compat helpers were removed from the runtime in the same commit
that landed this script. The tests exercise the script's embedded copies of
the legacy logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import update
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import LabelsSettings
from core.database import Clip, ClipLabel, StageState, init_db
from core.database.engine import get_engine
from modules.labels.state import STAGE_LABELS, clip_scope_for
from scripts.migrate_labels_legacy import (
    _legacy_video_clip_labels_config_payload,
    adopt_legacy_video_fingerprint,
    backfill_label_case_nulls,
)


def _make_clip(session: Session, *, clip_id: int, user_id: int) -> None:
    session.add(
        Clip(
            id=clip_id,
            user_id=user_id,
            is_selected=True,
            is_downloaded=True,
        )
    )


def _make_user(session: Session, *, user_id: int) -> None:
    from core.database import User

    session.add(User(id=user_id))


def _legacy_labels() -> LabelsSettings:
    """Minimal LabelsSettings carrying a non-empty video case prompt so the
    legacy payload is non-trivial."""
    return LabelsSettings(
        case_prompts={"video": "describe the clip"},
        cluster_case_prompts={"video": "cluster"},
    )


def _seed_legacy_stage_state(session: Session, labels: LabelsSettings) -> None:
    """Write a stage_state row in the legacy shape — empty data + dep
    hashes, config hash from the legacy payload."""
    empty = fp.hash_text("")
    config_hash = fp.hash_text(_legacy_video_clip_labels_config_payload(labels))
    session.add(
        StageState(
            stage_name=STAGE_LABELS,
            scope_key=clip_scope_for("video"),
            data_hash=empty,
            config_hash=config_hash,
            dependency_hash=empty,
        )
    )


def _fake_settings(labels: LabelsSettings, tmp_video_dir: str) -> SimpleNamespace:
    """Build a settings stand-in good enough for ``_current_fingerprint``.

    ``_current_fingerprint`` for the video case calls
    ``_data_hash_for_video`` which reads ``settings.paths.video_for(cid)``
    and stats each file. We supply a ``MagicMock``-based paths object that
    yields stable per-id paths so the hash is reproducible.
    """
    paths = MagicMock()
    paths.video_for = lambda cid: f"{tmp_video_dir}/{cid}.mp4"
    return SimpleNamespace(labels=labels, paths=paths)


def _seed_video_file(tmp_path, clip_id: int) -> None:
    """Write a tiny placeholder so ``file_stat_for_hash`` returns a stable
    (size, mtime) instead of raising."""
    p = tmp_path / f"{clip_id}.mp4"
    p.write_bytes(b"x")


# ── backfill_label_case_nulls ───────────────────────────────────────────────


def test_backfill_label_case_empty_string_rewrites_to_video() -> None:
    """The post-multimodal ORM enforces ``label_case NOT NULL`` so the NULL
    leg can only happen on a DB written by code that bypassed the ORM. The
    empty-string leg is reachable through the ORM (column has a default but
    accepts ``""``) so we cover it here; the NULL leg is covered by the
    ``or_(...is_(None)...)`` clause in the implementation and exercised by
    the legacy-shape conversion script when run against real production DBs.
    """
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        _make_clip(session, clip_id=10, user_id=1)
        _make_clip(session, clip_id=11, user_id=1)
        session.add_all(
            [
                ClipLabel(clip_id=10, label_case="video", status="success", attempts=1),
                ClipLabel(clip_id=11, label_case="video", status="success", attempts=1),
            ]
        )
        session.commit()
        session.execute(
            update(ClipLabel).where(ClipLabel.clip_id == 11).values(label_case="")
        )
        session.commit()

        affected = backfill_label_case_nulls(session)
        session.commit()
        assert affected == 1

        cases = sorted(
            (r.clip_id, r.label_case) for r in session.query(ClipLabel).all()
        )
        assert cases == [(10, "video"), (11, "video")]


def test_backfill_label_case_nulls_is_idempotent() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        _make_user(session, user_id=1)
        _make_clip(session, clip_id=10, user_id=1)
        session.add(
            ClipLabel(clip_id=10, label_case="video", status="success", attempts=1)
        )
        session.commit()

        first = backfill_label_case_nulls(session)
        session.commit()
        assert first == 0

        second = backfill_label_case_nulls(session)
        session.commit()
        assert second == 0


# ── adopt_legacy_video_fingerprint ──────────────────────────────────────────


def test_adopt_rewrites_legacy_shape(tmp_path) -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        labels = _legacy_labels()
        _seed_legacy_stage_state(session, labels)
        session.commit()

        settings = _fake_settings(labels, str(tmp_path))
        # No selected clips → data hash for video is hash_rows([]) which is
        # not the empty string, so the new fingerprint differs from the
        # legacy shape and the adoption MUST rewrite the row.
        rewritten = adopt_legacy_video_fingerprint(session, settings=settings)
        session.commit()
        assert rewritten is True

        row = session.get(StageState, (STAGE_LABELS, clip_scope_for("video")))
        assert row is not None
        # Adoption replaces the legacy empty data_hash with a fresh
        # hash_rows([]) digest (non-empty for non-empty inputs; here the
        # selected-clip set is empty so the digest is fp.hash_rows([])).
        assert row.data_hash == fp.hash_rows([])


def test_adopt_no_row_is_noop() -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        settings = _fake_settings(_legacy_labels(), "/tmp/nope")
        rewritten = adopt_legacy_video_fingerprint(session, settings=settings)
        session.commit()
        assert rewritten is False
        assert session.get(StageState, (STAGE_LABELS, clip_scope_for("video"))) is None


def test_adopt_already_new_shape_is_noop(tmp_path) -> None:
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        labels = _legacy_labels()
        # Seed a row whose data_hash is NOT hash_text("") so the legacy
        # detector skips it.
        session.add(
            StageState(
                stage_name=STAGE_LABELS,
                scope_key=clip_scope_for("video"),
                data_hash="not-empty",
                config_hash="not-empty",
                dependency_hash="not-empty",
            )
        )
        session.commit()

        settings = _fake_settings(labels, str(tmp_path))
        rewritten = adopt_legacy_video_fingerprint(session, settings=settings)
        session.commit()
        assert rewritten is False

        row = session.get(StageState, (STAGE_LABELS, clip_scope_for("video")))
        assert row is not None
        assert row.data_hash == "not-empty"


def test_adopt_legacy_other_variant_is_noop(tmp_path) -> None:
    """If the row is in a non-empty legacy variant (e.g. dependency_hash is
    populated), let the normal drift path handle it — adoption stays out."""
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        labels = _legacy_labels()
        empty = fp.hash_text("")
        legacy_config_hash = fp.hash_text(
            _legacy_video_clip_labels_config_payload(labels)
        )
        session.add(
            StageState(
                stage_name=STAGE_LABELS,
                scope_key=clip_scope_for("video"),
                data_hash=empty,
                config_hash=legacy_config_hash,
                dependency_hash="not-empty",  # diverges from legacy shape
            )
        )
        session.commit()

        settings = _fake_settings(labels, str(tmp_path))
        rewritten = adopt_legacy_video_fingerprint(session, settings=settings)
        session.commit()
        assert rewritten is False

        row = session.get(StageState, (STAGE_LABELS, clip_scope_for("video")))
        assert row is not None
        assert row.dependency_hash == "not-empty"


def test_adopt_is_idempotent_on_post_migration_row(tmp_path) -> None:
    """After a successful adoption, re-running must be a no-op."""
    init_db("sqlite:///:memory:", "sqlite:///:memory:")
    with Session(get_engine()) as session:
        labels = _legacy_labels()
        _seed_legacy_stage_state(session, labels)
        session.commit()

        settings = _fake_settings(labels, str(tmp_path))
        first = adopt_legacy_video_fingerprint(session, settings=settings)
        session.commit()
        assert first is True

        second = adopt_legacy_video_fingerprint(session, settings=settings)
        session.commit()
        assert second is False
