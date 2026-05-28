from datetime import datetime

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from core.database import (
    Base,
    Clip,
    ClipLabel,
    User,
    clip_label_done,
    clip_needs_label,
)


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed_clip(s: Session, *, clip_id: int, selected: bool) -> None:
    s.add(User(id=1, is_selected=True))
    s.add(
        Clip(
            id=clip_id,
            user_id=1,
            is_selected=selected,
            is_downloaded=True,
        )
    )
    s.commit()


def test_clip_label_round_trip() -> None:
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s, clip_id=10, selected=True)
        s.add(
            ClipLabel(
                clip_id=10,
                label_case="video",
                status="success",
                validation="ok",
                payload={"one_sentence_visual_reading": "hello"},
                warnings=[],
                attempts=1,
                source_hash="abc",
            )
        )
        s.commit()
        row = s.get(ClipLabel, (10, "video"))
        assert row is not None
        assert row.status == "success"
        assert row.validation == "ok"
        assert row.payload == {"one_sentence_visual_reading": "hello"}
        assert row.warnings == []
        assert row.attempts == 1
        assert row.source_hash == "abc"
        assert isinstance(row.updated_at, datetime)


def test_clip_label_composite_pk_is_clip_id_and_label_case() -> None:
    eng = _engine()
    insp = inspect(eng)
    pk = insp.get_pk_constraint("clip_labels")["constrained_columns"]
    assert pk == ["clip_id", "label_case"]


def test_clip_label_source_hash_column_present() -> None:
    eng = _engine()
    insp = inspect(eng)
    cols = {c["name"]: c for c in insp.get_columns("clip_labels")}
    assert "source_hash" in cols
    assert cols["source_hash"]["nullable"] is True


def test_clip_label_composite_pk_distinct_per_case() -> None:
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s, clip_id=10, selected=True)
        s.add(ClipLabel(clip_id=10, label_case="video", status="success", attempts=1))
        s.add(ClipLabel(clip_id=10, label_case="audio", status="success", attempts=1))
        s.commit()
        rows = (
            s.execute(
                select(ClipLabel)
                .where(ClipLabel.clip_id == 10)
                .order_by(ClipLabel.label_case)
            )
            .scalars()
            .all()
        )
        assert [r.label_case for r in rows] == ["audio", "video"]


def test_clip_needs_label_predicate() -> None:
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s, clip_id=10, selected=True)
        # No label row → needs labelling.
        ids = (
            s.execute(
                select(Clip.id)
                .outerjoin(ClipLabel, ClipLabel.clip_id == Clip.id)
                .where(*clip_needs_label())
            )
            .scalars()
            .all()
        )
        assert ids == [10]

        # Insert a pending row → still needs labelling.
        s.add(ClipLabel(clip_id=10, label_case="video", status="pending", attempts=0))
        s.commit()
        ids = (
            s.execute(
                select(Clip.id)
                .outerjoin(ClipLabel, ClipLabel.clip_id == Clip.id)
                .where(*clip_needs_label())
            )
            .scalars()
            .all()
        )
        assert ids == [10]

        # Mark success → no longer needs labelling.
        row = s.get(ClipLabel, (10, "video"))
        assert row is not None
        row.status = "success"
        s.commit()
        ids = (
            s.execute(
                select(Clip.id)
                .outerjoin(ClipLabel, ClipLabel.clip_id == Clip.id)
                .where(*clip_needs_label())
            )
            .scalars()
            .all()
        )
        assert ids == []


def test_clip_label_done_predicate() -> None:
    eng = _engine()
    with Session(eng) as s:
        _seed_clip(s, clip_id=10, selected=True)
        s.add(
            ClipLabel(clip_id=10, label_case="video", status="success", validation="ok")
        )
        s.commit()
        ids = (
            s.execute(
                select(Clip.id)
                .outerjoin(ClipLabel, ClipLabel.clip_id == Clip.id)
                .where(*clip_label_done())
            )
            .scalars()
            .all()
        )
        assert ids == [10]
