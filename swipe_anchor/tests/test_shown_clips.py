"""Recording the shown reel per creator + backfilling historical rows."""

from __future__ import annotations

from contextlib import contextmanager

from swipe_anchor.backend.backfill_reels import backfill
from swipe_anchor.backend.service import record_response
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope
from swipe_anchor.db.models import (
    Annotator,
    Assignment,
    Comparison,
    Creator,
    DigestClip,
    Response,
)


def _factory():
    engine = create_app_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)

    @contextmanager
    def sf():
        with session_scope(factory) as s:
            yield s

    return engine, sf


def _seed_comparison(s, cmp_id: str) -> None:
    s.add(Comparison(comparison_id=cmp_id, creator_a=10, creator_b=20, creator_c=30))
    for cid in (10, 20, 30):
        s.add(Creator(creator_id=cid))
    # creator 10 has two clips; clips[0] (lowest ord) is what the card shows.
    s.add(DigestClip(creator_id=10, clip_id=112, ord=1))
    s.add(DigestClip(creator_id=10, clip_id=111, ord=0))
    s.add(DigestClip(creator_id=20, clip_id=222, ord=0))
    s.add(DigestClip(creator_id=30, clip_id=333, ord=0))


def test_record_response_persists_shown_clips() -> None:
    _engine, sf = _factory()
    with sf() as s:
        s.add(Annotator(annotator_id="a"))
        _seed_comparison(s, "c1")
        s.add(Assignment(assignment_id="asg", comparison_id="c1", annotator_id="a"))
    with sf() as s:
        record_response(
            s,
            "asg",
            odd_id=10,
            shown_clips={"10": 111, "20": 222, "30": 333},
            expected_annotator_id="a",
        )
    with sf() as s:
        row = s.query(Response).one()
        assert row.shown_clips == {"10": 111, "20": 222, "30": 333}


def test_backfill_reconstructs_from_lowest_ord_clip() -> None:
    _engine, sf = _factory()
    with sf() as s:
        s.add(Annotator(annotator_id="a"))
        _seed_comparison(s, "c1")
        s.add(Assignment(assignment_id="asg", comparison_id="c1", annotator_id="a"))
        s.add(
            Response(
                response_id="r1",
                assignment_id="asg",
                comparison_id="c1",
                annotator_id="a",
                odd_creator_id=10,
                shown_clips=None,
            )
        )
    with sf() as s:
        updated, skipped = backfill(s)
        assert (updated, skipped) == (1, 0)
    with sf() as s:
        row = s.query(Response).one()
        assert row.shown_clips == {"10": 111, "20": 222, "30": 333}


def test_backfill_leaves_existing_and_skips_clipless() -> None:
    _engine, sf = _factory()
    with sf() as s:
        s.add(Annotator(annotator_id="a"))
        _seed_comparison(s, "c1")
        # A comparison whose creators have no digest clips → nothing to record.
        s.add(Comparison(comparison_id="c2", creator_a=70, creator_b=80, creator_c=90))
        s.add(Assignment(assignment_id="asg1", comparison_id="c1", annotator_id="a"))
        s.add(Assignment(assignment_id="asg2", comparison_id="c2", annotator_id="a"))
        s.add(
            Response(
                response_id="done",
                assignment_id="asg1",
                comparison_id="c1",
                annotator_id="a",
                shown_clips={"10": 999},  # already set — must not be overwritten
            )
        )
        s.add(
            Response(
                response_id="clipless",
                assignment_id="asg2",
                comparison_id="c2",
                annotator_id="a",
                shown_clips=None,
            )
        )
    with sf() as s:
        updated, skipped = backfill(s)
        assert updated == 0  # one already set (not selected), one has no clips
        assert skipped == 1
    with sf() as s:
        kept = s.get(Response, "done")
        assert kept.shown_clips == {"10": 999}


def test_dry_run_does_not_write(tmp_path) -> None:
    # A dry-run reports the same counts but leaves shown_clips untouched.
    engine = create_app_engine(f"sqlite:///{tmp_path / 'a.db'}")
    factory = make_session_factory(engine)

    @contextmanager
    def sf():
        with session_scope(factory) as s:
            yield s

    with sf() as s:
        s.add(Annotator(annotator_id="a"))
        _seed_comparison(s, "c1")
        s.add(Assignment(assignment_id="asg", comparison_id="c1", annotator_id="a"))
        s.add(
            Response(
                response_id="r1",
                assignment_id="asg",
                comparison_id="c1",
                annotator_id="a",
                shown_clips=None,
            )
        )
    with sf() as s:
        assert backfill(s, dry_run=True) == (1, 0)
    with sf() as s:
        assert s.get(Response, "r1").shown_clips in (None, {})
