"""Sweep owns reliability + reclaim; consensus re-materialized idempotently (§3.5)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.sweep import (
    response_count,
    run_sweep,
    sweep_tick,
)
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import (
    Annotator,
    Assignment,
    Comparison,
    GoldItem,
    ReliabilityEvent,
    Response,
    Triplet,
)


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def test_sweep_demotes_a_gold_failer(session: Session) -> None:
    session.add(Comparison(comparison_id="g1", creator_a=1, creator_b=2, creator_c=3, status="gold"))
    session.add(GoldItem(comparison_id="g1", known_odd=3))
    session.add(Annotator(annotator_id="bad", reliability=0.5))
    for _ in range(6):
        session.add(
            ReliabilityEvent(annotator_id="bad", comparison_id="g1", gold_correct=False)
        )
    session.flush()
    run_sweep(session, Settings(), now=datetime.now(UTC))
    assert session.get(Annotator, "bad").reliability < 0.5


def test_sweep_reclaims_expired_assignments(session: Session) -> None:
    session.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3))
    session.add(Annotator(annotator_id="ann"))
    past = datetime.now(UTC) - timedelta(hours=1)
    session.add(
        Assignment(
            assignment_id=str(uuid.uuid4()),
            comparison_id="c1",
            annotator_id="ann",
            status="issued",
            expires_at=past,
        )
    )
    session.flush()
    run_sweep(session, Settings(), now=datetime.now(UTC))
    asg = session.query(Assignment).filter_by(comparison_id="c1").one()
    assert asg.status == "expired"


def test_sweep_is_idempotent(session: Session) -> None:
    import uuid

    session.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3, target_k=5))
    for ann in ("a", "b"):
        session.add(Annotator(annotator_id=ann, reliability=0.5))
        aid = str(uuid.uuid4())
        session.add(Assignment(assignment_id=aid, comparison_id="c1", annotator_id=ann, status="answered"))
        session.add(Response(
            response_id=str(uuid.uuid4()),
            assignment_id=aid,
            comparison_id="c1",
            annotator_id=ann,
            odd_creator_id=3,
            reaction_time_ms=3000,
            card_dwell_ms={"0": 1500, "1": 1500, "2": 1500},
            expanded=True,
        ))
    session.flush()
    s = Settings(min_overlap=2, confidence_threshold=0.6, warmup_k=1)
    now = datetime.now(UTC)

    run_sweep(session, s, now=now)
    rel1 = {a.annotator_id: round(a.reliability, 9) for a in session.query(Annotator)}
    t1 = session.query(Triplet).count()
    run_sweep(session, s, now=now)  # second run must be stable
    rel2 = {a.annotator_id: round(a.reliability, 9) for a in session.query(Annotator)}
    t2 = session.query(Triplet).count()

    assert rel1 == rel2
    assert t1 == t2 == 2  # resolved → exactly 2 consensus triplets, stable across sweeps


def _seed_two_answers(session: Session, comparison_id: str = "c1") -> None:
    session.add(
        Comparison(comparison_id=comparison_id, creator_a=1, creator_b=2, creator_c=3, target_k=5)
    )
    for ann in ("a", "b"):
        session.add(Annotator(annotator_id=ann, reliability=0.5))
        aid = str(uuid.uuid4())
        session.add(
            Assignment(assignment_id=aid, comparison_id=comparison_id, annotator_id=ann, status="answered")
        )
        session.add(
            Response(
                response_id=str(uuid.uuid4()),
                assignment_id=aid,
                comparison_id=comparison_id,
                annotator_id=ann,
                odd_creator_id=3,
                reaction_time_ms=3000,
                card_dwell_ms={"0": 1500, "1": 1500, "2": 1500},
                expanded=True,
            )
        )
    session.flush()


def test_sweep_tick_skips_recompute_when_unchanged(session: Session) -> None:
    _seed_two_answers(session)
    s = Settings(min_overlap=2, confidence_threshold=0.6, warmup_k=1)

    # First tick (no prior count) runs a full recompute.
    n = sweep_tick(session, s, None)
    assert n == 2

    # Pin a sentinel reliability; an unchanged tick must NOT overwrite it.
    session.get(Annotator, "a").reliability = 0.123456
    session.flush()
    assert sweep_tick(session, s, n) == 2
    assert session.get(Annotator, "a").reliability == 0.123456  # recompute skipped

    # A new response moves the count → the next tick recomputes (clears sentinel).
    aid = str(uuid.uuid4())
    session.add(Annotator(annotator_id="c", reliability=0.5))
    session.add(
        Assignment(assignment_id=aid, comparison_id="c1", annotator_id="c", status="answered")
    )
    session.add(
        Response(
            response_id=str(uuid.uuid4()),
            assignment_id=aid,
            comparison_id="c1",
            annotator_id="c",
            odd_creator_id=3,
            reaction_time_ms=3000,
            card_dwell_ms={"0": 1500, "1": 1500, "2": 1500},
        )
    )
    session.flush()
    assert sweep_tick(session, s, n) == 3
    assert session.get(Annotator, "a").reliability != 0.123456  # recomputed


def test_sweep_tick_still_reclaims_when_idle(session: Session) -> None:
    _seed_two_answers(session)  # count is stable at 2
    # An expired hold on a *different* comparison (avoids the per-pair uniqueness).
    session.add(Comparison(comparison_id="c2", creator_a=4, creator_b=5, creator_c=6))
    exp_id = str(uuid.uuid4())
    session.add(
        Assignment(
            assignment_id=exp_id,
            comparison_id="c2",
            annotator_id="a",
            status="issued",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
    )
    session.flush()
    n = response_count(session)

    # Unchanged count → no full recompute, but the expired hold is still reclaimed.
    assert sweep_tick(session, Settings(warmup_k=1), n, now=datetime.now(UTC)) == n
    assert session.get(Assignment, exp_id).status == "expired"
