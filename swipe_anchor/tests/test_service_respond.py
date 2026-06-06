"""record_response: inline consensus, no per-response triplets, strict ownership."""

import uuid

import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import record_response
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import (
    Annotator,
    Assignment,
    Comparison,
    GoldItem,
    ReliabilityEvent,
    Triplet,
)


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def _assign(s: Session, cmp_id: str, ann: str) -> str:
    """Insert a comparison + annotator + an issued assignment directly (no next_batch dep)."""
    s.add(Comparison(comparison_id=cmp_id, creator_a=1, creator_b=2, creator_c=3, target_k=1))
    s.add(Annotator(annotator_id=ann))
    aid = str(uuid.uuid4())
    s.add(Assignment(assignment_id=aid, comparison_id=cmp_id, annotator_id=ann, status="issued"))
    s.flush()
    return aid


def test_respond_emits_no_per_response_triplets(session: Session) -> None:
    aid = _assign(session, "c1", "ann")
    record_response(session, aid, odd_id=3, settings=Settings(min_overlap=1, confidence_threshold=0.0))
    trips = session.query(Triplet).filter_by(comparison_id="c1").all()
    # min_overlap=1 + threshold 0 → resolves immediately → exactly the 2 consensus triplets.
    assert len(trips) == 2
    assert all(t.response_id is None for t in trips)  # not response-scoped


def test_respond_appends_gold_event_and_keeps_reliability_untouched(session: Session) -> None:
    aid = _assign(session, "g1", "ann")
    session.add(GoldItem(comparison_id="g1", known_odd=3))
    session.get(Comparison, "g1").status = "gold"
    session.flush()
    before = session.get(Annotator, "ann").reliability
    record_response(session, aid, odd_id=3, settings=Settings())
    ev = session.query(ReliabilityEvent).filter_by(comparison_id="g1", annotator_id="ann").one()
    assert ev.gold_correct is True
    assert session.get(Annotator, "ann").reliability == before  # sweep-owned; untouched


def test_respond_idempotent_still_holds(session: Session) -> None:
    aid = _assign(session, "c2", "ann")
    r1 = record_response(session, aid, odd_id=3, settings=Settings())
    r2 = record_response(session, aid, odd_id=3, settings=Settings())
    assert r1.accepted and not r2.accepted


def test_skip_on_gold_counts_as_failed_exposure(session: Session) -> None:
    # [P2] Skipping an obvious catch-trial is a QC failure, not a non-event.
    aid = _assign(session, "g1", "ann")
    session.add(GoldItem(comparison_id="g1", known_odd=3))
    session.get(Comparison, "g1").status = "gold"
    session.flush()
    record_response(session, aid, odd_id=None, settings=Settings())  # skip the gold trial
    ev = session.query(ReliabilityEvent).filter_by(comparison_id="g1", annotator_id="ann").one()
    assert ev.gold_correct is False  # seen + incorrect, not NULL (so it can't be dodged)


def test_skip_on_non_gold_is_not_a_gold_event(session: Session) -> None:
    # [P2] A skip on a normal item is still NULL (not a gold exposure).
    aid = _assign(session, "c1", "ann")
    record_response(session, aid, odd_id=None, settings=Settings())
    ev = session.query(ReliabilityEvent).filter_by(comparison_id="c1", annotator_id="ann").one()
    assert ev.gold_correct is None
