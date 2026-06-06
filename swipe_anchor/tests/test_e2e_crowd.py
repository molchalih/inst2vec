"""End-to-end: planted crowd → consensus + reliability separation + export (§7)."""

import random
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import next_batch, record_response
from swipe_anchor.backend.sweep import run_sweep
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import Annotator, Assignment, Comparison, GoldItem, Triplet
from swipe_anchor.export.anchor_export import export_anchor


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def test_planted_crowd_resolves_and_demotes_griefer(session: Session, tmp_path) -> None:
    session.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3, target_k=10))
    session.add(Comparison(comparison_id="c2", creator_a=10, creator_b=20, creator_c=30, target_k=10))
    session.add(Comparison(comparison_id="g1", creator_a=4, creator_b=5, creator_c=6, status="gold", target_k=10))
    session.add(GoldItem(comparison_id="g1", known_odd=6))
    session.commit()

    truth = {"c1": 3, "c2": 30}
    gold_odd = 6
    settings = Settings(min_overlap=4, confidence_threshold=0.6, warmup_k=1, p_gold=1.0, max_inflight=10)
    now = datetime.now(UTC)

    def answer_all(ann: str, correct: bool) -> None:
        next_batch(session, ann, n=5, rng=random.Random(0), settings=settings, now=now)
        session.flush()
        issued = list(session.query(Assignment).filter_by(annotator_id=ann, status="issued"))
        for asg in issued:
            cmp = session.get(Comparison, asg.comparison_id)
            if cmp.comparison_id == "g1":
                odd = gold_odd if correct else 4  # griefer fails the catch-trial
            else:
                t = truth[cmp.comparison_id]
                odd = t if correct else next(c for c in cmp.creators if c != t)
            if correct:
                record_response(
                    session, asg.assignment_id, odd_id=odd, settings=settings,
                    expected_annotator_id=ann, reaction_time_ms=3000,
                    card_dwell_ms={"0": 1500, "1": 1500, "2": 1500}, expanded=True,
                )
            else:
                record_response(
                    session, asg.assignment_id, odd_id=odd, settings=settings,
                    expected_annotator_id=ann, reaction_time_ms=80,
                    card_dwell_ms={"0": 0}, expanded=False,
                )
        session.commit()

    for ann in ("good1", "good2", "good3"):
        answer_all(ann, correct=True)
    answer_all("griefer", correct=False)

    run_sweep(session, settings, now=now)
    session.commit()

    # Both real comparisons resolved to the planted odd creator.
    assert session.get(Comparison, "c1").status == "retired"
    assert session.get(Comparison, "c2").status == "retired"

    # The griefer's reliability is below every good annotator's.
    g = session.get(Annotator, "griefer").reliability
    goods = [session.get(Annotator, a).reliability for a in ("good1", "good2", "good3")]
    assert g < min(goods)

    # Export emits exactly two consensus triplets per resolved comparison.
    meta = export_anchor(session, tmp_path / "anchor", build_timestamp=now)
    assert meta["counts"]["triplets"] == 4
    assert session.query(Triplet).count() == 4
