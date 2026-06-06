"""Backend assignment + respond logic (plan §4.4, §4.5 — Phase-1 subset).

Phase 1 balancer is deliberately simple (random over eligible); the active /
consensus machinery is Phases 2-4. These tests pin the invariants that must hold
from day one: eligibility, no self-collision, idempotent respond, judgment
counting, and consensus-based retirement.

Triplets are now consensus-materialized by recompute_item (Task 7/8), not emitted
per-response. Tests that assert triplet emission use
``settings=Settings(min_overlap=1, confidence_threshold=0.0)`` to force an
immediate confident retire in the inline-consensus path, producing 2 consensus
triplets (with response_id=None).
"""

import random

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import (
    InvalidOddCreatorError,
    UnknownAssignmentError,
    next_batch,
    record_response,
)
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import Annotator, Comparison, Response, Triplet


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def _add_comparison(
    s: Session, cid: str, a: int, b: int, c: int, target_k: int = 5
) -> None:
    s.add(
        Comparison(
            comparison_id=cid, creator_a=a, creator_b=b, creator_c=c, target_k=target_k
        )
    )


def test_next_batch_issues_assignments_for_eligible(session: Session) -> None:
    _add_comparison(session, "c1", 1, 2, 3)
    _add_comparison(session, "c2", 4, 5, 6)
    session.commit()

    picks = next_batch(session, "ann-1", n=2, rng=random.Random(0))
    session.commit()

    assert {c.comparison_id for c in picks} == {"c1", "c2"}
    # The annotator row is created on first contact (anonymous session).
    assert session.get(Annotator, "ann-1") is not None


def test_next_batch_excludes_already_seen_by_annotator(session: Session) -> None:
    _add_comparison(session, "c1", 1, 2, 3)
    session.commit()
    next_batch(session, "ann-1", n=5, rng=random.Random(0))
    session.commit()

    again = next_batch(session, "ann-1", n=5, rng=random.Random(0))
    session.commit()
    assert again == []


def test_next_batch_excludes_retired_and_capped(session: Session) -> None:
    _add_comparison(session, "c1", 1, 2, 3, target_k=1)
    session.add(
        Comparison(
            comparison_id="c2", creator_a=4, creator_b=5, creator_c=6, status="retired"
        )
    )
    session.commit()
    # "Full" now means the HARD cap (max_overlap), not target_k: an open item that
    # merely reached target_k without resolving must STAY eligible (see
    # test_service_batch.test_unresolved_item_past_target_k_stays_eligible). Only
    # at max_overlap (default 9) does it drop out of the draw.
    capped = session.get(Comparison, "c1")
    capped.n_judgments = 9
    session.commit()

    picks = next_batch(session, "ann-1", n=5, rng=random.Random(0))
    assert picks == []


def test_respond_non_skip_emits_two_triplets_and_counts(session: Session) -> None:
    # Triplets are now consensus-materialized; force immediate confident retire so
    # the 2 consensus triplets appear (response_id=None, not per-response-scoped).
    _add_comparison(session, "c1", 10, 20, 30)
    session.commit()
    next_batch(session, "ann-1", n=1, rng=random.Random(0))
    session.commit()
    assignment_id = _assignment_id_for(session, "ann-1", "c1")

    result = record_response(
        session, assignment_id, odd_id=30, confidence=1.0,
        settings=Settings(min_overlap=1, confidence_threshold=0.0),
    )
    session.commit()

    triplets = session.query(Triplet).all()
    assert len(triplets) == 2
    assert {(t.anchor_id, t.positive_id, t.negative_id) for t in triplets} == {
        (10, 20, 30),
        (20, 10, 30),
    }
    assert all(t.response_id is None for t in triplets)  # consensus-materialized
    assert session.get(Comparison, "c1").n_judgments == 1
    assert result.n_triplets == 2


def test_respond_skip_counts_but_emits_no_triplets(session: Session) -> None:
    _add_comparison(session, "c1", 10, 20, 30)
    session.commit()
    next_batch(session, "ann-1", n=1, rng=random.Random(0))
    session.commit()
    assignment_id = _assignment_id_for(session, "ann-1", "c1")

    result = record_response(session, assignment_id, odd_id=None, confidence=1.0)
    session.commit()

    assert session.query(Triplet).count() == 0
    assert session.get(Comparison, "c1").n_judgments == 1
    assert result.n_triplets == 0


def test_respond_is_idempotent(session: Session) -> None:
    # Force confident retire so consensus triplets appear; retry must not double them.
    _add_comparison(session, "c1", 10, 20, 30)
    session.commit()
    next_batch(session, "ann-1", n=1, rng=random.Random(0))
    session.commit()
    assignment_id = _assignment_id_for(session, "ann-1", "c1")

    settings = Settings(min_overlap=1, confidence_threshold=0.0)
    record_response(session, assignment_id, odd_id=30, confidence=1.0, settings=settings)
    session.commit()
    record_response(session, assignment_id, odd_id=30, confidence=1.0, settings=settings)  # retry
    session.commit()

    assert session.query(Triplet).count() == 2  # not doubled (consensus-materialized)
    assert session.query(Response).count() == 1  # exactly one judgment row
    assert session.get(Comparison, "c1").n_judgments == 1


def test_respond_unknown_assignment_raises_typed_error(session: Session) -> None:
    with pytest.raises(UnknownAssignmentError):
        record_response(session, "does-not-exist", odd_id=1, confidence=1.0)


def test_respond_odd_not_in_comparison_raises_typed_error(session: Session) -> None:
    _add_comparison(session, "c1", 10, 20, 30)
    session.commit()
    next_batch(session, "ann-1", n=1, rng=random.Random(0))
    session.commit()
    assignment_id = _assignment_id_for(session, "ann-1", "c1")

    with pytest.raises(InvalidOddCreatorError):
        record_response(session, assignment_id, odd_id=999, confidence=1.0)

    # A rejected bad submission must not consume the assignment.
    from swipe_anchor.db.models import Assignment

    assert session.get(Assignment, assignment_id).status == "issued"
    assert session.get(Comparison, "c1").n_judgments == 0


def test_respond_increments_judgments_from_db_not_stale_orm(tmp_path) -> None:
    # Item 4: a concurrent answer that already landed in the DB (committed by a
    # DIFFERENT session) must not be clobbered by a Python-side += on a stale
    # in-memory n_judgments. A file-backed DB gives the two sessions independent
    # connections/transactions (unlike a shared :memory: engine).
    engine = create_app_engine(str(tmp_path / "app.db"))
    with Session(engine) as s1:
        _add_comparison(s1, "c1", 10, 20, 30, target_k=10)
        s1.commit()
        next_batch(s1, "ann-1", n=1, rng=random.Random(0))
        s1.commit()
        assignment_id = _assignment_id_for(s1, "ann-1", "c1")

        cmp = s1.get(Comparison, "c1")  # s1 now caches n_judgments == 0
        assert cmp.n_judgments == 0

        # A concurrent writer commits a judgment in a separate session.
        with Session(engine) as s2:
            s2.execute(
                update(Comparison)
                .where(Comparison.comparison_id == "c1")
                .values(n_judgments=5)
            )
            s2.commit()

        # s1's identity map is still stale (0); the DB says 5.
        record_response(s1, assignment_id, odd_id=30, confidence=1.0)
        s1.commit()

        s1.refresh(cmp)
        assert cmp.n_judgments == 6  # 5 (concurrent) + 1, not 1


def test_respond_retires_comparison_at_quorum(session: Session) -> None:
    # Retirement is now consensus-driven; force confident resolve with 1 response.
    _add_comparison(session, "c1", 10, 20, 30, target_k=1)
    session.commit()
    next_batch(session, "ann-1", n=1, rng=random.Random(0))
    session.commit()
    assignment_id = _assignment_id_for(session, "ann-1", "c1")

    result = record_response(
        session, assignment_id, odd_id=30, confidence=1.0,
        settings=Settings(min_overlap=1, confidence_threshold=0.0),
    )
    session.commit()

    assert session.get(Comparison, "c1").status == "retired"
    assert result.retired is True


def _assignment_id_for(session: Session, annotator_id: str, comparison_id: str) -> str:
    from swipe_anchor.db.models import Assignment

    asg = (
        session.query(Assignment)
        .filter_by(annotator_id=annotator_id, comparison_id=comparison_id)
        .one()
    )
    return asg.assignment_id
