"""Per-item consensus glue (design §3.8, §4).

Maps creator ids <-> odd-one-out class indices (the canonical sorted order),
recomputes one comparison's consensus from its responses using the CURRENT
(frozen) annotator reliabilities, writes the Consensus row, applies §4
confident-consensus retirement, and materializes the two consensus triplets.

This is the single place that owns a comparison's Consensus row + status +
triplets — called inline (Task 8) and by the sweep (Task 10). It never writes
``annotators.reliability`` (sweep-owned).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from swipe_anchor.config import Settings
from swipe_anchor.core.consensus import Vote, update_item
from swipe_anchor.core.triplets import derive_triplets
from swipe_anchor.db.models import Annotator, Comparison, Consensus, Response, Triplet


def classes_for_comparison(cmp: Comparison) -> list[int]:
    """Canonical sorted creator order; class index == position in this list."""
    return sorted(cmp.creators)


def recompute_item(session: Session, comparison_id: str, settings: Settings) -> None:
    cmp = session.get(Comparison, comparison_id)
    if cmp is None or cmp.status == "gold":
        return  # gold items never resolve/retire (plan §4.3)
    order = classes_for_comparison(cmp)
    class_of = {cid: i for i, cid in enumerate(order)}

    responses = list(
        session.query(Response).filter_by(comparison_id=comparison_id).all()
    )
    votes = [
        Vote(
            annotator_id=r.annotator_id,
            label=None if r.odd_creator_id is None else class_of[r.odd_creator_id],
        )
        for r in responses
    ]
    voter_ids = {r.annotator_id for r in responses}
    rel = {
        a.annotator_id: a.reliability
        for a in session.query(Annotator).filter(Annotator.annotator_id.in_(voter_ids))
    } if voter_ids else {}
    est = update_item(votes, rel)

    consensus_odd = None if est.consensus_class is None else order[est.consensus_class]
    cons = session.get(Consensus, comparison_id)
    if cons is None:
        cons = Consensus(comparison_id=comparison_id)
        session.add(cons)
    cons.consensus_odd = consensus_odd
    cons.agreement = est.agreement
    cons.n_effective = est.n_effective

    n = len(responses)
    confident = (
        n >= settings.min_overlap
        and est.posterior_max >= settings.confidence_threshold
        and consensus_odd is not None
    )
    if confident:
        cmp.status = "retired"
        cons.resolved = True
        _materialize_triplets(session, cmp, consensus_odd, est.posterior_max)
    elif n >= settings.max_overlap:
        cmp.status = "ambiguous"  # near-tie kept as signal; no triplet emitted
        cons.resolved = False
        _clear_triplets(session, comparison_id)
    elif cmp.status == "retired":
        # Previously retired, but a reliability shift dropped it under-confident.
        # Keep it retired, but keep its triplets consistent with the CURRENT
        # consensus (refresh weight/direction, or drop if now ambiguous) so the
        # export never carries stale triplets (design §3.8).
        cons.resolved = consensus_odd is not None
        if consensus_odd is not None:
            _materialize_triplets(session, cmp, consensus_odd, est.posterior_max)
        else:
            _clear_triplets(session, comparison_id)
    # else: still open — no triplets yet
    session.flush()


def _clear_triplets(session: Session, comparison_id: str) -> None:
    session.query(Triplet).filter_by(comparison_id=comparison_id).delete()


def _materialize_triplets(
    session: Session, cmp: Comparison, consensus_odd: int, weight: float
) -> None:
    """Replace this comparison's triplets with the two derived from consensus."""
    _clear_triplets(session, cmp.comparison_id)
    for t in derive_triplets(cmp.creators, consensus_odd):
        session.add(
            Triplet(
                anchor_id=t.anchor,
                positive_id=t.positive,
                negative_id=t.negative,
                comparison_id=cmp.comparison_id,
                weight=weight,
            )
        )
