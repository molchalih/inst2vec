"""Aggregate collection stats for the admin ``/stats`` view.

Pure over a SQLAlchemy ``Session`` (no HTTP), so it is unit-testable and the
``/tg/stats`` endpoint stays a thin shell. Returns plain JSON-able dicts the bot
turns into a text overview + a few rendered charts.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from swipe_anchor.db.models import (
    AccessCode,
    Annotator,
    Comparison,
    Consensus,
    Response,
    Triplet,
)


def _short(code: str) -> str:
    """A compact, non-leaky fallback label when a code has no operator note."""
    return f"{code[:8]}…" if len(code) > 9 else code


def gather_stats(session: Session, *, top_contributors: int = 12) -> dict:
    """Snapshot the collection: headline totals, a time series, contributors."""

    def count(model, **filters) -> int:
        q = session.query(func.count()).select_from(model)
        return int(q.filter_by(**filters).scalar() if filters else q.scalar())

    comparisons = {
        "open": count(Comparison, status="open"),
        "retired": count(Comparison, status="retired"),
        "ambiguous": count(Comparison, status="ambiguous"),
        "gold": count(Comparison, status="gold"),
    }

    rels = [a.reliability for a in session.query(Annotator)]
    gold_seen = int(
        session.query(func.coalesce(func.sum(Annotator.n_gold_seen), 0)).scalar()
    )
    gold_correct = int(
        session.query(func.coalesce(func.sum(Annotator.n_gold_correct), 0)).scalar()
    )
    agreements = [
        c.agreement for c in session.query(Consensus).filter_by(resolved=True)
    ]

    # Sorted response timestamps drive the cumulative "datapoints over time" chart.
    response_times = [
        r.created_at.isoformat()
        for r in session.query(Response).order_by(Response.created_at)
        if r.created_at is not None
    ]

    # Top contributors, labelled by the operator note when present (this view is
    # admin-only) and otherwise a short, non-reversible id stub.
    notes = {c.code: c.note for c in session.query(AccessCode)}
    rows = (
        session.query(Response.annotator_id, func.count().label("n"))
        .group_by(Response.annotator_id)
        .order_by(func.count().desc())
        .limit(top_contributors)
        .all()
    )
    per_annotator = [
        {"label": (notes.get(ann_id) or _short(ann_id)), "n": int(n)}
        for ann_id, n in rows
    ]

    return {
        "totals": {
            "responses": count(Response),
            "triplets": count(Triplet),
            "annotators": count(Annotator),
            "comparisons": comparisons,
            "resolved": comparisons["retired"] + comparisons["ambiguous"],
            "mean_reliability": (sum(rels) / len(rels)) if rels else 0.0,
            "mean_agreement": (sum(agreements) / len(agreements))
            if agreements
            else 0.0,
            "gold_seen": gold_seen,
            "gold_correct": gold_correct,
        },
        "response_times": response_times,
        "per_annotator": per_annotator,
    }
