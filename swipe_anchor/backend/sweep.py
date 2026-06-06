"""Authoritative reconciliation pass (design §2, §3.5).

Sole owner of ``annotators.reliability``. Runs full Dawid-Skene EM over the
response history, blends gold-anchored reliability, refines every comparison's
consensus (re-materializing triplets), and reclaims expired assignments.

The background loop is **change-gated**: the full recompute only runs when the
response history actually changed since the last tick (a cheap COUNT). When idle
it does just the trivial expired-assignment reclaim, so there is no constant
EM/consensus churn. ``run_sweep`` itself remains a full pass for the CLI/cron and
tests.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from swipe_anchor.backend.consensus_writer import classes_for_comparison, recompute_item
from swipe_anchor.config import Settings
from swipe_anchor.core.consensus import Vote, full_em
from swipe_anchor.core.reliability import (
    behavioral_factor,
    gold_accuracy,
    is_trusted,
    reliability,
)
from swipe_anchor.db.models import (
    Annotator,
    Assignment,
    Comparison,
    GoldItem,
    ReliabilityEvent,
    Response,
)


def _gold_counts(session: Session) -> dict[str, tuple[int, int]]:
    """annotator_id -> (n_gold_seen, n_gold_correct) from the event log."""
    out: dict[str, list[int]] = {}
    for ev in session.query(ReliabilityEvent).filter(ReliabilityEvent.gold_correct.isnot(None)):
        seen_correct = out.setdefault(ev.annotator_id, [0, 0])
        seen_correct[0] += 1
        seen_correct[1] += 1 if ev.gold_correct else 0
    return {a: (s, c) for a, (s, c) in out.items()}


def _build_em_items(session: Session) -> dict[str, list[Vote]]:
    """Non-gold comparisons' responses, labeled by odd-one-out class."""
    items: dict[str, list[Vote]] = {}
    gold_ids = {g.comparison_id for g in session.query(GoldItem)}
    for cmp in session.query(Comparison):
        if cmp.comparison_id in gold_ids:
            continue
        order = classes_for_comparison(cmp)
        class_of = {cid: i for i, cid in enumerate(order)}
        votes = [
            Vote(
                annotator_id=r.annotator_id,
                label=None if r.odd_creator_id is None else class_of[r.odd_creator_id],
            )
            for r in session.query(Response).filter_by(comparison_id=cmp.comparison_id)
        ]
        if votes:
            items[cmp.comparison_id] = votes
    return items


def _behavioral(session: Session, annotator_id: str) -> float:
    rows = list(session.query(Response).filter_by(annotator_id=annotator_id))
    if not rows:
        return 1.0
    rts = [r.reaction_time_ms for r in rows if r.reaction_time_ms is not None]
    dwell = [sum((r.card_dwell_ms or {}).values()) for r in rows]
    streak = 1
    best = 1
    prev = object()
    for r in rows:
        if r.odd_creator_id == prev:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
        prev = r.odd_creator_id
    return behavioral_factor(
        min_rt_ms=min(rts) if rts else None,
        mean_dwell_ms=(sum(dwell) / len(dwell)) if dwell else 0.0,
        ever_expanded=any(r.expanded for r in rows),
        constant_streak=best,
    )


def run_sweep(session: Session, settings: Settings, *, now: datetime | None = None) -> None:
    now = now or datetime.now(UTC)
    gold = _gold_counts(session)
    items = _build_em_items(session)

    # n_eff per annotator = #gold + #non-gold responses (overlap proxy).
    n_eff: dict[str, float] = {a: float(s) for a, (s, _) in gold.items()}
    for votes in items.values():
        for v in votes:
            n_eff[v.annotator_id] = n_eff.get(v.annotator_id, 0.0) + 1.0

    trusted = {a for a, k in n_eff.items() if is_trusted(k, settings.warmup_k)}
    em = full_em(
        items,
        trusted=trusted,
        prior_reliability=0.5,
        dirichlet_conc=settings.dirichlet_conc,
    )

    # Write reliability (sole owner) + sweep-derived gold counters.
    for ann in session.query(Annotator):
        a = ann.annotator_id
        seen, correct = gold.get(a, (0, 0))
        ann.n_gold_seen = seen
        ann.n_gold_correct = correct
        g_acc = gold_accuracy(seen, correct, settings.beta_alpha0, settings.beta_beta0)
        comp = em.competence.get(a, None)
        ds_comp = 0.5 if comp is None else (comp - 1 / 3) / (2 / 3)  # map [1/3,1] -> [0,1]
        ann.reliability = reliability(
            gold_acc=g_acc,
            ds_comp=ds_comp,
            n_eff=n_eff.get(a, 0.0),
            warmup_k=settings.warmup_k,
            behavioral=_behavioral(session, a),
            floor=settings.gold_blend_floor,
        )
    session.flush()

    # Refine every non-gold comparison's consensus with fresh reliabilities.
    for cmp in session.query(Comparison).filter(Comparison.status != "gold"):
        recompute_item(session, cmp.comparison_id, settings)

    reclaim_expired(session, now=now)


def reclaim_expired(session: Session, *, now: datetime | None = None) -> int:
    """Mark issued-but-expired assignments ``expired`` (cheap, time-based).

    Independent of the EM recompute so the idle loop can keep the assignment
    lifecycle moving without re-running consensus. Returns the rows reclaimed.
    """
    now = now or datetime.now(UTC)
    result = session.execute(
        update(Assignment)
        .where(Assignment.status == "issued")
        .where(Assignment.expires_at.isnot(None))
        .where(Assignment.expires_at <= now)
        .values(status="expired")
    )
    session.flush()
    return int(result.rowcount or 0)


def response_count(session: Session) -> int:
    """Cheap change signal: the full recompute only matters when this moves."""
    return int(session.query(func.count()).select_from(Response).scalar())


def sweep_tick(
    session: Session,
    settings: Settings,
    last_count: int | None,
    *,
    now: datetime | None = None,
) -> int:
    """One change-gated tick; returns the response count to carry to the next.

    Full ``run_sweep`` (EM + reliability + consensus) runs only on the first tick
    or when the response count changed; otherwise just ``reclaim_expired``. With a
    fixed dataset and no new judgments this avoids re-deriving identical results.
    """
    n = response_count(session)
    if last_count is None or n != last_count:
        run_sweep(session, settings, now=now)
    else:
        reclaim_expired(session, now=now)
    return n


_log = logging.getLogger("swipe_anchor.sweep")


async def sweep_once(session_factory, settings: Settings) -> None:
    """Run one *full* sweep inside a fresh transaction (off the event loop)."""

    def _do() -> None:
        with session_factory() as s:
            run_sweep(s, settings)

    await asyncio.to_thread(_do)


async def run_sweep_loop(session_factory, settings: Settings) -> None:
    """Forever: change-gated tick, then sleep ``sweep_interval_s``.

    Carries the last response count across ticks so a full recompute fires only
    when new judgments landed; idle ticks just reclaim expired assignments.
    Cancelled on shutdown.
    """
    last_count: int | None = None
    while True:
        try:
            last_count = await asyncio.to_thread(
                _tick_once, session_factory, settings, last_count
            )
        except Exception:  # a bad tick must not kill the loop
            _log.exception("sweep tick failed")
        await asyncio.sleep(settings.sweep_interval_s)


def _tick_once(session_factory, settings: Settings, last_count: int | None) -> int:
    with session_factory() as s:
        return sweep_tick(s, settings, last_count)


def main(argv: list[str] | None = None) -> int:
    """One-shot sweep (cron/test). Uses the same env config as the server."""
    import os

    from swipe_anchor.db import create_app_engine, make_session_factory, session_scope

    engine = create_app_engine(os.environ.get("APP_DATABASE_URL") or "sqlite:///data/swipe_anchor.db")
    factory = make_session_factory(engine)
    with session_scope(factory) as s:
        run_sweep(s, Settings.from_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
