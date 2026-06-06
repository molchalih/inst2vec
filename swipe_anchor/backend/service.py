"""Assignment + respond service logic (plan §4).

Phase 1 implements the *invariants* that hold across all phases — eligibility,
no self-collision, idempotent respond, judgment counting — with a deliberately
trivial draw (random over eligible). Consensus-based retirement, gold injection,
and Dawid-Skene reliability (§4.1-4.3, §7) layer on top without changing these
contracts. Triplets are now consensus-materialized by ``recompute_item``, never
emitted per-response.

Functions take an explicit ``Session`` and do not commit — the caller (endpoint
or test) owns the transaction boundary, matching the §4.5 ``with tx():`` shape.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from swipe_anchor.backend.consensus_writer import recompute_item
from swipe_anchor.config import Settings
from swipe_anchor.core.balancer import (
    Features,
    score,
    softmax_sample_without_replacement,
)
from swipe_anchor.db.models import (
    AccessCode,
    Annotator,
    Assignment,
    Comparison,
    GoldItem,
    ReliabilityEvent,
    Response,
)


class UnknownAssignmentError(ValueError):
    """Raised when a respond references an assignment id that does not exist."""


class InvalidOddCreatorError(ValueError):
    """Raised when the crossed creator is not part of the assigned comparison."""


class ForbiddenError(ValueError):
    """Raised when a code tries to answer an assignment it does not own."""


def is_access_allowed(session: Session, code: str) -> bool:
    """Admit an access code (auth feature).

    Empty/whitespace codes are always rejected. When the ``access_codes`` table is
    empty the backend runs open (any non-empty code works, for bootstrap); once any
    row exists, only listed + active codes are admitted.
    """
    if not code or not code.strip():
        return False
    total = session.scalar(select(func.count()).select_from(AccessCode)) or 0
    if total == 0:
        return True
    row = session.get(AccessCode, code)
    return row is not None and bool(row.is_active)


@dataclass(frozen=True)
class RespondResult:
    """Outcome of recording a judgment."""

    accepted: bool
    n_triplets: int
    retired: bool


def get_or_create_annotator(session: Session, annotator_id: str) -> Annotator:
    """Anonymous session id is created on first contact (plan §3 annotators)."""
    annotator = session.get(Annotator, annotator_id)
    if annotator is None:
        annotator = Annotator(annotator_id=annotator_id)
        session.add(annotator)
        session.flush()
    return annotator


def _seen_comparison_ids(session: Session, annotator_id: str) -> set[str]:
    rows = session.execute(
        select(Assignment.comparison_id).where(Assignment.annotator_id == annotator_id)
    )
    return {r[0] for r in rows}


def _inflight_counts(session: Session, now: datetime) -> dict[str, int]:
    """Issued, not-yet-expired assignments per comparison (expired ignored)."""
    rows = session.execute(
        select(Assignment.comparison_id, func.count())
        .where(Assignment.status == "issued")
        .where((Assignment.expires_at.is_(None)) | (Assignment.expires_at > now))
        .group_by(Assignment.comparison_id)
    )
    return {cid: n for cid, n in rows}


def _annotator_group_histogram(session: Session, annotator_id: str) -> dict[str, int]:
    rows = session.execute(
        select(Comparison.seed_group, func.count())
        .join(Response, Response.comparison_id == Comparison.comparison_id)
        .where(Response.annotator_id == annotator_id)
        .group_by(Comparison.seed_group)
    )
    return {g: n for g, n in rows if g is not None}


def _features(cmp: Comparison, hist: dict[str, int], hist_total: int) -> Features:
    target = max(1, cmp.target_k)
    coverage_gap = max(0.0, (target - cmp.n_judgments) / target)
    is_boundary = 1.0 if cmp.kind == "boundary" else 0.0
    seen = hist.get(cmp.seed_group or "", 0)
    novelty = 1.0 - (seen / hist_total) if hist_total else 1.0
    return Features(coverage_gap, is_boundary, novelty, float(cmp.information or 0.0))


# Postgres deploys can wrap the candidate select in _scored_pool with
# .with_for_update(skip_locked=True) behind a
# ``session.bind.dialect.name == "postgresql"`` check to avoid hot-row
# contention under concurrent annotators. SQLite serializes writes so the
# plain select is correct for the current deployment.
def _scored_pool(
    session: Session,
    seen: set[str],
    now: datetime,
    settings: Settings,
    *,
    annotator_id: str,
) -> list[tuple[Comparison, float]]:
    inflight = _inflight_counts(session, now)
    hist = _annotator_group_histogram(session, annotator_id)
    hist_total = sum(hist.values())
    # Eligible up to the HARD cap, not target_k: a split item whose first
    # target_k votes miss confidence_threshold stays "open" and must keep
    # collecting overlap until it either turns confident or hits max_overlap
    # (where recompute_item flags it "ambiguous"). Confident/ambiguous items
    # already leave the pool via their non-"open" status.
    pool = session.scalars(
        select(Comparison)
        .where(Comparison.status == "open")
        .where(Comparison.n_judgments < settings.max_overlap)
    )
    scored: list[tuple[Comparison, float]] = []
    for cmp in pool:
        if cmp.comparison_id in seen:
            continue
        if inflight.get(cmp.comparison_id, 0) >= settings.max_inflight:
            continue
        scored.append((cmp, score(_features(cmp, hist, hist_total), settings)))
    return scored


def _sample_gold(session: Session, seen: set[str], settings: Settings) -> Comparison | None:
    # gold_exposure_cap bounds DISTINCT gold items shown to an annotator.
    # Each gold comparison has a unique (comparison_id, annotator_id) constraint
    # so it can only be issued once; this counts how many distinct gold items
    # this annotator has already been assigned.
    n_gold_seen = sum(1 for cid in seen if session.get(GoldItem, cid) is not None)
    if n_gold_seen >= settings.gold_exposure_cap:
        return None
    return session.scalars(
        select(Comparison)
        .where(Comparison.status == "gold")
        .where(Comparison.comparison_id.not_in(seen or {""}))
        .limit(1)
    ).first()


def next_batch(
    session: Session,
    annotator_id: str,
    n: int,
    *,
    rng: random.Random | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> list[Comparison]:
    """Scored, overlap-capped, expiry-stamped draw with gold injection (plan §4.1).

    The draw uses softmax sampling over a scored pool of eligible comparisons,
    respecting the max_inflight overlap cap and expiry-aware inflight counting.
    A gold comparison may be injected with probability p_gold, up to
    gold_exposure_cap distinct gold items per annotator lifetime.
    """
    rng = rng or random.Random()
    settings = settings or Settings()
    now = now or datetime.now(UTC)
    get_or_create_annotator(session, annotator_id)
    seen = _seen_comparison_ids(session, annotator_id)

    picks: list[Comparison] = []
    want = max(0, n)

    # Reserve a gold slot with probability p_gold (invisible to the annotator).
    if want > 0 and rng.random() < settings.p_gold:
        g = _sample_gold(session, seen, settings)
        if g is not None:
            picks.append(g)

    remaining = want - len(picks)
    if remaining > 0:
        scored = _scored_pool(session, seen, now, settings, annotator_id=annotator_id)
        chosen = softmax_sample_without_replacement(
            scored, remaining, temperature=settings.softmax_temperature, rng=rng
        )
        picks.extend(chosen)

    ttl = timedelta(seconds=settings.assign_ttl_s)
    for cmp in picks:
        session.add(
            Assignment(
                assignment_id=str(uuid.uuid4()),
                comparison_id=cmp.comparison_id,
                annotator_id=annotator_id,
                status="issued",
                expires_at=now + ttl,
            )
        )
    session.flush()
    return picks


def record_response(
    session: Session,
    assignment_id: str,
    odd_id: int | None,
    *,
    confidence: float = 1.0,
    reaction_time_ms: int | None = None,
    card_dwell_ms: dict | None = None,
    shown_clips: dict | None = None,
    expanded: bool = False,
    expected_annotator_id: str | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> RespondResult:
    """Record a judgment idempotently, then update this item's consensus inline.

    Strict write ownership (design §2): this path writes the Response, appends a
    ReliabilityEvent, recomputes THIS comparison's Consensus + status, and bumps
    only the annotator's monotonic counters. It never writes
    ``annotators.reliability`` — that is solely the sweep's (Task 10).

    Concurrency-safe by construction (plan §4.4):

    - the ``issued -> answered`` transition is a single atomic conditional UPDATE,
      so only one of two concurrent submissions for the same assignment wins; the
      loser returns ``accepted=False`` (idempotent, no double-count). The unique
      ``responses.assignment_id`` constraint is the DB-level backstop.
    - ``n_judgments`` is bumped with an atomic ``n_judgments = n_judgments + 1``
      so a concurrent answer for the same comparison cannot be lost to a stale
      read-modify-write.

    Validation is rejected *before* the assignment is claimed: an unknown
    assignment raises ``UnknownAssignmentError``, an assignment owned by a different
    code raises ``ForbiddenError``, and a crossed creator outside the comparison
    raises ``InvalidOddCreatorError`` (none consumes the assignment).
    A skip (``odd_id is None``) counts as a judgment but emits no triplets.
    Triplets are now consensus-materialized (two when the item resolves to a
    confident odd creator), never per-response.
    """
    settings = settings or Settings()
    now = now or datetime.now(UTC)

    assignment = session.get(Assignment, assignment_id)
    if assignment is None:
        raise UnknownAssignmentError(f"unknown assignment_id {assignment_id!r}")
    if expected_annotator_id is not None and assignment.annotator_id != expected_annotator_id:
        raise ForbiddenError(
            f"assignment {assignment_id!r} is not owned by {expected_annotator_id!r}"
        )

    comparison = session.get(Comparison, assignment.comparison_id)
    annotator = session.get(Annotator, assignment.annotator_id)
    assert comparison is not None and annotator is not None  # FK guarantees

    if odd_id is not None and odd_id not in comparison.creators:
        raise InvalidOddCreatorError(
            f"odd_id {odd_id!r} is not part of comparison {comparison.comparison_id!r}"
        )

    # Atomically claim the assignment: only the transition from 'issued' wins.
    claimed = session.execute(
        update(Assignment)
        .where(Assignment.assignment_id == assignment_id)
        .where(Assignment.status == "issued")
        .values(status="answered")
    ).rowcount
    if not claimed:
        return RespondResult(accepted=False, n_triplets=0, retired=False)

    session.add(
        Response(
            response_id=str(uuid.uuid4()),
            assignment_id=assignment_id,
            comparison_id=comparison.comparison_id,
            annotator_id=annotator.annotator_id,
            odd_creator_id=odd_id,
            confidence=confidence,
            reaction_time_ms=reaction_time_ms,
            card_dwell_ms=card_dwell_ms,
            shown_clips=shown_clips,
            expanded=expanded,
        )
    )
    annotator.n_responses += 1
    annotator.last_seen = now

    # Append a gold outcome event (sweep derives gold counters + reliability).
    # On a gold item a skip (odd_id is None) is a FAILED exposure, not a
    # non-event: an obvious catch-trial should be answerable, so skipping it
    # counts as seen+incorrect (gold_correct=False) rather than NULL — otherwise
    # an annotator could dodge the gold penalty by skipping every catch trial.
    # Non-gold items stay NULL (not a gold event), skip or not.
    gold = session.get(GoldItem, comparison.comparison_id)
    gold_correct = None if gold is None else (odd_id == gold.known_odd)
    session.add(
        ReliabilityEvent(
            annotator_id=annotator.annotator_id,
            comparison_id=comparison.comparison_id,
            gold_correct=gold_correct,
        )
    )

    # Atomic judgment count, then inline consensus + retirement for THIS item.
    session.execute(
        update(Comparison)
        .where(Comparison.comparison_id == comparison.comparison_id)
        .values(n_judgments=Comparison.n_judgments + 1)
    )
    session.flush()
    recompute_item(session, comparison.comparison_id, settings)

    session.refresh(comparison)
    retired = comparison.status in ("retired", "ambiguous")
    # Triplets are now consensus-materialized (two when the item resolves to a
    # confident odd creator), never per-response — so this reflects materialization.
    n_triplets = 2 if comparison.status == "retired" else 0
    return RespondResult(accepted=True, n_triplets=n_triplets, retired=retired)
