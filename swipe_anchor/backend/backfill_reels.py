"""Backfill ``Response.shown_clips`` for rows recorded before reels were captured.

The judge card always renders a creator's lowest-``ord`` digest clip (``clips[0]``),
so the reel shown for each creator in a historical comparison is deterministic and
can be reconstructed from the digests.

    uv run python -m swipe_anchor.backend.backfill_reels            # backfill
    uv run python -m swipe_anchor.backend.backfill_reels --dry-run  # report only

``--dry-run`` is strictly read-only: it opens the engine with ``migrate=False`` so
no table is created and no column is altered. The normal run opens with the
default ``migrate=True``, which reconciles the ``shown_clips`` column (the same
path the backend uses at startup) before filling rows.

``APP_DATABASE_URL`` selects the DB (default ``sqlite:///data/swipe_anchor.db``).
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import Comparison, DigestClip, Response


def _shown_clip(session: Session, creator_id: int, cache: dict[int, int | None]) -> int | None:
    if creator_id not in cache:
        cache[creator_id] = session.scalar(
            select(DigestClip.clip_id)
            .where(DigestClip.creator_id == creator_id)
            .order_by(DigestClip.ord)
            .limit(1)
        )
    return cache[creator_id]


def backfill(session: Session, *, dry_run: bool = False) -> tuple[int, int]:
    """Set ``shown_clips`` on every response missing it. Returns (updated, skipped).

    Reads in a column-safe way so a ``--dry-run`` works even before the column
    exists. A row is "missing" when its value is falsy — covering SQL NULL, JSON
    ``null`` (how SQLAlchemy stores a Python ``None`` JSON value), and ``{}``.
    """
    has_col = "shown_clips" in {
        c["name"] for c in inspect(session.bind).get_columns("responses")
    }
    if has_col:
        items = [
            (r, r.comparison_id, r.shown_clips)
            for r in session.scalars(select(Response)).all()
        ]
    else:
        # Column absent (dry-run on an un-migrated DB): select only safe columns.
        items = [
            (None, comparison_id, None)
            for _rid, comparison_id in session.execute(
                select(Response.response_id, Response.comparison_id)
            ).all()
        ]

    cache: dict[int, int | None] = {}
    updated = skipped = 0
    for entity, comparison_id, current in items:
        if current:  # already populated → leave it alone
            continue
        cmp = session.get(Comparison, comparison_id)
        if cmp is None:
            skipped += 1
            continue
        shown = {
            str(cr): clip
            for cr in cmp.creators
            if (clip := _shown_clip(session, cr, cache)) is not None
        }
        if not shown:
            skipped += 1
            continue
        if not dry_run and entity is not None:
            entity.shown_clips = shown
        updated += 1
    return updated, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swipe_anchor.backend.backfill_reels")
    parser.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = parser.parse_args(argv)

    url = os.environ.get("APP_DATABASE_URL") or "sqlite:///data/swipe_anchor.db"
    # Dry-run must not mutate: skip create_all + the additive column ALTER.
    engine = create_app_engine(url, migrate=not args.dry_run)
    with Session(engine) as session:
        updated, skipped = backfill(session, dry_run=args.dry_run)
        if not args.dry_run:
            session.commit()
    print(
        f"updated={updated} skipped={skipped} dry_run={args.dry_run}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
