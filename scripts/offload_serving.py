#!/usr/bin/env python
"""Offload the version-7 frontend contract into the normalised serving DB.

Reads the pipeline main DB (via ``modules.visualization.export``'s payload
builders), decomposes each exposed case's payloads into ``serving_*`` rows, and
writes them into the separate serving DB an API can serve.

Idempotent: each run is pruned (delete-then-insert) before reinsert, and runs
no longer present in the exposed case set are dropped entirely (R5), so the
serving tables always mirror the current DB / case set — the relational
analogue of the file exporter's stale-run pruning.

Run after the pipeline's visualization stage has populated the viz tables:

    uv run python scripts/offload_serving.py
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from core.config import load_runtime_config  # noqa: E402
from core.database import (  # noqa: E402
    get_serving_session,
    get_session,
    init_db,
    init_serving_db,
)
from core.database.serving_decompose import (  # noqa: E402
    SERVING_TABLES_PRUNE_ORDER,
    decompose_run,
)
from core.database.serving_models import ServingRun  # noqa: E402
from modules.embeddings.cases import default_cases  # noqa: E402
from modules.visualization.export import (  # noqa: E402
    _exposed_cases,
    build_case_payloads,
)


def _prune_run(session, run_id: str) -> None:
    for model in SERVING_TABLES_PRUNE_ORDER:
        session.query(model).filter(model.run_id == run_id).delete(
            synchronize_session=False
        )


# FK-safe INSERT order: parents before children. This is exactly the reverse of
# the FK-safe DELETE order. ServingRun is the root (no FK); everything else
# references it (or, transitively, ServingCluster / ServingUser).
SERVING_TABLES_INSERT_ORDER: tuple[Any, ...] = tuple(
    reversed(SERVING_TABLES_PRUNE_ORDER)
)


def _insert_run_rows(session, rows: list[Any]) -> None:
    """Persist one run's rows in FK-safe order.

    The serving_* models declare column-level ``ForeignKey``s but no ORM
    ``relationship()``, so SQLAlchemy's unit-of-work has no dependency edge to
    order parent INSERTs before children. An RDBMS that enforces FKs (Postgres)
    would then reject a child whose parent row has not landed yet. We instead
    flush tier by tier in the known FK-safe order so each parent is committed to
    the transaction before its dependents. (On SQLite FKs are off, so this is
    only load-bearing for the Postgres serving store, but it is correct on both.)
    """
    by_model: dict[type, list[Any]] = {}
    for r in rows:
        by_model.setdefault(type(r), []).append(r)
    for model in SERVING_TABLES_INSERT_ORDER:
        tier = by_model.pop(model, None)
        if tier:
            session.add_all(tier)
            session.flush()
    # Any model not listed in the prune/insert order (should be none) — add last.
    for tier in by_model.values():
        session.add_all(tier)
    session.flush()


def offload(settings, cases: tuple[str, ...]) -> None:
    """Decompose every exposed case into the serving DB. Idempotent."""
    viz_settings = settings.visualization
    exposed = _exposed_cases(cases)

    read = get_session()
    try:
        bundles = [
            b
            for case in exposed
            if (b := build_case_payloads(read, settings_viz=viz_settings, case=case))
            is not None
        ]
    finally:
        read.close()

    present = {b.case for b in bundles}
    with get_serving_session() as s:
        # Prune runs no longer present (dropped cases) plus every run we are
        # about to rewrite, then re-insert from scratch.
        stale = {r.run_id for r in s.query(ServingRun).all()} | present
        for run_id in stale:
            _prune_run(s, run_id)
        # Flush the prune DELETEs before re-inserting so the INSERT phase starts
        # from a clean slate (no delete-then-insert-same-PK churn in one flush).
        s.flush()
        for manifest_ord, b in enumerate(bundles):
            rows = decompose_run(
                b,
                is_default=b.case == viz_settings.default_case,
                manifest_ord=manifest_ord,
            )
            _insert_run_rows(s, rows)
        s.commit()


def main(cases: Sequence[str] | None = None) -> int:
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    init_serving_db(secrets.serving_database_url)
    # Default to every viewer-exposed case the frontend config knows about
    # (offload's _exposed_cases filters this to the expose_to_viewer set).
    resolved = tuple(cases) if cases else tuple(default_cases(settings))
    offload(settings, resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
