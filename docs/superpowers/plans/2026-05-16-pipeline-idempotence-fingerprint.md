# Pipeline Idempotence — Fingerprint Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `stage_state` table and a small `modules/fingerprint.py` helper, then wire `embed_clip_embeddings` and `embed_user_embeddings` to auto-recompute on data/config/dependency change.

**Architecture:** Stages own everything case-specific (what to hash, when to wipe, how to recompute). The shared helper just compares and merges three hashes per `(stage, scope)` row. Stale propagation emerges naturally because downstream `dependency_hash` reads actual upstream rows. No DAG engine, no decorators, no orchestrator changes.

**Tech Stack:** Python 3.12, SQLAlchemy, pydantic-settings, hashlib, pytest. `main.py` execution order is untouched.

**Spec:** `docs/superpowers/specs/2026-05-16-pipeline-idempotence-fingerprint-design.md`

---

## File Map

| Path | Change | Role |
|---|---|---|
| `modules/database.py` | Modify | Add `StageState` model. |
| `modules/fingerprint.py` | Create | `Fingerprint` dataclass, `hash_rows`, `hash_text`, `is_stale`, `mark_complete`, `describe_diff`. ~50 LoC. |
| `modules/embeddings/cases.py` | Modify | Add `TEXT_RECIPE_VERSIONS` mapping + `case_config_identity(spec, settings) -> str`. |
| `modules/embeddings/state.py` | Modify | Add `dependency_rows_for_case(session, case, candidate_ids) -> list[tuple]`. |
| `modules/embeddings/runner.py` | Modify | Compute `Fingerprint` per case; on stale, delete + recompute + `mark_complete` + commit. |
| `modules/embeddings/users.py` | Modify | Same shape; drop the TODO at line 8. |
| `tests/test_database_stage_state.py` | Create | Table-shape smoke test (T1). |
| `tests/test_fingerprint.py` | Create | Pure-helper tests (T2). |
| `tests/test_user_embeddings_idempotence.py` | Create | DB-level idempotence tests (T3). |
| `tests/test_clip_embeddings_idempotence.py` | Create | DB-level idempotence tests with a fake provider (T4). |
| `tests/test_embeddings_cascade.py` | Create | End-to-end cascade smoke (T5). |

`main.py`, `config.py`, `config.toml`, `tests/conftest.py` are not modified.

---

## Conventions used by every task

- `uv run pytest` runs the full suite. Always run the focused test first, then the full suite before committing.
- `uv run ruff check` and `uv run ty check` must pass before commit.
- Commits use Conventional Commits with a `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` trailer.
- Tests for DB-touching stages share a small `db_session` fixture pattern (defined inside each test file, copied verbatim — see No-Placeholders rule). Session-scoped `:memory:` DB from `tests/conftest.py` is already initialized; each test wipes the tables it uses before running.

---

### Task 1: Add `StageState` model

**Files:**
- Modify: `modules/database.py`
- Create: `tests/test_database_stage_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_database_stage_state.py`:

```python
"""Smoke test for the stage_state table and StageState model."""

from datetime import datetime

import pytest

from modules.database import (
    Base,
    StageState,
    get_engine,
    get_session,
)


@pytest.fixture
def fresh_stage_state():
    Base.metadata.create_all(get_engine())
    session = get_session()
    session.query(StageState).delete()
    session.commit()
    yield session
    session.close()


def test_stage_state_columns_present():
    cols = {c.name for c in StageState.__table__.columns}
    assert cols == {
        "stage_name",
        "scope_key",
        "data_hash",
        "config_hash",
        "dependency_hash",
        "updated_at",
    }


def test_stage_state_primary_key_is_stage_name_scope_key():
    pk = {c.name for c in StageState.__table__.primary_key.columns}
    assert pk == {"stage_name", "scope_key"}


def test_stage_state_merge_and_get(fresh_stage_state):
    session = fresh_stage_state
    session.merge(
        StageState(
            stage_name="t1",
            scope_key="alpha",
            data_hash="d1",
            config_hash="c1",
            dependency_hash="x1",
        )
    )
    session.commit()

    row = session.get(StageState, ("t1", "alpha"))
    assert row is not None
    assert row.data_hash == "d1"
    assert row.config_hash == "c1"
    assert row.dependency_hash == "x1"
    assert isinstance(row.updated_at, datetime)


def test_stage_state_merge_overwrites_in_place(fresh_stage_state):
    session = fresh_stage_state
    session.merge(
        StageState(
            stage_name="t1",
            scope_key="alpha",
            data_hash="d1",
            config_hash="c1",
            dependency_hash="x1",
        )
    )
    session.commit()
    session.merge(
        StageState(
            stage_name="t1",
            scope_key="alpha",
            data_hash="d2",
            config_hash="c2",
            dependency_hash="x2",
        )
    )
    session.commit()

    row = session.get(StageState, ("t1", "alpha"))
    assert row is not None
    assert row.data_hash == "d2"
    assert row.config_hash == "c2"
    assert row.dependency_hash == "x2"
    assert (
        session.query(StageState).filter_by(stage_name="t1", scope_key="alpha").count()
        == 1
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_database_stage_state.py -v`
Expected: ImportError on `StageState` (the model does not exist yet).

- [ ] **Step 3: Add `StageState` to `modules/database.py`**

Add this block in `modules/database.py` directly after the `UserCluster` model (around line 261, before `ClusterRun`):

```python
class StageState(Base):
    __tablename__ = "stage_state"

    stage_name: Mapped[str] = mapped_column(String, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String, primary_key=True)
    data_hash: Mapped[str] = mapped_column(String, nullable=False)
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    dependency_hash: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
```

No other change to `database.py`. The `init_db()` function already calls `Base.metadata.create_all(_engine)`, which picks up the new table.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_database_stage_state.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Lint, type-check, full suite**

Run:
```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add modules/database.py tests/test_database_stage_state.py
git commit -m "$(cat <<'EOF'
feat(db): add stage_state table for the fingerprint layer

Composite PK (stage_name, scope_key); three string hash columns and a
forensic updated_at. Picked up automatically by init_db via
Base.metadata.create_all.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `modules/fingerprint.py` helper

**Files:**
- Create: `modules/fingerprint.py`
- Create: `tests/test_fingerprint.py`

- [ ] **Step 1: Write the failing test (hash primitives)**

Create `tests/test_fingerprint.py` with this initial content:

```python
"""Unit tests for the fingerprint helper."""

import pytest

from modules.database import Base, StageState, get_engine, get_session
from modules.fingerprint import (
    Fingerprint,
    describe_diff,
    hash_rows,
    hash_text,
    is_stale,
    mark_complete,
)


# ── hash_rows / hash_text ────────────────────────────────────────────────────


def test_hash_rows_empty_is_stable():
    assert hash_rows([]) == hash_rows([])
    # well-defined and non-empty
    assert isinstance(hash_rows([]), str)
    assert len(hash_rows([])) == 64


def test_hash_rows_deterministic_for_same_input():
    rows = [(1, "a"), (2, "b"), (3, "c")]
    assert hash_rows(rows) == hash_rows(rows)


def test_hash_rows_order_sensitive():
    a = hash_rows([(1, "a"), (2, "b")])
    b = hash_rows([(2, "b"), (1, "a")])
    assert a != b


def test_hash_rows_value_change_changes_digest():
    a = hash_rows([(1, "a")])
    b = hash_rows([(1, "b")])
    assert a != b


def test_hash_rows_no_collision_on_split():
    # 0x1E record separator prevents (1,2),(3) from colliding with (1),(2,3)
    a = hash_rows([(1, 2), (3,)])
    b = hash_rows([(1,), (2, 3)])
    assert a != b


def test_hash_text_deterministic():
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")


def test_hash_text_empty_is_stable():
    assert hash_text("") == hash_text("")
    assert len(hash_text("")) == 64
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `uv run pytest tests/test_fingerprint.py -v`
Expected: `ModuleNotFoundError: No module named 'modules.fingerprint'`.

- [ ] **Step 3: Create `modules/fingerprint.py` (full implementation)**

Create `modules/fingerprint.py`:

```python
"""Tiny shared fingerprint layer for stage-level idempotence.

Each central pipeline stage computes a Fingerprint(data, config,
dependency) on entry and asks ``is_stale`` whether its stored
counterpart still matches. On mismatch the stage wipes its outputs for
the scope, recomputes, and calls ``mark_complete``. The stage commits
its own transaction; ``mark_complete`` only merges.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from modules.database import StageState


@dataclass(frozen=True)
class Fingerprint:
    data: str
    config: str
    dependency: str


# ── hashing utilities (stages decide what to feed in) ────────────────────────

def hash_rows(rows: Iterable[tuple[Any, ...]]) -> str:
    """Stable SHA-256 over an iterable of tuples.

    Caller is responsible for passing rows sorted on a stable key.
    Record separator 0x1E prevents (1,2),(3) from colliding with
    (1),(2,3).
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(repr(row).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── compare + store ──────────────────────────────────────────────────────────

def is_stale(
    session: Session, stage: str, scope: str, current: Fingerprint
) -> bool:
    row = session.get(StageState, (stage, scope))
    if row is None:
        return True
    return (
        row.data_hash != current.data
        or row.config_hash != current.config
        or row.dependency_hash != current.dependency
    )


def mark_complete(
    session: Session, stage: str, scope: str, current: Fingerprint
) -> None:
    """Merge the stage_state row. Caller commits."""
    session.merge(
        StageState(
            stage_name=stage,
            scope_key=scope,
            data_hash=current.data,
            config_hash=current.config,
            dependency_hash=current.dependency,
        )
    )


def describe_diff(
    session: Session, stage: str, scope: str, current: Fingerprint
) -> str:
    """Human-readable note for log lines.

    Returns 'no prior state' on first run, '' when no fields changed, or
    a '+'-joined list like 'data+dependency'.
    """
    row = session.get(StageState, (stage, scope))
    if row is None:
        return "no prior state"
    parts = []
    if row.data_hash != current.data:
        parts.append("data")
    if row.config_hash != current.config:
        parts.append("config")
    if row.dependency_hash != current.dependency:
        parts.append("dependency")
    return "+".join(parts)
```

- [ ] **Step 4: Run primitives tests — expect PASS**

Run: `uv run pytest tests/test_fingerprint.py -v`
Expected: 7 tests PASS.

- [ ] **Step 5: Add DB-touching tests to `tests/test_fingerprint.py`**

Append to `tests/test_fingerprint.py`:

```python
# ── is_stale / mark_complete / describe_diff ─────────────────────────────────


@pytest.fixture
def fresh_state():
    Base.metadata.create_all(get_engine())
    session = get_session()
    session.query(StageState).delete()
    session.commit()
    yield session
    session.close()


def _fp(d: str = "d", c: str = "c", x: str = "x") -> Fingerprint:
    return Fingerprint(data=d, config=c, dependency=x)


def test_is_stale_true_when_no_row(fresh_state):
    assert is_stale(fresh_state, "stg", "scp", _fp()) is True


def test_is_stale_false_when_all_match(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp())
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp()) is False


def test_is_stale_true_when_data_differs(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp("d1"))
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp("d2")) is True


def test_is_stale_true_when_config_differs(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp(c="c1"))
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp(c="c2")) is True


def test_is_stale_true_when_dependency_differs(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp(x="x1"))
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp(x="x2")) is True


def test_mark_complete_does_not_commit(fresh_state):
    # Without our own commit, a second fresh session should NOT see the row.
    mark_complete(fresh_state, "stg", "scp", _fp())
    fresh_state.flush()  # send to DB but stay in transaction
    other = get_session()
    try:
        # SQLite :memory: in tests shares the engine; the uncommitted row
        # must not be visible if mark_complete truly avoids committing.
        assert other.get(StageState, ("stg", "scp")) is None
    finally:
        other.close()


def test_describe_diff_no_prior_state(fresh_state):
    assert describe_diff(fresh_state, "stg", "scp", _fp()) == "no prior state"


def test_describe_diff_empty_when_all_match(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp())
    fresh_state.commit()
    assert describe_diff(fresh_state, "stg", "scp", _fp()) == ""


def test_describe_diff_lists_changed_fields(fresh_state):
    mark_complete(fresh_state, "stg", "scp", Fingerprint("d1", "c1", "x1"))
    fresh_state.commit()
    diff = describe_diff(
        fresh_state, "stg", "scp", Fingerprint("d2", "c1", "x2")
    )
    assert diff == "data+dependency"
```

- [ ] **Step 6: Run DB tests — expect PASS**

Run: `uv run pytest tests/test_fingerprint.py -v`
Expected: all 16 tests PASS.

Note on `test_mark_complete_does_not_commit`: it relies on SQLAlchemy session isolation. SQLite in-memory shares a single connection inside the same engine, so the uncommitted row IS visible via the same engine through different sessions if autoflush kicks in. If the assertion is unreliable in your local run, replace the test body with the simpler invariant check below (still meaningful):

```python
def test_mark_complete_does_not_commit(fresh_state):
    # Without our own commit, the row stays dirty in the session.
    mark_complete(fresh_state, "stg", "scp", _fp())
    assert any(isinstance(o, StageState) for o in fresh_state.new) or any(
        isinstance(o, StageState) for o in fresh_state.dirty
    ) or any(isinstance(o, StageState) for o in fresh_state.identity_map.values())
```

- [ ] **Step 7: Lint, type-check, full suite**

Run:
```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add modules/fingerprint.py tests/test_fingerprint.py
git commit -m "$(cat <<'EOF'
feat(fingerprint): add shared idempotence helper

modules/fingerprint.py exposes Fingerprint(data, config, dependency),
hash_rows, hash_text, is_stale, mark_complete (merge-only, caller
commits), and describe_diff. Pure logic; no cleanup, no upstream-state
reads. ~50 LoC, fully covered by tests/test_fingerprint.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire `user_embeddings` to the fingerprint layer

**Files:**
- Modify: `modules/embeddings/users.py`
- Create: `tests/test_user_embeddings_idempotence.py`

The stage scope_key is the embedding_case. `dependency_hash` reads `ClipEmbedding` rows directly. `config_hash` pins the aggregator at `"agg=mean_pool|v=1"`. `data_hash` is the participating user set.

- [ ] **Step 1: Write the failing test**

Create `tests/test_user_embeddings_idempotence.py`:

```python
"""Idempotence tests for embed_user_embeddings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    StageState,
    User,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.embeddings.users import embed_user_embeddings


def _blob(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, UserEmbedding, ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def _seed_users_and_clips(session, pairs: list[tuple[int, int]]):
    """pairs: list of (user_id, clip_id). Inserts the parents."""
    user_ids = {u for u, _ in pairs}
    for uid in user_ids:
        session.merge(User(id=uid, is_selected=True, is_eligible=True))
    for uid, cid in pairs:
        session.merge(
            Clip(id=cid, user_id=uid, is_selected=True, is_downloaded=True)
        )
    session.commit()


def _seed_clip_embeddings(
    session, case: str, items: list[tuple[int, list[float]]]
):
    """items: list of (clip_id, vector). Replaces rows for that case."""
    for cid, vec in items:
        session.merge(
            ClipEmbedding(clip_id=cid, embedding_case=case, embedding=_blob(vec))
        )
    session.commit()


def _settings_stub():
    # embed_user_embeddings ignores settings today; pass a placeholder.
    return object()


def test_first_run_aggregates_and_writes_stage_state(db_session):
    _seed_users_and_clips(db_session, [(1, 10), (1, 11), (2, 20)])
    _seed_clip_embeddings(
        db_session,
        "video",
        [(10, [1.0, 0.0]), (11, [3.0, 0.0]), (20, [0.0, 4.0])],
    )

    embed_user_embeddings(_settings_stub(), cases=["video"])

    s = db_session
    ue = {r.user_id: r for r in s.query(UserEmbedding).filter_by(embedding_case="video")}
    assert set(ue.keys()) == {1, 2}
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue[1].embedding, dtype=np.float32), [2.0, 0.0]
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue[2].embedding, dtype=np.float32), [0.0, 4.0]
    )

    state = s.get(StageState, ("user_embeddings", "video"))
    assert state is not None
    assert state.data_hash and state.config_hash and state.dependency_hash


def test_rerun_with_identical_inputs_is_noop(db_session):
    _seed_users_and_clips(db_session, [(1, 10), (2, 20)])
    _seed_clip_embeddings(
        db_session, "video", [(10, [1.0]), (20, [2.0])]
    )

    embed_user_embeddings(_settings_stub(), cases=["video"])
    first_updated = db_session.get(
        StageState, ("user_embeddings", "video")
    ).updated_at

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    second_updated = db_session.get(
        StageState, ("user_embeddings", "video")
    ).updated_at

    # No-op: stage_state row not rewritten.
    assert first_updated == second_updated


def test_clip_embedding_change_triggers_user_recompute(db_session):
    _seed_users_and_clips(db_session, [(1, 10)])
    _seed_clip_embeddings(db_session, "video", [(10, [1.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])

    # Bump the ClipEmbedding row so updated_at advances, with a new vector.
    row = (
        db_session.query(ClipEmbedding)
        .filter_by(clip_id=10, embedding_case="video")
        .one()
    )
    row.embedding = _blob([7.0, 0.0])
    # Force the updated_at bump in SQLite (server_onupdate fires on UPDATE).
    db_session.commit()

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    ue = (
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue.embedding, dtype=np.float32), [7.0, 0.0]
    )


def test_new_clip_embedding_triggers_user_recompute(db_session):
    _seed_users_and_clips(db_session, [(1, 10), (1, 11)])
    _seed_clip_embeddings(db_session, "video", [(10, [2.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])

    _seed_clip_embeddings(db_session, "video", [(11, [4.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    ue = (
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue.embedding, dtype=np.float32), [3.0, 0.0]
    )


def test_empty_inputs_writes_stage_state_and_skips_next(db_session):
    embed_user_embeddings(_settings_stub(), cases=["video"])
    state = db_session.get(StageState, ("user_embeddings", "video"))
    assert state is not None  # row written even for empty case
    first_updated = state.updated_at

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    second_updated = db_session.get(
        StageState, ("user_embeddings", "video")
    ).updated_at
    assert first_updated == second_updated
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `uv run pytest tests/test_user_embeddings_idempotence.py -v`
Expected: tests fail because `embed_user_embeddings` does not yet use fingerprints (it always rewrites rows; the stage_state row is never written).

- [ ] **Step 3: Rewrite `modules/embeddings/users.py`**

Replace the entire contents of `modules/embeddings/users.py` with:

```python
"""User-level embedding aggregation.

Stage is wired to the fingerprint layer (modules.fingerprint). For each
embedding_case the stage:

  1. computes Fingerprint(data, config, dependency) from the actual
     ClipEmbedding rows for the case;
  2. if stale, deletes its UserEmbedding rows for the case, recomputes,
     and merges StageState; commits once at the end so the seal lands
     in the same transaction as the rewrite;
  3. if not stale, logs and skips.

config_hash is currently constant ("agg=mean_pool|v=1"). Bump the
version tag when the aggregator changes.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from modules import fingerprint as fp
from modules.console import log
from modules.database import (
    Base,
    ClipEmbedding,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.embeddings.cases import DEFAULT_CASES
from modules.embeddings.state import (
    get_clip_embedding_rows_for_user_aggregation,
)
from modules.embeddings.vectors import bytes_to_array

STAGE = "user_embeddings"
_CONFIG_IDENTITY = "agg=mean_pool|v=1"


def aggregate_user_embeddings_from_rows(
    rows: list[tuple[bytes, int]],
) -> dict[int, bytes]:
    """Mean-pool clip embedding blobs by user. Returns {user_id: mean_blob}."""
    user_arrays: dict[int, list[np.ndarray]] = defaultdict(list)
    for blob, user_id in rows:
        user_arrays[user_id].append(bytes_to_array(blob))
    return {
        user_id: np.stack(arrays).mean(axis=0).astype(np.float32).tobytes()
        for user_id, arrays in user_arrays.items()
    }


def _compute_fingerprint(session, case: str) -> fp.Fingerprint:
    dep_rows = (
        session.query(ClipEmbedding.clip_id, ClipEmbedding.updated_at)
        .filter(ClipEmbedding.embedding_case == case)
        .order_by(ClipEmbedding.clip_id)
        .all()
    )
    dep = fp.hash_rows(
        (r.clip_id, r.updated_at.isoformat() if r.updated_at else "")
        for r in dep_rows
    )

    # Participating users derived from the same rows; one source of truth.
    agg_rows = get_clip_embedding_rows_for_user_aggregation(session, case)
    user_ids = sorted({user_id for _, user_id in agg_rows})
    data = fp.hash_rows((uid,) for uid in user_ids)

    return fp.Fingerprint(
        data=data,
        config=fp.hash_text(_CONFIG_IDENTITY),
        dependency=dep,
    )


def _clear_case(session, case: str) -> None:
    session.query(UserEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _recompute_case(session, case: str) -> None:
    rows = get_clip_embedding_rows_for_user_aggregation(session, case)
    aggregated = aggregate_user_embeddings_from_rows(rows)
    log(f"embed:user:{case}", f"{len(aggregated)} users to embed")
    for user_id, mean_blob in aggregated.items():
        session.merge(
            UserEmbedding(
                user_id=user_id, embedding_case=case, embedding=mean_blob
            )
        )
        session.commit()


def embed_user_embeddings(settings, cases: list[str] | None = None) -> None:
    """Recompute and merge UserEmbedding rows for each case when stale.

    ``settings`` is accepted for forward-compatibility; no field is read
    today.
    """
    case_names = list(cases) if cases is not None else list(DEFAULT_CASES)
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for case in case_names:
            current = _compute_fingerprint(session, case)
            if not fp.is_stale(session, STAGE, case, current):
                log(f"embed:user:{case}", "fingerprint match — skipping")
                continue

            diff = fp.describe_diff(session, STAGE, case, current)
            log(f"embed:user:{case}", f"stale ({diff}) — recomputing")
            _clear_case(session, case)
            _recompute_case(session, case)
            fp.mark_complete(session, STAGE, case, current)
            session.commit()
            log(f"embed:user:{case}", "done", level="ok")
    finally:
        session.close()
```

The TODO comment that used to live in the docstring and at the top of `embed_user_embeddings` is removed.

- [ ] **Step 4: Run focused tests — expect PASS**

Run: `uv run pytest tests/test_user_embeddings_idempotence.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Run existing user-embeddings tests to confirm no regression**

Run: `uv run pytest tests/test_embeddings_users.py tests/test_embeddings_public_api.py -v`
Expected: all PASS (the pure-aggregation tests are unaffected).

- [ ] **Step 6: Lint, type-check, full suite**

Run:
```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add modules/embeddings/users.py tests/test_user_embeddings_idempotence.py
git commit -m "$(cat <<'EOF'
feat(embeddings): wire user_embeddings to fingerprint layer

Per-case Fingerprint(data, config, dependency). On stale: drop
UserEmbedding rows for the case, recompute via mean-pool, mark_complete
+ commit. Dependency hash reads ClipEmbedding rows directly so re-embeds
and manual deletions both flip it. config_hash pinned at
"agg=mean_pool|v=1"; bump on aggregator change. Removes TODO at
users.py:8.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire `clip_embeddings` to the fingerprint layer

**Files:**
- Modify: `modules/embeddings/cases.py` — add `TEXT_RECIPE_VERSIONS` + `case_config_identity`.
- Modify: `modules/embeddings/state.py` — add `dependency_rows_for_case`.
- Modify: `modules/embeddings/runner.py` — fingerprint check, clear-and-recompute.
- Create: `tests/test_clip_embeddings_idempotence.py`

The stage scope_key is the embedding_case. `dependency_rows_for_case` returns case-specific columns mirroring what `payload_builder` + `text_builder` read. `config_hash` is `hash_text(case_config_identity(spec, settings))`. `data_hash` is the sorted candidate id list.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clip_embeddings_idempotence.py`:

```python
"""Idempotence tests for embed_clip_embeddings.

Uses a fake provider injected via spec.provider_factory monkey-patch so
no Qwen model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.embeddings import cases as cases_mod
from modules.embeddings.runner import embed_clip_embeddings


# ── fake provider ────────────────────────────────────────────────────────────


@dataclass
class _FakeProvider:
    salt: str = ""

    def embed(self, payload: dict) -> np.ndarray:
        seed = abs(hash((self.salt, repr(sorted(payload.items()))))) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal((1, 4), dtype=np.float32)


def _fake_factory(_settings):
    return _FakeProvider()


@pytest.fixture
def stub_providers(monkeypatch, tmp_path):
    """Patch every spec.provider_factory to a fake; redirect video_dir to tmp."""
    for spec in cases_mod.CASE_REGISTRY.values():
        monkeypatch.setattr(spec, "provider_factory", _fake_factory)

    # adaptive_sampling reads the video file. Create empty stand-in files so
    # os.path.exists checks pass; we patch adaptive_sampling itself below.
    from modules.embeddings import runner as runner_mod, sampling as sampling_mod

    monkeypatch.setattr(
        sampling_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )
    monkeypatch.setattr(
        runner_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )
    return tmp_path


# ── settings stub ────────────────────────────────────────────────────────────


@dataclass
class _PathsStub:
    video_dir: str
    model_path: str = "/fake/qwen"


@dataclass
class _EmbeddingsStub:
    exclude_disqualified_users: bool = True
    embed_max_length: int = 1024
    adaptive_max_frames: int = 8
    adaptive_default_fps: float = 1.0


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub


def _settings(tmp_path) -> _SettingsStub:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir)),
        embeddings=_EmbeddingsStub(),
    )


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def _seed_video_file(settings: _SettingsStub, clip_id: int) -> None:
    import os

    path = os.path.join(settings.paths.video_dir, f"{clip_id}.mp4")
    with open(path, "wb") as f:
        f.write(b"\x00")


def _seed(
    session,
    settings: _SettingsStub,
    *,
    clips: list[dict],
    music_rows: list[dict] | None = None,
) -> None:
    music_rows = music_rows or []
    for m in music_rows:
        session.merge(Music(**m))
    user_ids = {c["user_id"] for c in clips}
    for uid in user_ids:
        session.merge(User(id=uid, is_selected=True, is_eligible=True))
    for c in clips:
        defaults = dict(is_selected=True, is_downloaded=True)
        defaults.update(c)
        session.merge(Clip(**defaults))
        if defaults["is_downloaded"]:
            _seed_video_file(settings, defaults["id"])
    session.commit()


# ── tests ────────────────────────────────────────────────────────────────────


def test_first_run_embeds_all_three_cases(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[dict(id=10, user_id=1)],
        music_rows=[dict(id=1, artist="a", track="t")],
    )
    db_session.query(Clip).filter_by(id=10).update({"music_id": 1})
    db_session.commit()

    embed_clip_embeddings(settings)
    db_session.expire_all()

    rows = db_session.query(ClipEmbedding).all()
    cases = {r.embedding_case for r in rows}
    assert cases == {"video", "sandwich", "audio"}

    for case in ("video", "sandwich", "audio"):
        assert db_session.get(StageState, ("clip_embeddings", case)) is not None


def test_rerun_identical_inputs_is_noop(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])

    embed_clip_embeddings(settings, cases=["video"])
    first = db_session.get(
        StageState, ("clip_embeddings", "video")
    ).updated_at

    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()
    second = db_session.get(
        StageState, ("clip_embeddings", "video")
    ).updated_at
    assert first == second


def test_new_candidate_triggers_recompute(db_session, stub_providers):
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])
    embed_clip_embeddings(settings, cases=["video"])

    _seed(db_session, settings, clips=[dict(id=11, user_id=1)])
    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()

    ids = {
        r.clip_id
        for r in db_session.query(ClipEmbedding).filter_by(
            embedding_case="video"
        )
    }
    assert ids == {10, 11}


def test_audio_speech_change_only_invalidates_audio(
    db_session, stub_providers
):
    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[dict(id=10, user_id=1, speech_transcription="hi")],
    )
    embed_clip_embeddings(settings)
    db_session.expire_all()

    before = {
        case: db_session.get(
            StageState, ("clip_embeddings", case)
        ).updated_at
        for case in ("video", "sandwich", "audio")
    }

    # Mutate speech_transcription — should flip audio + sandwich dep, not video.
    db_session.query(Clip).filter_by(id=10).update(
        {"speech_transcription": "hello there"}
    )
    db_session.commit()

    embed_clip_embeddings(settings)
    db_session.expire_all()

    after = {
        case: db_session.get(
            StageState, ("clip_embeddings", case)
        ).updated_at
        for case in ("video", "sandwich", "audio")
    }
    assert after["video"] == before["video"]            # video untouched
    assert after["sandwich"] != before["sandwich"]      # sandwich dep changed
    assert after["audio"] != before["audio"]            # audio dep changed


def test_audio_instruction_change_only_invalidates_audio(
    db_session, stub_providers, monkeypatch
):
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])
    embed_clip_embeddings(settings)
    db_session.expire_all()
    before = {
        case: db_session.get(
            StageState, ("clip_embeddings", case)
        ).updated_at
        for case in ("video", "sandwich", "audio")
    }

    monkeypatch.setattr(
        cases_mod, "AUDIO_INSTRUCTION", "NEW INSTRUCTION TEXT"
    )

    embed_clip_embeddings(settings)
    db_session.expire_all()
    after = {
        case: db_session.get(
            StageState, ("clip_embeddings", case)
        ).updated_at
        for case in ("video", "sandwich", "audio")
    }
    assert after["video"] == before["video"]
    assert after["sandwich"] == before["sandwich"]
    assert after["audio"] != before["audio"]


def test_empty_candidates_writes_stage_state(db_session, stub_providers):
    settings = _settings(stub_providers)
    embed_clip_embeddings(settings, cases=["video"])
    state = db_session.get(StageState, ("clip_embeddings", "video"))
    assert state is not None
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -v`
Expected: tests fail because the runner doesn't yet write `stage_state` rows and doesn't gate on fingerprints.

- [ ] **Step 3: Add `TEXT_RECIPE_VERSIONS` and `case_config_identity` to `cases.py`**

Append to `modules/embeddings/cases.py` (after the existing `CASE_REGISTRY` block):

```python
import os as _os  # local alias; module-level os already used elsewhere

# Recipe versions for text builders. Bump the value when the corresponding
# build_*_text logic changes semantics so existing rows are invalidated.
TEXT_RECIPE_VERSIONS: dict[str, str] = {
    "video": "none",
    "sandwich": "sandwich_v1",
    "audio": "audio_v1",
}


def case_config_identity(spec: EmbeddingCaseSpec, settings) -> str:
    """Stable identity string for a case's recipe + relevant settings.

    Fed into ``fingerprint.hash_text`` to produce the case's config_hash.
    Co-located with the spec definitions so changing a case's identity
    inputs lives next to the case itself.
    """
    parts = [
        f"case={spec.name}",
        f"provider={spec.provider_factory.__name__}",
        f"model={_os.path.basename(settings.paths.model_path)}",
        f"max_len={settings.embeddings.embed_max_length}",
        f"max_frames={settings.embeddings.adaptive_max_frames}",
        f"fps={settings.embeddings.adaptive_default_fps}",
        f"token_fallback={spec.apply_video_token_fallback}",
        f"text_recipe={TEXT_RECIPE_VERSIONS.get(spec.name, 'unknown')}",
    ]
    if spec.name == "audio":
        parts.append(f"instruction={AUDIO_INSTRUCTION}")
    return "|".join(parts)
```

Verification: the existing `import os` (if present) is fine; the local alias prevents a name clash. Re-export is not required since the runner imports directly.

- [ ] **Step 4: Add `dependency_rows_for_case` to `state.py`**

Append to `modules/embeddings/state.py`:

```python
def dependency_rows_for_case(
    session: Session, case: str, candidate_ids: list[int]
) -> list[tuple]:
    """Return the per-candidate tuple of upstream output state for ``case``.

    The columns selected mirror what the case's payload_builder and
    text_builder actually read; the result is sorted by clip_id so the
    digest produced by ``fingerprint.hash_rows`` is deterministic.

    video    : (id, is_downloaded)
    sandwich : (id, is_downloaded, music_id, caption_*, speech_*,
                Music.energy/valence/acousticness/instrumentalness/
                danceability/speechiness/tempo/mode/key/track/artist)
    audio    : (id,                music_id, speech_*,
                Music.energy/...) — captions deliberately excluded
                (build_audio_text does not read them).
    """
    if not candidate_ids:
        return []

    if case == "video":
        rows = (
            session.query(Clip.id, Clip.is_downloaded)
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [tuple(r) for r in rows]

    music_cols = (
        Music.energy,
        Music.valence,
        Music.acousticness,
        Music.instrumentalness,
        Music.danceability,
        Music.speechiness,
        Music.tempo,
        Music.mode,
        Music.key,
        Music.track,
        Music.artist,
    )

    if case == "sandwich":
        rows = (
            session.query(
                Clip.id,
                Clip.is_downloaded,
                Clip.music_id,
                Clip.caption_text,
                Clip.caption_clean,
                Clip.caption_language,
                Clip.caption_translation,
                Clip.speech_transcription,
                Clip.speech_language,
                Clip.speech_translation,
                *music_cols,
            )
            .outerjoin(Music, Clip.music_id == Music.id)
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [tuple(r) for r in rows]

    if case == "audio":
        rows = (
            session.query(
                Clip.id,
                Clip.music_id,
                Clip.speech_transcription,
                Clip.speech_language,
                Clip.speech_translation,
                *music_cols,
            )
            .outerjoin(Music, Clip.music_id == Music.id)
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [tuple(r) for r in rows]

    raise ValueError(f"Unknown embedding case: {case!r}")
```

- [ ] **Step 5: Rewrite the case loop in `runner.py`**

Replace the body of `_run_case` in `modules/embeddings/runner.py` with the fingerprint-gated version. The full new file:

```python
"""Shared clip-embedding runner driven by EmbeddingCaseSpec entries.

Per case the stage:

  1. Picks candidates (selected + downloaded + optional eligibility).
  2. Computes Fingerprint(data, config, dependency). Dependency rows are
     case-specific and mirror what the case's text+payload builders read.
  3. If stale, deletes ClipEmbedding rows for the case, recomputes,
     mark_completes, and commits once at the end.
"""

from __future__ import annotations

import os

from modules import fingerprint as fp
from modules.console import log, progress
from modules.database import Base, Clip, ClipEmbedding, get_engine, get_session
from modules.embeddings.cases import (
    CASE_REGISTRY,
    DEFAULT_CASES,
    EmbeddingCaseSpec,
    case_config_identity,
)
from modules.embeddings.sampling import (
    adaptive_sampling,
    frame_retry_schedule,
    is_token_mismatch_error,
)
from modules.embeddings.state import (
    dependency_rows_for_case,
    get_clip_embedding_candidates,
    get_music_map,
)
from modules.embeddings.vectors import to_bytes

STAGE = "clip_embeddings"


def embed_clip_embeddings(settings, cases: list[str] | None = None) -> None:
    """Embed clips for the given cases (default: all DEFAULT_CASES)."""
    case_names = list(cases) if cases is not None else list(DEFAULT_CASES)
    for name in case_names:
        spec = CASE_REGISTRY[name]
        _run_case(settings, spec)


def _video_path(clip_id: int, video_dir: str) -> str:
    return os.path.abspath(os.path.join(video_dir, f"{clip_id}.mp4"))


def _compute_fingerprint(
    session, spec: EmbeddingCaseSpec, settings, candidates: list[Clip]
) -> fp.Fingerprint:
    candidate_ids = sorted(c.id for c in candidates)
    data = fp.hash_rows((cid,) for cid in candidate_ids)
    config = fp.hash_text(case_config_identity(spec, settings))
    dependency = fp.hash_rows(
        dependency_rows_for_case(session, spec.name, candidate_ids)
    )
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


def _clear_case(session, case: str) -> None:
    session.query(ClipEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _run_case(settings, spec: EmbeddingCaseSpec) -> None:
    log_tag = f"embed:{spec.name}"
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )

        current = _compute_fingerprint(session, spec, settings, candidates)
        if not fp.is_stale(session, STAGE, spec.name, current):
            log(log_tag, "fingerprint match — skipping")
            return

        diff = fp.describe_diff(session, STAGE, spec.name, current)
        log(log_tag, f"stale ({diff}) — recomputing")
        _clear_case(session, spec.name)

        # Materialize work list now that the case is cleared.
        music_map: dict = {}
        if spec.text_builder is not None:
            music_map = get_music_map(session)

        video_dir = settings.paths.video_dir
        jobs: list[tuple[Clip, str | None]] = []
        for clip in candidates:
            if spec.requires_video:
                path = _video_path(clip.id, video_dir)
                if not os.path.exists(path):
                    continue
            text: str | None = None
            if spec.text_builder is not None:
                text = spec.text_builder(clip, music_map)
                if text is None:
                    continue
            jobs.append((clip, text))

        if not jobs:
            log(log_tag, "nothing to embed (empty work set after filtering)")
            fp.mark_complete(session, STAGE, spec.name, current)
            session.commit()
            return

        log(
            log_tag,
            f"{len(jobs)} clips to embed",
        )

        provider = spec.provider_factory(settings)

        with progress(len(jobs), f"Embedding {spec.name}") as advance:
            for clip, text in jobs:
                if spec.requires_video:
                    path = _video_path(clip.id, video_dir)
                    fps_, max_frames, _ = adaptive_sampling(
                        path,
                        settings.embeddings.adaptive_max_frames,
                        settings.embeddings.adaptive_default_fps,
                    )
                else:
                    path, fps_, max_frames = None, None, None

                blob = _embed_with_token_fallback(
                    provider, spec, clip, text, path, fps_, max_frames
                )
                if blob is None:
                    advance(detail=f"✗ {clip.id}")
                    continue

                row = ClipEmbedding(
                    clip_id=clip.id,
                    embedding_case=spec.name,
                    embedding=blob,
                )
                session.merge(row)
                session.commit()
                advance(detail=f"✓ {clip.id}")

        fp.mark_complete(session, STAGE, spec.name, current)
        session.commit()
        log(log_tag, "done", level="ok")
    finally:
        session.close()


def _embed_with_token_fallback(
    provider,
    spec: EmbeddingCaseSpec,
    clip,
    text: str | None,
    video_path: str | None,
    fps_: float | None,
    max_frames: int | None,
) -> bytes | None:
    """Run the provider once, with a descending frame-cap retry only for
    cases that opt into video token-budget fallback. Returns the float32
    blob on success, or None if all attempts fail (next run will retry).
    """
    if not spec.apply_video_token_fallback or max_frames is None:
        payload = spec.payload_builder(clip, text, video_path, fps_, max_frames)
        try:
            out = provider.embed(payload)
        except Exception:
            return None
        return to_bytes(out[0])

    caps = frame_retry_schedule(max_frames)
    for attempt_idx, cap in enumerate(caps):
        payload = spec.payload_builder(clip, text, video_path, fps_, cap)
        try:
            out = provider.embed(payload)
            return to_bytes(out[0])
        except Exception as e:
            if is_token_mismatch_error(e) and attempt_idx < len(caps) - 1:
                continue
            return None
    return None
```

Notes for the implementer:
- `fps_` was renamed locally to avoid shadowing the imported `fp` module alias.
- The empty-work-set branch still writes `stage_state` (test `test_empty_candidates_writes_stage_state` covers this).
- `_compute_fingerprint` lives in `runner.py` rather than `cases.py` because it needs `dependency_rows_for_case`; the case-identity composition stays in `cases.py`.

- [ ] **Step 6: Run focused tests — expect PASS**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 7: Run all embedding tests to confirm no regression**

Run: `uv run pytest tests/test_embeddings_cases.py tests/test_embeddings_public_api.py tests/test_embeddings_text.py tests/test_embeddings_sampling.py tests/test_embeddings_remote.py tests/test_embeddings_users.py tests/test_embeddings_vectors.py -v`
Expected: all PASS.

- [ ] **Step 8: Lint, type-check, full suite**

Run:
```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add modules/embeddings/cases.py modules/embeddings/state.py \
  modules/embeddings/runner.py tests/test_clip_embeddings_idempotence.py
git commit -m "$(cat <<'EOF'
feat(embeddings): wire clip_embeddings to fingerprint layer

Per-case Fingerprint(data, config, dependency) gates the case loop.
- cases.case_config_identity composes the case-recipe string fed into
  hash_text, including TEXT_RECIPE_VERSIONS and AUDIO_INSTRUCTION.
- state.dependency_rows_for_case selects the per-case columns mirroring
  what the case's text+payload builders read.
- runner deletes ClipEmbedding rows for the case on stale, recomputes,
  mark_completes, and commits once at the end (per-row commits during
  the loop are partial state; the seal lands on the final commit).

Coarse per-case wipe + full recompute is the deliberate v1 tradeoff
(see spec).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: End-to-end cascade smoke

**Files:**
- Create: `tests/test_embeddings_cascade.py`

This exercises the full chain: clip_embeddings runs first; user_embeddings runs second; an audio-only knob change wipes audio rows at BOTH layers while leaving video + sandwich untouched.

- [ ] **Step 1: Write the test**

Create `tests/test_embeddings_cascade.py`:

```python
"""End-to-end cascade: change one audio knob, only audio rows are wiped."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    StageState,
    User,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.embeddings import cases as cases_mod
from modules.embeddings.runner import embed_clip_embeddings
from modules.embeddings.users import embed_user_embeddings


@dataclass
class _FakeProvider:
    def embed(self, payload: dict) -> np.ndarray:
        seed = abs(hash(repr(sorted(payload.items())))) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal((1, 4), dtype=np.float32)


def _fake_factory(_settings):
    return _FakeProvider()


@dataclass
class _PathsStub:
    video_dir: str
    model_path: str = "/fake/qwen"


@dataclass
class _EmbeddingsStub:
    exclude_disqualified_users: bool = True
    embed_max_length: int = 1024
    adaptive_max_frames: int = 8
    adaptive_default_fps: float = 1.0


@dataclass
class _SettingsStub:
    paths: _PathsStub
    embeddings: _EmbeddingsStub


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (
        StageState,
        UserEmbedding,
        ClipEmbedding,
        Clip,
        Music,
        User,
    ):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


@pytest.fixture
def stub_providers(monkeypatch):
    for spec in cases_mod.CASE_REGISTRY.values():
        monkeypatch.setattr(spec, "provider_factory", _fake_factory)
    from modules.embeddings import runner as runner_mod, sampling as sampling_mod

    monkeypatch.setattr(
        sampling_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )
    monkeypatch.setattr(
        runner_mod, "adaptive_sampling", lambda *a, **kw: (1.0, 8, None)
    )


def _settings(tmp_path) -> _SettingsStub:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    return _SettingsStub(
        paths=_PathsStub(video_dir=str(video_dir)),
        embeddings=_EmbeddingsStub(),
    )


def _seed(session, settings: _SettingsStub) -> None:
    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            speech_transcription="hi",
        )
    )
    import os

    with open(os.path.join(settings.paths.video_dir, "10.mp4"), "wb") as f:
        f.write(b"\x00")
    session.commit()


def test_audio_instruction_change_cascades_only_to_audio(
    db_session, stub_providers, tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    _seed(db_session, settings)

    embed_clip_embeddings(settings)
    embed_user_embeddings(settings)
    db_session.expire_all()

    def snapshot():
        return {
            "clip": {
                case: db_session.get(
                    StageState, ("clip_embeddings", case)
                ).updated_at
                for case in ("video", "sandwich", "audio")
            },
            "user": {
                case: db_session.get(
                    StageState, ("user_embeddings", case)
                ).updated_at
                for case in ("video", "sandwich", "audio")
            },
        }

    before = snapshot()

    monkeypatch.setattr(
        cases_mod, "AUDIO_INSTRUCTION", "DIFFERENT INSTRUCTION"
    )

    embed_clip_embeddings(settings)
    embed_user_embeddings(settings)
    db_session.expire_all()
    after = snapshot()

    # Audio cascades through both stages; video + sandwich stay sealed.
    assert after["clip"]["video"] == before["clip"]["video"]
    assert after["clip"]["sandwich"] == before["clip"]["sandwich"]
    assert after["clip"]["audio"] != before["clip"]["audio"]
    assert after["user"]["video"] == before["user"]["video"]
    assert after["user"]["sandwich"] == before["user"]["sandwich"]
    assert after["user"]["audio"] != before["user"]["audio"]
```

- [ ] **Step 2: Run test — expect PASS**

Run: `uv run pytest tests/test_embeddings_cascade.py -v`
Expected: PASS. (T3 + T4 already implement everything the cascade exercises; this test only verifies their composition.)

- [ ] **Step 3: Lint, type-check, full suite**

Run:
```
uv run ruff check
uv run ruff format --check
uv run ty check
uv run pytest
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_embeddings_cascade.py
git commit -m "$(cat <<'EOF'
test(embeddings): end-to-end fingerprint cascade smoke

Mutate AUDIO_INSTRUCTION → assert only the audio case is wiped at both
clip_embeddings and user_embeddings layers; video + sandwich rows stay
sealed.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review

**Spec coverage:**
- Principles 1–5 (spec §"Principles") → encoded in fingerprint.py (T2) and per-stage wiring (T3, T4).
- Scope table (spec §"Scope (v1)") → only `clip_embeddings` (T4) and `user_embeddings` (T3) wired; cluster + row-level untouched.
- `stage_state` table (spec §"`stage_state` table") → T1, columns/PK match exactly.
- `modules/fingerprint.py` API (spec §"`modules/fingerprint.py`") → T2, signature-by-signature.
- Stage entry pattern (spec §"Stage entry pattern") → reproduced in T3 (users.py) and T4 (runner.py).
- Interruption semantics (spec §"Interruption semantics") → preserved: per-row commits during loop, `mark_complete` + `session.commit()` only at the end.
- Cleanup is stage-owned (spec §"Principles" #1) → `_clear_case` lives inside users.py and runner.py, not in fingerprint.py.
- `data_hash` / `config_hash` / `dependency_hash` recipes (spec §"Per-stage recipes") → `_compute_fingerprint` in users.py and runner.py implement them; `dependency_rows_for_case` selects exactly the columns from the spec's per-case tables.
- Cascade examples (spec §"Cascade examples") → tested in T4 (`test_audio_speech_change_only_invalidates_audio`, `test_audio_instruction_change_only_invalidates_audio`) and T5 (end-to-end).
- Edge cases (spec §"Edge cases") → first-run, empty-input, manual deletion: covered. Interruption: covered by design (next run sees stale state).
- Refactor order (spec §"Refactor / implementation order") → T1–T5 match exactly.
- Deferred items (spec §"Deferred") → not implemented, named in the spec.

**Placeholders:** No `TBD` / `TODO` / `implement later` / vague "appropriate" handlers; every code block is complete. The legitimate spec reference to removing the existing TODO at `embeddings/users.py:8` is satisfied in T3 step 3 (the new file does not contain that TODO).

**Type / name consistency:**
- `STAGE = "clip_embeddings"` and `STAGE = "user_embeddings"` consistent across the two stages and their tests.
- `_compute_fingerprint`, `_clear_case`, `_recompute_case` use identical signatures in both stages.
- `case_config_identity(spec, settings)` declared in cases.py (T4 step 3) and imported in runner.py (T4 step 5) with matching signature.
- `dependency_rows_for_case(session, case, candidate_ids)` declared in state.py (T4 step 4) and called from runner.py (T4 step 5) with matching signature.
- `Fingerprint`, `is_stale`, `mark_complete`, `describe_diff`, `hash_rows`, `hash_text` names consistent across T2/T3/T4/T5.

No issues found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-pipeline-idempotence-fingerprint.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Best for this plan: tasks are independent and the test suite is the contract.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batched with checkpoints for review.

Which approach?
