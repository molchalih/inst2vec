# Filter Stage Fingerprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `modules/filter.py::process_dataset` skip itself when inputs and config are unchanged, by adding a fingerprint check that follows the same pattern as `clustering/search.py` and `clustering/validation.py`.

**Architecture:** Add a `_fingerprint(session, cfg) -> fp.Fingerprint` helper to `modules/filter.py`. Wrap the existing `process_dataset` body in a single-session `is_stale` check. On match, return early. On miss, run the existing reset+hard+stats+soft+sample pipeline, then `mark_complete` and commit. Stage name `"filter"`, fixed scope `"all"`.

**Tech Stack:** Python 3, SQLAlchemy ORM, Pydantic, pytest, `modules/fingerprint.py` (existing in-repo shared layer).

**Spec:** `docs/superpowers/specs/2026-05-16-filter-fingerprint-design.md`

---

## File Structure

- **Modify** `modules/filter.py` — add `_fingerprint` helper, `STAGE`/`SCOPE` constants, and the stale-check wrapping `process_dataset`.
- **Create** `tests/test_filter_fingerprint.py` — four tests covering skip-on-unchanged, rerun-on-config-change, rerun-on-new-clip, and `StageState` row creation. Uses the conftest in-memory DB (same engine pattern as `tests/test_cluster_search.py` fingerprint tests) so `StageState` rows land in the same engine as `Clip`/`User`/`UserStats` rows.

Nothing else needs changing: `process_dataset`'s public signature, return type, and `main.py` call site are all preserved. No migration (the `stage_state` table already exists).

---

## Task 1: Tests + implementation (single TDD cycle)

**Files:**
- Create: `tests/test_filter_fingerprint.py`
- Modify: `modules/filter.py` (add helper + wrap `process_dataset` at lines 334-350)

- [ ] **Step 1: Write the four failing tests**

Create `tests/test_filter_fingerprint.py` with the exact content below.

```python
"""Fingerprint-gated idempotence for modules/filter.py::process_dataset.

Uses the conftest in-memory DB (NOT a fresh create_engine) so that
StageState rows land in the same engine as Clip/User/UserStats rows.
Each test wipes the relevant tables before seeding to provide isolation.
"""

from modules.config import FilterSettings
from modules.database import (
    Base,
    Clip,
    StageState,
    User,
    UserStats,
    get_engine,
    get_session,
)
from modules.filter import process_dataset


def _wipe() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (UserStats, StageState, Clip, User):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()


def _seed_dataset(*, n_users: int = 3, n_clips_per_user: int = 12) -> None:
    """Seed users + clips that pass all hard policies under default FilterSettings.

    With default cfg (min_video_duration=3, max_video_duration=80, min_taken_at=1640995200,
    creator_min_median_views=10000, min_eligible_clips_per_user=10),
    these clips are eligible and select_clips will pick selected_clips_per_user of them.
    """
    session = get_session()
    try:
        clip_id = 1000
        for uid in range(n_users):
            session.add(User(id=uid))
            for _ in range(n_clips_per_user):
                session.add(
                    Clip(
                        id=clip_id,
                        user_id=uid,
                        play_count=100_000,
                        video_duration=15.0,
                        taken_at=1_700_000_000,
                        video_url=f"https://example.test/{clip_id}.mp4",
                        like_count=1_000,
                    )
                )
                clip_id += 1
        session.commit()
    finally:
        session.close()


def _default_cfg() -> FilterSettings:
    return FilterSettings()


def test_creates_stage_state_row_on_first_run():
    _wipe()
    _seed_dataset()
    process_dataset(_default_cfg())

    session = get_session()
    try:
        row = session.get(StageState, ("filter", "all"))
        assert row is not None
        assert row.data_hash
        assert row.config_hash
        assert row.dependency_hash
    finally:
        session.close()


def test_skips_on_unchanged_rerun():
    """Second call with identical inputs+cfg must not re-run the work.

    Verified by manually clearing every clip.is_selected after the first run
    and checking that the second run does NOT restore them.  (select_clips
    is deterministic given the seed, so a re-run would re-select the same
    clips; if our mutation persists, we skipped.)
    """
    _wipe()
    _seed_dataset()
    cfg = _default_cfg()
    process_dataset(cfg)

    session = get_session()
    try:
        first_selected = (
            session.query(Clip).filter(Clip.is_selected.is_(True)).count()
        )
        assert first_selected > 0  # sanity: first run did something
        session.query(Clip).update(
            {Clip.is_selected: False}, synchronize_session=False
        )
        session.commit()
    finally:
        session.close()

    process_dataset(cfg)

    session = get_session()
    try:
        second_selected = (
            session.query(Clip).filter(Clip.is_selected.is_(True)).count()
        )
    finally:
        session.close()

    assert second_selected == 0  # skipped → our mutation was preserved


def test_reruns_on_config_change():
    """Changing FilterSettings invalidates the fingerprint and recomputes."""
    _wipe()
    _seed_dataset()
    process_dataset(_default_cfg())

    session = get_session()
    try:
        before_too_short = (
            session.query(Clip).filter(Clip.is_too_short.is_(True)).count()
        )
    finally:
        session.close()
    assert before_too_short == 0  # 15.0s clips are not too short under default cfg

    # Raise min_video_duration above every clip's duration.
    stricter = _default_cfg().model_copy(update={"min_video_duration": 999})
    process_dataset(stricter)

    session = get_session()
    try:
        after_too_short = (
            session.query(Clip).filter(Clip.is_too_short.is_(True)).count()
        )
    finally:
        session.close()
    assert after_too_short > 0  # recomputed under stricter cfg


def test_reruns_on_new_clip():
    """Adding a new clip invalidates the data hash and recomputes."""
    _wipe()
    _seed_dataset()
    cfg = _default_cfg()
    process_dataset(cfg)

    # Sanity: confirm the fingerprint was written so we know we're testing
    # the stale-data path, not the "no prior state" path.
    session = get_session()
    try:
        assert session.get(StageState, ("filter", "all")) is not None
        # Add a new well-formed clip for an existing user.
        session.add(
            Clip(
                id=99_999,
                user_id=0,
                play_count=100_000,
                video_duration=20.0,
                taken_at=1_700_000_000,
                video_url="https://example.test/new.mp4",
                like_count=500,
            )
        )
        session.commit()
    finally:
        session.close()

    process_dataset(cfg)

    session = get_session()
    try:
        new_clip = session.get(Clip, 99_999)
        # If filter ran, the new clip got its is_garbage derived (False).
        # If filter skipped, is_garbage would still be NULL.
        assert new_clip.is_garbage is False
    finally:
        session.close()
```

- [ ] **Step 2: Run the tests to verify they all fail**

Run: `uv run pytest tests/test_filter_fingerprint.py -v`

Expected: 4 failures.
- `test_creates_stage_state_row_on_first_run` fails because no `StageState` row is written.
- `test_skips_on_unchanged_rerun` fails because the second run re-runs and restores `is_selected`.
- `test_reruns_on_config_change` passes coincidentally on the first call but the recomputation already runs unconditionally — this test will likely pass even before the change. (See note in Step 4.)
- `test_reruns_on_new_clip` may pass for the same reason.

The two definitive red signals are `test_creates_stage_state_row_on_first_run` and `test_skips_on_unchanged_rerun`. Those are the ones whose pass↔fail status flips with the implementation.

- [ ] **Step 3: Implement the helper and wrap `process_dataset`**

Modify `modules/filter.py`. Add a `json` import at the top of the file (alphabetically after `import math`, `import random`, `import statistics`) and the `fp` + `log` imports beside the existing imports.

Replace the existing imports block (lines 1-12) with:

```python
from __future__ import annotations

import json
import math
import random
import statistics
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from modules import fingerprint as fp
from modules.config import FilterSettings
from modules.console import log
from modules.database import Clip, StageState, User, UserStats
```

Note: `StageState` is added so `_fingerprint` test imports stay consistent — though `_fingerprint` itself doesn't need to import it directly (`fp` does). It's imported here only if needed elsewhere; if unused in `filter.py`, drop it. Keep the import minimal: leave `StageState` out unless it's actually referenced. The final import line in this step should be:

```python
from modules.database import Clip, User, UserStats
```

Add `STAGE` / `SCOPE` constants directly under the imports (above `HARD_CLIP_EXCLUSION_FLAGS`):

```python
STAGE = "filter"
SCOPE = "all"
```

Add the `_fingerprint` helper above `process_dataset` (just before line 334):

```python
def _fingerprint(session: Session, cfg: FilterSettings) -> fp.Fingerprint:
    user_rows = session.query(User.id).order_by(User.id).all()
    clip_rows = (
        session.query(
            Clip.id,
            Clip.user_id,
            Clip.play_count,
            Clip.video_duration,
            Clip.taken_at,
            Clip.video_url,
            Clip.like_count,
        )
        .order_by(Clip.id)
        .all()
    )
    rows = [("u", *r) for r in user_rows] + [("c", *r) for r in clip_rows]
    data = fp.hash_rows(rows)
    config = fp.hash_text(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str)
    )
    dependency = fp.hash_text("")
    return fp.Fingerprint(data=data, config=config, dependency=dependency)
```

Replace the current `process_dataset` body (lines 334-350) with:

```python
def process_dataset(
    cfg: FilterSettings,
    *,
    engine=None,
) -> None:
    from modules.database import get_engine

    eng = engine or get_engine()
    with Session(eng) as session:
        current = _fingerprint(session, cfg)
        if not fp.is_stale(session, STAGE, SCOPE, current):
            log("filter", "fingerprint match — skipping")
            return

        diff = fp.describe_diff(session, STAGE, SCOPE, current)
        log("filter", f"stale ({diff}) — recomputing")

        _reset_dataset_processing_state(session)
        _hard_preprocess(session, cfg)
        calculate_user_stats(session)
        _soft_preprocess(session, cfg)
        _random_sample(session, cfg)

        fp.mark_complete(session, STAGE, SCOPE, current)
        session.commit()
```

Leave every other function in `modules/filter.py` unchanged.

- [ ] **Step 4: Run the tests to verify they all pass**

Run: `uv run pytest tests/test_filter_fingerprint.py -v`

Expected: all four tests pass.

If `test_skips_on_unchanged_rerun` fails, the most likely cause is that the data fingerprint accidentally hashes a derived column. Re-check `_fingerprint`: the clip projection must list only `id, user_id, play_count, video_duration, taken_at, video_url, like_count`. Any other column (especially `is_*`) means the hash changes between the pre-recompute and post-recompute states.

- [ ] **Step 5: Run the existing filter tests to confirm no regression**

Run: `uv run pytest tests/test_filter.py -v`

Expected: every pre-existing test still passes. `process_dataset`'s observable behavior on the stale path is unchanged.

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest`

Expected: all tests pass. The fingerprint touches a brand-new `StageState` scope (`("filter", "all")`), so no other stage's state should be affected.

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff check modules/filter.py tests/test_filter_fingerprint.py && uv run ruff format modules/filter.py tests/test_filter_fingerprint.py`

Expected: no errors.

Run: `uv run ty check modules/filter.py tests/test_filter_fingerprint.py`

Expected: no errors. (`fp.Fingerprint`, `FilterSettings.model_dump`, and SQLAlchemy query types are all already used in identical patterns in `modules/clustering/search.py`, so the type checker should be happy.)

- [ ] **Step 8: Commit**

```bash
git add modules/filter.py tests/test_filter_fingerprint.py
git commit -m "feat(filter): fingerprint-gated process_dataset

Skip filter stage on re-run when raw clip inputs and FilterSettings are
unchanged. Single session, dataset-wide scope (\"all\"), hashes only raw
parse-derived clip fields plus user ids (no filter-derived columns)."
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Plan step |
| --- | --- |
| `_fingerprint(session, cfg) -> fp.Fingerprint` helper | Step 3 |
| `data` = hash_rows over users + clips sorted by id, raw input fields only | Step 3 (`_fingerprint` body) |
| `config` = `hash_text(json.dumps(...))` over all FilterSettings fields | Step 3 (`cfg.model_dump()` + `sort_keys=True`) |
| `dependency` = `hash_text("")` | Step 3 |
| Single-session stale-check wrapping `process_dataset` | Step 3 (replacement body) |
| Scope key = `"all"` | Step 3 (`SCOPE = "all"`) |
| Exclude all filter-derived columns from data hash | Step 3 (clip projection lists only 7 raw columns); Step 4 notes the consequence if violated |
| Stage commits its own transaction; `mark_complete` only merges | Step 3 (`fp.mark_complete` then `session.commit()`) |
| Test: skips on rerun | Step 1 (`test_skips_on_unchanged_rerun`) |
| Test: reruns on config change | Step 1 (`test_reruns_on_config_change`) |
| Test: reruns on new clip | Step 1 (`test_reruns_on_new_clip`) |
| StageState row gets created | Step 1 (`test_creates_stage_state_row_on_first_run`) — bonus check |

All covered.

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N". All code blocks are complete.

**Type consistency:** `STAGE`/`SCOPE` constants used identically in `_fingerprint` callers and in test lookups (`session.get(StageState, ("filter", "all"))`). `cfg.model_dump()` is the documented Pydantic v2 API and matches the version in `pyproject.toml`. `fp.Fingerprint`, `fp.is_stale`, `fp.mark_complete`, `fp.describe_diff`, `fp.hash_rows`, `fp.hash_text` are all imported via `from modules import fingerprint as fp` and called identically to `modules/clustering/search.py`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-16-filter-fingerprint.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
