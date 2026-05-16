# Filter Stage Fingerprint Idempotence

## Goal

Add fingerprint-based idempotence to `modules/filter.py::process_dataset` so the stage skips itself on re-run when nothing has changed, and recomputes only when the input data or config has changed. Pattern mirrors `modules/clustering/search.py` and `modules/clustering/validation.py`, adapted for the differences in filter's compute model.

## Why

`process_dataset` currently runs unconditionally on every pipeline invocation: full reset, hard preprocess, user stats, soft preprocess, random sample. It is deterministic given (clip inputs, user set, config), so re-running it without changes is pure waste. The fingerprint layer (`modules/fingerprint.py`) already exists and is used by three clustering stages; this brings filter into the same regime.

## Architecture

Single-session orchestration (in contrast to clustering's split read-only / compute / write structure). Filter's work is in-DB row mutation — there is no expensive in-memory compute phase to keep outside a write transaction — so splitting buys nothing.

Flow inside `process_dataset`:

1. Open one `Session`.
2. Compute `current = _fingerprint(session, cfg)`.
3. `fp.is_stale(session, "filter", "all", current)`:
   - If fresh: log skip, return.
   - If stale: log diff, run the existing reset+hard+stats+soft+sample pipeline, then `fp.mark_complete`, then `session.commit()`.

Stage name: `"filter"`. Scope key: `"all"` (fixed string — filter is dataset-wide, not per embedding case).

## `_fingerprint` helper

```python
def _fingerprint(session: Session, cfg: FilterSettings) -> fp.Fingerprint:
    user_rows = session.query(User.id).order_by(User.id).all()
    clip_rows = (
        session.query(
            Clip.id, Clip.user_id, Clip.play_count, Clip.video_duration,
            Clip.taken_at, Clip.video_url, Clip.like_count,
        )
        .order_by(Clip.id)
        .all()
    )
    data = fp.hash_rows(
        [("u", *r) for r in user_rows] + [("c", *r) for r in clip_rows]
    )
    config = fp.hash_text(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str)
    )
    dependency = fp.hash_text("")  # parse has no StageState row
    return fp.Fingerprint(data=data, config=config, dependency=dependency)
```

### Data hash

- **Per-user contribution**: `("u", user.id)` only. Filter reads no parse-derived User fields as input — it only writes `is_eligible`, `is_selected`, `is_low_plays_median`, `is_not_enough_clips`. User existence matters (`_derive_eligibility` looks up `user_map.get(clip.user_id)`), so membership-via-id is sufficient and avoids spurious invalidation when unrelated parse fields (`follower_count`, `parse_status`) change.
- **Per-clip contribution**: `("c", clip.id, clip.user_id, clip.play_count, clip.video_duration, clip.taken_at, clip.video_url, clip.like_count)`. These are exactly the fields `_is_garbage`, `_is_too_short`, `_is_too_long`, `_is_too_old`, `_flag_global_percentile_clips`, `_compute_creator_robust_stats`, and `select_clips` read.
- **Row-type tag** (`"u"` / `"c"`): cheap insurance against the (unlikely) case of a user id colliding with a clip id under `hash_rows`'s tuple-repr hashing.
- **Excluded**: every filter-derived column on Clip (`is_garbage`, `is_too_short`, `is_too_long`, `is_too_old`, `is_low_percentile`, `is_high_percentile`, `is_creator_low_outlier`, `log_plays`, `creator_relative_robust_z`, `is_preprocessed`, `is_eligible`, `is_selected`) and every derived column on User. Self-referential inputs would mean the fingerprint never stabilizes.

### Config hash

`fp.hash_text(json.dumps(cfg.model_dump(), sort_keys=True, default=str))` — covers all 11 fields of `FilterSettings` without manual enumeration. Any future field added to `FilterSettings` auto-invalidates.

### Dependency hash

`fp.hash_text("")` — parse has no `StageState` row (its idempotence is row-level via `parse_status`, not stage-level). Matches the documented behavior of `stage_dependency_hash` when no upstream row exists.

## `process_dataset` integration

```python
STAGE = "filter"
SCOPE = "all"

def process_dataset(cfg: FilterSettings, *, engine=None) -> None:
    from modules.database import get_engine

    eng = engine or get_engine()
    with Session(eng) as session:
        current = _fingerprint(session, cfg)
        stale = fp.is_stale(session, STAGE, SCOPE, current)
        if not stale:
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

Key invariants:

- The fingerprint is computed from the **pre-recompute** state. That is what determines staleness, and that is what gets stored on `mark_complete`. Filter writes do not affect any field that contributes to the data hash, so re-computing the fingerprint after the work would yield the same value — but pre-compute is simpler and matches clustering's pattern.
- The stage commits its own transaction; `mark_complete` only merges the `StageState` row.
- The `engine=None` kwarg is preserved so tests can inject in-memory engines.

## Test plan

New file: `tests/test_filter_fingerprint.py`. Three scenarios, each with its own in-memory engine via the existing `tests/conftest.py` fixtures.

1. **`test_skips_on_unchanged_rerun`** — Seed users + clips, run `process_dataset(cfg)`, capture derived state. Manually mutate a derived field (e.g., set every `clip.is_selected = False`) and commit. Call `process_dataset(cfg)` again; assert the mutation was **not** undone — proof the second run skipped.
2. **`test_reruns_on_config_change`** — Run with default `cfg`; assert clips pass `is_too_short` / `is_too_long` correctly. Then run with `cfg.model_copy(update={"min_video_duration": 999})`; assert every clip is now `is_too_short=True`.
3. **`test_reruns_on_new_clip`** — Run once, insert a new `Clip` row, run again; assert the new clip has its derived flags populated (`is_garbage is not None`).

## Non-goals

- No row-level idempotence; filter is dataset-wide.
- No change to `process_dataset`'s public signature, return type, or transaction semantics on the stale path.
- No migration — the `stage_state` table already exists.
- No change to clustering or any other stage.
