# Clip embeddings: incremental stage

## Background

`modules/embeddings/runner.py` embeds clips for each `embedding_case`
(`video`, `sandwich`, `audio`). It gates work with a stage-level
`Fingerprint(data, config, dependency)`:

- `data` — hash of sorted candidate clip ids.
- `config` — hash of recipe identity (model, max_len, frames, fps, text
  recipe version, audio instruction, …).
- `dependency` — hash of `dependency_rows_for_case()`, the per-candidate
  tuples of upstream fields the case's text + payload builders read.

On any mismatch the stage **wipes every `ClipEmbedding` row for the
case** and re-embeds every candidate. This is correct but wasteful: a
single new user, a single caption edit, or any candidate-set change
forces a full re-embed.

The per-row unique constraint `(clip_id, embedding_case)` on
`clip_embeddings` already supports per-row idempotence; the runner just
does not exploit it.

## Goal

Restructure clip embedding so that within a case:

1. **Recipe drift (`config_hash` change) still wipes** the case and
   re-embeds all candidates.
2. **Data / dependency drift** becomes incremental:
   - Embed clips that are new to the candidate set (no row).
   - Re-embed clips whose upstream data changed since they were
     embedded.
   - Leave clips whose upstream is unchanged untouched (no provider
     call).
   - Leave rows for clips that left the candidate set in place; the
     downstream user-embedding aggregation filters them out.

Failure semantics, the fingerprint layer, and the three case
definitions are unchanged.

## Design

### Per-row source hash

Add a nullable `source_hash` column to `ClipEmbedding`:

```python
source_hash: Mapped[str | None] = mapped_column(String, nullable=True)
```

`source_hash` is the SHA-256 of the case's per-row dependency tuple as
returned by `dependency_rows_for_case(session, case, [clip_id])`. It is
written every time the runner produces an embedding for that clip and
read every time the runner decides whether re-embedding is needed.

NULL means "unknown source state, treat as stale": a row with NULL is
always re-embedded. A one-shot migration script
(`scripts/migrate_clip_embeddings_source_hash.py`) backfills NULL rows
from current upstream so the first run after the schema change is fast.

### Single source of dependency truth

The existing `dependency_rows_for_case(session, case, candidate_ids)`
returns one tuple per candidate, in clip_id order, with the case-
specific upstream columns. A new helper in
`modules/embeddings/state.py` computes both per-clip hashes and the
stage-level aggregate from one call so they stay byte-identical:

```python
def per_clip_source_hashes_and_aggregate(
    session: Session, case: str, candidate_ids: list[int]
) -> tuple[dict[int, str], str]:
    rows = dependency_rows_for_case(session, case, candidate_ids)
    per_clip = {r[0]: fp.hash_rows([r]) for r in rows}
    aggregate = fp.hash_rows(rows)
    return per_clip, aggregate
```

A small read helper exposes the existing rows:

```python
def get_embedded_source_hashes(
    session: Session, case: str
) -> dict[int, str | None]:
    rows = (
        session.query(ClipEmbedding.clip_id, ClipEmbedding.source_hash)
        .filter(ClipEmbedding.embedding_case == case)
        .all()
    )
    return {r.clip_id: r.source_hash for r in rows}
```

`get_embedded_clip_ids()` stays as it is; no other dependent of the
state module changes.

### Runner control flow

`embed_clip_embeddings(settings, cases?)` keeps its signature and
per-case dispatch. `_run_case` becomes:

```python
def _run_case(settings, spec):
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )
        current, per_clip = _compute_fingerprint_and_per_clip(
            session, spec, settings, candidates
        )

        if not fp.is_stale(session, STAGE, spec.name, current):
            log(f"embed:{spec.name}", "fingerprint match — skipping")
            return

        stored = session.get(StageState, (STAGE, spec.name))
        if stored is not None and stored.config_hash != current.config:
            log(
                f"embed:{spec.name}",
                f"config drift ({fp.describe_diff(session, STAGE, spec.name, current)}) — wiping case",
            )
            _wipe_case(session, spec.name)

        embedded = get_embedded_source_hashes(session, spec.name)
        targets = _diff_targets(per_clip, embedded)
        log(
            f"embed:{spec.name}",
            f"{len(targets)} clip(s) to (re-)embed",
        )

        _embed_targets(
            session, spec, settings, candidates, targets, per_clip, current
        )
    finally:
        session.close()
```

Wipe only fires on `config_hash` drift; after the wipe `embedded` is
empty and the diff naturally targets every current candidate. Without
a wipe the diff narrows to only missing or hash-mismatched clips. A
partial failure leaves `stage_state` unsealed but keeps successfully
embedded rows + their `source_hash`; the next run's diff finds them
unchanged and retries only the still-missing/stale clips — true even
on the very first run after a partial failure.

Helpers in the same module:

```python
def _compute_fingerprint_and_per_clip(session, spec, settings, candidates):
    candidate_ids = sorted(c.id for c in candidates)
    per_clip, dep_agg = per_clip_source_hashes_and_aggregate(
        session, spec.name, candidate_ids
    )
    current = fp.Fingerprint(
        data=fp.hash_rows((cid,) for cid in candidate_ids),
        config=fp.hash_text(case_config_identity(spec, settings)),
        dependency=dep_agg,
    )
    return current, per_clip


def _wipe_case(session, case):
    session.query(ClipEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _diff_targets(per_clip, embedded):
    return {
        cid
        for cid, want in per_clip.items()
        if embedded.get(cid) != want
    }
```

`_embed_targets` is the existing inner work loop lifted whole: same
`_embed_with_token_fallback` retry, same per-row `session.merge` +
`commit`, same "don't seal on partial failure" semantics. Two changes:

- It iterates over `targets` (a subset of `candidates`) rather than all
  candidates.
- The `ClipEmbedding` instance it merges now carries
  `source_hash=per_clip[clip.id]`.

On full success it calls `fp.mark_complete` and commits. On any clip
failure it leaves the stage unsealed; the next run computes targets
again and the diff naturally narrows to only the still-missing /
still-stale clips — strictly better than today's "wipe and try the
whole set again."

When `targets` is empty (e.g. a clip was deselected — `data` drifts
but no current candidate needs work) the function still calls
`fp.mark_complete` and commits, so the stage seals to the new
fingerprint and subsequent runs hit the fast `is_stale` skip path.
This preserves the existing "empty work set" behaviour in the
wipe-and-recompute branch.

A useful side-effect: a clip that was deselected and later reselected
keeps its `ClipEmbedding` row. If its upstream data has not drifted
in the meantime, its stored `source_hash` matches the current
per-clip hash and the diff skips it — free reactivation, no
embedding call.

### Downstream filter

`get_clip_embedding_rows_for_user_aggregation` gains a candidate-
eligibility predicate so orphan rows for deselected clips don't
contaminate user-level means:

```python
def get_clip_embedding_rows_for_user_aggregation(session, case):
    return (
        session.query(ClipEmbedding.embedding, Clip.user_id)
        .join(Clip, ClipEmbedding.clip_id == Clip.id)
        .filter(
            ClipEmbedding.embedding_case == case,
            *clip_used_in_analysis(),
        )
        .all()
    )
```

`modules/embeddings/users.py` reads this function for both its
fingerprint dependency and its aggregation work set, so a clip leaving
or rejoining the candidate set automatically flips the user-embedding
dependency hash and triggers a recompute downstream — without further
wiring.

### Migration

`scripts/migrate_clip_embeddings_source_hash.py` runs once after
pulling the schema change:

1. For each distinct `embedding_case` present in `clip_embeddings`,
   gather the existing `clip_id`s.
2. Compute `per_clip_source_hashes_and_aggregate(session, case, ids)`.
3. For every row whose `source_hash` is currently NULL, write the new
   hash. Skip rows that already have a hash (idempotent re-run).
4. Commit per case.

The script never embeds; it only fingerprints what is already on disk.
If the user skips the migration, the runner still works — every
existing row counts as stale and gets re-embedded on first run.

## Component summary

| File | Change |
| ---- | ------ |
| `modules/database/models.py` | `ClipEmbedding.source_hash: str \| None` (nullable). |
| `modules/embeddings/state.py` | Add `per_clip_source_hashes_and_aggregate`, `get_embedded_source_hashes`. Filter `clip_used_in_analysis()` into `get_clip_embedding_rows_for_user_aggregation`. |
| `modules/embeddings/runner.py` | Replace `_compute_fingerprint`, `_clear_case`, `_run_case` with config-drift vs data/dependency-drift dispatch. Add `_diff_targets`, `_wipe_case`, `_compute_fingerprint_and_per_clip`. Write `source_hash` on every merged row. |
| `scripts/migrate_clip_embeddings_source_hash.py` | One-shot backfill. |
| `tests/test_clip_embeddings_idempotence.py` | Add incremental-path tests (see Testing). |
| `tests/test_migrate_clip_embeddings_source_hash.py` | New. |

`modules/fingerprint.py`, `modules/embeddings/cases.py`,
`modules/embeddings/users.py`, and the case-spec dataclass are all
untouched.

## Failure modes

- **Provider raises for clip C during incremental run.** C's row is not
  written; rows written before C are kept (each merge commits). The
  stage is not sealed because `failures > 0`. Next run: stage is stale
  (no fresh seal) so `is_stale` returns True. No config drift → no
  wipe. The diff reads back `embedded`: every successfully-embedded
  clip's stored `source_hash` matches the current per-clip hash → those
  clips are skipped. C has no row → it is the only target. Holds even
  when the partial failure was the very first run.
- **User deletes some `ClipEmbedding` rows manually.** Next run: stage
  fingerprint may match (no upstream change) and `is_stale` returns
  False → no work happens. This already-rare case is unchanged from
  today; recovering from manual DB edits is out of scope.
- **Schema upgrade without running the migration script.** All
  `source_hash` values are NULL. First run enters the data/dependency-
  drift branch (stored state mismatches current), every clip's
  `embedded.get(cid)` is NULL ≠ desired hash, so every clip is
  re-embedded. Correct, just slow.
- **`embedding_case` config changes (recipe bump).** Stored `config_hash`
  differs → wipe-and-recompute branch → full re-embed. Unchanged from
  today.

## Testing

`tests/test_clip_embeddings_idempotence.py` — added cases:

- `test_adding_new_candidate_only_embeds_the_new_one`: embed `{10}`,
  capture `ClipEmbedding(10).embedding` bytes, add clip `11`, rerun,
  assert clip 10's bytes are byte-identical and clip 11 has a new row
  with a non-NULL `source_hash`.
- `test_caption_change_reembeds_only_changed_clip_for_sandwich`: embed
  `{10, 11}` for the `sandwich` case, mutate clip 10's `caption_clean`,
  rerun, assert clip 10's bytes change, clip 11's bytes unchanged, both
  `source_hash` columns equal the current per-clip hash.
- `test_deselecting_a_clip_keeps_its_row_but_drops_it_from_aggregation`:
  embed `{10, 11}`, set clip 11 `is_selected=False`, rerun, assert
  clip 11's `ClipEmbedding` row still exists and that
  `get_clip_embedding_rows_for_user_aggregation` no longer returns it.
- `test_config_change_still_wipes`: embed, mutate `AUDIO_INSTRUCTION`
  via monkeypatch, rerun, assert all `audio`-case rows were rewritten
  (new bytes) and that the other cases' bytes are byte-identical.
- `test_partial_failure_seals_successes_and_narrows_retry`: with a
  flaky `_embed_with_token_fallback` that fails clip 11 on first
  attempt, embed `{10, 11}`, assert clip 10 sealed and stage unsealed;
  rerun and assert the provider was called for clip 11 only (clip 10
  was not re-embedded).

`tests/test_migrate_clip_embeddings_source_hash.py` — new file:

- Seed rows with `source_hash=NULL` plus upstream state, run the
  migration, assert every row's `source_hash` equals the per-clip hash
  from `dependency_rows_for_case`.
- Rerun the migration on a fully-backfilled DB and assert it makes no
  changes (idempotent).

Existing tests under `tests/test_clip_embeddings_idempotence.py`,
`tests/test_embeddings_users.py`, and the cascade tests stay green
without modification.

## Out of scope

- Changes to `modules/fingerprint.py` (the shared idempotence layer).
- Changes to user_embeddings, cluster_search, cluster_validation, or
  clustering proper.
- Garbage-collecting orphan `ClipEmbedding` rows. They cost storage but
  no compute and are correctly excluded from user aggregation; a future
  retention script can prune them if disk pressure becomes an issue.
