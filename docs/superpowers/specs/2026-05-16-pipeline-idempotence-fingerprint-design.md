# Pipeline Idempotence — Fingerprint Layer Design

**Date:** 2026-05-16
**Status:** Approved (design phase). Implementation plan to follow.

## Goal

Add a small, explicit, stage-driven idempotence layer so that each central pipeline stage can decide on entry whether its existing outputs are still valid, and automatically wipe-and-recompute its affected scope when they are not. The system tracks artifact validity, not execution history.

The motivating use case: change a config knob (e.g. an embedding text-builder version, or the audio instruction string), rerun `main.py`, and have only the affected scopes recompute while unchanged stages are skipped.

## Non-goals

- No DAG engine, orchestrator, or workflow framework.
- No decorators, no implicit instrumentation.
- No recursive invalidation engine. Stale propagation emerges naturally because each stage recomputes its own dependency hash on entry from actual upstream output rows.
- No automatic cleanup helpers in the shared layer. Cleanup is stage-owned.
- No changes to `main.py`'s execution order or stage call shape.
- No migration of row-level stages (`download`, `music`, `speech`, `captions`) — they keep their current output-existence / column-based skipping.
- No migration of `cluster_search`, `clustering`, `cluster_validation`, `visualization` in this pass — those stages have their own existing hash logic (`ClusterRun.dataset_hash`, `validation_config_hash`, `in_current_grid`) and will be unified into this layer when they are next refactored.
- No `Clip.downloaded_at` column. Deferred to v2.

## Principles

1. **Stages own everything case-specific**: choosing what data/config/dependency to hash, wiping outputs on mismatch, recomputing, and committing.
2. **The shared layer is dumb**: it compares two `Fingerprint` triples and stores them. It has no knowledge of stages, scopes, or what hashes mean.
3. **Stale means automatic recompute.** Not refuse, not log-and-skip. The whole point of the system is auto-propagation.
4. **Downstream `dependency_hash` is derived from actual upstream output rows, not from upstream `stage_state`.** A stage that mutates rows outside the framework must still trigger downstream recompute.
5. **Row-level stages stay as-is.** Their column-based skipping (`Clip.is_downloaded`, `Clip.is_speech_detected`, `Clip.caption_clean`, etc.) avoids recomputing existing rows but does NOT detect semantic staleness after config changes. This is acceptable for v1; the scope here is only `clip_embeddings` and `user_embeddings`.

## Scope (v1)

| Stage | Status | Scope key |
|---|---|---|
| `clip_embeddings` | **Wired** | `embedding_case` (`video`, `sandwich`, `audio`) |
| `user_embeddings` | **Wired** | `embedding_case` |
| `cluster_search`, `clustering`, `cluster_validation`, `visualization` | Deferred | n/a |
| `download`, `music`, `speech`, `captions`, `parse`, `filter` | Out of scope (row-level) | n/a |

## New artifacts

- One table: `stage_state` in the main DB (alongside `users` / `clips` / `clip_embeddings`).
- One file: `modules/fingerprint.py` (~50 lines).
- One pair of test files: `tests/test_fingerprint.py`, plus extensions to `tests/test_embeddings_*` for cascade behavior.

Nothing else changes.

## `stage_state` table

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

- Composite PK `(stage_name, scope_key)`. Matches the lookup in `fingerprint.py`: `session.get(StageState, (stage, scope))`. No surrogate id.
- `scope_key` is just a string the stage picks: `"sandwich"`, `"video"`, `"audio"`, or `"global"` for stages with no per-scope split.
- Three separate hash columns let `describe_diff` say "config changed" rather than "fingerprint changed".
- `updated_at` is forensic only — not used in staleness logic.

**Wiring**: `init_db()` already calls `Base.metadata.create_all(_engine)`. The new table is created automatically on next run. `create_all` is sufficient because `stage_state` is a wholly new table; this is NOT a general migration path for altering existing tables.

**Identity-DB coupling**: none. Lives in the main DB only.

**Manual reset / debug** (no new tooling required):

```sql
-- Force user_embeddings to recompute for the sandwich case on next run:
DELETE FROM stage_state WHERE stage_name='user_embeddings' AND scope_key='sandwich';

-- Inspect current state:
SELECT * FROM stage_state ORDER BY stage_name, scope_key;
```

## `modules/fingerprint.py`

```python
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from modules.database import StageState


@dataclass(frozen=True)
class Fingerprint:
    data: str         # hash of the input population for this stage+scope
    config: str       # hash of relevant config / recipe identity
    dependency: str   # hash of upstream output state the stage actually consumes


# ── hashing utilities (stages decide what to feed in) ────────────────────────

def hash_rows(rows: Iterable[tuple[Any, ...]]) -> str:
    """Stable SHA-256 over an iterable of tuples.

    Caller is responsible for passing rows sorted on a stable key.
    Record separator 0x1E prevents `(1,2),(3)` and `(1),(2,3)` from
    colliding.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(repr(row).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── compare + store ──────────────────────────────────────────────────────────

def is_stale(session: Session, stage: str, scope: str, current: Fingerprint) -> bool:
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
    """Merge the stage_state row but do NOT commit.

    The stage commits after its artifact writes + this merge, so the
    fingerprint and the artifacts land in one transaction. Committing
    here would allow a valid fingerprint to be recorded while later
    artifact writes in the same logical operation fail.
    """
    session.merge(StageState(
        stage_name=stage,
        scope_key=scope,
        data_hash=current.data,
        config_hash=current.config,
        dependency_hash=current.dependency,
    ))


def describe_diff(
    session: Session, stage: str, scope: str, current: Fingerprint
) -> str:
    """Short human-readable note for log lines: which of the three hashes changed.

    Returns 'no prior state' on first run, e.g. 'data+dependency' when
    both differ, '' if all match. Pure formatting; no side effects.
    """
    row = session.get(StageState, (stage, scope))
    if row is None:
        return "no prior state"
    parts = []
    if row.data_hash != current.data: parts.append("data")
    if row.config_hash != current.config: parts.append("config")
    if row.dependency_hash != current.dependency: parts.append("dependency")
    return "+".join(parts)
```

**Deliberately absent**:

- No cleanup helpers — stages own deletion (`DELETE FROM clip_embeddings WHERE embedding_case = ?`).
- No upstream-state reads — each stage's `dependency_hash` is built from actual upstream artifact rows.
- No retry/lock/transaction wrapping.
- No file-system access.

## Stage entry pattern

Identical shape in both `clip_embeddings` and `user_embeddings`:

```python
case_name = ...                                    # scope_key
fp = Fingerprint(
    data=fingerprint.hash_rows(...),               # input population
    config=fingerprint.hash_text("..."),           # recipe identity
    dependency=fingerprint.hash_rows(...),         # upstream output state
)
if fingerprint.is_stale(session, STAGE, case_name, fp):
    diff = fingerprint.describe_diff(session, STAGE, case_name, fp)
    log(f"{STAGE}:{case_name}", f"stale ({diff}) — recomputing")
    _clear_case(session, case_name)                # stage-owned cleanup
    _recompute_case(session, case_name, ...)       # existing loop, per-row commits OK
    fingerprint.mark_complete(session, STAGE, case_name, fp)
    session.commit()                               # seals the fingerprint
else:
    log(f"{STAGE}:{case_name}", "fingerprint match — skipping")
```

### Interruption semantics

Per-row commits during recompute land in `clip_embeddings` / `user_embeddings`, but `stage_state` is only merged-and-committed at the end. If the run is interrupted mid-loop, the next run sees `is_stale=True` (no stored fingerprint for the current recipe), `_clear_case` wipes the partial rows, and the case restarts cleanly.

**Deliberate v1 tradeoff:** coarse per-case wipe + full recompute. This sacrifices partial reuse on interruption (and on any non-zero recipe change), but it keeps the model simple and deterministic: at any point in time, a case's outputs are either fully consistent with the recorded fingerprint or fully absent. Per-row reuse on partial recipe changes is a deferred v2 optimization, not an accidental limitation.

## Per-stage recipes

### `clip_embeddings` — per `embedding_case`

`STAGE = "clip_embeddings"`, `scope_key = case_name`.

**Cleanup**:

```python
session.query(ClipEmbedding).filter_by(embedding_case=case_name).delete()
session.commit()
```

**`data_hash`** — the candidate set (independent of case):

```python
candidate_ids = sorted(c.id for c in candidates)   # from get_clip_embedding_candidates
data = hash_rows((cid,) for cid in candidate_ids)
```

**`config_hash`** — recipe identity for this case. Stable string composed from spec + settings, then hashed:

```python
config = hash_text(
    f"case={spec.name}"
    f"|provider={spec.provider_factory.__name__}"
    f"|model={os.path.basename(settings.paths.model_path)}"
    f"|max_len={settings.embeddings.embed_max_length}"
    f"|max_frames={settings.embeddings.adaptive_max_frames}"
    f"|fps={settings.embeddings.adaptive_default_fps}"
    f"|token_fallback={spec.apply_video_token_fallback}"
    f"|text_recipe={_text_recipe_tag(spec)}"       # "none" / "sandwich_v1" / "audio_v1"
    f"|instruction={AUDIO_INSTRUCTION if spec.name == 'audio' else ''}"
)
```

`_text_recipe_tag` is a small lookup added in `cases.py`. Bumping the version string is the explicit, human-controlled trigger for "the text-builder semantics changed; invalidate".

**`dependency_hash`** — upstream output state the case actually consumes. Case-specific tuple per candidate clip, hashed in clip_id order:

| case | tuple per candidate |
|---|---|
| `video` | `(clip_id, is_downloaded)` |
| `sandwich` | `(clip_id, is_downloaded, music_id, music_features_extracted)` |
| `audio` | `(clip_id, music_id, music_features_extracted, speech_transcription, speech_translation, caption_translation_or_clean)` |

These tuples cover exactly what each case's `payload_builder` and `text_builder` actually read. Any mutation to any of these fields for any participating clip changes the digest.

A new helper `dependency_rows_for_case(session, case_name) -> list[tuple]` lives in `modules/embeddings/state.py` so the recipe stays co-located with the rest of the case's state code.

**Known limitation (caveat #2):** the `video`/`sandwich` cases use `Clip.is_downloaded` rather than file mtime. If a video file on disk is semantically replaced without nuking `Clip.is_downloaded`, the case will not notice. The recommended v2 fix is a `Clip.downloaded_at` column; until then, file replacement must be paired with `UPDATE clips SET is_downloaded=NULL WHERE id IN (...)`.

### `user_embeddings` — per `embedding_case`

`STAGE = "user_embeddings"`, `scope_key = case_name`.

**Cleanup**:

```python
session.query(UserEmbedding).filter_by(embedding_case=case_name).delete()
session.commit()
```

**`data_hash`** — participating user set:

```python
rows = get_clip_embedding_rows_for_user_aggregation(session, case_name)
user_ids = sorted({user_id for _, user_id in rows})
data = hash_rows((uid,) for uid in user_ids)
```

**`config_hash`** — currently no knobs; pin the recipe so future changes bump it:

```python
config = hash_text("agg=mean_pool|v=1")
```

When a weighted aggregator or alternative pooler lands, bump `v=`.

**`dependency_hash`** — actual upstream `ClipEmbedding` row state (NOT upstream `stage_state`):

```python
rows = session.query(
    ClipEmbedding.clip_id, ClipEmbedding.updated_at
).filter(ClipEmbedding.embedding_case == case_name) \
 .order_by(ClipEmbedding.clip_id).all()

dependency = hash_rows((r.clip_id, r.updated_at.isoformat()) for r in rows)
```

One cheap aggregate query, no blob hashing. The existing `onupdate=func.now()` on `ClipEmbedding.updated_at` makes any re-embed automatically bump the digest; manual deletions also flip it. The TODO comment at `modules/embeddings/users.py:8` is removed when this lands.

## Cascade examples

The traces the design is supposed to deliver:

- **Bump `EmbeddingsSettings.adaptive_max_frames`** → `config_hash` flips for all three clip cases → all three wipe + recompute → all three `user_embeddings` cases see a new `dependency_hash` → all three user cases wipe + recompute.
- **Tweak only `AUDIO_INSTRUCTION`** → only `audio` clip `config_hash` changes → only audio clip rows wiped → only audio user-embeddings recompute. Video + sandwich untouched.
- **Re-translate captions for N clips** → audio clip `dependency_hash` changes → only audio clip rows for those N+downstream wipe (entire audio case wipes per v1 coarse-recompute tradeoff). Video + sandwich unaffected.
- **Add a new selected+downloaded clip** → all cases' `data_hash` changes → all wipe + recompute. (Acceptable: the candidate set genuinely shifted.)
- **No change at all between runs** → both stages log "fingerprint match — skipping" and return immediately.

## Edge cases

| Case | Behavior |
|---|---|
| First run (no `stage_state` row) | `is_stale=True` → recompute → first `mark_complete` writes the row. |
| Empty candidate set | Stage still computes a fingerprint (`hash_rows([])` is the well-defined empty SHA-256), writes the row, logs "nothing to do". A second no-input run is correctly a no-op. |
| Interruption mid-recompute | `stage_state` never sealed. Next run wipes and restarts. Lost work is the v1 tradeoff. |
| Manual artifact edit / row deletion | `dependency_hash` or `data_hash` naturally changes; stage recomputes that scope. |
| Manual `stage_state` row deletion | Documented force-recompute escape hatch. |
| Stale recipe rename (a case dropped from `DEFAULT_CASES`) | Orphan row persists. Harmless; can be hand-deleted. |
| File-replacement without DB flag change | Not detected (caveat #2). Documented; v2 mitigation noted. |

## Testing strategy

### `tests/test_fingerprint.py` (new)

- `hash_rows` is deterministic and order-sensitive (order of rows matters).
- `hash_rows([])` is stable across calls.
- `is_stale` returns `True` when no row exists.
- `is_stale` returns `True` when any single hash differs.
- `is_stale` returns `False` when all three match.
- `mark_complete` issues a `merge` but does NOT commit (assert via `session.dirty` or in-memory inspection).
- `describe_diff` returns `"no prior state"`, `""`, `"data"`, `"config"`, `"dependency"`, or combinations like `"data+config"`.

### `tests/test_clip_embeddings_idempotence.py` (extends existing embedding tests)

- Fresh DB → all candidates embedded for each case; `stage_state` row written per case.
- Rerun with identical inputs → 0 new embeddings, log includes `"match"` per case.
- Bump `embeddings.adaptive_max_frames` → all three cases wipe + recompute.
- Mutate `Clip.speech_translation` for one candidate → only `audio` case wipes + recompute.
- Add a new selected+downloaded clip → all three cases' `data_hash` flips → all wipe + recompute.
- Change `AUDIO_INSTRUCTION` → only audio case wipes.

### `tests/test_user_embeddings_idempotence.py` (new)

- After clip_embeddings completes, user_embeddings runs and writes its `stage_state`.
- Rerun → no-op (log includes `"match"`).
- Force `updated_at` bump on a single `ClipEmbedding` row for the `audio` case → only audio `user_embeddings` recomputed.

### End-to-end cascade smoke

A single test that walks the full cascade: mutate `AUDIO_INSTRUCTION`, run `embed_clip_embeddings` then `embed_user_embeddings`, assert that only audio rows are wiped at both layers; video + sandwich rows untouched.

## Refactor / implementation order

Five small commits, each independently testable. No long-lived branch.

1. **T1 — schema**: Add `StageState` model in `modules/database.py`. Tests assert the table is created in `:memory:` DB. No other code touched.
2. **T2 — helper**: Add `modules/fingerprint.py` + `tests/test_fingerprint.py`. Pure logic, no callers yet. Lint + ty + tests.
3. **T3 — `user_embeddings`**: Wire `modules/embeddings/users.py` (smaller, already has the TODO at line 8). Add `tests/test_user_embeddings_idempotence.py`. Safe to do before T4 because `user_embeddings.dependency_hash` reads `ClipEmbedding` rows directly — it does not require `clip_embeddings` to be wired.
4. **T4 — `clip_embeddings`**: Wire `modules/embeddings/runner.py`. Add `dependency_rows_for_case` helper in `embeddings/state.py` and `_text_recipe_tag` mapping in `cases.py`. Add `tests/test_clip_embeddings_idempotence.py`. All three cases at once (uniform per-case logic).
5. **T5 — cascade test**: End-to-end smoke from §"End-to-end cascade smoke". Drop the TODO comment in `users.py`.

`main.py` is not touched at any point.

## Deferred (named, not done in v1)

- **`Clip.downloaded_at` column** — fixes the file-replacement blind spot. Tiny migration when the time comes.
- **Unifying `cluster_search` / `clustering` / `cluster_validation`** with `stage_state` — those stages have their own `dataset_hash` / `validation_config_hash` / `in_current_grid` machinery on `ClusterRun`. Migrate when those stages are next refactored.
- **Per-row reuse on partial recipe changes** — v2 optimization. v1's coarse wipe-and-recompute is the deliberate simplicity tradeoff.
- **Orphan `stage_state` cleanup script** — only if it becomes noisy in practice.
- **Adoption by row-level stages** — `download`, `music`, `speech`, `captions` keep column-based skipping. They can opt in later by calling the same helper; nothing in the layer prevents it.
