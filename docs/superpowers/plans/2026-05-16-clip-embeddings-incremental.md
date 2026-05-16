# Clip embeddings: incremental stage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `embed_clip_embeddings` incremental: only embed clips that are new or whose upstream data changed; leave deselected clips' rows in place; preserve full-wipe semantics only for recipe (config) drift.

**Architecture:** Add a nullable `source_hash` column to `ClipEmbedding`. Compute per-clip source hashes from the existing `dependency_rows_for_case()` helper. The runner dispatches on `config_hash` drift (wipe + full recompute, current behaviour) vs `data`/`dependency` drift (compute per-clip diff, embed only missing or hash-mismatched clips). The downstream user-aggregation query filters orphan rows so leaving them in place is safe. A one-shot migration script backfills existing rows.

**Tech Stack:** Python 3.13, SQLAlchemy 2.x ORM, pytest, ruff, ty, uv. Project layout per `CLAUDE.md`: `modules/` for production code, `scripts/` for orchestration, `tests/` for pytest.

**Spec:** `docs/superpowers/specs/2026-05-16-clip-embeddings-incremental-design.md` (commit `d29eb8b`).

---

## File map

| File | Action | Responsibility |
| ---- | ------ | -------------- |
| `modules/database/models.py` | Modify | Add `source_hash` column to `ClipEmbedding`. |
| `modules/embeddings/state.py` | Modify | Add `get_embedded_source_hashes`, `per_clip_source_hashes_and_aggregate`; filter orphans in `get_clip_embedding_rows_for_user_aggregation`. |
| `modules/embeddings/runner.py` | Modify | Rename `_clear_case` → `_wipe_case`; replace `_compute_fingerprint` with `_compute_fingerprint_and_per_clip`; add `_diff_targets`; refactor `_run_case` to dispatch config-drift vs data/dep-drift; write `source_hash` on merge. |
| `scripts/migrate_clip_embeddings_source_hash.py` | Create | One-shot schema-and-backfill script (idempotent). |
| `tests/test_clip_embeddings_idempotence.py` | Modify | Add five incremental-path tests; update `test_partial_failure_does_not_seal_stage` to assert narrow retry. |
| `tests/test_migrate_clip_embeddings_source_hash.py` | Create | Unit tests for migration script (schema add + backfill + idempotence). |

`modules/fingerprint.py`, `modules/embeddings/cases.py`, `modules/embeddings/users.py`, and the case-spec dataclass are untouched.

---

## Task 1: Add `source_hash` column to `ClipEmbedding`

**Files:**
- Modify: `modules/database/models.py:175-198`

- [ ] **Step 1: Read the current ClipEmbedding model**

Run: `sed -n '175,200p' modules/database/models.py`
Expected: shows the existing class with `clip_id`, `embedding_case`, `embedding`, `created_at`, `updated_at`.

- [ ] **Step 2: Add `source_hash` column**

In `modules/database/models.py`, inside the `ClipEmbedding` class, immediately after the `embedding: Mapped[bytes] = ...` line:

```python
    source_hash: Mapped[str | None] = mapped_column(String, nullable=True)
```

The class then continues with `created_at: Mapped[DateTime] = ...`.

- [ ] **Step 3: Run the existing suite as a smoke test**

Run: `uv run pytest tests/test_embeddings_public_api.py tests/test_clip_embeddings_idempotence.py -q`
Expected: PASS (the new column is nullable; nothing else reads or writes it yet).

- [ ] **Step 4: Commit**

```bash
git add modules/database/models.py
git commit -m "feat(db): add ClipEmbedding.source_hash column"
```

---

## Task 2: Filter orphans out of `get_clip_embedding_rows_for_user_aggregation`

**Files:**
- Modify: `modules/embeddings/state.py:53-62`
- Test: `tests/test_embeddings_users.py` (or add new file `tests/test_embeddings_state.py` if no good home exists)

- [ ] **Step 1: Locate or create a home for the test**

Run: `grep -n "get_clip_embedding_rows_for_user_aggregation" tests/`
If a test file already exercises this helper, reuse it. Otherwise create `tests/test_embeddings_state.py`.

- [ ] **Step 2: Write the failing test**

In the chosen test file:

```python
from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    User,
    get_engine,
    get_session,
)
from modules.embeddings.state import get_clip_embedding_rows_for_user_aggregation


def test_aggregation_excludes_orphan_rows():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=False, is_downloaded=True))
    session.merge(ClipEmbedding(clip_id=10, embedding_case="video", embedding=b"\x00" * 16))
    session.merge(ClipEmbedding(clip_id=11, embedding_case="video", embedding=b"\x00" * 16))
    session.commit()

    rows = get_clip_embedding_rows_for_user_aggregation(session, "video")
    user_ids_seen = {user_id for _, user_id in rows}
    clip_ids_seen = {
        ce.clip_id
        for ce in session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert user_ids_seen == {1}            # user 1 still contributes
    assert clip_ids_seen == {10, 11}        # row for orphan clip 11 still on disk
    assert len(rows) == 1                   # but only clip 10 reached the aggregation
    session.close()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_embeddings_state.py::test_aggregation_excludes_orphan_rows -v`
Expected: FAIL with `assert len(rows) == 1` (currently returns 2 because both rows are joined regardless of `is_selected`).

- [ ] **Step 4: Update the helper to filter by candidate eligibility**

Replace the current body of `get_clip_embedding_rows_for_user_aggregation` in `modules/embeddings/state.py` with:

```python
def get_clip_embedding_rows_for_user_aggregation(
    session: Session, case: str
) -> list[tuple[bytes, int]]:
    """Return (embedding_blob, user_id) rows for the given case.

    Filters out clips that are no longer in the candidate set so orphan
    rows (e.g. clips later deselected) do not contaminate user means.
    """
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

At the top of `modules/embeddings/state.py`, ensure `clip_used_in_analysis` is in the import list — it already is. Confirm with `grep clip_used_in_analysis modules/embeddings/state.py`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_embeddings_state.py::test_aggregation_excludes_orphan_rows -v`
Expected: PASS.

- [ ] **Step 6: Run the broader embeddings test set as a smoke check**

Run: `uv run pytest tests/test_embeddings_users.py tests/test_user_embeddings_idempotence.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add modules/embeddings/state.py tests/test_embeddings_state.py
git commit -m "fix(embeddings): exclude orphan clips from user aggregation"
```

---

## Task 3: Add `get_embedded_source_hashes` helper

**Files:**
- Modify: `modules/embeddings/state.py`
- Test: `tests/test_embeddings_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embeddings_state.py`:

```python
from modules.embeddings.state import get_embedded_source_hashes


def test_get_embedded_source_hashes_returns_clip_id_to_hash_map():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(
        ClipEmbedding(
            clip_id=10, embedding_case="video", embedding=b"\x00" * 4, source_hash="abc",
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=11, embedding_case="video", embedding=b"\x00" * 4, source_hash=None,
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=10, embedding_case="audio", embedding=b"\x00" * 4, source_hash="zzz",
        )
    )
    session.commit()

    out = get_embedded_source_hashes(session, "video")
    assert out == {10: "abc", 11: None}
    session.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_embeddings_state.py::test_get_embedded_source_hashes_returns_clip_id_to_hash_map -v`
Expected: FAIL with `ImportError: cannot import name 'get_embedded_source_hashes'`.

- [ ] **Step 3: Implement the helper**

Add to `modules/embeddings/state.py`, immediately after `get_embedded_clip_ids`:

```python
def get_embedded_source_hashes(
    session: Session, case: str
) -> dict[int, str | None]:
    """Map clip_id → stored source_hash for every ClipEmbedding row of ``case``.

    Used by the incremental runner to decide which clips need re-embedding.
    A row that exists with source_hash=None is treated as stale: a previous
    pre-incremental run wrote it without the hash, and we cannot prove it
    still matches current upstream.
    """
    rows = (
        session.query(ClipEmbedding.clip_id, ClipEmbedding.source_hash)
        .filter(ClipEmbedding.embedding_case == case)
        .all()
    )
    return {r.clip_id: r.source_hash for r in rows}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_embeddings_state.py::test_get_embedded_source_hashes_returns_clip_id_to_hash_map -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/state.py tests/test_embeddings_state.py
git commit -m "feat(embeddings): add get_embedded_source_hashes helper"
```

---

## Task 4: Add `per_clip_source_hashes_and_aggregate` helper

**Files:**
- Modify: `modules/embeddings/state.py`
- Test: `tests/test_embeddings_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embeddings_state.py`:

```python
from modules import fingerprint as fp
from modules.database import Music
from modules.embeddings.state import (
    dependency_rows_for_case,
    per_clip_source_hashes_and_aggregate,
)


def test_per_clip_source_hashes_match_dependency_rows():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
    session.commit()

    per_clip, aggregate = per_clip_source_hashes_and_aggregate(
        session, "video", [10, 11]
    )

    # Per-clip hash must equal hash_rows of the single dependency row.
    rows = dependency_rows_for_case(session, "video", [10, 11])
    by_id = {r[0]: r for r in rows}
    assert per_clip == {
        10: fp.hash_rows([by_id[10]]),
        11: fp.hash_rows([by_id[11]]),
    }
    # Aggregate must equal hash_rows over the full ordered row list.
    assert aggregate == fp.hash_rows(rows)
    session.close()


def test_per_clip_source_hashes_with_no_candidates():
    Base.metadata.create_all(get_engine())
    session = get_session()
    per_clip, aggregate = per_clip_source_hashes_and_aggregate(session, "video", [])
    assert per_clip == {}
    assert aggregate == fp.hash_rows([])
    session.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_embeddings_state.py::test_per_clip_source_hashes_match_dependency_rows tests/test_embeddings_state.py::test_per_clip_source_hashes_with_no_candidates -v`
Expected: FAIL with `ImportError: cannot import name 'per_clip_source_hashes_and_aggregate'`.

- [ ] **Step 3: Implement the helper**

Add to `modules/embeddings/state.py`. First add the import at the top (if not already present):

```python
from modules import fingerprint as fp
```

Then append at the end of the file:

```python
def per_clip_source_hashes_and_aggregate(
    session: Session, case: str, candidate_ids: list[int]
) -> tuple[dict[int, str], str]:
    """Return ({clip_id: per_clip_hash}, aggregate_hash) for ``case``.

    Both values are derived from the same call to ``dependency_rows_for_case``
    so the per-clip hashes and the stage-level aggregate stay byte-identical.
    """
    rows = dependency_rows_for_case(session, case, candidate_ids)
    per_clip = {r[0]: fp.hash_rows([r]) for r in rows}
    aggregate = fp.hash_rows(rows)
    return per_clip, aggregate
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_embeddings_state.py -v`
Expected: PASS for all tests in the file.

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/state.py tests/test_embeddings_state.py
git commit -m "feat(embeddings): add per_clip_source_hashes_and_aggregate"
```

---

## Task 5: Add `_diff_targets` and rename `_clear_case` → `_wipe_case`

**Files:**
- Modify: `modules/embeddings/runner.py`
- Test: `tests/test_clip_embeddings_idempotence.py`

- [ ] **Step 1: Write a unit test for `_diff_targets`**

Append at the bottom of `tests/test_clip_embeddings_idempotence.py`:

```python
from modules.embeddings.runner import _diff_targets


def test_diff_targets_picks_missing_and_changed():
    per_clip = {10: "h10", 11: "h11", 12: "h12"}
    embedded = {10: "h10", 11: "old", 13: "h13"}  # 12 missing, 11 stale, 13 orphan
    assert _diff_targets(per_clip, embedded) == {11, 12}


def test_diff_targets_treats_none_as_stale():
    per_clip = {10: "h10"}
    embedded = {10: None}
    assert _diff_targets(per_clip, embedded) == {10}


def test_diff_targets_empty_per_clip_returns_empty():
    assert _diff_targets({}, {10: "h10"}) == set()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py::test_diff_targets_picks_missing_and_changed tests/test_clip_embeddings_idempotence.py::test_diff_targets_treats_none_as_stale tests/test_clip_embeddings_idempotence.py::test_diff_targets_empty_per_clip_returns_empty -v`
Expected: FAIL with `ImportError: cannot import name '_diff_targets'`.

- [ ] **Step 3: Add `_diff_targets` and rename `_clear_case` in `modules/embeddings/runner.py`**

Rename the function `_clear_case` to `_wipe_case` (only the definition; the only caller on line 85 currently is `_clear_case(session, spec.name)` — update it too):

```python
def _wipe_case(session, case: str) -> None:
    session.query(ClipEmbedding).filter_by(embedding_case=case).delete()
    session.commit()
```

Add directly below `_wipe_case`:

```python
def _diff_targets(per_clip: dict[int, str], embedded: dict[int, str | None]) -> set[int]:
    """Clip ids that need (re-)embedding: missing rows or stored hash != desired."""
    return {cid for cid, want in per_clip.items() if embedded.get(cid) != want}
```

In `_run_case`, replace the line `_clear_case(session, spec.name)` with `_wipe_case(session, spec.name)`.

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -q -k "diff_targets"`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full clip-embeddings idempotence suite to confirm rename did not break anything**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -q`
Expected: PASS (existing tests + 3 new).

- [ ] **Step 6: Commit**

```bash
git add modules/embeddings/runner.py tests/test_clip_embeddings_idempotence.py
git commit -m "refactor(embeddings): add _diff_targets, rename _clear_case to _wipe_case"
```

---

## Task 6: Replace `_compute_fingerprint` with `_compute_fingerprint_and_per_clip`

**Files:**
- Modify: `modules/embeddings/runner.py:52-61` (existing `_compute_fingerprint`)

- [ ] **Step 1: Update the imports in `modules/embeddings/runner.py`**

The current import block on line 30-34 reads:

```python
from modules.embeddings.state import (
    dependency_rows_for_case,
    get_clip_embedding_candidates,
    get_music_map,
)
```

Replace it with:

```python
from modules.embeddings.state import (
    get_clip_embedding_candidates,
    get_embedded_source_hashes,
    get_music_map,
    per_clip_source_hashes_and_aggregate,
)
```

`dependency_rows_for_case` is no longer used directly by `runner.py` (the new helper wraps it).

- [ ] **Step 2: Replace `_compute_fingerprint`**

Replace the entire `_compute_fingerprint` function in `modules/embeddings/runner.py` with:

```python
def _compute_fingerprint_and_per_clip(
    session, spec: EmbeddingCaseSpec, settings, candidates: list[Clip]
) -> tuple[fp.Fingerprint, dict[int, str]]:
    """Return (Fingerprint, {clip_id: per_clip_source_hash}) for ``case``.

    Both share the same ``dependency_rows_for_case`` source of truth so the
    aggregate ``Fingerprint.dependency`` and the per-row hashes never drift.
    """
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
```

- [ ] **Step 3: Update the call site in `_run_case`**

The current line near the top of `_run_case` reads:

```python
        current = _compute_fingerprint(session, spec, settings, candidates)
```

Replace with:

```python
        current, per_clip = _compute_fingerprint_and_per_clip(
            session, spec, settings, candidates
        )
```

(The `per_clip` map is consumed by the next task — for now leave it bound but unused. Existing tests still pass because the `current` value is byte-identical to what `_compute_fingerprint` returned.)

- [ ] **Step 4: Run the full clip-embeddings idempotence suite**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -q`
Expected: PASS — fingerprints are computed from the same dependency rows as before, just in a different shape.

- [ ] **Step 5: Commit**

```bash
git add modules/embeddings/runner.py
git commit -m "refactor(embeddings): _compute_fingerprint returns per-clip hashes too"
```

---

## Task 7: Refactor `_run_case` for incremental dispatch

**Files:**
- Modify: `modules/embeddings/runner.py` — `_run_case` and its inner work loop
- Test: `tests/test_clip_embeddings_idempotence.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_clip_embeddings_idempotence.py`:

```python
def test_adding_new_candidate_only_embeds_the_new_one(
    db_session, stub_providers, monkeypatch
):
    """Adding a clip to the candidate set must not re-embed existing clips."""
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1)])
    embed_clip_embeddings(settings, cases=["video"])

    from modules.embeddings import runner as runner_mod

    original = runner_mod._embed_with_token_fallback
    call_log: list[int] = []

    def tracked(provider, spec, clip, *args, **kwargs):
        call_log.append(clip.id)
        return original(provider, spec, clip, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "_embed_with_token_fallback", tracked)

    # Add a second clip without touching the first.
    _seed(db_session, settings, clips=[dict(id=11, user_id=1)])
    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()

    assert call_log == [11], "only the new clip should hit the provider"
    rows = {
        r.clip_id: r.source_hash
        for r in db_session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert set(rows) == {10, 11}
    assert all(v is not None for v in rows.values()), (
        "every freshly written row must carry a source_hash"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py::test_adding_new_candidate_only_embeds_the_new_one -v`
Expected: FAIL with `assert call_log == [11]` (today it equals `[10, 11]` because the wipe-and-recompute branch re-embeds clip 10).

- [ ] **Step 3: Refactor `_run_case` to dispatch config-drift vs data/dep-drift**

In `modules/embeddings/runner.py`, replace the current `_run_case` with the following. Keep `embed_clip_embeddings`, `_video_path`, `_compute_fingerprint_and_per_clip`, `_wipe_case`, `_diff_targets`, and `_embed_with_token_fallback` as-is.

The dispatch wipes only on config drift; after the wipe (or when no wipe is needed) the incremental diff against `get_embedded_source_hashes` decides the work set. This unifies "first run", "data/dep drift", and "retry after partial failure" through the same diff and naturally narrows partial-failure retries to only the still-missing/stale clips.

```python
def _run_case(settings, spec: EmbeddingCaseSpec) -> None:
    log_tag = f"embed:{spec.name}"
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
            log(log_tag, "fingerprint match — skipping")
            return

        stored = session.get(StageState, (STAGE, spec.name))
        if stored is not None and stored.config_hash != current.config:
            diff = fp.describe_diff(session, STAGE, spec.name, current)
            log(log_tag, f"config drift ({diff}) — wiping case")
            _wipe_case(session, spec.name)

        embedded = get_embedded_source_hashes(session, spec.name)
        target_ids = _diff_targets(per_clip, embedded)
        log(log_tag, f"{len(target_ids)} clip(s) to (re-)embed")

        _embed_targets(
            session, spec, settings, log_tag, candidates, target_ids, per_clip, current
        )
    finally:
        session.close()


def _embed_targets(
    session,
    spec: EmbeddingCaseSpec,
    settings,
    log_tag: str,
    candidates: list[Clip],
    target_ids: set[int],
    per_clip: dict[int, str],
    current: fp.Fingerprint,
) -> None:
    """Embed the subset of ``candidates`` whose ids are in ``target_ids``.

    Writes ``source_hash`` on every merged row so future runs can diff. On
    full success seals the stage; on any failure leaves stage unsealed so the
    next run retries only the still-missing/stale clips.
    """
    targets = [c for c in candidates if c.id in target_ids]

    # Materialize the work list (skip clips missing video files or text).
    music_map: dict = {}
    if spec.text_builder is not None:
        music_map = get_music_map(session)

    video_dir = settings.paths.video_dir
    jobs: list[tuple[Clip, str | None]] = []
    for clip in targets:
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

    log(log_tag, f"{len(jobs)} clips to embed")

    provider = spec.provider_factory(settings)
    failures = 0
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
                failures += 1
                advance(detail=f"✗ {clip.id}")
                continue

            row = ClipEmbedding(
                clip_id=clip.id,
                embedding_case=spec.name,
                embedding=blob,
                source_hash=per_clip[clip.id],
            )
            session.merge(row)
            session.commit()
            advance(detail=f"✓ {clip.id}")

    if failures:
        log(
            log_tag,
            f"{failures}/{len(jobs)} failed — leaving stage stale for retry",
            level="warn",
        )
    else:
        fp.mark_complete(session, STAGE, spec.name, current)
        session.commit()
    log(log_tag, "done", level="ok")
```

Make sure `StageState` is imported at the top of `runner.py`:

```python
from modules.database import Base, Clip, ClipEmbedding, StageState, get_engine, get_session
```

(The current import line on `runner.py:18` is `from modules.database import Base, Clip, ClipEmbedding, get_engine, get_session` — add `StageState`.)

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py::test_adding_new_candidate_only_embeds_the_new_one -v`
Expected: PASS.

- [ ] **Step 5: Run the full clip-embeddings idempotence suite**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -q`
Expected: PASS. Note: `test_new_candidate_triggers_recompute` (existing) still passes — it only asserts end state.

- [ ] **Step 6: Commit**

```bash
git add modules/embeddings/runner.py tests/test_clip_embeddings_idempotence.py
git commit -m "feat(embeddings): incremental dispatch for clip embeddings"
```

---

## Task 8: Add the remaining incremental-behaviour tests

**Files:**
- Modify: `tests/test_clip_embeddings_idempotence.py`

- [ ] **Step 1: Add `test_caption_change_reembeds_only_changed_clip_for_sandwich`**

Append to `tests/test_clip_embeddings_idempotence.py`:

```python
def test_caption_change_reembeds_only_changed_clip_for_sandwich(
    db_session, stub_providers, monkeypatch
):
    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[
            dict(id=10, user_id=1, caption_clean="alpha"),
            dict(id=11, user_id=1, caption_clean="beta"),
        ],
    )
    embed_clip_embeddings(settings, cases=["sandwich"])

    from modules.embeddings import runner as runner_mod

    original = runner_mod._embed_with_token_fallback
    call_log: list[int] = []

    def tracked(provider, spec, clip, *args, **kwargs):
        call_log.append(clip.id)
        return original(provider, spec, clip, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "_embed_with_token_fallback", tracked)

    # Mutate clip 10's caption_clean only; clip 11's upstream unchanged.
    db_session.query(Clip).filter_by(id=10).update({"caption_clean": "ALPHA-EDIT"})
    db_session.commit()

    embed_clip_embeddings(settings, cases=["sandwich"])
    db_session.expire_all()

    assert call_log == [10], "only the clip whose upstream changed should be re-embedded"
    rows = {
        r.clip_id: r.source_hash
        for r in db_session.query(ClipEmbedding).filter_by(embedding_case="sandwich")
    }
    assert set(rows) == {10, 11}
    assert all(v is not None for v in rows.values())
```

- [ ] **Step 2: Add `test_deselecting_a_clip_keeps_its_row_but_drops_it_from_aggregation`**

```python
def test_deselecting_a_clip_keeps_its_row_but_drops_it_from_aggregation(
    db_session, stub_providers
):
    from modules.embeddings.state import get_clip_embedding_rows_for_user_aggregation

    settings = _settings(stub_providers)
    _seed(
        db_session,
        settings,
        clips=[dict(id=10, user_id=1), dict(id=11, user_id=1)],
    )
    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()

    # Deselect clip 11.
    db_session.query(Clip).filter_by(id=11).update({"is_selected": False})
    db_session.commit()

    embed_clip_embeddings(settings, cases=["video"])
    db_session.expire_all()

    row_ids = {
        r.clip_id
        for r in db_session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert row_ids == {10, 11}, "orphan rows must persist"

    agg = get_clip_embedding_rows_for_user_aggregation(db_session, "video")
    # The orphan must not contribute to aggregation. User 1 still has clip 10,
    # so exactly one row comes back.
    assert len(agg) == 1
```

- [ ] **Step 3: Add `test_config_change_still_wipes`**

```python
def test_config_change_still_wipes(db_session, stub_providers, monkeypatch):
    """A config-hash drift (e.g. AUDIO_INSTRUCTION edit) must wipe + recompute."""
    settings = _settings(stub_providers)
    _seed(db_session, settings, clips=[dict(id=10, user_id=1), dict(id=11, user_id=1)])
    embed_clip_embeddings(settings)
    db_session.expire_all()

    from modules.embeddings import runner as runner_mod

    original = runner_mod._embed_with_token_fallback
    call_log: list[tuple[str, int]] = []

    def tracked(provider, spec, clip, *args, **kwargs):
        call_log.append((spec.name, clip.id))
        return original(provider, spec, clip, *args, **kwargs)

    monkeypatch.setattr(runner_mod, "_embed_with_token_fallback", tracked)

    # Mutate AUDIO_INSTRUCTION → only audio's config_hash changes.
    monkeypatch.setattr(cases_mod, "AUDIO_INSTRUCTION", "NEW INSTRUCTION TEXT")

    embed_clip_embeddings(settings)
    db_session.expire_all()

    audio_calls = sorted(cid for case, cid in call_log if case == "audio")
    non_audio_calls = [pair for pair in call_log if pair[0] != "audio"]
    assert audio_calls == [10, 11], "audio: full wipe-and-recompute"
    assert non_audio_calls == [], "video and sandwich: fingerprint match → skip"
```

- [ ] **Step 4: Run the three new tests to verify they pass**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -v -k "caption_change or deselecting or config_change_still_wipes"`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_clip_embeddings_idempotence.py
git commit -m "test(embeddings): cover incremental data/dep/config drift paths"
```

---

## Task 9: Strengthen the partial-failure test to assert narrow retry

**Files:**
- Modify: `tests/test_clip_embeddings_idempotence.py` — existing `test_partial_failure_does_not_seal_stage`

- [ ] **Step 1: Locate the existing test**

Run: `grep -n "test_partial_failure_does_not_seal_stage" tests/test_clip_embeddings_idempotence.py`
Expected: a single match around line 316.

- [ ] **Step 2: Append a narrow-retry assertion**

Just before the final assertion `assert db_session.get(StageState, ("clip_embeddings", "video")) is not None`, insert a check that the retry hit only clip 11 (not clip 10):

```python
    # The retry must touch clip 11 only — clip 10's row + source_hash are
    # already sealed by the first run's per-row commit.
    retry_calls = call_log[2:]  # first two entries are clip 10's success + clip 11's first failure
    assert retry_calls == [11], (
        f"expected the retry to call clip 11 only, got {retry_calls!r}"
    )
```

Note: the current `call_log` records calls inside `flaky`, which wraps `_embed_with_token_fallback`. With the existing test's seed order (10, 11), the first run produces `call_log == [10, 11]` (10 succeeds, 11 fails). The second run, under incremental dispatch, must only call 11 because clip 10's row + source_hash are sealed. So `call_log[2:]` should equal `[11]`.

If the seed order is reversed (11 then 10) by SQLAlchemy ordering on a future change, this slice math will fail loudly — that's intentional. The implementer should confirm the order with a `print(call_log)` if it breaks.

- [ ] **Step 3: Run the updated test**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py::test_partial_failure_does_not_seal_stage -v`
Expected: PASS.

- [ ] **Step 4: Run the full idempotence suite**

Run: `uv run pytest tests/test_clip_embeddings_idempotence.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_clip_embeddings_idempotence.py
git commit -m "test(embeddings): assert partial-failure retry narrows to failed clips"
```

---

## Task 10: Migration script `migrate_clip_embeddings_source_hash.py`

**Files:**
- Create: `scripts/migrate_clip_embeddings_source_hash.py`
- Create: `tests/test_migrate_clip_embeddings_source_hash.py`

- [ ] **Step 1: Write the failing migration test**

Create `tests/test_migrate_clip_embeddings_source_hash.py`:

```python
"""Tests for scripts/migrate_clip_embeddings_source_hash.py.

The migration must:
  1. Add the source_hash column to clip_embeddings if missing.
  2. Backfill NULL source_hash values with the per-clip dependency hash
     computed from current upstream state.
  3. Be idempotent: re-running on a fully-backfilled DB must change nothing.
"""

from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from modules.database import Base
from scripts.migrate_clip_embeddings_source_hash import migrate_database


def _legacy_clip_embeddings_ddl() -> str:
    """Pre-source_hash schema for clip_embeddings on SQLite."""
    return """
    CREATE TABLE clip_embeddings (
        clip_id BIGINT NOT NULL,
        embedding_case TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
        PRIMARY KEY (clip_id, embedding_case),
        FOREIGN KEY (clip_id) REFERENCES clips(id),
        CONSTRAINT uq_clip_embeddings_clip_case UNIQUE (clip_id, embedding_case)
    )
    """


def _seed_minimal_upstream(conn) -> None:
    # The migration computes source_hash via dependency_rows_for_case, which
    # queries clips and music. Seed the minimum the helper needs.
    conn.execute(
        text(
            "INSERT INTO users (id, is_selected, is_eligible) VALUES (1, 1, 1)"
        )
    )
    conn.execute(
        text(
            "INSERT INTO clips (id, user_id, is_selected, is_downloaded) "
            "VALUES (10, 1, 1, 1), (11, 1, 1, 1)"
        )
    )


def test_migration_adds_column_and_backfills(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    # Build the full schema EXCEPT clip_embeddings, then create the legacy
    # clip_embeddings table by hand so the migration has something to ALTER.
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE clip_embeddings"))
        conn.execute(text(_legacy_clip_embeddings_ddl()))
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding) "
                "VALUES (10, 'video', X'00'), (11, 'video', X'00')"
            )
        )

    # Pre-condition: column does not exist.
    cols_before = {c["name"] for c in inspect(eng).get_columns("clip_embeddings")}
    assert "source_hash" not in cols_before

    migrate_database(eng)

    # Column exists and rows are backfilled.
    cols_after = {c["name"] for c in inspect(eng).get_columns("clip_embeddings")}
    assert "source_hash" in cols_after

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT clip_id, source_hash FROM clip_embeddings "
                "WHERE embedding_case = 'video' ORDER BY clip_id"
            )
        ).all()
    assert len(rows) == 2
    assert all(r.source_hash is not None for r in rows), (
        "every row must carry a non-NULL source_hash after backfill"
    )

    # Sanity: per-clip hashes match what the runner would compute.
    from modules.database import get_session_factory  # noqa: PLC0415
    from modules.embeddings.state import per_clip_source_hashes_and_aggregate  # noqa: PLC0415

    Session = get_session_factory(eng)
    with Session() as session:
        per_clip, _ = per_clip_source_hashes_and_aggregate(session, "video", [10, 11])
    by_id = dict(rows)
    assert by_id[10] == per_clip[10]
    assert by_id[11] == per_clip[11]


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    Base.metadata.create_all(eng)  # new-schema clip_embeddings (source_hash already present)
    with eng.begin() as conn:
        _seed_minimal_upstream(conn)
        conn.execute(
            text(
                "INSERT INTO clip_embeddings (clip_id, embedding_case, embedding, source_hash) "
                "VALUES (10, 'video', X'00', 'preexisting'), (11, 'video', X'00', NULL)"
            )
        )

    migrate_database(eng)

    with eng.connect() as conn:
        rows = {
            r.clip_id: r.source_hash
            for r in conn.execute(
                text("SELECT clip_id, source_hash FROM clip_embeddings")
            ).all()
        }
    # Existing non-NULL hash must not be overwritten; NULL is backfilled.
    assert rows[10] == "preexisting"
    assert rows[11] is not None

    # Second run: nothing changes.
    migrate_database(eng)
    with eng.connect() as conn:
        rows_after = {
            r.clip_id: r.source_hash
            for r in conn.execute(
                text("SELECT clip_id, source_hash FROM clip_embeddings")
            ).all()
        }
    assert rows_after == rows
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_migrate_clip_embeddings_source_hash.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.migrate_clip_embeddings_source_hash'`.

- [ ] **Step 3: Check whether `get_session_factory` is exported, and add it if missing**

Run: `grep -n "get_session_factory" modules/database/*.py`
If not present, the existing helper in `modules/database/__init__.py` uses something equivalent (e.g. `get_session`). If the project has no factory helper, replace the `with Session() as session:` block in the test with `from modules.database import get_session; session = get_session()` and bind the engine via `DATABASE_URL`. The simplest portable approach for the test:

Replace the `Session = get_session_factory(eng)` block with:

```python
from sqlalchemy.orm import Session as _Session  # noqa: PLC0415
with _Session(eng) as session:
    per_clip, _ = per_clip_source_hashes_and_aggregate(session, "video", [10, 11])
```

Use this form if `get_session_factory` does not exist.

- [ ] **Step 4: Implement the migration script**

Create `scripts/migrate_clip_embeddings_source_hash.py`:

```python
"""Production migration: backfill ClipEmbedding.source_hash.

The clip-embedding stage is now incremental: it re-embeds only clips whose
per-row source hash differs from what is stored on the row. After this
schema change every existing row needs its ``source_hash`` populated from
current upstream state. Without backfill the stage still works — every
NULL counts as "stale" and gets re-embedded on first run — but that costs
hours on large datasets.

This script:
  1. Adds the ``source_hash`` column to ``clip_embeddings`` if missing
     (SQLite + PostgreSQL both accept ``ALTER TABLE ADD COLUMN``).
  2. For each ``embedding_case`` present in the table, computes per-clip
     dependency hashes via ``per_clip_source_hashes_and_aggregate`` and
     writes them into rows whose ``source_hash`` is NULL.

Idempotent: re-running on a fully-backfilled DB is a no-op.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
        uv run python scripts/migrate_clip_embeddings_source_hash.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.database import ClipEmbedding  # noqa: E402
from modules.embeddings.state import (  # noqa: E402
    per_clip_source_hashes_and_aggregate,
)

TABLE = "clip_embeddings"
NEW_COLUMN = "source_hash"


def _ensure_column(engine: Engine) -> None:
    inspector = inspect(engine)
    if TABLE not in inspector.get_table_names():
        print(f"Table {TABLE!r} does not exist — nothing to migrate.")
        return
    existing = {col["name"] for col in inspector.get_columns(TABLE)}
    if NEW_COLUMN in existing:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {NEW_COLUMN} TEXT"))
    print(f"  OK: added {TABLE}.{NEW_COLUMN}")


def _backfill(engine: Engine) -> None:
    inspector = inspect(engine)
    if TABLE not in inspector.get_table_names():
        return
    with Session(engine) as session:
        cases = [
            r.embedding_case
            for r in session.query(ClipEmbedding.embedding_case)
            .distinct()
            .all()
        ]
        for case in cases:
            null_ids = [
                r.clip_id
                for r in session.query(ClipEmbedding.clip_id)
                .filter(
                    ClipEmbedding.embedding_case == case,
                    ClipEmbedding.source_hash.is_(None),
                )
                .all()
            ]
            if not null_ids:
                print(f"  case={case!r}: no NULL rows.")
                continue

            per_clip, _ = per_clip_source_hashes_and_aggregate(
                session, case, sorted(null_ids)
            )
            updated = 0
            for clip_id, h in per_clip.items():
                session.query(ClipEmbedding).filter_by(
                    clip_id=clip_id, embedding_case=case
                ).update({ClipEmbedding.source_hash: h})
                updated += 1
            session.commit()
            print(f"  case={case!r}: backfilled {updated} rows.")


def migrate_database(engine: Engine) -> None:
    _ensure_column(engine)
    _backfill(engine)
    print("Migration complete.")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL environment variable.", file=sys.stderr)
        raise SystemExit(1)
    migrate_database(create_engine(url))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the migration tests**

Run: `uv run pytest tests/test_migrate_clip_embeddings_source_hash.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/migrate_clip_embeddings_source_hash.py tests/test_migrate_clip_embeddings_source_hash.py
git commit -m "feat(scripts): one-shot migration for ClipEmbedding.source_hash"
```

---

## Task 11: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS — no failures, no errors. Note any flaky or skipped tests in the commit message if relevant.

- [ ] **Step 2: Lint**

Run: `uv run ruff check`
Expected: no issues. If issues are reported, fix them inline (typically import ordering or unused imports — the refactor in Task 6 removes `dependency_rows_for_case` from `runner.py`'s imports).

- [ ] **Step 3: Format**

Run: `uv run ruff format`
Expected: zero or minimal reformatting. Re-run `uv run ruff check` to confirm clean.

- [ ] **Step 4: Type-check**

Run: `uv run ty check`
Expected: no new errors. The new dict typing (`dict[int, str | None]`) is supported by Python 3.13.

- [ ] **Step 5: Commit any formatting / lint adjustments**

If steps 2–4 produced changes:

```bash
git add -A
git commit -m "chore: ruff format + lint adjustments"
```

Otherwise skip this step.

- [ ] **Step 6: Summarise the diff**

Run: `git log --oneline c460857..HEAD`
Expected: roughly nine commits — one per task, plus optional formatting. The diff should match the file map at the top of this plan.

---

## Self-review notes

- Every spec section maps to at least one task: schema (Task 1), helpers (Tasks 3–4), runner refactor (Tasks 5–7), behavioural tests (Tasks 7–9), aggregation filter (Task 2), migration (Task 10), verification (Task 11).
- No placeholders, every code step shows the actual code.
- Function names match across tasks: `_wipe_case`, `_diff_targets`, `_compute_fingerprint_and_per_clip`, `_embed_targets`, `get_embedded_source_hashes`, `per_clip_source_hashes_and_aggregate`, `migrate_database`.
- Test names match the spec's "Testing" section verbatim.
- `dependency_rows_for_case` is removed from `runner.py`'s imports in Task 6; no other module-level import drift.
