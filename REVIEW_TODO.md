# Cleanup Batch 4 — Audit Final Cleanup

Branch: `refactor/cleanup-batch-4`
Shape: 5 sequenced commits, one PR.
Spec: `docs/superpowers/specs/2026-05-18-audit-final-cleanup-design.md`.
Plan: `docs/superpowers/plans/2026-05-18-audit-final-cleanup.md` (pending).

## Items

- [x] **1. Delete one-shot migrate scripts and tests**
  - Delete 11 `scripts/migrate_*.py` + matching `tests/test_migrate_*.py`.
  - Verify no live imports first.
  - Commit: `chore(scripts): drop one-shot migrate_* scripts and tests`

- [ ] **2. Extract Clip filter scratch state (audit 3.2 / 3.8)**
  - New `clip_filter_scratch` table (1:1 on `clip_id`).
  - Move `log_plays`, `creator_relative_robust_z`, `is_creator_low_outlier` off `Clip`.
  - Update filter stage writers/readers; no migration (throwaway DB).
  - Commit: `refactor(filter): extract scratch state into ClipFilterScratch table`

- [ ] **3. Identity-DB boundary — identity-first + orphan sweep (audit 3.11)**
  - Document identity-first invariant on `core/database/identity.py`.
  - Add `sweep_orphans()` helper.
  - Add `scripts/sweep_identity_orphans.py` CLI shell.
  - Commit: `refactor(database): identity-first contract + orphan sweep`

- [ ] **4. Centralize EN-detection helper (audit 2.9)**
  - Create `core/lang.py` with `is_english` + `sql_is_english`.
  - Replace inline checks in `modules/embeddings/text.py`, captions/translate, speech/translate, `core/database/predicates.py`.
  - Commit: `refactor: centralize EN-detection helper in core.lang`

- [ ] **5. Move retry bodies into stage modules (audit 3.9)**
  - New `modules/{music,speech,captions}/retry.py` with function bodies.
  - `scripts/retry_failed_*.py` become CLI shells.
  - Commit: `refactor(scripts): move retry bodies into modules/<stage>/retry.py`

## Field test (gate before merge)

- Delete `data/*.db`.
- Run `uv run python main.py` against a small (3-5 user) CSV.
- Spot-check row counts at each stage.

## Out of scope

- 3.17 visualization `eng` param (per audit review).
- 1.1 `_phase_*` / `_select_best` — already removed; stale docstring comment in `tests/test_cluster_validation.py:190` left as-is.
- 2.8 music upload-fallback `min(Clip.id)` — already fixed in cleanup-batch-2.
- 2.11, 2.13, 2.15, 2.16, 1.16 — already done in prior batches.
