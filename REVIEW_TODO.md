# REVIEW TODO

Tracking sheet for the 56 findings in [REVIEW_ANSWERED.md](./REVIEW_ANSWERED.md).
IDs only — full evidence and remediation direction live in REVIEW_ANSWERED.md.

**Totals:** 56 IDs across Phase 1 (23) / Phase 2 (16) / Phase 3 (17).
**Status as of 2026-05-18:** 49 IDs done, 7 deferred.

The prose "47 of 56" total counts unique *fixes*; two ID pairs collapse:
1.2 ≡ 2.6 (audit duplicate), 2.10 was subsumed by 2.4's fix, and
1.22 ≡ 3.10 (same fix in `refactor/cleanup-batch-3`).
49 IDs → 47 unique fixes.

---

## Done — merged in `refactor/cleanup` (2026-05-18)

24 unique fixes (26 IDs). Categories from the merge plan:

- **Correctness (6):** 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 *(2.6 ≡ 1.2)*
- **Deletions (8):** 1.1, 1.3, 1.5, 1.6, 1.18, 1.19, 1.20, 1.21
- **Small dead-value cleanups (8):** 1.14, 1.15, 1.16, 1.17, 2.11, 2.13, 2.15, 2.16
- **Side-effects of Task 6 (DEFAULT_CASES single source):** 1.4, 3.16
- **Subsumed by Tasks 2+3 (speech as one atomic block):** 2.10

---

## Done — merged in `refactor/cleanup-batch-2` (2026-05-18)

8 IDs from batch-2 spec at
[`docs/superpowers/specs/2026-05-18-cleanup-batch-2-design.md`](docs/superpowers/specs/2026-05-18-cleanup-batch-2-design.md).

- **Phase 1 dedup/cleanup (5):** 1.7 *(revisited and fully closed by batch-3 A2)*, 1.9, 1.10, 1.11, 1.12
- **Phase 2 logic fixes (3):** 2.8, 2.9, 2.12
- **Architectural addressed in-line by 1.10:** 3.4

---

## Done — merged in `refactor/cleanup-batch-3` (2026-05-18)

15 IDs (14 unique fixes since 1.22 ≡ 3.10) across 14 commits.
Plan at [`docs/superpowers/plans/2026-05-18-cleanup-batch-3.md`](docs/superpowers/plans/2026-05-18-cleanup-batch-3.md).

### Phase A — hash-reset block (data-affecting)
- 1.13 drop `_legacy_provider_name` shim
- 1.7 (revisit) consolidate qwen factories into single `qwen_provider`
- 3.3 centralize case metadata on `EmbeddingCaseSpec`; rename `gemini_mm` → `gemini`
- 3.6 `dependency_rows_for_case` reduces to a CASE_REGISTRY-driven dispatcher
- 3.7 replace `_NO_MATCH` sentinel with `Music.recognition_status` enum
- 1.23 split `is_not_enough_clips` into `is_not_enough_preprocessed` + `is_not_enough_eligible`

### Phase B — stage convention
- 3.12 introduce `core.pipeline.Stage` enum; drop bare-string stage names
- 3.1 standardize stage entry signature to `run(settings, secrets)`

### Phase C — cosmetic cleanup
- 1.8 delete unused `ParseSettings.fetch_retry_delays_sec`
- 3.5 delete unused `OverridesSettings`
- 3.14 prune `modules.clustering` package re-exports to public API
- 1.22 ≡ 3.10 drop underscore on re-exported filter helpers

### Phase D — structural cleanup
- 3.15 lift audio-extraction settings off `EmbeddingsSettings`
- 3.13 add `paths.video_for/audio_for/thumbnail_for` helpers

### Local-only (not committed; file gitignored)
- 3.9 update `CLAUDE.md` `scripts/` description to reflect dual entry-and-library role.
  The local edit is applied; no commit because `CLAUDE.md` is in `.gitignore`.

---

## Deferred — to be picked up later

7 IDs, grouped by reason.

### Logic / operational — fix only on signal

- 2.7 identity-DB vs main-DB commit ordering — only if ops report ghosts
- 2.14 `audio_extract_stage` fingerprint covers video stats only

### Architectural — defer to dedicated cycle

Material design changes, not safe to bundle with cleanup batches.

- 3.2 `Clip` is a god-table holding per-stage scratch state
- 3.8 filter writes scratch state directly on `Clip`
- 3.11 identity-DB integration has no central transactional bracket

### Do-not-touch (per original scope)

- 3.17 visualization tables read DB via passed `eng` instead of `get_engine()`

---

## Accounting check

| Phase   | Total | Done | Deferred |
|---------|------:|-----:|---------:|
| Phase 1 |    23 |   23 |        0 |
| Phase 2 |    16 |   14 |        2 |
| Phase 3 |    17 |   12 \* |     4 \*\* |
| **All** |**56** |**49**|     **6**|

\* Phase 3 done counts 11 committed (3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.12, 3.13,
3.14, 3.15, 3.16) plus 3.9 applied locally only (CLAUDE.md is gitignored).
3.4 was addressed in-line by 1.10 (batch-2). 3.10 is the same fix as 1.22.

\*\* Counts only architectural-deferred (3.2, 3.8, 3.11) and do-not-touch
(3.17). 3.9 is "done locally, not committed".
