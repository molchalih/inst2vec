# REVIEW TODO

Tracking sheet for the 56 findings in [REVIEW_ANSWERED.md](./REVIEW_ANSWERED.md).
IDs only — full evidence and remediation direction live in REVIEW_ANSWERED.md.

**Totals:** 56 IDs across Phase 1 (23) / Phase 2 (16) / Phase 3 (17).
**Status as of 2026-05-18:** 26 IDs done, 9 in progress, 21 deferred.

The prose "24 of 56" total counts unique *fixes*; two ID pairs collapse:
1.2 ≡ 2.6 (audit duplicate) and 2.10 was subsumed by 2.4's fix.
26 IDs → 24 unique fixes.

---

## Done — merged in `refactor/cleanup` (2026-05-18)

24 unique fixes (26 IDs). Categories from the merge plan:

- **Correctness (6):** 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 *(2.6 ≡ 1.2)*
- **Deletions (8):** 1.1, 1.3, 1.5, 1.6, 1.18, 1.19, 1.20, 1.21
- **Small dead-value cleanups (8):** 1.14, 1.15, 1.16, 1.17, 2.11, 2.13, 2.15, 2.16
- **Side-effects of Task 6 (DEFAULT_CASES single source):** 1.4, 3.16
- **Subsumed by Tasks 2+3 (speech as one atomic block):** 2.10

---

## In progress — `refactor/cleanup-batch-2`

8 findings, spec at
[`docs/superpowers/specs/2026-05-18-cleanup-batch-2-design.md`](docs/superpowers/specs/2026-05-18-cleanup-batch-2-design.md).
Architectural item 3.4 is addressed in-line by 1.10 (same helper, same migration).

- **Phase 1 dedup/cleanup (5):** 1.7, 1.9, 1.10, 1.11, 1.12
- **Phase 2 logic fixes (3):** 2.8, 2.9, 2.12
- **Effectively addressed by 1.10:** 3.4

---

## Deferred — to be picked up later

22 IDs, grouped by reason.

### Rename-later (DB hash / column-name cost)
Action gated on a clean reset window.

- 1.13 `_legacy_provider_name` shim
- 1.22 `modules/filter/__init__.py` re-exports private helpers
- 1.23 dual `_flag_users_without_enough_clips` call (rename has DB cost)
- 3.10 14 underscored filter helpers re-exported

### Delete-later (cosmetic / configured-but-unused)

- 1.8 `ParseSettings.fetch_retry_delays_sec` configured but never read
- 3.5 `OverridesSettings` configured but unused
- 3.14 `modules.clustering.__init__` re-exports 13 symbols

### Logic / operational — fix only on signal

- 2.7 identity-DB vs main-DB commit ordering — only if ops report ghosts
- 2.14 `audio_extract_stage` fingerprint covers video stats only

### Architectural — defer to dedicated cycle

Material design changes, not safe to bundle with cleanup batches.

- 3.1 stage entry-point signatures have no convention
- 3.2 `Clip` is a god-table holding per-stage scratch state
- 3.3 `embedding_case` requires 4+ updates per new case
- 3.6 `dependency_rows_for_case` concentrates cross-stage column knowledge
- 3.7 `_NO_MATCH = "none"` sentinel leaks across module boundary
- 3.8 filter writes scratch state directly on `Clip`
- 3.9 `scripts/` is half script, half library
- 3.11 identity-DB integration has no central transactional bracket
- 3.12 stage names are stringly-typed across `stage_dependency_hash` callers
- 3.13 `paths` is a flat bag with no clip→path helper
- 3.15 ingest stage reads `embeddings.audio_*` config (boundary violation)

### Do-not-touch (per original scope)

- 3.17 visualization tables read DB via passed `eng` instead of `get_engine()`

---

## Accounting check

| Phase   | Total | Done | Batch-2 | Deferred |
|---------|------:|-----:|--------:|---------:|
| Phase 1 |    23 |   14 |       5 |        4 |
| Phase 2 |    16 |   11 |       3 |        2 |
| Phase 3 |    17 |    1 |       1\* |       15 |
| **All** |**56** |**26**|     **9**|     **21**|

\* 3.4 is addressed in-line by 1.10. Counts above use IDs; the prose total
"24 of 56 done" treats 1.2 ≡ 2.6 and 2.4 ≡ 2.10 as single fixes.
