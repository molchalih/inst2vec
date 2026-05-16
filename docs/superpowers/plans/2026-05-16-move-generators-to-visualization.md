# Move generators/ to modules/visualization/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the seven top-level `generators/` modules into a new `modules/visualization/` package, splitting markdown table generators (`tables/`) from plot generators (`plots/`), while preserving the public `plot_clusters` entry point.

**Architecture:** Convert `modules/visualization.py` into a package `modules/visualization/` whose `__init__.py` carries the old module's contents. Add two subpackages: `tables/` (six markdown table generators) and `plots/` (the matplotlib figure generator). Use `git mv` to preserve history. Rewrite imports in `modules/visualization/__init__.py`, `docs/quarto_helpers.py`, and seven test files.

**Tech Stack:** Python 3, pytest, ruff, ty, git.

**Spec:** `docs/superpowers/specs/2026-05-16-move-generators-to-visualization-design.md`

---

## File Structure

**Created:**
- `modules/visualization/__init__.py` — package entry; carries `plot_clusters` (moved from `modules/visualization.py`).
- `modules/visualization/tables/__init__.py` — empty.
- `modules/visualization/plots/__init__.py` — empty.

**Moved (via `git mv`, preserves history):**
- `generators/cluster_results_all.py`     → `modules/visualization/tables/cluster_results_all.py`
- `generators/cluster_results_best.py`    → `modules/visualization/tables/cluster_results_best.py`
- `generators/dataset_counts.py`          → `modules/visualization/tables/dataset_counts.py`
- `generators/dataset_summary_clips.py`   → `modules/visualization/tables/dataset_summary_clips.py`
- `generators/dataset_summary_users.py`   → `modules/visualization/tables/dataset_summary_users.py`
- `generators/spotify_feature_metrics.py` → `modules/visualization/tables/spotify_feature_metrics.py`
- `generators/cluster_case_plots.py`      → `modules/visualization/plots/cluster_case_plots.py`

**Deleted:**
- `modules/visualization.py` (replaced by package `__init__.py`).
- `generators/` directory (after the moves).
- `generators/__pycache__/`, `modules/__pycache__/visualization.cpython-*.pyc` (untracked; cleaned with `find` or ignored).

**Modified (import rewrites only — no behavior change):**
- `docs/quarto_helpers.py` — 8 lazy imports.
- `tests/test_cluster_case_plots.py`
- `tests/test_visualization.py`
- `tests/test_cluster_results_all.py`
- `tests/test_cluster_results_best.py`
- `tests/test_dataset_summary_clips.py`
- `tests/test_dataset_summary_users.py`
- `tests/test_spotify_feature_metrics.py`

---

## Task 1: Capture baseline test status

Run the full test suite first so any later failure can be cleanly attributed to this refactor.

**Files:** None (read-only check).

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -q
```

Expected: All tests pass (497 today). Record the exact pass count for comparison after the refactor.

- [ ] **Step 2: Run ruff and ty**

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
```

Expected: All clean. If anything is already failing, stop and report — this plan assumes a green baseline.

---

## Task 2: Create the `modules/visualization/` package skeleton

Create the new package directory with its three `__init__.py` files. The top-level `__init__.py` carries the old `modules/visualization.py` contents, with the one internal `generators.cluster_case_plots` import rewritten to its new path. Subpackage `__init__.py` files are empty.

We create `__init__.py` first (before moving files in Task 3) so the new import path resolves immediately when later commits land.

**Files:**
- Create: `modules/visualization/__init__.py`
- Create: `modules/visualization/tables/__init__.py`
- Create: `modules/visualization/plots/__init__.py`

- [ ] **Step 1: Create `modules/visualization/__init__.py`**

Write file with the following exact contents (this is the old `modules/visualization.py` body with the import rewritten):

```python
"""Save UMAP cluster scatter plots to data/plots/ as PNG files."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.console import log
from modules.database import UserCluster, get_session
from modules.visualization.plots.cluster_case_plots import cluster_plot_figure_for_case

PLOTS_DIR = "data/plots"


def plot_clusters() -> None:
    """Load user_clusters from DB and save one PNG per embedding case."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    session = get_session()
    try:
        eng = session.get_bind()
        cases = sorted(
            r[0] for r in session.query(UserCluster.embedding_case).distinct().all()
        )
        for case in cases:
            fig = cluster_plot_figure_for_case(eng, case)
            path = os.path.join(PLOTS_DIR, f"clusters_{case}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log("viz", f"saved {path}", level="ok")
    finally:
        session.close()
```

Note: imports are alphabetised within their group per the existing ruff config (`modules.console` → `modules.database` → `modules.visualization.plots.cluster_case_plots`). `ruff format` will normalise if needed.

- [ ] **Step 2: Create `modules/visualization/tables/__init__.py`**

Write an empty file (zero bytes).

- [ ] **Step 3: Create `modules/visualization/plots/__init__.py`**

Write an empty file (zero bytes).

- [ ] **Step 4: Verify package is importable (target import will fail until Task 3, that's expected)**

Just confirm files exist on disk:

```bash
ls -la modules/visualization/ modules/visualization/tables/ modules/visualization/plots/
```

Expected: All three directories exist; each contains `__init__.py`. Do NOT run pytest yet — `modules/visualization/__init__.py` imports from `modules.visualization.plots.cluster_case_plots`, which doesn't exist until Task 3.

Do not commit yet — this commit will be paired with Task 3 to keep the tree green between commits.

---

## Task 3: Move generator files and delete old `modules/visualization.py`

Move all seven generator files into their new homes with `git mv` (preserves rename history). Delete the old `modules/visualization.py` (its contents are now in the package `__init__.py`). Tests will fail after this step because of stale import paths — that's expected; Tasks 4–5 fix them. Don't commit until after Task 5.

**Files:**
- Move: `generators/cluster_results_all.py`     → `modules/visualization/tables/cluster_results_all.py`
- Move: `generators/cluster_results_best.py`    → `modules/visualization/tables/cluster_results_best.py`
- Move: `generators/dataset_counts.py`          → `modules/visualization/tables/dataset_counts.py`
- Move: `generators/dataset_summary_clips.py`   → `modules/visualization/tables/dataset_summary_clips.py`
- Move: `generators/dataset_summary_users.py`   → `modules/visualization/tables/dataset_summary_users.py`
- Move: `generators/spotify_feature_metrics.py` → `modules/visualization/tables/spotify_feature_metrics.py`
- Move: `generators/cluster_case_plots.py`      → `modules/visualization/plots/cluster_case_plots.py`
- Delete: `modules/visualization.py`

- [ ] **Step 1: `git mv` the six table generators**

```bash
git mv generators/cluster_results_all.py     modules/visualization/tables/cluster_results_all.py
git mv generators/cluster_results_best.py    modules/visualization/tables/cluster_results_best.py
git mv generators/dataset_counts.py          modules/visualization/tables/dataset_counts.py
git mv generators/dataset_summary_clips.py   modules/visualization/tables/dataset_summary_clips.py
git mv generators/dataset_summary_users.py   modules/visualization/tables/dataset_summary_users.py
git mv generators/spotify_feature_metrics.py modules/visualization/tables/spotify_feature_metrics.py
```

Expected: No output, exit 0 for each.

- [ ] **Step 2: `git mv` the plot generator**

```bash
git mv generators/cluster_case_plots.py modules/visualization/plots/cluster_case_plots.py
```

Expected: No output, exit 0.

- [ ] **Step 3: Delete the old `modules/visualization.py`**

```bash
git rm modules/visualization.py
```

Expected: `rm 'modules/visualization.py'`.

- [ ] **Step 4: Remove leftover `generators/` directory if it lingers**

After all seven `git mv` commands, `generators/` should be empty of tracked files but may still hold `__pycache__/`. Remove it:

```bash
rm -rf generators
```

Expected: directory is gone. (`__pycache__` is gitignored, so no `git rm` is needed.)

- [ ] **Step 5: Confirm git sees seven renames + one delete**

```bash
git status
```

Expected output should list:
- `renamed: generators/cluster_case_plots.py    -> modules/visualization/plots/cluster_case_plots.py`
- `renamed: generators/cluster_results_all.py   -> modules/visualization/tables/cluster_results_all.py`
- `renamed: generators/cluster_results_best.py  -> modules/visualization/tables/cluster_results_best.py`
- `renamed: generators/dataset_counts.py        -> modules/visualization/tables/dataset_counts.py`
- `renamed: generators/dataset_summary_clips.py -> modules/visualization/tables/dataset_summary_clips.py`
- `renamed: generators/dataset_summary_users.py -> modules/visualization/tables/dataset_summary_users.py`
- `renamed: generators/spotify_feature_metrics.py -> modules/visualization/tables/spotify_feature_metrics.py`
- `deleted: modules/visualization.py`
- `new file: modules/visualization/__init__.py`
- `new file: modules/visualization/tables/__init__.py`
- `new file: modules/visualization/plots/__init__.py`

If git shows any of the moves as "deleted + new file" instead of "renamed", that means it didn't detect the rename — verify the file contents are identical to before the move. If they are, this is just git's rename-detection threshold; it will still apply cleanly. No action needed.

Do not commit yet — Tasks 4–5 still need to fix consumer imports.

---

## Task 4: Rewrite imports in `docs/quarto_helpers.py`

`docs/quarto_helpers.py` lazy-imports from `generators` in eight places. Rewrite each to the new module path. All eight imports are inside function bodies (lazy), so the file structure is unchanged — only the `from generators.X import Y` lines get updated.

**Files:**
- Modify: `docs/quarto_helpers.py:53`
- Modify: `docs/quarto_helpers.py:70`
- Modify: `docs/quarto_helpers.py:81`
- Modify: `docs/quarto_helpers.py:96`
- Modify: `docs/quarto_helpers.py:107`
- Modify: `docs/quarto_helpers.py:121`
- Modify: `docs/quarto_helpers.py:132`
- Modify: `docs/quarto_helpers.py:143`
- Modify: `docs/quarto_helpers.py:156`

- [ ] **Step 1: Replace line 53 import**

```python
    from modules.visualization.tables.cluster_results_all import summarize_all_to_markdown
```

Replaces:

```python
    from generators.cluster_results_all import summarize_all_to_markdown
```

- [ ] **Step 2: Replace line 70 import**

```python
    from modules.visualization.tables.cluster_results_all import summarize_to_markdown
```

Replaces:

```python
    from generators.cluster_results_all import summarize_to_markdown
```

- [ ] **Step 3: Replace line 81 import**

```python
    from modules.visualization.tables.cluster_results_best import best_runs_all_to_markdown
```

Replaces:

```python
    from generators.cluster_results_best import best_runs_all_to_markdown
```

- [ ] **Step 4: Replace line 96 import**

```python
    from modules.visualization.tables.dataset_summary_users import users_summary_to_markdown
```

Replaces:

```python
    from generators.dataset_summary_users import users_summary_to_markdown
```

- [ ] **Step 5: Replace line 107 import**

```python
    from modules.visualization.tables.dataset_summary_users import get_users_summary_cells
```

Replaces:

```python
    from generators.dataset_summary_users import get_users_summary_cells
```

- [ ] **Step 6: Replace line 121 import**

```python
    from modules.visualization.tables.dataset_summary_clips import clips_summary_to_markdown
```

Replaces:

```python
    from generators.dataset_summary_clips import clips_summary_to_markdown
```

- [ ] **Step 7: Replace line 132 import**

```python
    from modules.visualization.tables.dataset_summary_clips import get_clips_summary_cells
```

Replaces:

```python
    from generators.dataset_summary_clips import get_clips_summary_cells
```

- [ ] **Step 8: Replace line 143 import**

```python
    from modules.visualization.tables.spotify_feature_metrics import spotify_feature_metrics_to_markdown
```

Replaces:

```python
    from generators.spotify_feature_metrics import spotify_feature_metrics_to_markdown
```

- [ ] **Step 9: Replace line 156 import**

```python
    from modules.visualization.plots.cluster_case_plots import cluster_plot_figure_for_case
```

Replaces:

```python
    from generators.cluster_case_plots import cluster_plot_figure_for_case
```

- [ ] **Step 10: Sanity check — no stale `generators.` references remain in `docs/quarto_helpers.py`**

```bash
grep -n "generators" docs/quarto_helpers.py
```

Expected: no output.

---

## Task 5: Rewrite imports in test files

Rewrite every `from generators.X import Y` in `tests/` to point at `modules.visualization.tables.X` (or `modules.visualization.plots.X` for `cluster_case_plots`). Test imports are split across module-level and function-level locations; the rewrites are textual and identical in either context.

**Files:**
- Modify: `tests/test_cluster_case_plots.py` (3 occurrences)
- Modify: `tests/test_visualization.py` (1 occurrence)
- Modify: `tests/test_cluster_results_all.py` (5 occurrences)
- Modify: `tests/test_cluster_results_best.py` (4 occurrences)
- Modify: `tests/test_dataset_summary_clips.py` (3 occurrences)
- Modify: `tests/test_dataset_summary_users.py` (6 occurrences)
- Modify: `tests/test_spotify_feature_metrics.py` (2 occurrences)

- [ ] **Step 1: Inventory every `from generators` line in `tests/`**

```bash
grep -rn "from generators" tests/
```

Record the count. Expected (current state, before this task): 24 occurrences across 7 files.

- [ ] **Step 2: Rewrite all `from generators.cluster_case_plots` references**

In `tests/test_cluster_case_plots.py` and `tests/test_visualization.py`, replace each occurrence of:

```python
from generators.cluster_case_plots import …
```

with:

```python
from modules.visualization.plots.cluster_case_plots import …
```

(The `…` represents the existing imported names, which stay unchanged.)

You can do this safely with `sed`, then verify:

```bash
sed -i '' 's|from generators\.cluster_case_plots|from modules.visualization.plots.cluster_case_plots|g' \
  tests/test_cluster_case_plots.py tests/test_visualization.py
grep -n "generators" tests/test_cluster_case_plots.py tests/test_visualization.py
```

Expected: `grep` returns no output.

- [ ] **Step 3: Rewrite all `from generators.<table>` references**

For each of the six table modules, rewrite `from generators.<name>` → `from modules.visualization.tables.<name>` in the corresponding test file:

```bash
for name in cluster_results_all cluster_results_best dataset_counts dataset_summary_clips dataset_summary_users spotify_feature_metrics; do
  sed -i '' "s|from generators\\.${name}|from modules.visualization.tables.${name}|g" tests/*.py
done
grep -rn "from generators" tests/
```

Expected: final `grep` returns no output.

- [ ] **Step 4: Confirm test file structures are otherwise unchanged**

```bash
git diff --stat tests/
```

Expected: seven files changed, with a small line count per file (one line replaced per occurrence, so 24 modifications total across the 7 files).

---

## Task 6: Search for any remaining `generators` references in the tree

Defence-in-depth: catch any reference the plan missed.

**Files:** None (read-only check).

- [ ] **Step 1: Grep for `generators` across code, docs, and config**

```bash
grep -rn "generators" --include="*.py" --include="*.toml" --include="*.cfg" --include="*.ini" .
```

Expected: no output (everything we care about lives in `.py` files; the moved files themselves don't self-reference `generators`).

If anything turns up that isn't inside `docs/superpowers/plans/` or `docs/superpowers/specs/` (those are historical and out of scope per the spec), stop and rewrite it the same way as Task 4 or Task 5.

- [ ] **Step 2: Grep historical docs for awareness only (no edits)**

```bash
grep -rn "generators" docs/superpowers/ | head
```

Expected: matches in old plans/specs — confirmed out of scope per the spec. No action.

---

## Task 7: Run the verification suite

This is the gate. Tests must pass and lint/type checks must be clean before we commit. If any check fails, fix it before proceeding.

**Files:** None (read-only check).

- [ ] **Step 1: Run pytest**

```bash
uv run pytest -q
```

Expected: same pass count as Task 1 (497). Zero failures, zero errors. If any test fails, diagnose using `uv run pytest <path>::<name> -v`; the most likely cause is a missed import rewrite.

- [ ] **Step 2: Run ruff**

```bash
uv run ruff check
uv run ruff format
```

Expected: `ruff check` reports `All checks passed!`; `ruff format` may reflow `modules/visualization/__init__.py` (the import block) — that's fine.

- [ ] **Step 3: Re-run ruff format-check**

```bash
uv run ruff format --check
```

Expected: `<N> files already formatted`. If anything is unformatted at this point, run `uv run ruff format` and re-verify.

- [ ] **Step 4: Run type check**

```bash
uv run ty check
```

Expected: no new errors versus Task 1's baseline.

- [ ] **Step 5: Re-run pytest after format**

```bash
uv run pytest -q
```

Expected: still all green (in case `ruff format` touched anything that mattered).

---

## Task 8: Commit

A single commit captures the whole rename. Because Tasks 2–5 leave the tree in an inconsistent state mid-refactor, we explicitly did not commit until verification was green.

**Files:** All staged changes from Tasks 2–5.

- [ ] **Step 1: Stage everything**

```bash
git add modules/visualization/ tests/ docs/quarto_helpers.py
# git rm and git mv from Task 3 have already staged the deletes/renames
```

- [ ] **Step 2: Confirm the staged diff matches expectations**

```bash
git status
git diff --cached --stat
```

Expected:
- 7 renames (one per generator file).
- 1 deletion: `modules/visualization.py`.
- 3 new files: `modules/visualization/__init__.py`, `modules/visualization/tables/__init__.py`, `modules/visualization/plots/__init__.py`.
- ~9 modified lines in `docs/quarto_helpers.py`.
- Small modifications across 7 test files.

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor: move generators/ to modules/visualization/{tables,plots}

Relocates the seven generator modules into a new modules/visualization/
package, splitting markdown table generators (tables/) from the plot
generator (plots/). The old modules/visualization.py becomes the package's
__init__.py so `from modules.visualization import plot_clusters` (used by
main.py) keeps working. All consumers in docs/quarto_helpers.py and tests/
are updated to the new import paths. No behaviour change.
EOF
)"
```

Expected: commit created with one summary line indicating the file changes.

- [ ] **Step 4: Final post-commit sanity**

```bash
uv run pytest -q
git log --oneline -1
```

Expected: pytest green; the new commit appears at HEAD.

---

## Done

Verification:
- `uv run pytest` green (same pass count as baseline).
- `uv run ruff check && uv run ruff format --check` clean.
- `uv run ty check` clean.
- `git log` shows one new commit; `generators/` no longer exists; `modules/visualization/` package exists with `__init__.py`, `tables/`, and `plots/` subpackages.
