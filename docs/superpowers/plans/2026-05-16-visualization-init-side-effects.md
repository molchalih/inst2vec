# Scope `matplotlib.use("Agg")` Out Of `modules.visualization` Package Root — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `plot_clusters()` from `modules/visualization/__init__.py` into a new `modules/visualization/plots/cluster_plots.py`, centralize the `matplotlib.use("Agg")` backend selection in `modules/visualization/plots/__init__.py`, and parameterize `plot_clusters(plots_dir: str)` so the `main.py` call site works.

**Architecture:** The visualization package root currently runs three classes of import-time work (matplotlib backend, DB imports, plot module imports) that ripple into every `modules.visualization.tables.*` consumer. Relocating `plot_clusters` into the `plots/` subpackage and putting `matplotlib.use("Agg")` in `plots/__init__.py` keeps the side-effect surface confined to code that actually renders plots. Table consumers (Quarto + dataset/cluster-results tests) load a no-op package `__init__.py` only.

**Tech Stack:** Python, matplotlib, SQLAlchemy, pytest, ruff, ty, uv.

**Design spec:** `docs/superpowers/specs/2026-05-16-visualization-init-side-effects-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `modules/visualization/__init__.py` | Modify (shrink to docstring) | Marker for the visualization package. No imports, no side effects. |
| `modules/visualization/plots/__init__.py` | Modify (replace contents) | Set the matplotlib backend once for the entire plot subpackage; re-export `plot_clusters` as the package's public entry point. |
| `modules/visualization/plots/cluster_plots.py` | Create | Pipeline entry point: load `UserCluster` rows, render one figure per embedding case, write PNGs to a caller-provided directory. |
| `modules/visualization/plots/cluster_case_plots.py` | Modify (delete its own `matplotlib.use("Agg")` block) | Build a `Figure` for a single embedding case from `UserCluster` rows. Unchanged in behavior; loses redundant backend call. |
| `main.py` | Modify (import line) | Pipeline orchestrator. Updates the `plot_clusters` import path. |
| `tests/test_visualization.py` | Modify (import line, monkeypatch targets, call site) | Updates patch paths to the relocated module and passes `plots_dir=tmp_path` to the call. |

No other files change. `tests/test_main_runtime.py` patches `main.plot_clusters` — that binding still resolves correctly after the import path updates in `main.py`, so no change is needed there.

---

## Pre-flight

The design spec (`docs/superpowers/specs/2026-05-16-visualization-init-side-effects-design.md`) and this plan both live on `main`. The implementation goes on its own branch.

- [ ] **Step 0a: Confirm clean working tree on `main`**

Run: `git status`
Expected: working tree clean (the spec and plan are already committed). If anything is modified or staged, stop and surface it.

- [ ] **Step 0b: Create the working branch from `main`**

Run:
```bash
git fetch origin
git checkout -b refactor/visualization-init-side-effects main
```
Expected: switched to a new branch starting at `main`. Both the design spec and this plan come along automatically.

- [ ] **Step 0c: Baseline tests pass before any code change**

Run: `uv run pytest tests/test_visualization.py tests/test_cluster_case_plots.py tests/test_main_runtime.py -v`
Expected: all green. If any test fails, stop and surface — this plan assumes a passing baseline.

---

## Task 1: Update the failing test for the relocated `plot_clusters`

The existing test in `tests/test_visualization.py` monkeypatches symbols on `modules.visualization` (`get_session`, `PLOTS_DIR`, `cluster_plot_figure_for_case`) and calls `plot_clusters()` with no arguments. After the refactor those symbols live on `modules.visualization.plots.cluster_plots`, and `plot_clusters` requires `plots_dir`. Update the test first (TDD: red → green).

**Files:**
- Modify: `tests/test_visualization.py:15-16, 89-96`

- [ ] **Step 1: Replace the import block and the `plot_clusters` test body**

Open `tests/test_visualization.py`. Change line 15 from:

```python
from modules.visualization import plot_clusters
```

to:

```python
from modules.visualization.plots import plot_clusters
```

Line 16 stays as-is (`from modules.visualization.plots.cluster_case_plots import _plot_case`).

Then replace the body of `test_plot_clusters_uses_shared_cluster_plot_generator` (lines 61–102) with the version below. The differences from the current body: monkeypatch targets are now `modules.visualization.plots.cluster_plots.*`, the `PLOTS_DIR` patch is dropped, and `plot_clusters` is called with `plots_dir=str(tmp_path)`.

```python
def test_plot_clusters_uses_shared_cluster_plot_generator(monkeypatch, tmp_path):
    fake_rows = [
        SimpleNamespace(embedding_case="audio"),
        SimpleNamespace(embedding_case="video"),
        SimpleNamespace(embedding_case="sandwich"),
    ]

    query = MagicMock()
    query.distinct.return_value.all.return_value = [
        ("audio",),
        ("video",),
        ("sandwich",),
    ]
    query.filter.return_value.all.side_effect = [
        [fake_rows[0]],
        [fake_rows[1]],
        [fake_rows[2]],
    ]

    fake_session = MagicMock()
    fake_session.query.return_value = query

    called_cases = []

    def fake_cluster_plot_figure_for_case(eng, case: str, *, title_label=None):
        called_cases.append((eng, case, title_label))
        return plt.figure()

    monkeypatch.setattr(
        "modules.visualization.plots.cluster_plots.get_session",
        lambda: fake_session,
    )
    monkeypatch.setattr(
        "modules.visualization.plots.cluster_plots.cluster_plot_figure_for_case",
        fake_cluster_plot_figure_for_case,
    )

    plot_clusters(plots_dir=str(tmp_path))

    assert called_cases == [
        (fake_session.get_bind.return_value, "audio", None),
        (fake_session.get_bind.return_value, "sandwich", None),
        (fake_session.get_bind.return_value, "video", None),
    ]
```

- [ ] **Step 2: Run the test and confirm it fails (red)**

Run: `uv run pytest tests/test_visualization.py::test_plot_clusters_uses_shared_cluster_plot_generator -v`
Expected: FAIL — `ImportError: cannot import name 'plot_clusters' from 'modules.visualization.plots'` (or an `AttributeError` on the monkeypatch path). Either way, the test is correctly red.

Do **not** commit yet — the failing test will go in the same commit as the implementation.

---

## Task 2: Create `modules/visualization/plots/cluster_plots.py`

Move `plot_clusters` out of the package root and parameterize `plots_dir`.

**Files:**
- Create: `modules/visualization/plots/cluster_plots.py`

- [ ] **Step 1: Write the new module**

Create `modules/visualization/plots/cluster_plots.py` with exactly this content:

```python
"""Pipeline entry point: render and save UMAP cluster scatter plots as PNGs."""

import os

import matplotlib.pyplot as plt

from modules.console import log
from modules.database import UserCluster, get_session
from modules.visualization.plots.cluster_case_plots import cluster_plot_figure_for_case

__all__ = ("plot_clusters",)


def plot_clusters(plots_dir: str) -> None:
    """Load user_clusters from DB and save one PNG per embedding case."""
    os.makedirs(plots_dir, exist_ok=True)
    session = get_session()
    try:
        eng = session.get_bind()
        cases = sorted(
            r[0] for r in session.query(UserCluster.embedding_case).distinct().all()
        )
        for case in cases:
            fig = cluster_plot_figure_for_case(eng, case)
            path = os.path.join(plots_dir, f"clusters_{case}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log("viz", f"saved {path}", level="ok")
    finally:
        session.close()
```

Differences from the current `modules/visualization/__init__.py` body:
- `plots_dir: str` parameter replaces the module-level `PLOTS_DIR = "data/plots"` constant.
- No `matplotlib.use("Agg")` — `plots/__init__.py` will own that (Task 3).

---

## Task 3: Replace `modules/visualization/plots/__init__.py`

Make `plots/` own the backend selection and re-export `plot_clusters`.

**Files:**
- Modify: `modules/visualization/plots/__init__.py`

- [ ] **Step 1: Overwrite the file**

Current content is a single blank line. Replace with exactly:

```python
"""Plot subpackage: scatter plots produced by the pipeline and by Quarto."""

import matplotlib

matplotlib.use("Agg")

from modules.visualization.plots.cluster_plots import plot_clusters

__all__ = ("plot_clusters",)
```

Order matters: `matplotlib.use("Agg")` must run before anything imports `matplotlib.pyplot`. Because Python executes `plots/__init__.py` before any of its submodules, this guarantees the backend is set before `cluster_plots` or `cluster_case_plots` import `pyplot`.

---

## Task 4: Shrink `modules/visualization/__init__.py` to a docstring

**Files:**
- Modify: `modules/visualization/__init__.py`

- [ ] **Step 1: Overwrite the file**

Replace the entire contents with exactly:

```python
"""Visualization package: paper-facing tables and pipeline plots."""
```

No imports, no `plot_clusters`, no matplotlib, no DB references.

---

## Task 5: Remove the redundant `matplotlib.use("Agg")` from `cluster_case_plots.py`

**Files:**
- Modify: `modules/visualization/plots/cluster_case_plots.py:5-8`

- [ ] **Step 1: Delete lines 5–8**

Current lines 5–8:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Replace with:

```python
import matplotlib.pyplot as plt
```

The `matplotlib` import and `use("Agg")` call are gone; `pyplot` is still imported. Backend has already been set by `plots/__init__.py`.

Leave the rest of the file (line 9 onward) untouched.

---

## Task 6: Update `main.py` import

**Files:**
- Modify: `main.py:15`

- [ ] **Step 1: Edit the import line**

Change line 15 from:

```python
from modules.visualization import plot_clusters
```

to:

```python
from modules.visualization.plots import plot_clusters
```

The call site at `main.py:162` (`plot_clusters(plots_dir=settings.paths.plots_dir)`) is already correct — do not touch it.

---

## Task 7: Run the targeted test suite (green)

- [ ] **Step 1: Run the visualization tests**

Run: `uv run pytest tests/test_visualization.py tests/test_cluster_case_plots.py -v`
Expected: all green. In particular `test_plot_clusters_uses_shared_cluster_plot_generator` now passes against the new monkeypatch paths and the parameterized signature.

- [ ] **Step 2: Run the runtime test that patches `main.plot_clusters`**

Run: `uv run pytest tests/test_main_runtime.py -v`
Expected: all green. The patch `monkeypatch.setattr(main, "plot_clusters", lambda **kwargs: calls.append("viz"))` continues to work because `main` still binds the name `plot_clusters`, just sourced from a different module.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: all green. This catches anything else that may have imported from the old location.

---

## Task 8: Lint and type-check

- [ ] **Step 1: ruff**

Run: `uv run ruff check`
Expected: no errors. If imports are reported as unsorted in the modified files, run `uv run ruff check --fix`.

- [ ] **Step 2: ruff format**

Run: `uv run ruff format`
Expected: no files reformatted (or only the modified ones, with cosmetic changes only). Inspect the diff before staging.

- [ ] **Step 3: ty**

Run: `uv run ty check`
Expected: no errors.

---

## Task 9: Manual smoke verification

Confirm the side-effect scope actually shrank.

- [ ] **Step 1: Verify that importing a tables module no longer pulls matplotlib**

Run:
```bash
uv run python -c "import sys; import modules.visualization.tables.dataset_summary_users; assert 'matplotlib' not in sys.modules, sorted(k for k in sys.modules if 'matplotlib' in k); print('OK')"
```
Expected: `OK`. If matplotlib appears in `sys.modules`, something in the tables module or the package `__init__.py` is still importing it — investigate before proceeding.

- [ ] **Step 2: Verify the plots subpackage still sets the backend**

Run:
```bash
uv run python -c "import matplotlib; import modules.visualization.plots; assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend(); print('OK')"
```
Expected: `OK`.

---

## Task 10: Commit

Single atomic commit — test and implementation move together (TDD discipline: red commit only after green is achieved).

- [ ] **Step 1: Review the staged change**

Run:
```bash
git status
git diff --stat
```
Expected modified/created files:
- `M main.py`
- `M modules/visualization/__init__.py`
- `M modules/visualization/plots/__init__.py`
- `A modules/visualization/plots/cluster_plots.py`
- `M modules/visualization/plots/cluster_case_plots.py`
- `M tests/test_visualization.py`

Nothing else. If `modules/clustering/assign.py` shows as modified, that is the pre-existing unrelated change from before the branch was created — `git restore` it on this branch or move it aside; it does not belong in this commit.

- [ ] **Step 2: Stage the change**

Run:
```bash
git add main.py modules/visualization/__init__.py modules/visualization/plots/__init__.py modules/visualization/plots/cluster_plots.py modules/visualization/plots/cluster_case_plots.py tests/test_visualization.py
```

- [ ] **Step 3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
refactor(visualization): scope matplotlib backend out of package root

Move plot_clusters from modules/visualization/__init__.py into
modules/visualization/plots/cluster_plots.py and centralize
matplotlib.use("Agg") in modules/visualization/plots/__init__.py.

Importing modules.visualization.tables.* no longer pulls matplotlib,
the DB layer, or sets a global rendering backend — only consumers of
modules.visualization.plots.* pay that cost.

plot_clusters now accepts a plots_dir argument, fixing main.py:162
which already passes settings.paths.plots_dir but had been silently
swallowed by the old no-arg signature with a hardcoded PLOTS_DIR.
EOF
)"
```

- [ ] **Step 4: Confirm the commit landed**

Run: `git log --oneline -3`
Expected: the new commit at HEAD, the design-spec cherry-pick beneath it, and the `main` tip below that.

---

## Spec coverage check (self-review during plan writing)

- Spec "Goals" → backend scope shrunk (Tasks 3, 4, 5, 9); single-location backend selection (Task 3); `plot_clusters(plots_dir=…)` (Task 2, 6).
- Spec "Non-goals" → no renames (verified across tasks); no table-generator changes (Task 9 step 1 confirms via smoke test); `cluster_case_plots.py` only loses redundant backend call (Task 5).
- Spec "File layout" → matches Tasks 2–5 exactly.
- Spec "Per-file changes" → covered file-by-file in Tasks 2–6.
- Spec "Callers and ripple effects" → `main.py` Task 6; `tests/test_visualization.py` Task 1; `tests/test_main_runtime.py` Task 7 step 2; Quarto + table tests unchanged and verified by Task 7 step 3 plus Task 9 smoke.
- Spec "Test plan" → Task 7 (pytest), Task 8 (ruff/ty), Task 9 (manual smoke).
- Spec "Implementation order" → mirrored in Task numbering (test first, then create, then re-export, then shrink root, then drop redundant call, then update import).
