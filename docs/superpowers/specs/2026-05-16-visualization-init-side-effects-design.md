# Scope matplotlib backend selection out of `modules.visualization` package root

**Status:** Design — awaiting user review
**Date:** 2026-05-16
**Branch:** `feature/music-clients-relocation` (will be carved off; this work is unrelated to that branch's scope)

## Context

The generators-to-visualization refactor (merged in `0568fc9`) relocated `plot_clusters()` into `modules/visualization/__init__.py`, inheriting the original `matplotlib.use("Agg")` call from the standalone `modules/visualization.py` it replaced. As a side effect, `modules/visualization/` became a package whose `__init__.py` performs three classes of work at import time:

1. Imports matplotlib and selects the `Agg` backend globally.
2. Imports `modules.database` and `modules.console` (for `plot_clusters`'s body).
3. Imports `modules.visualization.plots.cluster_case_plots`.

Because Python executes a package's `__init__.py` whenever any submodule is imported, every `import modules.visualization.tables.X` — used by `docs/quarto_helpers.py` and the `tests/test_*_summary_*.py` / `tests/test_cluster_results_*.py` / `tests/test_spotify_feature_metrics.py` suites — now triggers all three. None of those callers need matplotlib or the DB session.

The reviewer flagged this as advisory; it is harmless in practice (tests run headless, Quarto saves via Agg-compatible paths). The user has elected to fix it cleanly now.

A separate latent bug was uncovered in the same area: `main.py:162` calls `plot_clusters(plots_dir=settings.paths.plots_dir)` but the current signature is `plot_clusters() -> None` with a hardcoded `PLOTS_DIR = "data/plots"`. The pipeline would raise `TypeError` on the visualization stage. The user has elected to fold the fix into this change.

## Goals

- Importing any `modules.visualization.tables.*` submodule must not pull in matplotlib, the DB, or set a global rendering backend.
- The `Agg` backend selection must remain effective for every code path that produces plots (pipeline `plot_clusters`, Quarto `cluster_plot_figure_for_case`, and any future plot module).
- The backend selection must live in exactly one place, where forgetting it on a future submodule is impossible.
- `plot_clusters()` accepts a `plots_dir` argument so `main.py:162` works.

## Non-goals

- No renames of public functions (`plot_clusters`, `cluster_plot_figure_for_case`).
- No changes to table generators, Quarto helpers, or their import paths.
- No refactor of `cluster_case_plots.py` internals beyond removing its redundant `matplotlib.use("Agg")` call.

## Design

### File layout

```
modules/visualization/
├── __init__.py                       # docstring only, no imports, no side effects
├── tables/
│   └── …                              # unchanged
└── plots/
    ├── __init__.py                   # matplotlib.use("Agg") + re-export plot_clusters
    ├── cluster_plots.py              # NEW — houses plot_clusters()
    └── cluster_case_plots.py         # drop its own matplotlib.use("Agg")
```

### Per-file changes

**`modules/visualization/__init__.py`** — reduce to:

```python
"""Visualization package: paper-facing tables and pipeline plots."""
```

No imports. No side effects.

**`modules/visualization/plots/__init__.py`** — own the backend selection for the whole plot subpackage and re-export the pipeline entry point:

```python
"""Plot subpackage: scatter plots produced by the pipeline and by Quarto."""

import matplotlib

matplotlib.use("Agg")

from modules.visualization.plots.cluster_plots import plot_clusters

__all__ = ("plot_clusters",)
```

Because Python executes a subpackage's `__init__.py` before any of its submodules, every `import modules.visualization.plots…` path is guaranteed to have called `matplotlib.use("Agg")` before any `import matplotlib.pyplot` runs. A new plot module can simply `import matplotlib.pyplot as plt` without ceremony.

**`modules/visualization/plots/cluster_plots.py`** — new file holding the relocated `plot_clusters()` with `plots_dir` parameterized:

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

Differences from the existing `__init__.py` body:

- `plots_dir: str` is now a required keyword/positional argument; the hardcoded `PLOTS_DIR = "data/plots"` constant is removed.
- The module-level `matplotlib.use("Agg")` is gone — `plots/__init__.py` has already called it.

**`modules/visualization/plots/cluster_case_plots.py`** — delete lines 5–7:

```python
import matplotlib

matplotlib.use("Agg")
```

The first matplotlib-touching import (`import matplotlib.pyplot as plt`) remains; the backend has already been set by `plots/__init__.py`.

**`main.py:15`** — update the import:

```python
from modules.visualization.plots import plot_clusters
```

The call at `main.py:162` is already correct (`plot_clusters(plots_dir=settings.paths.plots_dir)`); no change there.

### Callers and ripple effects

| Caller | Current import | New import | Notes |
|---|---|---|---|
| `main.py` | `from modules.visualization import plot_clusters` | `from modules.visualization.plots import plot_clusters` | Only caller of `plot_clusters`. |
| `tests/test_visualization.py` | `from modules.visualization import plot_clusters` | `from modules.visualization.plots import plot_clusters` | Test file. Also imports `_plot_case` from `cluster_case_plots` — unchanged. |
| `tests/test_main_runtime.py:158` | `monkeypatch.setattr(main, "plot_clusters", …)` | unchanged | Patches the binding on `main`, which now resolves to the new location. Works as-is. |
| `docs/quarto_helpers.py:170` | imports `cluster_plot_figure_for_case` from `cluster_case_plots` | unchanged | Path unchanged. |
| `tests/test_cluster_case_plots.py` | imports from `…plots.cluster_case_plots` | unchanged | |
| `tests/test_*` (tables) | imports from `…tables.X` | unchanged | These will now load *only* the empty package `__init__.py` and the targeted submodule. Behavioral win. |

### Behavior preservation

- `plot_clusters()` produces identical output: same DB query, same per-case scatter generation via `cluster_plot_figure_for_case`, same PNG path scheme `{plots_dir}/clusters_{case}.png`, same DPI, same logging.
- `cluster_plot_figure_for_case` is untouched apart from removal of a redundant backend call.
- Table modules' behavior is unchanged; only their import-time cost shrinks.

### Test plan

- `uv run pytest tests/test_visualization.py` — confirms `plot_clusters` still wires `cluster_plot_figure_for_case` and writes PNGs.
- `uv run pytest tests/test_cluster_case_plots.py` — confirms the figure generator still works with the relocated backend call.
- `uv run pytest tests/test_main_runtime.py` — confirms `monkeypatch.setattr(main, "plot_clusters", …)` still patches successfully.
- `uv run pytest` — full suite to catch any missed import.
- `uv run ruff check && uv run ruff format && uv run ty check`.
- Manual smoke (optional): `python -c "import modules.visualization.tables.dataset_summary_users; import sys; assert 'matplotlib' not in sys.modules"` — confirms the side-effect scope has actually shrunk.

### Edge cases & risks

- **`plot_clusters` signature breaks any caller that omits `plots_dir`.** Only `main.py:162` calls it, and that call already passes the argument. The test in `tests/test_visualization.py:96` (`plot_clusters()`) will need `plots_dir=tmp_path` added. Quick read of that test will confirm.
- **`plots/__init__.py` re-exports `plot_clusters` which imports `cluster_case_plots`.** This means `import modules.visualization.plots` still pulls matplotlib and the DB. That is the *correct* scope — anyone touching `plots/` is here to render plots. The win is that `tables/` consumers no longer pay this cost.
- **Backend set twice if someone imports `cluster_case_plots` directly without going through `plots/`?** Cannot happen: importing any submodule of `plots/` runs `plots/__init__.py` first.

## Implementation order

1. Create `modules/visualization/plots/cluster_plots.py` with the relocated, parameterized `plot_clusters`.
2. Replace `modules/visualization/plots/__init__.py` with the new content (backend selection + re-export).
3. Reduce `modules/visualization/__init__.py` to docstring only.
4. Remove the redundant `matplotlib.use("Agg")` block from `modules/visualization/plots/cluster_case_plots.py`.
5. Update `main.py:15` import.
6. Update `tests/test_visualization.py:15` import and pass `plots_dir=tmp_path` at line 96.
7. Run the full test plan.
