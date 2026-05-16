# Move `generators/` to `modules/visualization/`

## Goal

Relocate the top-level `generators/` namespace package into the `modules/`
tree under a new `modules/visualization/` package, separating markdown table
generators from plot generators by sub-directory. This brings the directory
layout in line with the rest of the codebase, where production code lives
under `modules/`.

## Motivation

`generators/` sits at the project root alongside `modules/`, but its contents
are project-internal helpers, not a separate distribution. The current
`modules/visualization.py` already depends on `generators.cluster_case_plots`,
so the two are coupled but kept apart. Folding the generators into a
`modules/visualization/` package makes that coupling explicit and removes a
root-level directory that exists for historical reasons.

## Target layout

```
modules/visualization/
├── __init__.py                          # was modules/visualization.py
│                                        # exports plot_clusters
├── tables/
│   ├── __init__.py                      # empty
│   ├── cluster_results_all.py
│   ├── cluster_results_best.py
│   ├── dataset_counts.py
│   ├── dataset_summary_clips.py
│   ├── dataset_summary_users.py
│   └── spotify_feature_metrics.py
└── plots/
    ├── __init__.py                      # empty
    └── cluster_case_plots.py
```

`tables/` holds the six modules that produce paper-facing Markdown/LaTeX
tables. `plots/` holds the matplotlib figure generator currently consumed by
`modules/visualization.py`. The split is by output medium so the directory
names match contents.

The existing public entry point `from modules.visualization import
plot_clusters` is preserved by promoting `modules/visualization.py` to
`modules/visualization/__init__.py`.

## Edits

1. **Create package skeleton.**
   - `modules/visualization/__init__.py` ← contents of
     `modules/visualization.py`, with internal import rewritten:
     `from generators.cluster_case_plots import …` →
     `from modules.visualization.plots.cluster_case_plots import …`.
   - `modules/visualization/tables/__init__.py` (empty).
   - `modules/visualization/plots/__init__.py` (empty).
2. **Move files via `git mv`** (preserves history):
   - `generators/cluster_results_all.py`        → `modules/visualization/tables/`
   - `generators/cluster_results_best.py`       → `modules/visualization/tables/`
   - `generators/dataset_counts.py`             → `modules/visualization/tables/`
   - `generators/dataset_summary_clips.py`      → `modules/visualization/tables/`
   - `generators/dataset_summary_users.py`      → `modules/visualization/tables/`
   - `generators/spotify_feature_metrics.py`    → `modules/visualization/tables/`
   - `generators/cluster_case_plots.py`         → `modules/visualization/plots/`
3. **Delete `modules/visualization.py`** (replaced by the new package
   `__init__.py`) and remove the now-empty `generators/` directory plus its
   `__pycache__`.
4. **Rewrite imports** in consumers:
   - `docs/quarto_helpers.py` — 8 `from generators.X import Y` rewritten to
     `from modules.visualization.tables.X import Y` (or `…plots…` for
     `cluster_case_plots`).
   - `tests/test_cluster_case_plots.py` — `from generators.cluster_case_plots`
     → `from modules.visualization.plots.cluster_case_plots`.
   - `tests/test_visualization.py` — same rewrite for `_plot_case` import.
   - `tests/test_cluster_results_all.py`,
     `tests/test_cluster_results_best.py`,
     `tests/test_dataset_summary_clips.py`,
     `tests/test_dataset_summary_users.py`,
     `tests/test_spotify_feature_metrics.py` — rewrite to
     `from modules.visualization.tables.X import Y`.
   - Test file names stay as-is.

## Public API

Unchanged. `main.py` imports `from modules.visualization import
plot_clusters`; that path remains valid because the package's `__init__.py`
defines `plot_clusters` (carried over from the old module file).

## Out of scope

- Historical plan docs under `docs/superpowers/plans/` that mention old
  `generators/` paths. Those are records of past work, not live code, and
  rewriting them would muddle the project's history.
- Renaming, splitting, or otherwise restructuring any generator module's
  internals. This change is purely relocation + import-path updates.
- Adding re-exports in the new `__init__.py` files. Consumers import the
  fully-qualified module paths.

## Verification

- `uv run ruff check`
- `uv run ruff format --check`
- `uv run ty check`
- `uv run pytest` — full suite must pass (497 tests today). The generator
  tests will exercise the new import paths; `tests/test_visualization.py`
  exercises both `plot_clusters` (via the new package `__init__.py`) and the
  private `_plot_case` helper under `plots/`.

## Risk and rollback

Risk is low: this is mechanical file relocation plus import rewrites with no
behavior change. Rollback is a single `git revert` of the resulting commit.
