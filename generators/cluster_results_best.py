#!/usr/bin/env python
"""Show top-N cluster runs per embedding case, aligned with validate_clustering.

Default sort matches main.py: plateau survivors first (drop <= threshold), then by
DBCv descending — so rank 1 is the same run cluster_users() uses. Use --sort dbcv
for a raw DBCV-only leaderboard (those rows may all be ok=N while a lower-DBCv run wins).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.orm import Session

from modules.database import engine, ClusterRun


CASES = ["video", "sandwich", "audio"]

_HEADER = (
    f"{'rk':>2}  {'id':>5}  {'dbcv':>7}  {'plat':>7}  {'drop':>6}  {'ok':>2}  {'sil':>7}"
    f"  {'k':>3}  {'noise%':>6}"
    f"  {'nc':>3}  {'nn':>3}  {'md':>5}  {'u1m':<8}"
    f"  {'u2n':>3}  {'u2d':>5}  {'u2m':<8}"
    f"  {'mcs':>4}  {'ms':>4}  {'sel':<4}  {'hdbm':<8}  {'*':>1}"
)
_SEP = "-" * len(_HEADER)


def _fmt(val: float | None) -> str:
    return f"{val:.4f}" if val is not None else "    —"


def _fmt_ms(val: int | None) -> str:
    if val is None:
        return "   —"
    return f"{val:>4}"


def _pick_best(rows: list[ClusterRun], threshold: float) -> ClusterRun | None:
    """Same survivor rule as modules.cluster_validation._select_best."""
    if not rows:
        return None
    survivors = [r for r in rows if (r.dbcv - r.param_plateau_score) <= threshold]
    if not survivors:
        survivors = rows
    return max(survivors, key=lambda r: r.dbcv)


def _passes_plateau(r: ClusterRun, threshold: float) -> bool:
    return (r.dbcv - r.param_plateau_score) <= threshold


def _sort_rows(
    rows: list[ClusterRun],
    threshold: float,
    sort: str,
) -> list[ClusterRun]:
    if sort == "dbcv":
        rows.sort(key=lambda r: r.dbcv, reverse=True)
        return rows
    # validation: survivors first (same ordering as _select_best’s pool), then rest
    rows.sort(
        key=lambda r: (
            0 if _passes_plateau(r, threshold) else 1,
            -r.dbcv,
        ),
    )
    return rows


def show(
    case: str,
    num: int = 10,
    *,
    sort: str = "validation",
) -> None:
    threshold = float(os.environ.get("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05"))

    with Session(engine) as session:
        rows = (
            session.query(ClusterRun)
            .filter(
                ClusterRun.embedding_case == case,
                ClusterRun.in_current_grid == 1,
                ClusterRun.disqualified == 0,
                ClusterRun.dbcv.isnot(None),
                ClusterRun.param_plateau_score.isnot(None),
            )
            .all()
        )

    best = _pick_best(rows, threshold)
    _sort_rows(rows, threshold, sort)
    rows = rows[:num]

    print(f"\nTop {num} cluster runs — case={case} (current grid)\n")
    print(f"plateau drop threshold (VALIDATION_PLATEAU_DROP_THRESHOLD) = {threshold:g}")
    if best is not None:
        print(
            f"validation pick: id={best.id} dbcv={best.dbcv:.4f} "
            f"plat={best.param_plateau_score:.4f} "
            f"drop={best.dbcv - best.param_plateau_score:.4f}"
        )
    print()
    print(_HEADER)
    print(_SEP)
    for rank, r in enumerate(rows, 1):
        drop = r.dbcv - r.param_plateau_score
        ok = "Y" if drop <= threshold else "N"
        star = "*" if best is not None and r.id == best.id else " "
        print(
            f"{rank:>2}  {r.id:>5}  {_fmt(r.dbcv):>7}  {_fmt(r.param_plateau_score):>7}"
            f"  {drop:>6.4f}  {ok:>2}  {_fmt(r.silhouette):>7}"
            f"  {r.n_clusters:>3}  {r.noise_ratio:>6.1%}"
            f"  {r.umap_n_components:>3}  {r.umap_n_neighbors:>3}  {r.umap_min_dist:>5.3f}"
            f"  {r.umap_metric:<8}"
            f"  {r.umap2d_n_neighbors:>3}  {r.umap2d_min_dist:>5.3f}  {r.umap2d_metric:<8}"
            f"  {r.hdbscan_min_cluster_size:>4}  {_fmt_ms(r.hdbscan_min_samples)}"
            f"  {r.hdbscan_cluster_selection_method:<4}  {r.hdbscan_metric:<8}  {star:>1}"
        )
    if not rows:
        print("  (no eligible runs — need cluster_search + validate_clustering)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show top cluster runs per embedding case from the DB, "
            "matching validate_clustering filters and selection."
        )
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=CASES,
        help="Embedding case to inspect (video, sandwich, audio)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of top runs to show (default: 10)",
    )
    parser.add_argument(
        "--sort",
        choices=("validation", "dbcv"),
        default="validation",
        help=(
            "validation: plateau-stable runs first, then by DBCV (rank 1 = pipeline run). "
            "dbcv: pure DBCV descending (may disagree with cluster_users)."
        ),
    )
    args = parser.parse_args()
    show(args.case, args.num, sort=args.sort)


if __name__ == "__main__":
    main()
