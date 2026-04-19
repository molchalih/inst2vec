#!/usr/bin/env python
"""Show top-N cluster runs per embedding case, ranked by final selection score."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.orm import Session

from modules.database import engine, ClusterRun

CASES = ["video", "sandwich", "audio"]

_HEADER = (
    f"{'rk':>2}  {'id':>5}  {'composite':>9}  {'plateau':>7}  {'final':>7}"
    f"  {'dbcv':>7}  {'sil':>7}  {'stab':>7}"
    f"  {'k':>3}  {'noise%':>6}"
    f"  {'nc':>3}  {'nn':>3}  {'md':>5}  {'metric':<10}  {'mcs':>4}  {'ms':>4}"
)
_SEP = "-" * len(_HEADER)


def _fmt(val: float | None) -> str:
    return f"{val:.4f}" if val is not None else "    —"


def show(case: str, num: int = 10) -> None:
    with Session(engine) as session:
        rows = (
            session.query(ClusterRun)
            .filter(
                ClusterRun.embedding_case == case,
                ClusterRun.disqualified == 0,
                ClusterRun.composite_score.isnot(None),
                ClusterRun.param_plateau_score.isnot(None),
            )
            .all()
        )

    rows.sort(
        key=lambda r: 0.7 * r.composite_score + 0.3 * r.param_plateau_score,
        reverse=True,
    )
    rows = rows[:num]

    print(f"\nTop {num} cluster runs — case={case}\n")
    print(_HEADER)
    print(_SEP)
    for rank, r in enumerate(rows, 1):
        final = 0.7 * r.composite_score + 0.3 * r.param_plateau_score
        print(
            f"{rank:>2}  {r.id:>5}  {r.composite_score:>9.4f}  {r.param_plateau_score:>7.4f}"
            f"  {final:>7.4f}"
            f"  {_fmt(r.dbcv):>7}  {_fmt(r.silhouette):>7}  {_fmt(r.bootstrap_stability):>7}"
            f"  {r.n_clusters:>3}  {r.noise_ratio:>6.1%}"
            f"  {r.umap_n_components:>3}  {r.umap_n_neighbors:>3}  {r.umap_min_dist:>5.3f}"
            f"  {r.umap_metric:<10}  {r.hdbscan_min_cluster_size:>4}"
            f"  {str(r.hdbscan_min_samples):>4}"
        )
    if not rows:
        print("  (no eligible runs)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show top cluster runs per embedding case from the DB"
    )
    parser.add_argument("--case", required=True, choices=CASES,
                        help="Embedding case to inspect (video, sandwich, audio)")
    parser.add_argument("--num", type=int, default=10,
                        help="Number of top runs to show (default: 10)")
    args = parser.parse_args()
    show(args.case, args.num)


if __name__ == "__main__":
    main()
