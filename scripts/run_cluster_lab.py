"""CLI driver for the cluster lab.

Usage:
    uv run --group analysis python scripts/run_cluster_lab.py \\
        --grid all --workers 5

    uv run --group analysis python scripts/run_cluster_lab.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_lab.grids import ALL_GRIDS, GRID_REGISTRY  # noqa: E402
from scripts.cluster_lab.loader import load_sandwich_matrix  # noqa: E402
from scripts.cluster_lab.orchestrator import run_grid  # noqa: E402


def _resolve_grids(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_GRIDS)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [n for n in names if n not in GRID_REGISTRY]
    if bad:
        raise SystemExit(f"Unknown grids: {bad}; available: {list(GRID_REGISTRY)}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--grid",
        default="all",
        help="Comma-separated grid names or 'all'.",
    )
    ap.add_argument("--db", default="data/cluster_testing.db")
    ap.add_argument("--src-db", default="data/old/inst2vec.db")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = _resolve_grids(args.grid)

    if args.dry_run:
        total = 0
        for name in names:
            n = sum(1 for _ in GRID_REGISTRY[name]())
            total += n
            print(f"  {name}: {n} configs")
        print(f"TOTAL: {total} configs across {len(names)} grids")
        return

    print(f"Loading sandwich embeddings from {args.src_db} …")
    matrix, user_ids = load_sandwich_matrix(args.src_db)
    print(f"  matrix shape = {matrix.shape}, {len(user_ids)} users")

    summaries = []
    for name in names:
        print(f"\n=== {name} ===")
        s = run_grid(
            matrix,
            GRID_REGISTRY[name](),
            db_path=args.db,
            name=name,
            max_workers=args.workers,
            verbose=True,
        )
        summaries.append(s)

    print()
    print("=" * 60)
    print(f"{'grid':<32} {'sub':>6} {'skip':>6} {'err':>6} {'time':>8}")
    for s in summaries:
        print(
            f"{s['name']:<32} {s['submitted']:>6d} {s['skipped']:>6d} "
            f"{s['errors']:>6d} {s['elapsed']:>7.1f}s"
        )


if __name__ == "__main__":
    main()
