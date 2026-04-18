"""Analyse audio cluster_runs and output a reduced hyperparameter grid to CSV.

Scoring: rank all audio runs by (noise_ratio ASC) among runs with n_clusters >= 3.
For each hyperparameter, count how often each value appears in the top-N ranked runs.
Keep the top-K most frequent values per param, then emit the Cartesian product.

Output: scripts/audio_best_params.csv
"""
import csv
import sqlite3
from collections import Counter
from itertools import product

TOP_N = 100   # runs to consider when tallying param frequencies
TOP_K = 2     # max distinct values to keep per param in the reduced grid

# Values forced regardless of frequency (when the data is highly skewed)
FORCED: dict[str, list] = {
    "hdbscan_cluster_selection_method": ["eom"],  # 96/100 top runs use eom
}

PARAM_COLS = [
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
]

OUTPUT = "scripts/audio_best_params.csv"
DB = "data/inst2vec.db"


def main() -> None:
    con = sqlite3.connect(DB)
    cur = con.cursor()

    rows = cur.execute(
        f"""
        SELECT {', '.join(PARAM_COLS)}, n_clusters, noise_ratio
        FROM cluster_runs
        WHERE embedding_case = 'audio' AND n_clusters >= 3
        ORDER BY noise_ratio ASC
        LIMIT {TOP_N}
        """
    ).fetchall()

    if not rows:
        print("No audio runs with n_clusters >= 3 found.")
        return

    print(f"Analysing top {len(rows)} audio runs (n_clusters >= 3, lowest noise).")

    # Tally value frequencies per param
    counters: dict[str, Counter] = {col: Counter() for col in PARAM_COLS}
    for row in rows:
        for i, col in enumerate(PARAM_COLS):
            counters[col][row[i]] += 1

    # Select top-K values per param, print summary
    selected: dict[str, list] = {}
    for col in PARAM_COLS:
        if col in FORCED:
            top_vals = FORCED[col]
        else:
            top_vals = [v for v, _ in counters[col].most_common(TOP_K)]
        selected[col] = sorted(top_vals, key=lambda x: (x is None, x))
        print(f"  {col}: {selected[col]}  (full dist: {dict(counters[col].most_common())})")

    # Cartesian product → reduced grid
    keys = list(selected.keys())
    combos = list(product(*[selected[k] for k in keys]))
    print(f"\nReduced grid: {len(combos)} combinations (from {TOP_N} top runs, TOP_K={TOP_K})")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for combo in combos:
            writer.writerow(dict(zip(keys, combo)))

    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
