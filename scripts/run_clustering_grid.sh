#!/usr/bin/env bash
#
# full overnight grid: sweeps umap_n_components, pass-1 UMAP + HDBSCAN structure, and metrics.
#
# dimensions (edit arrays below):
#   --embedding-case filters which modalities run (default: all three)
#   umap_n_components × umap_n_neighbors × umap_min_dist
#   × hdbscan_min_cluster_size × hdbscan_cluster_selection_method
#   × umap_metric × umap2d_metric × hdbscan_metric
#
# umap2d layout (viz only): fixed here (not swept) — umap2d_n_neighbors / umap2d_min_dist
#
# usage:
#   ./scripts/run_clustering_grid.sh
#   ./scripts/run_clustering_grid.sh --embedding-case video
#   GRID_JOBS=4 ./scripts/run_clustering_grid.sh
#   OUT_CSV=/tmp/runs.csv ./scripts/run_clustering_grid.sh
#   nohup ./scripts/run_clustering_grid.sh >data/grid_overnight.log 2>&1 &
#
# concurrent runs (default GRID_JOBS=1 = sequential). with GRID_JOBS>1, tune_clustering
# uses file locking on the csv; optional: cap blas threads per process to avoid
# oversubscription, e.g. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNE="${ROOT}/scripts/tune_clustering.py"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

EMBED="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-case)
      EMBED="${2:?}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,32p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

case "$EMBED" in
  all) CASES=(video sandwich audio) ;;
  video|sandwich|audio) CASES=("$EMBED") ;;
  *)
    echo "--embedding-case must be video, sandwich, audio, or all" >&2
    exit 1
    ;;
esac

OUT_CSV="${OUT_CSV:-${ROOT}/scripts/clustering_results.csv}"
GRID_JOBS="${GRID_JOBS:-1}"
mkdir -p "$(dirname "$OUT_CSV")"

# --- edit these ---
UMAP_N_COMPONENTS=(10 15 20 30 35 40)
UMAP_N_NEIGHBORS=(10 15 30)
UMAP_MIN_DIST=(0.0 0.05)
HDBSCAN_MIN_CLUSTER_SIZE=(10 15 25)
HDBSCAN_SELECTION=(eom leaf)

UMAP_METRICS=(cosine euclidean correlation)
UMAP2D_METRICS=(cosine euclidean)
HDBSCAN_METRICS=(euclidean cosine)

# fixed 2D UMAP (not swept); tune_clustering passes explicit values
UMAP2D_N_NEIGHBORS=15
UMAP2D_MIN_DIST=0.1
# ---

export OUT_CSV PYTHON TUNE UMAP2D_N_NEIGHBORS UMAP2D_MIN_DIST GRID_JOBS

total=$((${#CASES[@]} * ${#UMAP_N_COMPONENTS[@]} * ${#UMAP_N_NEIGHBORS[@]} * ${#UMAP_MIN_DIST[@]} * ${#HDBSCAN_MIN_CLUSTER_SIZE[@]} * ${#HDBSCAN_SELECTION[@]} * ${#UMAP_METRICS[@]} * ${#UMAP2D_METRICS[@]} * ${#HDBSCAN_METRICS[@]}))

echo "[grid] output: $OUT_CSV"
echo "[grid] GRID_JOBS=$GRID_JOBS (set GRID_JOBS=N for N concurrent workers)"
echo "[grid] runs: $total  (cases: ${CASES[*]})"
echo "[grid] umap_n_components=${UMAP_N_COMPONENTS[*]}"
echo "[grid] umap_n_neighbors=${UMAP_N_NEIGHBORS[*]} min_dist=${UMAP_MIN_DIST[*]}"
echo "[grid] hdbscan_min_cluster_size=${HDBSCAN_MIN_CLUSTER_SIZE[*]} selection=${HDBSCAN_SELECTION[*]}"
echo "[grid] umap_metric=${UMAP_METRICS[*]} | umap2d_metric=${UMAP2D_METRICS[*]} (2d nn=$UMAP2D_N_NEIGHBORS min_dist=$UMAP2D_MIN_DIST) | hdbscan_metric=${HDBSCAN_METRICS[*]}"
echo

for case in "${CASES[@]}"; do
  for ucomp in "${UMAP_N_COMPONENTS[@]}"; do
    for nn in "${UMAP_N_NEIGHBORS[@]}"; do
      for md in "${UMAP_MIN_DIST[@]}"; do
        for mcs in "${HDBSCAN_MIN_CLUSTER_SIZE[@]}"; do
          for sel in "${HDBSCAN_SELECTION[@]}"; do
            for um in "${UMAP_METRICS[@]}"; do
              for u2m in "${UMAP2D_METRICS[@]}"; do
                for hm in "${HDBSCAN_METRICS[@]}"; do
                  echo "$case" "$ucomp" "$nn" "$md" "$mcs" "$sel" "$um" "$u2m" "$hm"
                done
              done
            done
          done
        done
      done
    done
  done
done | xargs -r -P "$GRID_JOBS" -n 9 bash -c '
  echo "[grid] $1  ucomp=$2 nn=$3 md=$4 mcs=$5 sel=$6 | umap=$7 umap2d=$8 hdbscan_m=$9"
  "$PYTHON" "$TUNE" \
    --embedding-case "$1" \
    --csv "$OUT_CSV" \
    --umap-n-components "$2" \
    --umap-n-neighbors "$3" \
    --umap-min-dist "$4" \
    --umap-metric "$7" \
    --umap2d-n-neighbors "$UMAP2D_N_NEIGHBORS" \
    --umap2d-min-dist "$UMAP2D_MIN_DIST" \
    --umap2d-metric "$8" \
    --hdbscan-min-cluster-size "$5" \
    --hdbscan-cluster-selection-method "$6" \
    --hdbscan-metric "$9"
' _

echo
echo "========== summary: $OUT_CSV =========="
"$PYTHON" - "$OUT_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
if not rows:
    print("(empty)")
    sys.exit(0)

cols = [
    "embedding_case",
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "umap2d_n_neighbors",
    "umap2d_min_dist",
    "umap2d_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
    "n_clusters",
    "noise_ratio",
    "min_size",
    "median_size",
    "max_size",
]
cols = [c for c in cols if c in rows[0]]

def key(r):
    return (
        r.get("embedding_case", ""),
        float(r.get("noise_ratio", 0)),
        -int(r.get("n_clusters", 0) or 0),
    )

rows.sort(key=key)

widths = []
for c in cols:
    w = len(c)
    for r in rows:
        w = max(w, len(str(r.get(c, ""))))
    widths.append(w)

def line(cells):
    return " | ".join(str(x).ljust(w) for x, w in zip(cells, widths))

print(line(cols))
print("-+-".join("-" * w for w in widths))
for r in rows:
    print(line([r.get(c, "") for c in cols]))
PY

echo
echo "[grid] done. full csv: $OUT_CSV"
