#!/usr/bin/env python
"""Plain stdout summary of all clustering runs for one embedding case from the DB.

Reads every `cluster_runs` row for `--case` with no plateau/validation/grid filtering
unless `--no-include-filtered` drops disqualified or stale (`in_current_grid == 0`) rows.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sqlalchemy.orm import Session

from modules.database import ClusterRun, engine

CASES = ("video", "sandwich", "audio")


def _std_ddof1(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var)


def _rows_after_filter(
    rows: list[ClusterRun],
    *,
    include_filtered: bool,
) -> list[ClusterRun]:
    if include_filtered:
        return rows
    out: list[ClusterRun] = []
    for r in rows:
        if r.disqualified == 1:
            continue
        if r.in_current_grid == 0:
            continue
        out.append(r)
    return out


def stats_lines_for_runs(rows: list[ClusterRun]) -> list[str]:
    """Aggregate metrics for an already-filtered sequence of `ClusterRun` rows."""
    n = len(rows)
    denom = float(n) if n else 1.0

    dbcv_vals: list[float] = []
    sil_vals: list[float] = []
    k_vals: list[float] = []
    noise_pct_vals: list[float] = []

    neg_sil = 0
    lt01 = 0
    lt0 = 0
    k_le2 = 0
    k_le3 = 0

    for r in rows:
        dv = r.dbcv
        if dv is not None and math.isfinite(float(dv)):
            fv = float(dv)
            dbcv_vals.append(fv)
            if fv < 0.1:
                lt01 += 1
            if fv < 0:
                lt0 += 1
        sv = r.silhouette
        if sv is not None and math.isfinite(float(sv)):
            fv = float(sv)
            sil_vals.append(fv)
            if fv < 0:
                neg_sil += 1
        kv = r.n_clusters
        if kv is not None:
            k_vals.append(float(kv))
            if kv <= 2:
                k_le2 += 1
            if kv <= 3:
                k_le3 += 1
        nv = r.noise_ratio
        if nv is not None and math.isfinite(float(nv)):
            noise_pct_vals.append(float(nv) * 100.0)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    dbcv_a = np.array(dbcv_vals, dtype=float) if dbcv_vals else np.array([], dtype=float)
    k_a = np.array(k_vals, dtype=float) if k_vals else np.array([], dtype=float)

    k_med = float(np.median(k_a)) if k_a.size else 0.0

    def pct(count: int) -> float:
        return round(100.0 * count / denom, 1)

    return [
        f"n_runs: {n}",
        f"dbcv_mean: {mean(dbcv_vals):.3f}",
        f"dbcv_std: {_std_ddof1(dbcv_vals):.3f}",
        f"dbcv_min: {float(np.min(dbcv_a)):.3f}" if dbcv_a.size else "dbcv_min: 0.000",
        f"dbcv_max: {float(np.max(dbcv_a)):.3f}" if dbcv_a.size else "dbcv_max: 0.000",
        f"silhouette_mean: {mean(sil_vals):.3f}",
        f"silhouette_std: {_std_ddof1(sil_vals):.3f}",
        f"k_mean: {mean(k_vals):.3f}" if k_vals else "k_mean: 0.000",
        f"k_median: {k_med:.3f}",
        f"pct_k_le_2: {pct(k_le2)}",
        f"pct_k_le_3: {pct(k_le3)}",
        f"noise_pct_mean: {mean(noise_pct_vals):.3f}",
        f"noise_pct_std: {_std_ddof1(noise_pct_vals):.3f}",
        f"pct_negative_silhouette: {pct(neg_sil)}",
        f"pct_dbcv_lt_0_1: {pct(lt01)}",
        f"pct_dbcv_lt_0: {pct(lt0)}",
    ]


def summarize_to_lines(
    eng,
    *,
    case: str,
    include_filtered: bool,
) -> list[str]:
    with Session(eng) as session:
        rows = (
            session.query(ClusterRun)
            .filter(ClusterRun.embedding_case == case)
            .all()
        )
    filtered = _rows_after_filter(rows, include_filtered=include_filtered)
    body = stats_lines_for_runs(filtered)
    return [f"case: {case}", *body]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summary stats for all cluster runs in the DB (one embedding case).",
    )
    p.add_argument("--case", choices=CASES, default="audio")
    p.add_argument("--include-filtered", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    for line in summarize_to_lines(engine, case=args.case, include_filtered=args.include_filtered):
        print(line)


if __name__ == "__main__":
    main()
