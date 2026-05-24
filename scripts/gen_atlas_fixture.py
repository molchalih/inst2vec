"""Generate a synthetic atlas fixture for the frontend Phase 1 viewer.

Writes three runs (`video-1`, `sandwich-1`, `audio-1`) to the given
output directory using `modules.export.writer`. Deterministic given a
seed — re-running produces byte-identical files.

Usage (from repo root):

    uv run python scripts/gen_atlas_fixture.py
    uv run python scripts/gen_atlas_fixture.py --out frontend/public/data
    uv run python scripts/gen_atlas_fixture.py --seed 7
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

from modules.export import (
    BoundsModel,
    ClusterModel,
    ClustersFile,
    EmbeddingCase,
    ManifestRun,
    RunPayload,
    UsersFile,
    write_dataset,
)

DEFAULT_OUT_DIR = Path("frontend/public/data")


@dataclass
class CaseSpec:
    run_id: str
    case: EmbeddingCase
    label: str
    n_clusters: int
    n_users: int
    noise_fraction: float


CASE_SPECS: list[CaseSpec] = [
    CaseSpec(
        "video-1", "video", "Visual", n_clusters=12, n_users=1000, noise_fraction=0.05
    ),
    CaseSpec(
        "sandwich-1",
        "sandwich",
        "Visual + Music",
        n_clusters=10,
        n_users=1000,
        noise_fraction=0.07,
    ),
    CaseSpec(
        "audio-1", "audio", "Speech", n_clusters=14, n_users=1000, noise_fraction=0.10
    ),
]


def _gaussian_blob(
    rng: random.Random,
    cx: float,
    cy: float,
    sigma_x: float,
    sigma_y: float,
    angle: float,
    n: int,
) -> list[tuple[float, float]]:
    """Sample `n` points from a 2-D Gaussian rotated by `angle`."""
    points: list[tuple[float, float]] = []
    cos = math.cos(angle)
    sin = math.sin(angle)
    for _ in range(n):
        u = rng.gauss(0.0, sigma_x)
        v = rng.gauss(0.0, sigma_y)
        x = cx + u * cos - v * sin
        y = cy + u * sin + v * cos
        points.append((x, y))
    return points


def _fit_ellipse(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float, float]:
    """Axis-aligned ellipse fit: centre = mean, rx/ry = 2 stdev. angle = 0.

    Phase 3's real export will reuse this same helper (and may add
    rotated PCA later); Phase 1 only needs axis-aligned because the
    blobs are intentionally axis-aligned at generation time.
    """
    if not points:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    var_x = sum((p[0] - mean_x) ** 2 for p in points) / max(len(points), 1)
    var_y = sum((p[1] - mean_y) ** 2 for p in points) / max(len(points), 1)
    return mean_x, mean_y, 2 * math.sqrt(var_x), 2 * math.sqrt(var_y), 0.0


def _build_run(spec: CaseSpec, rng: random.Random) -> RunPayload:
    """Build one run: cluster centres on a coarse ring, blobs around each,
    plus noise points scattered across the bounding box."""
    n_signal = int(spec.n_users * (1.0 - spec.noise_fraction))
    per_cluster = max(n_signal // spec.n_clusters, 1)
    cluster_users: list[tuple[float, float, int]] = []
    cluster_meta: list[ClusterModel] = []

    ring_radius = 6.0
    for cluster_id in range(spec.n_clusters):
        theta = 2 * math.pi * cluster_id / spec.n_clusters + rng.uniform(-0.15, 0.15)
        cx = ring_radius * math.cos(theta)
        cy = ring_radius * math.sin(theta)
        sigma_x = rng.uniform(0.7, 1.4)
        sigma_y = rng.uniform(0.7, 1.4)
        angle = rng.uniform(0.0, math.pi)
        pts = _gaussian_blob(rng, cx, cy, sigma_x, sigma_y, angle, per_cluster)
        for x, y in pts:
            cluster_users.append((x, y, cluster_id))
        fcx, fcy, rx, ry, fang = _fit_ellipse(pts)
        cluster_meta.append(
            ClusterModel(
                id=cluster_id,
                label=f"Cluster {cluster_id + 1}",
                cx=fcx,
                cy=fcy,
                rx=rx,
                ry=ry,
                angle=fang,
                size=len(pts),
            )
        )

    n_noise = spec.n_users - len(cluster_users)
    bound = ring_radius + 3.0
    for _ in range(n_noise):
        cluster_users.append(
            (rng.uniform(-bound, bound), rng.uniform(-bound, bound), -1)
        )

    rng.shuffle(cluster_users)
    users_tuples: list[tuple[int, float, float, int]] = [
        (i, x, y, cid) for i, (x, y, cid) in enumerate(cluster_users)
    ]
    xs = [u[1] for u in users_tuples]
    ys = [u[2] for u in users_tuples]
    bounds = BoundsModel(minX=min(xs), maxX=max(xs), minY=min(ys), maxY=max(ys))

    return RunPayload(
        meta=ManifestRun(
            id=spec.run_id, case=spec.case, label=spec.label, size=len(users_tuples)
        ),
        users=UsersFile(run_id=spec.run_id, bounds=bounds, users=users_tuples),
        clusters=ClustersFile(run_id=spec.run_id, clusters=cluster_meta),
    )


def build_dataset(out_dir: Path, *, seed: int = 42) -> None:
    """Build and write the full synthetic dataset to `out_dir`."""
    runs: list[RunPayload] = []
    for i, spec in enumerate(CASE_SPECS):
        # One rng per case so adding a case doesn't shift earlier seeds.
        rng = random.Random(seed * 1000 + i)
        runs.append(_build_run(spec, rng))
    write_dataset(out_dir, default_run_id="video-1", runs=runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic generation (default: 42)",
    )
    args = parser.parse_args()
    build_dataset(args.out, seed=args.seed)
    print(f"Wrote synthetic fixture to {args.out}")


if __name__ == "__main__":
    main()
