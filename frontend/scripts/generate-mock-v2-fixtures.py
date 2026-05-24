#!/usr/bin/env python3
"""Generate v2 mock fixtures for local frontend testing.

Reads the existing v1 manifest/users/clusters fixtures under
`public/data/` and upgrades them in place:

  * `manifest.json`         — version → 2, each run gets `details_available`.
  * `runs/{id}/users.json`  — version → 2, every tuple grows to
                              `[id, x, y, cluster_id, has_detail]`.
  * `runs/{id}/clusters.json` — version → 2, each cluster gains `has_detail`.

For the default run only (`details_available: true`), it also emits
per-id detail files:

  * `runs/{id}/clusters/{cluster_id}.json`
  * `runs/{id}/users/{user_id}.json`   — only for users marked
                                          `has_detail: true`

Everything is deterministic (seeded random) so re-running yields the
same output.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # frontend/
DATA = ROOT / "public" / "data"

# Which run gets full detail payloads. The others stay v2 but with
# details_available: false so we can verify the killswitch path too.
RUN_WITH_DETAILS = "video-1"

# Fraction of users in the detail run that gets `has_detail: true`.
# Pick ~30 % so there's a healthy mix of detail-rich and detail-less dots.
DETAIL_USER_FRACTION = 0.30

GENRES = [
    "house",
    "techno",
    "pop",
    "ambient",
    "hip-hop",
    "rock",
    "jazz",
    "lo-fi",
    "drum-and-bass",
    "trance",
    "indie",
    "soul",
]
INSTRUMENTS = [
    "synth",
    "drums",
    "guitar",
    "piano",
    "bass",
    "strings",
    "vocals",
    "percussion",
    "brass",
    "pads",
]
LANG_CODES = ["en", "es", "pt", "fr", "de", "ja", "ko", "it", "ru", "zh"]
FOLLOWER_BUCKETS = [
    "<1k",
    "1k–2k",
    "2k–5k",
    "5k–10k",
    "10k–20k",
    "20k–50k",
    "50k–100k",
    "100k–500k",
    "500k+",
]

DISTINCTIVENESS_FEATURES = [
    "is_electronic",
    "is_acoustic",
    "is_instrumental",
    "is_happy",
    "is_sad",
    "is_relaxed",
    "is_aggressive",
    "is_party",
    "is_bright",
    "is_tonal",
    "is_female_voice",
    "danceability",
    "engagement",
    "approachability",
    "median_clip_duration_s",
    "median_clips_per_week",
    "engagement_shape_ratio",
    "activity_span_months",
]


def seeded(seed: int) -> random.Random:
    return random.Random(seed)


def pick_weighted_labels(rng: random.Random, pool: list[str], k: int) -> list[dict]:
    """Pick k distinct labels and assign monotonically decreasing weights."""
    sample = rng.sample(pool, k)
    weights = sorted([rng.uniform(0.3, 1.0) for _ in range(k)], reverse=True)
    return [
        {"label": s, "weight": round(w, 3)}
        for s, w in zip(sample, weights, strict=True)
    ]


def lang_shares(rng: random.Random, k: int) -> list[dict]:
    sample = rng.sample(LANG_CODES, k)
    raw = [rng.uniform(0.1, 1.0) for _ in range(k)]
    total = sum(raw)
    shares = sorted([r / total for r in raw], reverse=True)
    return [
        {"code": c, "share": round(s, 3)} for c, s in zip(sample, shares, strict=True)
    ]


def distinctiveness(rng: random.Random) -> list[dict]:
    """1-3 entries, each a different feature with a plausible z."""
    k = rng.randint(1, 3)
    features = rng.sample(DISTINCTIVENESS_FEATURES, k)
    out = []
    for f in features:
        z = round(rng.uniform(-2.6, 2.6), 2)
        cohort = round(rng.uniform(0.1, 0.9), 3)
        base_mean = round(cohort - z * 0.15, 3)
        base_std = round(rng.uniform(0.05, 0.25), 3)
        out.append(
            {
                "feature": f,
                "cohort_value": cohort,
                "baseline_mean": base_mean,
                "baseline_std": base_std,
                "z": z,
            }
        )
    return out


def audio_scores(rng: random.Random) -> dict:
    return {
        "approachability": round(rng.uniform(0.2, 0.9), 3),
        "engagement": round(rng.uniform(0.2, 0.95), 3),
        "danceability": round(rng.uniform(0.1, 0.95), 3),
    }


def mood_shares(rng: random.Random) -> dict:
    return {
        "happy": round(rng.uniform(0, 0.85), 3),
        "sad": round(rng.uniform(0, 0.55), 3),
        "relaxed": round(rng.uniform(0, 0.8), 3),
        "aggressive": round(rng.uniform(0, 0.5), 3),
        "party": round(rng.uniform(0, 0.9), 3),
    }


def timbre_shares(rng: random.Random) -> dict:
    return {
        "acoustic": round(rng.uniform(0, 0.85), 3),
        "electronic": round(rng.uniform(0, 0.95), 3),
        "instrumental": round(rng.uniform(0, 0.85), 3),
        "female_voice": round(rng.uniform(0, 0.85), 3),
        "bright": round(rng.uniform(0.1, 0.95), 3),
        "tonal": round(rng.uniform(0.1, 0.95), 3),
    }


def posting_stats(rng: random.Random) -> dict:
    return {
        "median_plays": rng.choice(
            [
                120,
                540,
                1200,
                4800,
                8400,
                12000,
                18000,
                26000,
                41000,
                75000,
                120000,
                250000,
                480000,
                900000,
                1_400_000,
                3_200_000,
            ]
        ),
        "median_clip_duration_s": round(rng.uniform(8.0, 60.0), 1),
        "median_clips_per_week": round(rng.uniform(0.5, 12.0), 1),
        "engagement_shape_ratio": round(rng.uniform(0.6, 4.5), 1),
    }


@dataclass
class ClusterInfo:
    id: int
    label: str
    size: int
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float


def nearest(clusters: list[ClusterInfo], cid: int, k: int) -> list[dict]:
    me = next(c for c in clusters if c.id == cid)
    others = [c for c in clusters if c.id != cid and c.id >= 0]
    others.sort(key=lambda c: math.hypot(c.cx - me.cx, c.cy - me.cy))
    return [
        {
            "cluster_id": c.id,
            "label": c.label,
            "distance": round(math.hypot(c.cx - me.cx, c.cy - me.cy), 3),
        }
        for c in others[:k]
    ]


def build_cluster_detail(c: ClusterInfo, clusters: list[ClusterInfo]) -> dict:
    rng = seeded(20260523 + c.id * 17)
    return {
        "version": 2,
        "cluster_id": c.id,
        "label": c.label,
        "size": c.size,
        "ellipse": {
            "cx": c.cx,
            "cy": c.cy,
            "rx": c.rx,
            "ry": c.ry,
            "angle": c.angle,
        },
        "audio": audio_scores(rng),
        "mood_shares": mood_shares(rng),
        "timbre_shares": timbre_shares(rng),
        "genre_top": pick_weighted_labels(rng, GENRES, rng.randint(2, 4)),
        "instrument_top": pick_weighted_labels(rng, INSTRUMENTS, rng.randint(2, 3)),
        "speech": {
            "detected_share": round(rng.uniform(0.05, 0.85), 3),
            "top_langs": lang_shares(rng, rng.randint(1, 3)),
        },
        "caption": {
            "top_langs": lang_shares(rng, rng.randint(1, 3)),
        },
        "posting": posting_stats(rng),
        "follower_bucket": rng.choice(FOLLOWER_BUCKETS),
        "activity_span_months": rng.randint(3, 48),
        "distinctiveness": distinctiveness(rng),
        "spatial": {
            "compactness": round(rng.uniform(0.005, 0.08), 4),
            "nearest_clusters": nearest(clusters, c.id, 3),
        },
    }


def build_creator_detail(
    user_row: list, clusters: list[ClusterInfo], borderline: bool
) -> dict:
    uid, x, y, cid, _ = user_row
    rng = seeded(99250523 + uid * 31)
    spatial = {
        "distance_from_centroid": round(rng.uniform(0.05, 1.2), 3),
        "distance_from_centroid_percentile": rng.randint(5, 99),
        "nearest_other_cluster": None,
    }
    if borderline:
        # pick the geometrically closest non-own cluster
        nearest_list = nearest(clusters, cid, 1)
        if nearest_list:
            spatial["nearest_other_cluster"] = nearest_list[0]
    return {
        "version": 2,
        "user_id": uid,
        "cluster_id": cid,
        "x": x,
        "y": y,
        "n_clips": rng.randint(2, 60),
        "audio": audio_scores(rng),
        "mood_shares": mood_shares(rng),
        "timbre_shares": timbre_shares(rng),
        "genre_top": pick_weighted_labels(rng, GENRES, rng.randint(1, 3)),
        "instrument_top": pick_weighted_labels(rng, INSTRUMENTS, rng.randint(1, 2)),
        "speech": {
            "detected_share": round(rng.uniform(0.0, 0.95), 3),
            "top_langs": lang_shares(rng, rng.randint(1, 2)),
        },
        "caption": {
            "top_langs": lang_shares(rng, rng.randint(1, 2)),
        },
        "posting": posting_stats(rng),
        "follower_bucket": rng.choice(FOLLOWER_BUCKETS),
        "activity_span_months": rng.randint(2, 42),
        "distinctiveness": distinctiveness(rng),
        "spatial": spatial,
    }


def process_run(run_id: str, has_details: bool) -> None:
    run_dir = DATA / "runs" / run_id
    users_path = run_dir / "users.json"
    clusters_path = run_dir / "clusters.json"

    users_doc = json.loads(users_path.read_text())
    clusters_doc = json.loads(clusters_path.read_text())

    rng = seeded(int(sum(ord(c) for c in run_id)))

    # Pick detail user ids only for the detail-rich run.
    detail_ids: set[int] = set()
    if has_details:
        all_ids = [row[0] for row in users_doc["users"] if row[3] >= 0]
        n = int(len(all_ids) * DETAIL_USER_FRACTION)
        detail_ids = set(rng.sample(all_ids, n))

    # v2 users: 5-wide tuples.
    new_users = []
    for row in users_doc["users"]:
        # Existing rows are [id, x, y, cluster_id]; preserve any extras.
        uid, x, y, cid = row[:4]
        new_users.append([uid, x, y, cid, uid in detail_ids])
    users_doc["users"] = new_users
    users_doc["version"] = 2
    users_path.write_text(json.dumps(users_doc) + "\n")

    # v2 clusters: each gains has_detail. In the detail run, every
    # non-noise cluster gets has_detail: true. In other runs, false.
    new_clusters = []
    cluster_infos: list[ClusterInfo] = []
    for c in clusters_doc["clusters"]:
        hd = has_details and c["id"] >= 0
        c2 = {**c, "has_detail": hd}
        new_clusters.append(c2)
        cluster_infos.append(
            ClusterInfo(
                id=c["id"],
                label=c["label"],
                size=c["size"],
                cx=c["cx"],
                cy=c["cy"],
                rx=c["rx"],
                ry=c["ry"],
                angle=c.get("angle", 0.0),
            )
        )
    clusters_doc["clusters"] = new_clusters
    clusters_doc["version"] = 2
    clusters_path.write_text(json.dumps(clusters_doc) + "\n")

    if not has_details:
        # Remove any stale detail directories from previous runs so the
        # tree only carries v2 detail data where it's supposed to.
        for sub in ("clusters", "users"):
            d = run_dir / sub
            if d.is_dir():
                for f in d.glob("*.json"):
                    f.unlink()
                d.rmdir()
        return

    # Cluster detail files.
    cluster_dir = run_dir / "clusters"
    cluster_dir.mkdir(exist_ok=True)
    for c in cluster_infos:
        if c.id < 0:
            continue
        payload = build_cluster_detail(c, cluster_infos)
        (cluster_dir / f"{c.id}.json").write_text(json.dumps(payload) + "\n")

    # Creator detail files.
    user_dir = run_dir / "users"
    user_dir.mkdir(exist_ok=True)
    # A creator is "borderline" if its row index is divisible by 5;
    # roughly 20 % of detail-rich creators get a nearest_other_cluster.
    for row in new_users:
        uid, _, _, cid, has_detail = row
        if not has_detail:
            continue
        borderline = uid % 5 == 0
        payload = build_creator_detail(row, cluster_infos, borderline)
        (user_dir / f"{uid}.json").write_text(json.dumps(payload) + "\n")


def upgrade_manifest() -> None:
    path = DATA / "manifest.json"
    doc = json.loads(path.read_text())
    doc["version"] = 2
    for run in doc["runs"]:
        run["details_available"] = run["id"] == RUN_WITH_DETAILS
    path.write_text(json.dumps(doc, indent=2) + "\n")


def main() -> None:
    print(f"Upgrading fixtures under {DATA}")
    upgrade_manifest()
    manifest = json.loads((DATA / "manifest.json").read_text())
    for run in manifest["runs"]:
        process_run(run["id"], run["details_available"])
        print(f"  - {run['id']:11s}  details_available={run['details_available']}")
    print("Done. Restart `bun run dev` to load v2 fixtures.")


if __name__ == "__main__":
    main()
