"""Pure compute helpers for the visualization stage.

No DB, no filesystem — functions take primitive inputs and return
primitive outputs / dataclasses, so they are trivially unit-tested.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.database import UserCluster, VisualizationCluster, VisualizationUser
from modules.visualization.schema import SCHEMA_VERSION

# Edges follow a 1–1.5–2–2.5–3–4–5–7 progression per decade, formatted with
# k/M suffixes. Used only for the follower_bucket field in detail payloads.
_FOLLOWER_MANTISSAS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0)
_FOLLOWER_EDGES: tuple[float, ...] = (
    *tuple(m * (10**k) for k in range(0, 9) for m in _FOLLOWER_MANTISSAS),
    1e9,  # sentinel above 100M so the 70M–100M bucket closes cleanly.
)


def _format_si(value: float) -> str:
    if value >= 1_000_000:
        n = value / 1_000_000
        suffix = "M"
    elif value >= 1_000:
        n = value / 1_000
        suffix = "k"
    else:
        return f"{int(value)}" if value == int(value) else f"{value:g}"
    s = f"{n:.1f}".rstrip("0").rstrip(".")
    return f"{s}{suffix}"


def bucket_followers(n: int) -> str:
    """Bucket a follower count into a human-readable band string.

    Edges follow a 1–1.5–2–2.5–3–4–5–7 progression per decade up to 100M.
    Anything ≥ 100M collapses to ``"100M+"``. Values < 1 collapse to ``"<1"``.
    """
    if n < 1:
        return "<1"
    if n >= 100_000_000:
        return "100M+"
    for i in range(len(_FOLLOWER_EDGES) - 1):
        lo = _FOLLOWER_EDGES[i]
        hi = _FOLLOWER_EDGES[i + 1]
        if lo <= n < hi:
            return f"{_format_si(lo)}–{_format_si(hi)}"
    raise ValueError(f"unreachable for n={n}")


def aggregate_boolean_share(values: Sequence[bool | None]) -> float:
    """Share of non-None values that are True. Returns 0.0 if all None or empty."""
    denom = sum(1 for v in values if v is not None)
    if denom == 0:
        return 0.0
    numer = sum(1 for v in values if v is True)
    return numer / denom


def parse_label_score_csv(
    labels_csv: str | None, scores_csv: str | None
) -> list[tuple[str, float]]:
    """Parse comma-separated MIR label/score strings into ``(label, score)`` pairs.

    Returns an empty list when either input is empty/None or lengths mismatch.
    Mirrors the writer in ``modules.mir.descriptors.topk_csv``.
    """
    if not labels_csv or not scores_csv:
        return []
    labels = [s.strip() for s in labels_csv.split(",") if s.strip()]
    score_strs = [s.strip() for s in scores_csv.split(",") if s.strip()]
    if len(labels) != len(score_strs):
        return []
    try:
        return [(lab, float(sc)) for lab, sc in zip(labels, score_strs, strict=True)]
    except ValueError:
        return []


def aggregate_label_scores(rows: Sequence[tuple[str, float]], top_k: int) -> list[dict]:
    """Sum scores per label, sort descending, take top_k, normalize so top=1.0.

    Returns ``[{"label", "weight"}]``. Empty input → ``[]``. Weights rounded
    to 4 decimals.
    """
    if not rows:
        return []
    totals: dict[str, float] = defaultdict(float)
    for label, score in rows:
        totals[label] += score
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    top_weight = ranked[0][1]
    if top_weight == 0.0:
        return [{"label": lab, "weight": 0.0} for lab, _ in ranked]
    return [{"label": lab, "weight": round(sc / top_weight, 4)} for lab, sc in ranked]


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float  # radians; principal axis from +x


def fit_cluster_ellipse(xs: np.ndarray, ys: np.ndarray, sigma: float = 2.0) -> Ellipse:
    """2σ covariance ellipse over a cluster's member positions.

    Degenerate input (n<2) returns a tiny non-zero ellipse at the
    centroid so downstream consumers always see rx, ry > 0.
    """
    n = len(xs)
    cx = float(xs.mean())
    cy = float(ys.mean())
    if n < 2:
        return Ellipse(cx=cx, cy=cy, rx=1e-6, ry=1e-6, angle=0.0)
    cov = np.cov(np.vstack([xs, ys]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rx = float(sigma * np.sqrt(max(float(eigvals[0]), 0.0)))
    ry = float(sigma * np.sqrt(max(float(eigvals[1]), 0.0)))
    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return Ellipse(cx=cx, cy=cy, rx=rx, ry=ry, angle=angle)


@dataclass(frozen=True)
class CasePayload:
    case: str
    label: str
    users: list[VisualizationUser]
    clusters: list[VisualizationCluster]


def build_case_payload(
    case: str, label: str, user_rows: Sequence[UserCluster]
) -> CasePayload:
    """Group UserCluster rows by cluster_id, fit ellipses for real
    clusters (id >= 0), skip noise (id == -1) entirely from the cluster
    table. Noise users still appear in `users` with cluster_id = -1.
    """
    users = [
        VisualizationUser(
            user_id=r.user_id,
            embedding_case=case,
            x=float(r.umap_x),
            y=float(r.umap_y),
            cluster_id=int(r.cluster_id),
            centrality=float(r.centrality) if r.centrality is not None else 0.0,
        )
        for r in user_rows
    ]
    by_cluster: dict[int, list[UserCluster]] = defaultdict(list)
    for r in user_rows:
        if r.cluster_id >= 0:
            by_cluster[int(r.cluster_id)].append(r)
    clusters: list[VisualizationCluster] = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        xs = np.fromiter(
            (float(m.umap_x) for m in members), dtype=np.float64, count=len(members)
        )
        ys = np.fromiter(
            (float(m.umap_y) for m in members), dtype=np.float64, count=len(members)
        )
        e = fit_cluster_ellipse(xs, ys)
        clusters.append(
            VisualizationCluster(
                embedding_case=case,
                cluster_id=cid,
                cx=e.cx,
                cy=e.cy,
                rx=e.rx,
                ry=e.ry,
                angle=e.angle,
                size=len(members),
                label=f"Cluster {cid + 1}",
            )
        )
    return CasePayload(case=case, label=label, users=users, clusters=clusters)


def top_languages(codes: Sequence[str | None], top_k: int = 3) -> list[dict]:
    """Top language codes by frequency among non-None values.

    Returns ``[{"code", "share"}]`` with ``share`` rounded to 4 decimals.
    """
    present = [c for c in codes if c]
    if not present:
        return []
    counts: dict[str, int] = defaultdict(int)
    for c in present:
        counts[c] += 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    denom = len(present)
    return [{"code": c, "share": round(n / denom, 4)} for c, n in ranked]


def distinctiveness(
    cohort: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    top_k: int,
    z_min: float,
) -> list[dict]:
    """Per-feature standardized mean diff between cohort and baseline.

    Skips features with empty cohort, empty baseline, or zero baseline std.
    Filters ``|z| < z_min``, returns top_k by ``|z|``, all floats rounded
    to 4 decimals.
    """
    entries: list[dict] = []
    for feature, cohort_arr in cohort.items():
        if cohort_arr.size == 0:
            continue
        base_arr = baseline.get(feature)
        if base_arr is None or base_arr.size == 0:
            continue
        base_mean = float(base_arr.mean())
        base_std = float(base_arr.std())
        if base_std == 0.0:
            continue
        cohort_mean = float(cohort_arr.mean())
        z = (cohort_mean - base_mean) / base_std
        if abs(z) < z_min:
            continue
        entries.append(
            {
                "feature": feature,
                "cohort_value": round(cohort_mean, 4),
                "baseline_mean": round(base_mean, 4),
                "baseline_std": round(base_std, 4),
                "z": round(z, 4),
            }
        )
    entries.sort(key=lambda d: abs(d["z"]), reverse=True)
    return entries[:top_k]


def nearest_clusters(
    centroids: dict[int, tuple[float, float]],
    labels: dict[int, str],
    target_id: int,
    top_k: int = 3,
) -> list[dict]:
    """Top-K clusters nearest to ``target_id`` by Euclidean centroid distance.

    Excludes the target itself. Returns ``[{"cluster_id", "label", "distance"}]``.
    Distance rounded to 4 decimals.
    """
    if target_id not in centroids:
        return []
    tx, ty = centroids[target_id]
    candidates = []
    for cid, (cx, cy) in centroids.items():
        if cid == target_id:
            continue
        d = float(np.hypot(cx - tx, cy - ty))
        candidates.append((d, cid))
    candidates.sort()
    return [
        {"cluster_id": cid, "label": labels.get(cid, ""), "distance": round(d, 4)}
        for d, cid in candidates[:top_k]
    ]


def centroid_percentile(distances: np.ndarray, target_distance: float) -> int:
    """Rank percentile of ``target_distance`` within ``distances``.

    Empty input → 0. Values rank-ordered (smaller distance = lower percentile).
    """
    n = distances.size
    if n == 0:
        return 0
    rank = int(np.sum(distances < target_distance))
    if n == 1:
        return 0
    return min(100, round(100 * rank / (n - 1)))


def compactness(rx: float, ry: float, size: int) -> float:
    """Ellipse area per member: ``π·rx·ry / max(size, 1)``."""
    return float(np.pi * rx * ry / max(size, 1))


@dataclass(frozen=True)
class ClusterDetail:
    cluster_id: int
    label: str
    size: int
    ellipse: dict
    audio: dict
    mood_shares: dict
    timbre_shares: dict
    genre_top: list[dict]
    instrument_top: list[dict]
    speech: dict
    caption: dict
    posting: dict
    follower_bucket: str
    activity_span_months: int | None
    distinctiveness: list[dict]
    spatial: dict

    def to_json(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "cluster_id": self.cluster_id,
            "label": self.label,
            "size": self.size,
            "ellipse": self.ellipse,
            "audio": self.audio,
            "mood_shares": self.mood_shares,
            "timbre_shares": self.timbre_shares,
            "genre_top": self.genre_top,
            "instrument_top": self.instrument_top,
            "speech": self.speech,
            "caption": self.caption,
            "posting": self.posting,
            "follower_bucket": self.follower_bucket,
            "activity_span_months": self.activity_span_months,
            "distinctiveness": self.distinctiveness,
            "spatial": self.spatial,
        }


@dataclass(frozen=True)
class UserDetail:
    user_id: int
    cluster_id: int
    x: float
    y: float
    n_clips: int
    audio: dict
    mood_shares: dict
    timbre_shares: dict
    genre_top: list[dict]
    instrument_top: list[dict]
    speech: dict
    caption: dict
    posting: dict
    follower_bucket: str
    activity_span_months: int | None
    distinctiveness: list[dict]
    spatial: dict

    def to_json(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "user_id": self.user_id,
            "cluster_id": self.cluster_id,
            "x": self.x,
            "y": self.y,
            "n_clips": self.n_clips,
            "audio": self.audio,
            "mood_shares": self.mood_shares,
            "timbre_shares": self.timbre_shares,
            "genre_top": self.genre_top,
            "instrument_top": self.instrument_top,
            "speech": self.speech,
            "caption": self.caption,
            "posting": self.posting,
            "follower_bucket": self.follower_bucket,
            "activity_span_months": self.activity_span_months,
            "distinctiveness": self.distinctiveness,
            "spatial": self.spatial,
        }


@dataclass(frozen=True)
class ClusterMemberClip:
    """Per-clip slice needed by the detail builder. Plain Python types only."""

    approachability: float | None
    engagement: float | None
    danceability: float | None
    is_happy: bool | None
    is_sad: bool | None
    is_relaxed: bool | None
    is_aggressive: bool | None
    is_party: bool | None
    is_acoustic: bool | None
    is_electronic: bool | None
    is_instrumental: bool | None
    is_female_voice: bool | None
    is_bright_timbre: bool | None
    is_tonal: bool | None
    is_speech_detected: bool | None
    speech_language: str | None
    caption_language: str | None
    genre_pairs: list[tuple[str, float]]
    instrument_pairs: list[tuple[str, float]]


@dataclass(frozen=True)
class ClusterMember:
    user_id: int
    follower_count: int | None
    n_clips: int
    median_plays: int | None
    median_clips_per_week: float | None
    engagement_shape_ratio: float | None
    median_video_duration: float | None
    activity_span_months: int | None
    clips: list[ClusterMemberClip]


@dataclass(frozen=True)
class DatasetBaseline:
    """Per-case dataset baselines for distinctiveness."""

    numeric: dict[str, np.ndarray]
    boolean: dict[str, np.ndarray]


_MOOD_FLAGS: tuple[tuple[str, str], ...] = (
    ("happy", "is_happy"),
    ("sad", "is_sad"),
    ("relaxed", "is_relaxed"),
    ("aggressive", "is_aggressive"),
    ("party", "is_party"),
)
_TIMBRE_FLAGS: tuple[tuple[str, str], ...] = (
    ("acoustic", "is_acoustic"),
    ("electronic", "is_electronic"),
    ("instrumental", "is_instrumental"),
    ("female_voice", "is_female_voice"),
    ("bright", "is_bright_timbre"),
    ("tonal", "is_tonal"),
)
_AUDIO_SCORES: tuple[str, ...] = ("approachability", "engagement", "danceability")


def _share_dict(
    clips: Sequence[ClusterMemberClip], flag_pairs: Sequence[tuple[str, str]]
) -> dict[str, float]:
    return {
        out_key: round(aggregate_boolean_share([getattr(c, attr) for c in clips]), 4)
        for out_key, attr in flag_pairs
    }


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return None if arr.size == 0 else float(arr.mean())


def _median_or_none(values: Sequence[float | None]) -> float | None:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return None if arr.size == 0 else float(np.median(arr))


def _cohort_numeric_arrays(
    clips: Sequence[ClusterMemberClip],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in _AUDIO_SCORES:
        out[name] = np.array(
            [getattr(c, name) for c in clips if getattr(c, name) is not None],
            dtype=np.float64,
        )
    return out


def _cohort_boolean_arrays(
    clips: Sequence[ClusterMemberClip],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for _, attr in _MOOD_FLAGS + _TIMBRE_FLAGS:
        out[attr] = np.array(
            [bool(getattr(c, attr)) for c in clips if getattr(c, attr) is not None],
            dtype=np.float64,
        )
    return out


def build_cluster_detail(
    *,
    cluster_id: int,
    cluster_label: str,
    cluster_size: int,
    ellipse: dict,
    members: Sequence[ClusterMember],
    baseline: DatasetBaseline,
    cluster_centroids: dict[int, tuple[float, float]],
    cluster_labels: dict[int, str],
    z_min: float,
    distinctiveness_top_k: int,
    genre_top_k: int,
    instrument_top_k: int,
    languages_top_k: int,
) -> ClusterDetail:
    all_clips = [c for m in members for c in m.clips]
    audio = {
        name: round(v, 4)
        if (v := _mean_or_none([getattr(c, name) for c in all_clips])) is not None
        else 0.0
        for name in _AUDIO_SCORES
    }
    mood_shares = _share_dict(all_clips, _MOOD_FLAGS)
    timbre_shares = _share_dict(all_clips, _TIMBRE_FLAGS)
    genre_top = aggregate_label_scores(
        [p for c in all_clips for p in c.genre_pairs], top_k=genre_top_k
    )
    instrument_top = aggregate_label_scores(
        [p for c in all_clips for p in c.instrument_pairs], top_k=instrument_top_k
    )
    speech = {
        "detected_share": round(
            aggregate_boolean_share([c.is_speech_detected for c in all_clips]), 4
        ),
        "top_langs": top_languages(
            [c.speech_language for c in all_clips], top_k=languages_top_k
        ),
    }
    caption = {
        "top_langs": top_languages(
            [c.caption_language for c in all_clips], top_k=languages_top_k
        )
    }
    posting = {
        "median_plays": round(_median_or_none([m.median_plays for m in members]) or 0),
        "median_clip_duration_s": round(
            _median_or_none([m.median_video_duration for m in members]) or 0.0, 4
        ),
        "median_clips_per_week": round(
            _median_or_none([m.median_clips_per_week for m in members]) or 0.0, 4
        ),
        "engagement_shape_ratio": round(
            _median_or_none([m.engagement_shape_ratio for m in members]) or 0.0, 4
        ),
    }
    median_followers = _median_or_none([m.follower_count for m in members])
    follower_bucket = bucket_followers(int(median_followers or 0))
    activity_span = _median_or_none([m.activity_span_months for m in members])
    activity_span_months = None if activity_span is None else round(activity_span)

    cohort_numeric = _cohort_numeric_arrays(all_clips)
    cohort_boolean = _cohort_boolean_arrays(all_clips)
    distinct = distinctiveness(
        cohort={**cohort_numeric, **cohort_boolean},
        baseline={**baseline.numeric, **baseline.boolean},
        top_k=distinctiveness_top_k,
        z_min=z_min,
    )
    spatial = {
        "compactness": round(
            compactness(ellipse["rx"], ellipse["ry"], cluster_size), 4
        ),
        "nearest_clusters": nearest_clusters(
            cluster_centroids, cluster_labels, target_id=cluster_id, top_k=3
        ),
    }
    return ClusterDetail(
        cluster_id=cluster_id,
        label=cluster_label,
        size=cluster_size,
        ellipse=ellipse,
        audio=audio,
        mood_shares=mood_shares,
        timbre_shares=timbre_shares,
        genre_top=genre_top,
        instrument_top=instrument_top,
        speech=speech,
        caption=caption,
        posting=posting,
        follower_bucket=follower_bucket,
        activity_span_months=activity_span_months,
        distinctiveness=distinct,
        spatial=spatial,
    )


def _cluster_baseline_from_members(
    members: Sequence[ClusterMember],
) -> DatasetBaseline:
    all_clips = [c for m in members for c in m.clips]
    return DatasetBaseline(
        numeric=_cohort_numeric_arrays(all_clips),
        boolean=_cohort_boolean_arrays(all_clips),
    )


def build_user_detail(
    *,
    user_id: int,
    cluster_id: int,
    x: float,
    y: float,
    self_member: ClusterMember,
    own_cluster_members_excl_self: Sequence[ClusterMember],
    own_cluster_centroid: tuple[float, float],
    own_cluster_member_distances: np.ndarray,
    other_cluster_centroids: dict[int, tuple[float, float]],
    other_cluster_labels: dict[int, str],
    edge_percentile: int,
    z_min: float,
    distinctiveness_top_k: int,
    genre_top_k: int,
    instrument_top_k: int,
    languages_top_k: int,
) -> UserDetail:
    clips = self_member.clips
    audio = {
        name: round(v, 4)
        if (v := _mean_or_none([getattr(c, name) for c in clips])) is not None
        else 0.0
        for name in _AUDIO_SCORES
    }
    mood_shares = _share_dict(clips, _MOOD_FLAGS)
    timbre_shares = _share_dict(clips, _TIMBRE_FLAGS)
    genre_top = aggregate_label_scores(
        [p for c in clips for p in c.genre_pairs], top_k=genre_top_k
    )
    instrument_top = aggregate_label_scores(
        [p for c in clips for p in c.instrument_pairs], top_k=instrument_top_k
    )
    speech = {
        "detected_share": round(
            aggregate_boolean_share([c.is_speech_detected for c in clips]), 4
        ),
        "top_langs": top_languages(
            [c.speech_language for c in clips], top_k=languages_top_k
        ),
    }
    caption = {
        "top_langs": top_languages(
            [c.caption_language for c in clips], top_k=languages_top_k
        )
    }
    posting = {
        "median_plays": self_member.median_plays or 0,
        "median_clip_duration_s": round(self_member.median_video_duration or 0.0, 4),
        "median_clips_per_week": round(self_member.median_clips_per_week or 0.0, 4),
        "engagement_shape_ratio": round(self_member.engagement_shape_ratio or 0.0, 4),
    }
    follower_bucket = bucket_followers(self_member.follower_count or 0)

    baseline = _cluster_baseline_from_members(own_cluster_members_excl_self)
    cohort_numeric = _cohort_numeric_arrays(clips)
    cohort_boolean = _cohort_boolean_arrays(clips)
    distinct = distinctiveness(
        cohort={**cohort_numeric, **cohort_boolean},
        baseline={**baseline.numeric, **baseline.boolean},
        top_k=distinctiveness_top_k,
        z_min=z_min,
    )

    dist = float(np.hypot(x - own_cluster_centroid[0], y - own_cluster_centroid[1]))
    percentile = centroid_percentile(own_cluster_member_distances, dist)
    nearest_other = None
    if percentile >= edge_percentile and other_cluster_centroids:
        ranked = sorted(
            (
                (float(np.hypot(x - cx, y - cy)), cid)
                for cid, (cx, cy) in other_cluster_centroids.items()
            )
        )
        d, cid = ranked[0]
        nearest_other = {
            "cluster_id": cid,
            "label": other_cluster_labels.get(cid, ""),
            "distance": round(d, 4),
        }
    spatial = {
        "distance_from_centroid": round(dist, 4),
        "distance_from_centroid_percentile": percentile,
        "nearest_other_cluster": nearest_other,
    }
    return UserDetail(
        user_id=user_id,
        cluster_id=cluster_id,
        x=x,
        y=y,
        n_clips=self_member.n_clips,
        audio=audio,
        mood_shares=mood_shares,
        timbre_shares=timbre_shares,
        genre_top=genre_top,
        instrument_top=instrument_top,
        speech=speech,
        caption=caption,
        posting=posting,
        follower_bucket=follower_bucket,
        activity_span_months=self_member.activity_span_months,
        distinctiveness=distinct,
        spatial=spatial,
    )
