"""Tests for detail-payload pure helpers in modules.visualization.compute."""

from __future__ import annotations

import numpy as np
import pytest

from modules.visualization.compute import (
    ClusterDetail,
    ClusterMember,
    ClusterMemberClip,
    DatasetBaseline,
    UserDetail,
    aggregate_boolean_share,
    aggregate_label_scores,
    bucket_followers,
    build_cluster_detail,
    build_user_detail,
    centroid_percentile,
    compactness,
    distinctiveness,
    nearest_clusters,
    parse_label_score_csv,
    top_languages,
)
from modules.visualization.schema import SCHEMA_VERSION


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "<1"),
        (1, "1–1.5"),
        (999, "700–1k"),
        (1_000, "1k–1.5k"),
        (1_499, "1k–1.5k"),
        (1_500, "1.5k–2k"),
        (9_999, "7k–10k"),
        (10_000, "10k–15k"),
        (99_999, "70k–100k"),
        (100_000, "100k–150k"),
        (1_500_000, "1.5M–2M"),
        (99_999_999, "70M–100M"),
        (100_000_000, "100M+"),
        (5_000_000_000, "100M+"),
    ],
)
def test_bucket_followers_known_values(n, expected):
    assert bucket_followers(n) == expected


def test_aggregate_boolean_share_mixed():
    assert aggregate_boolean_share([True, True, False, None]) == pytest.approx(2 / 3)


def test_aggregate_boolean_share_all_none_is_zero():
    assert aggregate_boolean_share([None, None]) == 0.0


def test_aggregate_boolean_share_empty_is_zero():
    assert aggregate_boolean_share([]) == 0.0


def test_aggregate_boolean_share_all_true():
    assert aggregate_boolean_share([True, True, True]) == 1.0


def test_parse_label_score_csv_basic():
    pairs = parse_label_score_csv("house, techno, pop", "0.85, 0.12, 0.03")
    assert pairs == [("house", 0.85), ("techno", 0.12), ("pop", 0.03)]


def test_parse_label_score_csv_none_inputs_returns_empty():
    assert parse_label_score_csv(None, "0.5") == []
    assert parse_label_score_csv("a", None) == []
    assert parse_label_score_csv(None, None) == []
    assert parse_label_score_csv("", "") == []


def test_parse_label_score_csv_mismatched_lengths_returns_empty():
    # Defensive: never raise from a single bad MIR row.
    assert parse_label_score_csv("a, b", "0.5") == []


def test_aggregate_label_scores_sums_and_normalizes():
    rows = [
        ("house", 0.5),
        ("techno", 0.3),
        ("house", 0.4),
        ("pop", 0.2),
    ]
    out = aggregate_label_scores(rows, top_k=3)
    # house=0.9, techno=0.3, pop=0.2; normalized so top=1.0.
    assert out == [
        {"label": "house", "weight": 1.0},
        {"label": "techno", "weight": pytest.approx(0.3 / 0.9, abs=1e-4)},
        {"label": "pop", "weight": pytest.approx(0.2 / 0.9, abs=1e-4)},
    ]


def test_aggregate_label_scores_top_k_truncates():
    rows = [("a", 0.5), ("b", 0.4), ("c", 0.3), ("d", 0.2), ("e", 0.1)]
    assert [d["label"] for d in aggregate_label_scores(rows, top_k=2)] == ["a", "b"]


def test_aggregate_label_scores_empty_returns_empty():
    assert aggregate_label_scores([], top_k=5) == []


def test_aggregate_label_scores_zero_top_score_does_not_div_zero():
    # Score 0 is unusual but should not blow up.
    assert aggregate_label_scores([("a", 0.0)], top_k=1) == [
        {"label": "a", "weight": 0.0}
    ]


def test_top_languages_ranks_and_rounds_share():
    out = top_languages(["en", "en", "es", "en", "es", "pt"], top_k=2)
    # en=3/6=0.5, es=2/6≈0.3333, pt=1/6 (cut by top_k=2).
    assert out == [
        {"code": "en", "share": 0.5},
        {"code": "es", "share": 0.3333},
    ]


def test_top_languages_ignores_none():
    out = top_languages([None, "en", None, "en"], top_k=3)
    assert out == [{"code": "en", "share": 1.0}]


def test_top_languages_all_none_returns_empty():
    assert top_languages([None, None]) == []


def test_top_languages_empty_returns_empty():
    assert top_languages([]) == []


def test_distinctiveness_ranks_by_abs_z_and_returns_raw_stats():
    cohort = {
        "danceability": np.array([0.6, 0.6, 0.6]),
        "approach": np.array([0.5, 0.5, 0.5]),
    }
    baseline = {
        "danceability": np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        + np.array([0.05, -0.05, 0.05, -0.05, 0.05, -0.05]),
        "approach": np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50]),
    }
    out = distinctiveness(cohort, baseline, top_k=2, z_min=0.0)
    assert len(out) == 1  # approach has std=0 → skipped
    d = out[0]
    assert d["feature"] == "danceability"
    assert d["cohort_value"] == pytest.approx(0.6, abs=1e-4)
    assert d["baseline_mean"] == pytest.approx(0.4, abs=1e-4)
    assert d["z"] > 0


def test_distinctiveness_filters_below_z_min():
    cohort = {"a": np.array([1.0, 1.0])}
    baseline = {"a": np.array([0.0, 1.0, 2.0])}  # mean=1, std≈0.816
    out = distinctiveness(cohort, baseline, top_k=3, z_min=5.0)
    assert out == []


def test_distinctiveness_skips_empty_cohort():
    out = distinctiveness(
        {"a": np.array([])}, {"a": np.array([0.0, 1.0])}, top_k=1, z_min=0.0
    )
    assert out == []


def test_nearest_clusters_excludes_target_and_orders_by_distance():
    centroids = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 2.0), 3: (3.0, 4.0)}
    labels = {0: "C0", 1: "C1", 2: "C2", 3: "C3"}
    out = nearest_clusters(centroids, labels, target_id=0, top_k=2)
    assert [d["cluster_id"] for d in out] == [1, 2]
    assert out[0]["distance"] == pytest.approx(1.0)
    assert out[1]["distance"] == pytest.approx(2.0)
    assert out[0]["label"] == "C1"


def test_nearest_clusters_top_k_clamped_to_available():
    centroids = {0: (0.0, 0.0), 1: (1.0, 0.0)}
    labels = {0: "C0", 1: "C1"}
    out = nearest_clusters(centroids, labels, target_id=0, top_k=5)
    assert len(out) == 1


def test_centroid_percentile_bounds():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert centroid_percentile(d, 0.0) == 0
    assert centroid_percentile(d, 4.0) == 100
    assert 30 < centroid_percentile(d, 2.0) < 70


def test_compactness_zero_size_does_not_div_zero():
    assert compactness(1.0, 1.0, 0) > 0


def test_compactness_known_value():
    import math

    assert compactness(0.82, 0.31, 42) == pytest.approx(
        math.pi * 0.82 * 0.31 / 42, rel=1e-6
    )


def test_cluster_detail_to_json_shape():
    d = ClusterDetail(
        cluster_id=7,
        label="Cluster 8",
        size=42,
        ellipse={"cx": 1.0, "cy": 2.0, "rx": 0.5, "ry": 0.3, "angle": 0.1},
        audio={"approachability": 0.6, "engagement": 0.7, "danceability": 0.5},
        mood_shares={"happy": 0.4},
        timbre_shares={"acoustic": 0.2},
        genre_top=[{"label": "house", "weight": 1.0}],
        instrument_top=[],
        speech={"detected_share": 0.3, "top_langs": [{"code": "en", "share": 1.0}]},
        caption={"top_langs": []},
        posting={
            "median_plays": 100,
            "median_clip_duration_s": 20.0,
            "median_clips_per_week": 2.0,
            "engagement_shape_ratio": 1.5,
        },
        follower_bucket="10k–15k",
        activity_span_months=12,
        distinctiveness=[],
        spatial={"compactness": 0.01, "nearest_clusters": []},
    )
    j = d.to_json()
    assert j["version"] == SCHEMA_VERSION
    assert j["cluster_id"] == 7
    assert j["label"] == "Cluster 8"
    assert set(j["audio"].keys()) == {"approachability", "engagement", "danceability"}
    assert j["follower_bucket"] == "10k–15k"


def test_user_detail_to_json_shape():
    d = UserDetail(
        user_id=12345,
        cluster_id=7,
        x=1.0,
        y=-1.0,
        n_clips=9,
        audio={"approachability": 0.5, "engagement": 0.6, "danceability": 0.5},
        mood_shares={},
        timbre_shares={},
        genre_top=[],
        instrument_top=[],
        speech={"detected_share": 0.0, "top_langs": []},
        caption={"top_langs": []},
        posting={
            "median_plays": 50,
            "median_clip_duration_s": 10.0,
            "median_clips_per_week": 1.0,
            "engagement_shape_ratio": 1.2,
        },
        follower_bucket="1k–1.5k",
        activity_span_months=3,
        distinctiveness=[],
        spatial={
            "distance_from_centroid": 0.1,
            "distance_from_centroid_percentile": 25,
            "nearest_other_cluster": None,
        },
    )
    j = d.to_json()
    assert j["version"] == SCHEMA_VERSION
    assert j["user_id"] == 12345
    assert j["spatial"]["nearest_other_cluster"] is None


def _baseline_zero_var() -> DatasetBaseline:
    # Single feature with std > 0 so distinctiveness can produce entries.
    return DatasetBaseline(
        numeric={"danceability": np.array([0.0, 0.5, 1.0])},
        boolean={"is_electronic": np.array([False, False, True])},
    )


def _member(
    uid: int, *, follower_count: int = 12_000, clip_count: int = 3
) -> ClusterMember:
    return ClusterMember(
        user_id=uid,
        follower_count=follower_count,
        n_clips=clip_count,
        median_plays=10_000,
        median_clips_per_week=2.0,
        engagement_shape_ratio=1.5,
        median_video_duration=20.0,
        activity_span_months=6,
        clips=[
            ClusterMemberClip(
                approachability=0.6,
                engagement=0.7,
                danceability=0.5,
                is_happy=True,
                is_sad=False,
                is_relaxed=False,
                is_aggressive=False,
                is_party=True,
                is_acoustic=False,
                is_electronic=True,
                is_instrumental=False,
                is_female_voice=True,
                is_bright_timbre=True,
                is_tonal=True,
                is_speech_detected=False,
                speech_language=None,
                caption_language="en",
                genre_pairs=[("house", 0.9)],
                instrument_pairs=[("synth", 0.8)],
            )
            for _ in range(clip_count)
        ],
    )


def test_build_cluster_detail_basic_shape():
    members = [_member(uid=1), _member(uid=2)]
    cluster_centroids = {7: (1.0, 2.0), 1: (1.5, 2.5)}
    cluster_labels = {7: "Cluster 8", 1: "Cluster 2"}
    detail = build_cluster_detail(
        cluster_id=7,
        cluster_label="Cluster 8",
        cluster_size=2,
        ellipse={"cx": 1.0, "cy": 2.0, "rx": 0.5, "ry": 0.3, "angle": 0.0},
        members=members,
        baseline=_baseline_zero_var(),
        cluster_centroids=cluster_centroids,
        cluster_labels=cluster_labels,
        z_min=0.0,
        distinctiveness_top_k=3,
        genre_top_k=5,
        instrument_top_k=3,
        languages_top_k=3,
    )
    j = detail.to_json()
    assert j["cluster_id"] == 7
    assert j["size"] == 2
    assert j["audio"]["danceability"] == pytest.approx(0.5)
    assert j["timbre_shares"]["electronic"] == 1.0
    assert j["mood_shares"]["happy"] == 1.0
    assert j["genre_top"][0]["label"] == "house"
    assert j["follower_bucket"] == "10k–15k"
    assert j["activity_span_months"] == 6
    assert j["spatial"]["compactness"] > 0
    assert j["spatial"]["nearest_clusters"] == [
        {
            "cluster_id": 1,
            "label": "Cluster 2",
            "distance": pytest.approx(0.7071, abs=1e-3),
        }
    ]
    # All clips have caption_language="en" → coverage is 1.0 over all clips,
    # not 1.0 over only-captioned clips.
    assert j["caption"]["detected_share"] == pytest.approx(1.0)
    assert j["caption"]["top_langs"] == [{"code": "en", "share": 1.0}]


def test_build_user_detail_borderline_dot_includes_nearest_other():
    self_member = _member(uid=99)
    cluster_members = [_member(uid=1), _member(uid=2), _member(uid=3)]
    detail = build_user_detail(
        user_id=99,
        cluster_id=7,
        x=2.5,  # far from cluster centroid at (1.0, 2.0)
        y=4.0,
        self_member=self_member,
        own_cluster_members_excl_self=cluster_members,
        own_cluster_centroid=(1.0, 2.0),
        own_cluster_member_distances=np.array([0.1, 0.2, 0.3]),
        other_cluster_centroids={1: (3.0, 4.0)},
        other_cluster_labels={1: "Cluster 2"},
        edge_percentile=66,
        z_min=0.0,
        distinctiveness_top_k=3,
        genre_top_k=5,
        instrument_top_k=3,
        languages_top_k=3,
    )
    j = detail.to_json()
    assert j["user_id"] == 99
    assert j["cluster_id"] == 7
    assert j["n_clips"] == 3
    assert j["follower_bucket"] == "10k–15k"
    assert j["spatial"]["distance_from_centroid_percentile"] == 100
    # Past edge → nearest_other_cluster shipped.
    assert j["spatial"]["nearest_other_cluster"] == {
        "cluster_id": 1,
        "label": "Cluster 2",
        "distance": pytest.approx(float(np.hypot(2.5 - 3.0, 4.0 - 4.0)), abs=1e-3),
    }


def test_build_user_detail_core_dot_no_nearest_other():
    self_member = _member(uid=99)
    cluster_members = [_member(uid=1), _member(uid=2), _member(uid=3)]
    detail = build_user_detail(
        user_id=99,
        cluster_id=7,
        x=1.0,
        y=2.0,
        self_member=self_member,
        own_cluster_members_excl_self=cluster_members,
        own_cluster_centroid=(1.0, 2.0),
        own_cluster_member_distances=np.array([0.5, 1.0, 1.5]),
        other_cluster_centroids={1: (3.0, 4.0)},
        other_cluster_labels={1: "Cluster 2"},
        edge_percentile=66,
        z_min=0.0,
        distinctiveness_top_k=3,
        genre_top_k=5,
        instrument_top_k=3,
        languages_top_k=3,
    )
    j = detail.to_json()
    assert j["spatial"]["distance_from_centroid"] == pytest.approx(0.0, abs=1e-6)
    assert j["spatial"]["nearest_other_cluster"] is None
