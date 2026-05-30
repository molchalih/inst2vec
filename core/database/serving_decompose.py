"""Decompose version-6 payload dicts into normalised serving rows.

The exact inverse of the atlas API's reconstruction layer. The offload calls
``decompose_run`` with the builder's ``CasePayloadBundle`` and gets back every
``serving_*`` ORM instance for that run; both directions round-trip the
builder dict losslessly (guarded by the golden-equality test).

Pure: takes plain dicts + the run id, returns ORM instances. No session, no IO.
Array order is captured into dense 0-based ``ord`` columns in the payload's
own array order (which is the exporter's emission order), so reconstruction
can restore it without re-sorting.
"""

from __future__ import annotations

from typing import Any

from core.database.serving_models import (
    ServingCluster,
    ServingClusterDetail,
    ServingClusterLabel,
    ServingClusterLabelAesthetic,
    ServingClusterLabelRepertoire,
    ServingClusterLabelTooltag,
    ServingClusterLabelVariations,
    ServingClusterNearest,
    ServingDistinctiveness,
    ServingLangShare,
    ServingRun,
    ServingRunBounds,
    ServingUser,
    ServingUserClip,
    ServingUserClipTag,
    ServingUserClipWarning,
    ServingUserDetail,
    ServingWeightedTag,
)
from modules.visualization.schema import SCHEMA_VERSION


def _weighted_tags(
    run_id: str, owner_kind: str, owner_id: int, detail: dict
) -> list[ServingWeightedTag]:
    rows: list[ServingWeightedTag] = []
    for field in ("genre", "instrument"):
        for ord_, entry in enumerate(detail[f"{field}_top"]):
            rows.append(
                ServingWeightedTag(
                    run_id=run_id,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    field=field,
                    ord=ord_,
                    label=entry["label"],
                    weight=entry["weight"],
                )
            )
    return rows


def _lang_shares(
    run_id: str, owner_kind: str, owner_id: int, detail: dict
) -> list[ServingLangShare]:
    rows: list[ServingLangShare] = []
    for block in ("speech", "caption"):
        for ord_, entry in enumerate(detail[block]["top_langs"]):
            rows.append(
                ServingLangShare(
                    run_id=run_id,
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    block=block,
                    ord=ord_,
                    code=entry["code"],
                    share=entry["share"],
                )
            )
    return rows


def _distinctiveness(
    run_id: str, owner_kind: str, owner_id: int, detail: dict
) -> list[ServingDistinctiveness]:
    return [
        ServingDistinctiveness(
            run_id=run_id,
            owner_kind=owner_kind,
            owner_id=owner_id,
            ord=ord_,
            feature=e["feature"],
            cohort_value=e["cohort_value"],
            baseline_mean=e["baseline_mean"],
            baseline_std=e["baseline_std"],
            z=e["z"],
        )
        for ord_, e in enumerate(detail["distinctiveness"])
    ]


def _cluster_label_rows(run_id: str, cluster_id: int, label: dict) -> list[Any]:
    rows: list[Any] = [
        ServingClusterLabel(
            run_id=run_id,
            cluster_id=cluster_id,
            label=label["label"],
            summary=label["summary"],
            modality=label["modality"],
            taste_signalling_label=label["taste_signalling"]["label"],
            taste_signalling_description=label["taste_signalling"]["description"],
            taste_signalling_confidence=label["taste_signalling"]["confidence"],
            visibility_orientation_label=label["visibility_orientation"]["label"],
            visibility_orientation_description=label["visibility_orientation"][
                "description"
            ],
            visibility_orientation_confidence=label["visibility_orientation"][
                "confidence"
            ],
            boundary_notes=label["boundary_notes"],
            validation=label["validation"],
        )
    ]
    for ord_, e in enumerate(label["repertoire"]):
        rows.append(
            ServingClusterLabelRepertoire(
                run_id=run_id,
                cluster_id=cluster_id,
                ord=ord_,
                tag=e["tag"],
                description=e["description"],
                recurrence=e["recurrence"],
            )
        )
    for ord_, e in enumerate(label["aesthetic_logic"]):
        rows.append(
            ServingClusterLabelAesthetic(
                run_id=run_id,
                cluster_id=cluster_id,
                ord=ord_,
                tag=e["tag"],
                grounded_in=e["grounded_in"],
                description=e["description"],
            )
        )
    for ord_, e in enumerate(label["internal_variations"]):
        rows.append(
            ServingClusterLabelVariations(
                run_id=run_id,
                cluster_id=cluster_id,
                ord=ord_,
                variation=e["variation"],
                description=e["description"],
            )
        )
    for kind, key in (("tool_tag", "tool_tags"), ("warning", "warnings")):
        for ord_, value in enumerate(label[key]):
            rows.append(
                ServingClusterLabelTooltag(
                    run_id=run_id,
                    cluster_id=cluster_id,
                    kind=kind,
                    ord=ord_,
                    value=value,
                )
            )
    return rows


def _cluster_detail_rows(run_id: str, cluster_id: int, detail: dict) -> list[Any]:
    ell = detail["ellipse"]
    audio = detail["audio"]
    mood = detail["mood_shares"]
    timbre = detail["timbre_shares"]
    posting = detail["posting"]
    rows: list[Any] = [
        ServingClusterDetail(
            run_id=run_id,
            cluster_id=cluster_id,
            size=detail["size"],
            ellipse_cx=ell["cx"],
            ellipse_cy=ell["cy"],
            ellipse_rx=ell["rx"],
            ellipse_ry=ell["ry"],
            ellipse_angle=ell["angle"],
            audio_approachability=audio["approachability"],
            audio_engagement=audio["engagement"],
            audio_danceability=audio["danceability"],
            mood_happy=mood["happy"],
            mood_sad=mood["sad"],
            mood_relaxed=mood["relaxed"],
            mood_aggressive=mood["aggressive"],
            mood_party=mood["party"],
            timbre_acoustic=timbre["acoustic"],
            timbre_electronic=timbre["electronic"],
            timbre_instrumental=timbre["instrumental"],
            timbre_female_voice=timbre["female_voice"],
            timbre_bright=timbre["bright"],
            timbre_tonal=timbre["tonal"],
            speech_detected_share=detail["speech"]["detected_share"],
            caption_detected_share=detail["caption"].get("detected_share"),
            posting_median_plays=posting["median_plays"],
            posting_median_clip_duration_s=posting["median_clip_duration_s"],
            posting_median_clips_per_week=posting["median_clips_per_week"],
            posting_engagement_shape_ratio=posting["engagement_shape_ratio"],
            follower_bucket=detail["follower_bucket"],
            activity_span_months=detail["activity_span_months"],
            spatial_compactness=detail["spatial"]["compactness"],
        )
    ]
    rows += _weighted_tags(run_id, "cluster", cluster_id, detail)
    rows += _lang_shares(run_id, "cluster", cluster_id, detail)
    rows += _distinctiveness(run_id, "cluster", cluster_id, detail)
    for ord_, e in enumerate(detail["spatial"]["nearest_clusters"]):
        rows.append(
            ServingClusterNearest(
                run_id=run_id,
                cluster_id=cluster_id,
                ord=ord_,
                nearest_cluster_id=e["cluster_id"],
                label=e["label"],
                distance=e["distance"],
            )
        )
    label = detail.get("label")
    if label is not None:
        rows += _cluster_label_rows(run_id, cluster_id, label)
    return rows


def _clip_rows(run_id: str, user_id: int, clips: list[dict]) -> list[Any]:
    rows: list[Any] = []
    for clip_ord, clip in enumerate(clips):
        rows.append(
            ServingUserClip(
                run_id=run_id,
                user_id=user_id,
                ord=clip_ord,
                clip_id=clip["clip_id"],
                shortcode=clip["shortcode"],
                thumbnail_url=clip["thumbnail_url"],
                sentence=clip["sentence"],
                validation=clip["validation"],
            )
        )
        tags = clip["tags"]
        for ord_, t in enumerate(tags["observable"]):
            rows.append(
                ServingUserClipTag(
                    run_id=run_id,
                    user_id=user_id,
                    clip_ord=clip_ord,
                    kind="observable",
                    ord=ord_,
                    tag=t["tag"],
                    evidence=t["evidence"],
                    grounded_in=None,
                    confidence=None,
                )
            )
        for kind in ("aesthetic", "community"):
            for ord_, t in enumerate(tags[kind]):
                rows.append(
                    ServingUserClipTag(
                        run_id=run_id,
                        user_id=user_id,
                        clip_ord=clip_ord,
                        kind=kind,
                        ord=ord_,
                        tag=t["tag"],
                        evidence=None,
                        grounded_in=t["grounded_in"],
                        confidence=t["confidence"],
                    )
                )
        for ord_, w in enumerate(clip["warnings"]):
            rows.append(
                ServingUserClipWarning(
                    run_id=run_id,
                    user_id=user_id,
                    clip_ord=clip_ord,
                    ord=ord_,
                    value=w,
                )
            )
    return rows


def _user_detail_rows(run_id: str, user_id: int, detail: dict) -> list[Any]:
    audio = detail["audio"]
    mood = detail["mood_shares"]
    timbre = detail["timbre_shares"]
    posting = detail["posting"]
    spatial = detail["spatial"]
    nearest = spatial["nearest_other_cluster"]
    rows: list[Any] = [
        ServingUserDetail(
            run_id=run_id,
            user_id=user_id,
            x=detail["x"],
            y=detail["y"],
            n_clips=detail["n_clips"],
            cluster_id=detail["cluster_id"],
            audio_approachability=audio["approachability"],
            audio_engagement=audio["engagement"],
            audio_danceability=audio["danceability"],
            mood_happy=mood["happy"],
            mood_sad=mood["sad"],
            mood_relaxed=mood["relaxed"],
            mood_aggressive=mood["aggressive"],
            mood_party=mood["party"],
            timbre_acoustic=timbre["acoustic"],
            timbre_electronic=timbre["electronic"],
            timbre_instrumental=timbre["instrumental"],
            timbre_female_voice=timbre["female_voice"],
            timbre_bright=timbre["bright"],
            timbre_tonal=timbre["tonal"],
            speech_detected_share=detail["speech"]["detected_share"],
            caption_detected_share=detail["caption"].get("detected_share"),
            posting_median_plays=posting["median_plays"],
            posting_median_clip_duration_s=posting["median_clip_duration_s"],
            posting_median_clips_per_week=posting["median_clips_per_week"],
            posting_engagement_shape_ratio=posting["engagement_shape_ratio"],
            follower_bucket=detail["follower_bucket"],
            activity_span_months=detail["activity_span_months"],
            spatial_distance_from_centroid=spatial["distance_from_centroid"],
            spatial_distance_percentile=spatial["distance_from_centroid_percentile"],
            spatial_nearest_cluster_id=(
                nearest["cluster_id"] if nearest is not None else None
            ),
            spatial_nearest_label=(nearest["label"] if nearest is not None else None),
            spatial_nearest_distance=(
                nearest["distance"] if nearest is not None else None
            ),
        )
    ]
    rows += _weighted_tags(run_id, "user", user_id, detail)
    rows += _lang_shares(run_id, "user", user_id, detail)
    rows += _distinctiveness(run_id, "user", user_id, detail)
    rows += _clip_rows(run_id, user_id, detail["clips"])
    return rows


def decompose_run(bundle, *, is_default: bool, manifest_ord: int) -> list[Any]:
    """All ``serving_*`` ORM instances for one run (the bundle's case).

    ``manifest_ord`` records the run's position in the file exporter's manifest
    (its index in the exposed-case list) so the reconstructed manifest keeps
    that order for byte parity.
    """
    run_id = bundle.case
    rows: list[Any] = []

    rows.append(
        ServingRun(
            run_id=run_id,
            case=bundle.manifest_entry["case"],
            label=bundle.manifest_entry["label"],
            size=bundle.manifest_entry["size"],
            details_available=bundle.manifest_entry["details_available"],
            manifest_ord=manifest_ord,
            is_default=is_default,
            schema_version=SCHEMA_VERSION,
        )
    )
    b = bundle.users["bounds"]
    rows.append(
        ServingRunBounds(
            run_id=run_id,
            min_x=b["minX"],
            max_x=b["maxX"],
            min_y=b["minY"],
            max_y=b["maxY"],
        )
    )
    for u in bundle.users["users"]:
        rows.append(
            ServingUser(
                run_id=run_id,
                user_id=u[0],
                x=u[1],
                y=u[2],
                cluster_id=u[3],
                has_detail=u[4],
                centrality=u[5],
            )
        )
    for c in bundle.clusters["clusters"]:
        rows.append(
            ServingCluster(
                run_id=run_id,
                cluster_id=c["id"],
                label=c["label"],
                cx=c["cx"],
                cy=c["cy"],
                rx=c["rx"],
                ry=c["ry"],
                angle=c["angle"],
                size=c["size"],
                has_detail=c["has_detail"],
            )
        )
    for cluster_id, detail in bundle.cluster_details.items():
        rows += _cluster_detail_rows(run_id, cluster_id, detail)
    for user_id, detail in bundle.creator_details.items():
        rows += _user_detail_rows(run_id, user_id, detail)
    return rows


# Tables to clear (in FK-safe order) when pruning a run before re-inserting.
SERVING_TABLES_PRUNE_ORDER: tuple[Any, ...] = (
    ServingUserClipWarning,
    ServingUserClipTag,
    ServingUserClip,
    ServingClusterLabelTooltag,
    ServingClusterLabelVariations,
    ServingClusterLabelAesthetic,
    ServingClusterLabelRepertoire,
    ServingClusterLabel,
    ServingClusterNearest,
    ServingDistinctiveness,
    ServingLangShare,
    ServingWeightedTag,
    ServingUserDetail,
    ServingClusterDetail,
    ServingUser,
    ServingCluster,
    ServingRunBounds,
    ServingRun,
)
