"""Reconstruct version-6 payload dicts from normalised serving rows.

The round-trip inverse of ``core.database.serving_decompose``: each function
rebuilds one payload dict, ordering child collections by ``ord`` and gating
optional blocks on row-presence / NULL, then validates it against the Pydantic
mirror before returning. Detail reconstructors return ``None`` when the id has
no detail row (→ API 404). ``reconstruct_users`` / ``reconstruct_clusters``
raise ``KeyError`` for an unknown run.

Byte parity is owned by ``serialize.to_bytes`` + ``sort_keys=True``; this layer
only needs correct values and array order, not key order.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

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
from modules.visualization.contract import (
    ClusterDetailModel,
    ClustersFileModel,
    CreatorDetailModel,
    ManifestModel,
    UsersFileModel,
)
from modules.visualization.schema import SCHEMA_VERSION


def reconstruct_manifest(session: Session) -> dict:
    runs = session.query(ServingRun).order_by(ServingRun.manifest_ord).all()
    if not runs:
        raise KeyError("no runs in serving database")
    default = next((r.run_id for r in runs if r.is_default), runs[0].run_id)
    # Runs are ordered by `manifest_ord` — the file exporter's `cases` order —
    # so the reconstructed manifest array is byte-identical to the on-disk file.
    payload = {
        "version": SCHEMA_VERSION,
        "default_run_id": default,
        "runs": [
            {
                "id": r.run_id,
                "case": r.case,
                "label": r.label,
                "size": r.size,
                "details_available": r.details_available,
            }
            for r in runs
        ],
    }
    ManifestModel.model_validate(payload)
    return payload


def _require_run(session: Session, run_id: str) -> ServingRun:
    run = session.get(ServingRun, run_id)
    if run is None:
        raise KeyError(run_id)
    return run


def reconstruct_users(session: Session, run_id: str) -> dict:
    _require_run(session, run_id)
    bounds = session.get(ServingRunBounds, run_id)
    rows = (
        session.query(ServingUser)
        .filter_by(run_id=run_id)
        .order_by(ServingUser.user_id)
        .all()
    )
    payload = {
        "version": SCHEMA_VERSION,
        "run_id": run_id,
        "bounds": {
            "minX": bounds.min_x,
            "maxX": bounds.max_x,
            "minY": bounds.min_y,
            "maxY": bounds.max_y,
        },
        "users": [
            [r.user_id, r.x, r.y, r.cluster_id, r.has_detail, r.centrality]
            for r in rows
        ],
    }
    UsersFileModel.model_validate(payload)
    return payload


def reconstruct_clusters(session: Session, run_id: str) -> dict:
    _require_run(session, run_id)
    rows = (
        session.query(ServingCluster)
        .filter_by(run_id=run_id)
        .order_by(ServingCluster.cluster_id)
        .all()
    )
    payload = {
        "version": SCHEMA_VERSION,
        "run_id": run_id,
        "clusters": [
            {
                "id": r.cluster_id,
                "label": r.label,
                "cx": r.cx,
                "cy": r.cy,
                "rx": r.rx,
                "ry": r.ry,
                "angle": r.angle,
                "size": r.size,
                "has_detail": r.has_detail,
            }
            for r in rows
        ],
    }
    ClustersFileModel.model_validate(payload)
    return payload


def _weighted_top(
    session: Session, run_id: str, owner_kind: str, owner_id: int, field: str
) -> list[dict]:
    rows = (
        session.query(ServingWeightedTag)
        .filter_by(run_id=run_id, owner_kind=owner_kind, owner_id=owner_id, field=field)
        .order_by(ServingWeightedTag.ord)
        .all()
    )
    return [{"label": r.label, "weight": r.weight} for r in rows]


def _top_langs(
    session: Session, run_id: str, owner_kind: str, owner_id: int, block: str
) -> list[dict]:
    rows = (
        session.query(ServingLangShare)
        .filter_by(run_id=run_id, owner_kind=owner_kind, owner_id=owner_id, block=block)
        .order_by(ServingLangShare.ord)
        .all()
    )
    return [{"code": r.code, "share": r.share} for r in rows]


def _distinctiveness(
    session: Session, run_id: str, owner_kind: str, owner_id: int
) -> list[dict]:
    rows = (
        session.query(ServingDistinctiveness)
        .filter_by(run_id=run_id, owner_kind=owner_kind, owner_id=owner_id)
        .order_by(ServingDistinctiveness.ord)
        .all()
    )
    return [
        {
            "feature": r.feature,
            "cohort_value": r.cohort_value,
            "baseline_mean": r.baseline_mean,
            "baseline_std": r.baseline_std,
            "z": r.z,
        }
        for r in rows
    ]


def _audio(d) -> dict:
    return {
        "approachability": d.audio_approachability,
        "engagement": d.audio_engagement,
        "danceability": d.audio_danceability,
    }


def _mood(d) -> dict:
    return {
        "happy": d.mood_happy,
        "sad": d.mood_sad,
        "relaxed": d.mood_relaxed,
        "aggressive": d.mood_aggressive,
        "party": d.mood_party,
    }


def _timbre(d) -> dict:
    return {
        "acoustic": d.timbre_acoustic,
        "electronic": d.timbre_electronic,
        "instrumental": d.timbre_instrumental,
        "female_voice": d.timbre_female_voice,
        "bright": d.timbre_bright,
        "tonal": d.timbre_tonal,
    }


def _posting(d) -> dict:
    return {
        "median_plays": d.posting_median_plays,
        "median_clip_duration_s": d.posting_median_clip_duration_s,
        "median_clips_per_week": d.posting_median_clips_per_week,
        "engagement_shape_ratio": d.posting_engagement_shape_ratio,
    }


def _cluster_label_block(session: Session, run_id: str, cluster_id: int) -> dict | None:
    row = session.get(ServingClusterLabel, {"run_id": run_id, "cluster_id": cluster_id})
    if row is None:
        return None
    repertoire = (
        session.query(ServingClusterLabelRepertoire)
        .filter_by(run_id=run_id, cluster_id=cluster_id)
        .order_by(ServingClusterLabelRepertoire.ord)
        .all()
    )
    aesthetic = (
        session.query(ServingClusterLabelAesthetic)
        .filter_by(run_id=run_id, cluster_id=cluster_id)
        .order_by(ServingClusterLabelAesthetic.ord)
        .all()
    )
    variations = (
        session.query(ServingClusterLabelVariations)
        .filter_by(run_id=run_id, cluster_id=cluster_id)
        .order_by(ServingClusterLabelVariations.ord)
        .all()
    )
    tooltags = (
        session.query(ServingClusterLabelTooltag)
        .filter_by(run_id=run_id, cluster_id=cluster_id)
        .order_by(ServingClusterLabelTooltag.ord)
        .all()
    )
    tool_tags = [r.value for r in tooltags if r.kind == "tool_tag"]
    warnings = [r.value for r in tooltags if r.kind == "warning"]
    return {
        "label": row.label,
        "summary": row.summary,
        "modality": row.modality,
        "repertoire": [
            {"tag": r.tag, "description": r.description, "recurrence": r.recurrence}
            for r in repertoire
        ],
        "aesthetic_logic": [
            {"tag": r.tag, "grounded_in": r.grounded_in, "description": r.description}
            for r in aesthetic
        ],
        "taste_signalling": {
            "label": row.taste_signalling_label,
            "description": row.taste_signalling_description,
            "confidence": row.taste_signalling_confidence,
        },
        "visibility_orientation": {
            "label": row.visibility_orientation_label,
            "description": row.visibility_orientation_description,
            "confidence": row.visibility_orientation_confidence,
        },
        "internal_variations": [
            {"variation": r.variation, "description": r.description} for r in variations
        ],
        "boundary_notes": row.boundary_notes,
        "tool_tags": tool_tags,
        "validation": row.validation,
        "warnings": warnings,
    }


def reconstruct_cluster_detail(
    session: Session, run_id: str, cluster_id: int
) -> dict | None:
    d = session.get(ServingClusterDetail, {"run_id": run_id, "cluster_id": cluster_id})
    if d is None:
        return None
    nearest = (
        session.query(ServingClusterNearest)
        .filter_by(run_id=run_id, cluster_id=cluster_id)
        .order_by(ServingClusterNearest.ord)
        .all()
    )
    payload = {
        "version": SCHEMA_VERSION,
        "cluster_id": cluster_id,
        "size": d.size,
        "ellipse": {
            "cx": d.ellipse_cx,
            "cy": d.ellipse_cy,
            "rx": d.ellipse_rx,
            "ry": d.ellipse_ry,
            "angle": d.ellipse_angle,
        },
        "audio": _audio(d),
        "mood_shares": _mood(d),
        "timbre_shares": _timbre(d),
        "genre_top": _weighted_top(session, run_id, "cluster", cluster_id, "genre"),
        "instrument_top": _weighted_top(
            session, run_id, "cluster", cluster_id, "instrument"
        ),
        "speech": {
            "detected_share": d.speech_detected_share,
            "top_langs": _top_langs(session, run_id, "cluster", cluster_id, "speech"),
        },
        "caption": {
            "detected_share": d.caption_detected_share,
            "top_langs": _top_langs(session, run_id, "cluster", cluster_id, "caption"),
        },
        "posting": _posting(d),
        "follower_bucket": d.follower_bucket,
        "activity_span_months": d.activity_span_months,
        "distinctiveness": _distinctiveness(session, run_id, "cluster", cluster_id),
        "spatial": {
            "compactness": d.spatial_compactness,
            "nearest_clusters": [
                {
                    "cluster_id": r.nearest_cluster_id,
                    "label": r.label,
                    "distance": r.distance,
                }
                for r in nearest
            ],
        },
    }
    label = _cluster_label_block(session, run_id, cluster_id)
    if label is not None:
        payload["label"] = label
    ClusterDetailModel.model_validate(payload)
    return payload


def _clip_block(session: Session, run_id: str, user_id: int) -> list[dict]:
    clips = (
        session.query(ServingUserClip)
        .filter_by(run_id=run_id, user_id=user_id)
        .order_by(ServingUserClip.ord)
        .all()
    )
    tags = (
        session.query(ServingUserClipTag)
        .filter_by(run_id=run_id, user_id=user_id)
        .order_by(ServingUserClipTag.clip_ord, ServingUserClipTag.ord)
        .all()
    )
    warnings = (
        session.query(ServingUserClipWarning)
        .filter_by(run_id=run_id, user_id=user_id)
        .order_by(ServingUserClipWarning.clip_ord, ServingUserClipWarning.ord)
        .all()
    )
    out: list[dict] = []
    for clip in clips:
        clip_tags = [t for t in tags if t.clip_ord == clip.ord]
        observable = [
            {"tag": t.tag, "evidence": t.evidence}
            for t in clip_tags
            if t.kind == "observable"
        ]
        aesthetic = [
            {"tag": t.tag, "grounded_in": t.grounded_in, "confidence": t.confidence}
            for t in clip_tags
            if t.kind == "aesthetic"
        ]
        community = [
            {"tag": t.tag, "grounded_in": t.grounded_in, "confidence": t.confidence}
            for t in clip_tags
            if t.kind == "community"
        ]
        out.append(
            {
                "clip_id": clip.clip_id,
                "shortcode": clip.shortcode,
                "thumbnail_url": clip.thumbnail_url,
                "sentence": clip.sentence,
                "tags": {
                    "observable": observable,
                    "aesthetic": aesthetic,
                    "community": community,
                },
                "validation": clip.validation,
                "warnings": [w.value for w in warnings if w.clip_ord == clip.ord],
            }
        )
    return out


def reconstruct_creator_detail(
    session: Session, run_id: str, user_id: int
) -> dict | None:
    d = session.get(ServingUserDetail, {"run_id": run_id, "user_id": user_id})
    if d is None:
        return None
    if d.spatial_nearest_cluster_id is None:
        nearest_other = None
    else:
        nearest_other = {
            "cluster_id": d.spatial_nearest_cluster_id,
            "label": d.spatial_nearest_label,
            "distance": d.spatial_nearest_distance,
        }
    payload = {
        "version": SCHEMA_VERSION,
        "user_id": user_id,
        "cluster_id": d.cluster_id,
        "x": d.x,
        "y": d.y,
        "n_clips": d.n_clips,
        "audio": _audio(d),
        "mood_shares": _mood(d),
        "timbre_shares": _timbre(d),
        "genre_top": _weighted_top(session, run_id, "user", user_id, "genre"),
        "instrument_top": _weighted_top(session, run_id, "user", user_id, "instrument"),
        "speech": {
            "detected_share": d.speech_detected_share,
            "top_langs": _top_langs(session, run_id, "user", user_id, "speech"),
        },
        "caption": {
            "detected_share": d.caption_detected_share,
            "top_langs": _top_langs(session, run_id, "user", user_id, "caption"),
        },
        "posting": _posting(d),
        "follower_bucket": d.follower_bucket,
        "activity_span_months": d.activity_span_months,
        "distinctiveness": _distinctiveness(session, run_id, "user", user_id),
        "spatial": {
            "distance_from_centroid": d.spatial_distance_from_centroid,
            "distance_from_centroid_percentile": d.spatial_distance_percentile,
            "nearest_other_cluster": nearest_other,
        },
        "clips": _clip_block(session, run_id, user_id),
    }
    CreatorDetailModel.model_validate(payload)
    return payload
