"""JSON exporter for the visualization stage.

Reads the three visualization tables and writes the bulk-JSON tree the
frontend expects (manifest.json + runs/{case}/{users,clusters}.json).
Filters out cases whose EmbeddingCaseSpec.expose_to_viewer is False so
hidden cases never leak into the user-facing artifact.

Stale run directories and manifests from previous exports are pruned so
the on-disk tree always reflects the current DB / case set.

No fingerprint: always runs after the DB-write stage. A schema bump
(see schema.py SCHEMA_VERSION) thus rewrites every file on the next
pipeline call with no DB change.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.database import (
    AudioMIR,
    Clip,
    ClipLabel,
    ClusterLabel,
    User,
    UserCluster,
    UserStats,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_session,
)
from core.log import event, scope
from modules.embeddings.cases import CASE_REGISTRY
from modules.labels.cases import REGISTRY as LABEL_CASE_REGISTRY
from modules.labels.validation import clip_role_keys
from modules.visualization.cluster_label_render import (
    display_label_for,
    render_label_block,
)
from modules.visualization.compute import (
    ClusterMember,
    ClusterMemberClip,
    _cluster_baseline_from_members,
    build_cluster_detail,
    build_user_detail,
)
from modules.visualization.schema import SCHEMA_VERSION

# Placeholder thumbnail served for every clip for now: we deliberately do NOT
# emit real per-clip thumbnail URLs into the published payload. Swap back to the
# per-clip ``clip.thumbnail_url`` here (and re-export / re-offload) to restore
# real thumbnails.
BLANK_THUMBNAIL_URL = "https://cdn.240.agency/blank.jpg"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )


# Mapping from validation rule codes (modules/labels/validation.py) to
# human-readable warning strings consumed by the frontend / Quarto.
_WARNING_LABELS: dict[str, str] = {
    "S1": "tag_count_out_of_range",
    "S2": "tag_length_out_of_range",
    "S3": "duplicate_tag_within_kind",
    "S4": "hashtag_like_tag",
    "S6": "invalid_confidence",
    "S7": "ungrounded_tag_reference",
    "S8": "sentence_length_out_of_range",
}


def _render_user_clips_block(
    session: Session, user_id: int, *, case: str
) -> list[dict]:
    """Build the ``clips: [...]`` JSON block for one creator.

    Joins ``Clip`` (selected) → ``ClipLabel`` (status=success) restricted
    to ``label_case == case`` so the per-case export does not 4×-fan-out
    when multiple modality labels exist for the same clip. Payload keys
    for the case-specific observable-tag list and one-sentence reading are
    derived from ``LabelCaseSpec`` (via ``clip_role_keys``) — the
    case-agnostic ``aesthetic_tags`` / ``community_signalling_tags``
    payload keys are stable per SPEC.

    Stage-1-skipped cases (sandwich/audio/maest/gemini) carry no
    ``ClipLabel`` rows of their own. Their per-clip visual tag block
    falls back to the case-agnostic video labels (same clip, same
    frames) so the creator pane shows tags in every run, not only in
    the video run.
    """
    spec = LABEL_CASE_REGISTRY[case]
    if not spec.runs_clip_pass:
        spec = LABEL_CASE_REGISTRY["video"]
    observable_key, sentence_key = clip_role_keys(spec)
    rows = (
        session.query(Clip, ClipLabel)
        .join(ClipLabel, ClipLabel.clip_id == Clip.id)
        .filter(
            Clip.user_id == user_id,
            Clip.is_selected.is_(True),
            ClipLabel.label_case == spec.name,
            ClipLabel.status == "success",
        )
        .order_by(Clip.id)
        .all()
    )
    out: list[dict] = []
    for clip, label in rows:
        payload = label.payload or {}
        out.append(
            {
                "clip_id": clip.id,
                "shortcode": getattr(clip, "shortcode", None),
                "thumbnail_url": BLANK_THUMBNAIL_URL,
                "sentence": payload.get(sentence_key, ""),
                "tags": {
                    "observable": payload.get(observable_key, []),
                    "aesthetic": payload.get("aesthetic_tags", []),
                    "community": payload.get("community_signalling_tags", []),
                },
                "validation": label.validation or "ok",
                "warnings": [
                    _WARNING_LABELS.get(code, code) for code in (label.warnings or [])
                ],
            }
        )
    return out


def _bounds(users: list[VisualizationUser]) -> dict[str, float]:
    if not users:
        return {"minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0}
    xs = [u.x for u in users]
    ys = [u.y for u in users]
    return {
        "minX": float(min(xs)),
        "maxX": float(max(xs)),
        "minY": float(min(ys)),
        "maxY": float(max(ys)),
    }


def _exposed_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        c for c in cases if c in CASE_REGISTRY and CASE_REGISTRY[c].expose_to_viewer
    )


def _prune_stale_run_dirs(runs_dir: Path, keep: set[str]) -> None:
    if not runs_dir.exists():
        return
    for child in runs_dir.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)


def _prune_stale_entity_files(dir_path: Path, keep_ids: set[int]) -> None:
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if not child.is_file() or child.suffix != ".json":
            continue
        try:
            cid = int(child.stem)
        except ValueError:
            child.unlink()
            continue
        if cid not in keep_ids:
            child.unlink()


@lru_cache(maxsize=1)
def _genre_leaf_map() -> dict[str, str]:
    """Map flattened Discogs genre labels to their leaf genre.

    MIR stores ``Parent---Child`` taxonomy entries with the separator
    flattened to a space (``Electronic---House`` → ``Electronic House``);
    the viewer shows only the leaf (``House``). Flattened forms are unique
    across the 519-entry taxonomy, so the mapping is unambiguous. Labels
    absent from the taxonomy pass through unchanged.

    TODO(bad-design): the root problem is upstream — ``mir.descriptors.topk_csv``
    discards the ``---`` separator before persisting genre labels, so the
    parent/child structure is lost at the source and this stage has to
    reconstruct the leaf by reloading and re-parsing the taxonomy. The
    label CSV doubles as embedding input text, so we can't change the
    persisted form without perturbing embeddings. The real fix (store the
    structured label, or a separate leaf column, independent of the
    embedding text) is deferred — revisit when the embedding text contract
    is decoupled from the stored MIR descriptors.
    """
    from modules.mir.descriptors import load_labels

    path = (
        Path(__file__).resolve().parents[1] / "mir" / "labels" / "genre_discogs519.json"
    )
    leaf: dict[str, str] = {}
    for raw in load_labels(path):
        _, sep, child = raw.partition("---")
        if sep:
            leaf[raw.replace("---", " ")] = child
    return leaf


def _genre_only_pairs(
    labels_csv: str | None, scores_csv: str | None
) -> list[tuple[str, float]]:
    """Parse genre label/score CSVs, reducing each label to its leaf genre."""
    from modules.visualization.compute import parse_label_score_csv

    leaf = _genre_leaf_map()
    return [
        (leaf.get(label, label), score)
        for label, score in parse_label_score_csv(labels_csv, scores_csv)
    ]


def _make_member_clip(*, clip: Clip, mir: AudioMIR) -> ClusterMemberClip:
    from modules.visualization.compute import parse_label_score_csv

    return ClusterMemberClip(
        approachability=mir.approachability,
        engagement=mir.engagement,
        danceability=mir.danceability,
        is_happy=mir.is_happy,
        is_sad=mir.is_sad,
        is_relaxed=mir.is_relaxed,
        is_aggressive=mir.is_aggressive,
        is_party=mir.is_party,
        is_acoustic=mir.is_acoustic,
        is_electronic=mir.is_electronic,
        is_instrumental=mir.is_instrumental,
        is_female_voice=mir.is_female_voice,
        is_bright_timbre=mir.is_bright_timbre,
        is_tonal=mir.is_tonal,
        is_speech_detected=clip.is_speech_detected,
        speech_language=clip.speech_language,
        caption_language=clip.caption_language,
        genre_pairs=_genre_only_pairs(mir.genre_labels, mir.genre_scores),
        instrument_pairs=parse_label_score_csv(
            mir.instrument_labels, mir.instrument_scores
        ),
    )


def _load_case_detail_inputs(
    session, case: str
) -> tuple[dict[int, ClusterMember], dict[int, list[int]]]:
    """Return (member_by_user_id, member_ids_by_cluster_id) for one case.

    Every user with a UserCluster row becomes a ClusterMember; the
    ``clips`` list contains only that user's selected clips with a
    successfully extracted AudioMIR row, while ``n_clips`` reports the
    creator's actual selected-clip count so non-MIR-dependent cluster
    aggregates (followers, posting stats) still include them.
    """
    uc_rows = (
        session.query(UserCluster).filter(UserCluster.embedding_case == case).all()
    )
    if not uc_rows:
        return {}, {}
    user_ids = [r.user_id for r in uc_rows]

    users = {u.id: u for u in session.query(User).filter(User.id.in_(user_ids)).all()}
    stats = {
        s.user_id: s
        for s in session.query(UserStats).filter(UserStats.user_id.in_(user_ids)).all()
    }
    selected_count_by_user: dict[int, int] = dict(
        session.query(Clip.user_id, func.count(Clip.id))
        .filter(Clip.user_id.in_(user_ids))
        .filter(Clip.is_selected.is_(True))
        .group_by(Clip.user_id)
        .all()
    )
    clip_rows = (
        session.query(Clip, AudioMIR)
        .join(AudioMIR, AudioMIR.clip_id == Clip.id)
        .filter(Clip.user_id.in_(user_ids))
        .filter(Clip.is_selected.is_(True))
        .filter(AudioMIR.is_mir_extracted.is_(True))
        .all()
    )
    clips_by_user: dict[int, list[ClusterMemberClip]] = defaultdict(list)
    for clip, mir in clip_rows:
        clips_by_user[clip.user_id].append(_make_member_clip(clip=clip, mir=mir))

    member_by_user: dict[int, ClusterMember] = {}
    members_by_cluster: dict[int, list[int]] = defaultdict(list)
    for uc in uc_rows:
        clips = clips_by_user.get(uc.user_id, [])
        u = users.get(uc.user_id)
        s = stats.get(uc.user_id)
        member_by_user[uc.user_id] = ClusterMember(
            user_id=uc.user_id,
            follower_count=(u.follower_count if u else None),
            n_clips=selected_count_by_user.get(uc.user_id, 0),
            median_plays=(
                int(s.median_plays) if s and s.median_plays is not None else None
            ),
            median_clips_per_week=(s.approx_clips_per_week if s else None),
            engagement_shape_ratio=(s.top_to_median_plays_ratio if s else None),
            median_video_duration=(s.median_video_duration if s else None),
            activity_span_months=(
                round(s.clip_time_span_days / 30.0)
                if s and s.clip_time_span_days is not None
                else None
            ),
            clips=clips,
        )
        members_by_cluster[uc.cluster_id].append(uc.user_id)

    return member_by_user, members_by_cluster


def _build_cluster_detail_payloads(
    *,
    settings_viz,
    case: str,
    cluster_rows: Sequence[VisualizationCluster],
    member_by_user: dict[int, ClusterMember],
    members_by_cluster: dict[int, list[int]],
    cluster_label_rows: dict[int, ClusterLabel],
) -> dict[int, dict]:
    """Per-cluster detail dicts keyed by cluster_id (exporter emission order)."""
    baseline = _cluster_baseline_from_members(list(member_by_user.values()))
    centroids = {c.cluster_id: (c.cx, c.cy) for c in cluster_rows}
    labels = {c.cluster_id: c.label for c in cluster_rows}

    out: dict[int, dict] = {}
    for c in cluster_rows:
        member_ids = members_by_cluster.get(c.cluster_id, [])
        members = [member_by_user[uid] for uid in member_ids if uid in member_by_user]
        detail = build_cluster_detail(
            cluster_id=c.cluster_id,
            cluster_label=c.label,
            cluster_size=c.size,
            ellipse={"cx": c.cx, "cy": c.cy, "rx": c.rx, "ry": c.ry, "angle": c.angle},
            members=members,
            baseline=baseline,
            cluster_centroids=centroids,
            cluster_labels=labels,
            z_min=settings_viz.distinctiveness_z_min,
            distinctiveness_top_k=settings_viz.distinctiveness_top_k,
            genre_top_k=settings_viz.genre_top_k,
            instrument_top_k=settings_viz.instrument_top_k,
            languages_top_k=settings_viz.languages_top_k,
        )
        cluster_payload = detail.to_json()
        # The frontend's clusterDetailSchema reserves the ``label`` key for
        # the case-agnostic label block (Phase E). ``detail.to_json()`` writes
        # the cluster's display-label string there; we overwrite it with the
        # block when available, and drop the key entirely otherwise so the
        # Zod schema's ``label?: ClusterLabel`` shape parses.
        label_block = render_label_block(
            cluster_label_rows.get(c.cluster_id),
            spec=LABEL_CASE_REGISTRY[case],
        )
        if label_block is not None:
            cluster_payload["label"] = label_block
        else:
            cluster_payload.pop("label", None)
        out[c.cluster_id] = cluster_payload
    return out


def _build_creator_detail_payloads(
    *,
    session: Session,
    settings_viz,
    case: str,
    cluster_rows: Sequence[VisualizationCluster],
    user_rows: Sequence[VisualizationUser],
    member_by_user: dict[int, ClusterMember],
    members_by_cluster: dict[int, list[int]],
) -> dict[int, dict]:
    """Per-creator detail dicts keyed by user_id (exporter emission order).

    Only users the exporter actually writes are present (skips users with no
    centroid or no MIR-backed clips), so the keyset doubles as ``has_detail``.
    """
    centroids = {c.cluster_id: (c.cx, c.cy) for c in cluster_rows}
    labels = {c.cluster_id: c.label for c in cluster_rows}
    user_row_by_id = {u.user_id: u for u in user_rows}
    centroid_dist_by_cluster: dict[int, np.ndarray] = {}
    for cid, member_ids in members_by_cluster.items():
        if cid not in centroids:
            continue
        cx, cy = centroids[cid]
        dists = []
        for uid in member_ids:
            uc = user_row_by_id.get(uid)
            if uc is None or uc.cluster_id != cid:
                continue
            dists.append(float(np.hypot(uc.x - cx, uc.y - cy)))
        centroid_dist_by_cluster[cid] = np.array(dists, dtype=np.float64)

    out: dict[int, dict] = {}
    for u in user_rows:
        if u.user_id not in member_by_user or u.cluster_id not in centroids:
            continue
        self_member = member_by_user[u.user_id]
        if not self_member.clips:
            continue
        own_member_ids = members_by_cluster.get(u.cluster_id, [])
        own_excl_self = [
            member_by_user[m]
            for m in own_member_ids
            if m != u.user_id and m in member_by_user
        ]
        other_centroids = {
            cid: xy for cid, xy in centroids.items() if cid != u.cluster_id
        }
        other_labels = {cid: labels[cid] for cid in other_centroids}
        detail = build_user_detail(
            user_id=u.user_id,
            cluster_id=u.cluster_id,
            x=u.x,
            y=u.y,
            self_member=self_member,
            own_cluster_members_excl_self=own_excl_self,
            own_cluster_centroid=centroids[u.cluster_id],
            own_cluster_member_distances=centroid_dist_by_cluster.get(
                u.cluster_id, np.array([], dtype=np.float64)
            ),
            other_cluster_centroids=other_centroids,
            other_cluster_labels=other_labels,
            edge_percentile=settings_viz.edge_percentile,
            z_min=settings_viz.distinctiveness_z_min,
            distinctiveness_top_k=settings_viz.distinctiveness_top_k,
            genre_top_k=settings_viz.genre_top_k,
            instrument_top_k=settings_viz.instrument_top_k,
            languages_top_k=settings_viz.languages_top_k,
        )
        payload = detail.to_json()
        payload["clips"] = _render_user_clips_block(session, u.user_id, case=case)
        out[u.user_id] = payload
    return out


@dataclass(frozen=True)
class CasePayloadBundle:
    """Every version-6 payload for one run/case, built once from the DB.

    The single producer of payload shape shared by the file exporter and the
    serving offload: ``manifest_entry`` + ``users`` + ``clusters`` are the bulk
    files; ``cluster_details`` / ``creator_details`` are keyed by id in the
    exporter's emission order (clusters by cluster_id, users by user_id). A
    user/cluster id is present in the details dict iff the exporter writes its
    file, so the keyset is the source of ``has_detail``.
    """

    case: str
    manifest_entry: dict
    users: dict
    clusters: dict
    cluster_details: dict[int, dict]
    creator_details: dict[int, dict]


def build_case_payloads(
    session: Session, *, settings_viz, case: str
) -> CasePayloadBundle | None:
    """Build all version-6 payloads for one case, or None if it has no row."""
    viz = session.get(Visualization, case)
    if viz is None:
        return None
    users = (
        session.query(VisualizationUser)
        .filter_by(embedding_case=case)
        .order_by(VisualizationUser.user_id)
        .all()
    )
    clusters = (
        session.query(VisualizationCluster)
        .filter_by(embedding_case=case)
        .order_by(VisualizationCluster.cluster_id)
        .all()
    )
    member_by_user, members_by_cluster = _load_case_detail_inputs(session, case)
    cluster_label_rows = {
        r.cluster_id: r
        for r in session.query(ClusterLabel)
        .filter(ClusterLabel.embedding_case == case)
        .all()
    }
    cluster_details = _build_cluster_detail_payloads(
        settings_viz=settings_viz,
        case=case,
        cluster_rows=clusters,
        member_by_user=member_by_user,
        members_by_cluster=members_by_cluster,
        cluster_label_rows=cluster_label_rows,
    )
    creator_details = _build_creator_detail_payloads(
        session=session,
        settings_viz=settings_viz,
        case=case,
        cluster_rows=clusters,
        user_rows=users,
        member_by_user=member_by_user,
        members_by_cluster=members_by_cluster,
    )
    written_users = set(creator_details)
    written_clusters = set(cluster_details)

    users_payload = {
        "version": SCHEMA_VERSION,
        "run_id": case,
        "bounds": _bounds(users),
        "users": [
            [
                u.user_id,
                u.x,
                u.y,
                u.cluster_id,
                u.user_id in written_users,
                float(u.centrality) if u.centrality is not None else 0.0,
            ]
            for u in users
        ],
    }
    clusters_payload = {
        "version": SCHEMA_VERSION,
        "run_id": case,
        "clusters": [
            {
                "id": c.cluster_id,
                "label": display_label_for(
                    cluster_label_rows.get(c.cluster_id), c.label
                ),
                "cx": c.cx,
                "cy": c.cy,
                "rx": c.rx,
                "ry": c.ry,
                "angle": c.angle,
                "size": c.size,
                "has_detail": c.cluster_id in written_clusters,
            }
            for c in clusters
        ],
    }
    manifest_entry = {
        "id": case,
        "case": case,
        # Label is presentation metadata, not fingerprinted data: read it live
        # from the case spec so renaming a display_label lands on the next
        # export without forcing the (data-only) viz fingerprint stale.
        "label": CASE_REGISTRY[case].display_label,
        "size": viz.size,
        "details_available": True,
    }
    return CasePayloadBundle(
        case=case,
        manifest_entry=manifest_entry,
        users=users_payload,
        clusters=clusters_payload,
        cluster_details=cluster_details,
        creator_details=creator_details,
    )


def build_manifest_payload(default_case: str, entries: list[dict]) -> dict:
    """Assemble the manifest.json payload from per-case manifest entries."""
    return {
        "version": SCHEMA_VERSION,
        "default_run_id": default_case,
        "runs": entries,
    }


def _write_case_bundle(bundle: CasePayloadBundle, export_dir: Path) -> None:
    """Write one case's detail + bulk files; prune stale per-entity files."""
    case = bundle.case
    for cluster_id, payload in bundle.cluster_details.items():
        _write_json(
            export_dir / "runs" / case / "clusters" / f"{cluster_id}.json", payload
        )
    for user_id, payload in bundle.creator_details.items():
        _write_json(export_dir / "runs" / case / "users" / f"{user_id}.json", payload)
    _prune_stale_entity_files(
        export_dir / "runs" / case / "clusters", set(bundle.cluster_details)
    )
    _prune_stale_entity_files(
        export_dir / "runs" / case / "users", set(bundle.creator_details)
    )
    _write_json(export_dir / "runs" / case / "users.json", bundle.users)
    _write_json(export_dir / "runs" / case / "clusters.json", bundle.clusters)


@scope("visualization:export")
def export_visualization_json(settings, cases: tuple[str, ...]) -> None:
    """Read DB → write the frontend bulk-JSON tree. Idempotent."""
    viz_settings = settings.visualization
    export_dir = Path(viz_settings.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = export_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    manifest_path = export_dir / "manifest.json"

    cases_to_export = _exposed_cases(cases)
    session = get_session()
    try:
        manifest_runs: list[dict] = []
        written_cases: set[str] = set()
        for case in cases_to_export:
            bundle = build_case_payloads(session, settings_viz=viz_settings, case=case)
            if bundle is None:
                event("SKIP", case)
                continue
            _write_case_bundle(bundle, export_dir)
            manifest_runs.append(bundle.manifest_entry)
            written_cases.add(case)
            event(
                "WRITE",
                case,
                stats={
                    "users": len(bundle.users["users"]),
                    "clusters": len(bundle.clusters["clusters"]),
                },
            )

        _prune_stale_run_dirs(runs_dir, written_cases)

        if not manifest_runs:
            if manifest_path.exists():
                manifest_path.unlink()
            event("SKIP", "manifest")
            return

        _write_json(
            manifest_path,
            build_manifest_payload(viz_settings.default_case, manifest_runs),
        )
        event("SEAL", "manifest", stats={"runs": len(manifest_runs)})
    finally:
        session.close()
