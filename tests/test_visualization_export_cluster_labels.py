"""Tests for cluster-label overlay in export_visualization_json.

Covers:
1. clusters.json falls back to VisualizationCluster.label when no ClusterLabel row.
2. clusters.json uses payload.cluster_label when status=success, and detail gets label block.
3. cluster detail omits ``label`` key when there is no ClusterLabel row.
4. Warning codes in a warn-status ClusterLabel row are translated to human strings.
5. The exported block is keyed ``label`` (not ``visual``) and carries
   ``modality`` + flattened ``repertoire`` per Phase E1.
6. The exported schema version matches Phase E's bump to 6.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.database import (
    Base,
    ClusterLabel,
    User,
    UserCluster,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_engine,
    get_session,
)
from modules.visualization.export import export_visualization_json


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        visualization=SimpleNamespace(
            export_dir=str(tmp_path),
            default_case="video",
            distinctiveness_z_min=0.0,
            distinctiveness_top_k=3,
            genre_top_k=5,
            instrument_top_k=3,
            languages_top_k=3,
            edge_percentile=66,
        )
    )


def _clear() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (
            ClusterLabel,
            VisualizationUser,
            VisualizationCluster,
            Visualization,
            UserCluster,
            User,
        ):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()


def _seed(case: str = "video", n_clusters: int = 2, n_users: int = 4) -> None:
    """Seed the minimum rows needed for export to produce clusters.json and detail files."""
    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case=case,
                label="Visual",
                size=n_users,
                source_hash="hash-" + case,
            )
        )
        for uid in range(n_users):
            session.merge(User(id=uid))
            cid = uid % n_clusters
            session.add(
                UserCluster(
                    user_id=uid,
                    embedding_case=case,
                    cluster_id=cid,
                    umap_x=float(uid),
                    umap_y=float(-uid),
                )
            )
            session.add(
                VisualizationUser(
                    user_id=uid,
                    embedding_case=case,
                    x=float(uid),
                    y=float(-uid),
                    cluster_id=cid,
                )
            )
        for cid in range(n_clusters):
            session.add(
                VisualizationCluster(
                    embedding_case=case,
                    cluster_id=cid,
                    cx=float(cid),
                    cy=float(cid * 2),
                    rx=1.0,
                    ry=2.0,
                    angle=0.0,
                    size=n_users // n_clusters,
                    label=f"Cluster {cid + 1}",
                )
            )
        session.commit()
    finally:
        session.close()


def _add_cluster_label(
    case: str,
    cluster_id: int,
    *,
    status: str,
    payload: dict | None = None,
    validation: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    session = get_session()
    try:
        session.add(
            ClusterLabel(
                embedding_case=case,
                cluster_id=cluster_id,
                status=status,
                payload=payload,
                validation=validation,
                warnings=warnings,
                attempts=1,
            )
        )
        session.commit()
    finally:
        session.close()


# ── Test 1 ────────────────────────────────────────────────────────────────────


def test_clusters_json_falls_back_to_placeholder_without_label(tmp_path):
    """When no ClusterLabel row exists, clusters.json uses VisualizationCluster.label."""
    _clear()
    _seed("video", n_clusters=2)

    export_visualization_json(_settings(tmp_path), cases=("video",))

    clusters = json.loads((tmp_path / "runs" / "video" / "clusters.json").read_text())
    labels = {c["id"]: c["label"] for c in clusters["clusters"]}
    assert labels[0] == "Cluster 1"
    assert labels[1] == "Cluster 2"


# ── Test 2 ────────────────────────────────────────────────────────────────────


def test_clusters_json_uses_generated_label_when_success(tmp_path):
    """Success ClusterLabel → clusters.json shows generated label and detail has label block."""
    _clear()
    _seed("video", n_clusters=2)
    _add_cluster_label(
        "video",
        0,
        status="success",
        payload={
            "cluster_label": "Ambient Architects",
            "cluster_summary": "Creators who favour soft, layered soundscapes.",
            "dominant_visual_repertoire": ["mist", "bloom"],
            "dominant_aesthetic_logic": ["restraint"],
            "taste_signalling": {"mode": "understated"},
            "visibility_orientation": {"reach": "niche"},
            "internal_variations": ["sub-genre A"],
            "boundary_notes": "Overlaps with cluster 1.",
            "tool_tags": ["reverb", "delay"],
        },
        validation="ok",
        warnings=[],
    )

    export_visualization_json(_settings(tmp_path), cases=("video",))

    # clusters.json: cluster 0 uses generated label, cluster 1 falls back
    clusters = json.loads((tmp_path / "runs" / "video" / "clusters.json").read_text())
    label_by_id = {c["id"]: c["label"] for c in clusters["clusters"]}
    assert label_by_id[0] == "Ambient Architects"
    assert label_by_id[1] == "Cluster 2"

    # detail file: cluster 0 has label block
    detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    assert "label" in detail
    label = detail["label"]
    assert label["label"] == "Ambient Architects"
    assert label["summary"] == "Creators who favour soft, layered soundscapes."
    assert label["repertoire"] == ["mist", "bloom"]
    assert label["validation"] == "ok"
    assert label["warnings"] == []


# ── Test 3 ────────────────────────────────────────────────────────────────────


def test_cluster_detail_omits_label_when_no_label_row(tmp_path):
    """When no ClusterLabel row exists, cluster detail file has no ``label`` key."""
    _clear()
    _seed("video", n_clusters=2)

    export_visualization_json(_settings(tmp_path), cases=("video",))

    detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    assert "label" not in detail


# ── Test 4 ────────────────────────────────────────────────────────────────────


def test_cluster_warning_codes_translated_in_label_block(tmp_path):
    """A warn-status ClusterLabel row → label block present with translated warnings."""
    _clear()
    _seed("video", n_clusters=2)
    _add_cluster_label(
        "video",
        0,
        status="success",
        payload={
            "cluster_label": "Bass Minimalists",
            "cluster_summary": "Low-frequency focused creators.",
        },
        validation="warn",
        warnings=["SC1", "SC5"],
    )

    export_visualization_json(_settings(tmp_path), cases=("video",))

    detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    assert "label" in detail
    label = detail["label"]
    assert label["validation"] == "warn"
    assert "tag_count_out_of_range" in label["warnings"]
    assert "invalid_confidence" in label["warnings"]


# ── Phase E1 tests ────────────────────────────────────────────────────────────


def _success_payload(repertoire_key: str) -> dict:
    return {
        "cluster_label": "Test Cluster",
        "cluster_summary": "A summary.",
        repertoire_key: [
            {"tag": "thing", "description": "x", "recurrence": "dominant"}
        ],
        "dominant_aesthetic_logic": [
            {"tag": "logic", "grounded_in": ["thing"], "description": "y"}
        ],
        "taste_signalling": {
            "label": "t",
            "description": "d",
            "confidence": "medium",
        },
        "visibility_orientation": {
            "label": "v",
            "description": "d",
            "confidence": "low",
        },
        "internal_variations": [],
        "boundary_notes": "",
        "tool_tags": [],
    }


def test_export_writes_label_block_not_visual_block(tmp_path):
    """Phase E1: the per-cluster JSON uses key ``label`` (never ``visual``)."""
    _clear()
    _seed("video", n_clusters=2)
    _add_cluster_label(
        "video",
        0,
        status="success",
        payload=_success_payload("dominant_visual_repertoire"),
        validation="ok",
        warnings=[],
    )

    export_visualization_json(_settings(tmp_path), cases=("video",))

    detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    assert "label" in detail
    assert "visual" not in detail


def test_export_label_block_has_modality_field_matching_case(tmp_path):
    """Phase E1: ``modality`` is stamped from LabelCaseSpec — video → visual."""
    _clear()
    _seed("video", n_clusters=2)
    _add_cluster_label(
        "video",
        0,
        status="success",
        payload=_success_payload("dominant_visual_repertoire"),
        validation="ok",
        warnings=[],
    )

    export_visualization_json(_settings(tmp_path), cases=("video",))

    detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    assert detail["label"]["modality"] == "visual"


def test_export_label_block_flattens_repertoire_field(tmp_path):
    """Phase E1: the case-specific ``dominant_*_repertoire`` key is renamed to
    ``repertoire`` at the render boundary; the case-specific name does not
    leak through."""
    _clear()
    _seed("video", n_clusters=2)
    _add_cluster_label(
        "video",
        0,
        status="success",
        payload=_success_payload("dominant_visual_repertoire"),
        validation="ok",
        warnings=[],
    )

    export_visualization_json(_settings(tmp_path), cases=("video",))

    detail = json.loads(
        (tmp_path / "runs" / "video" / "clusters" / "0.json").read_text()
    )
    label = detail["label"]
    assert "repertoire" in label
    assert "dominant_visual_repertoire" not in label
    assert label["repertoire"][0]["tag"] == "thing"


def test_export_schema_version_is_6(tmp_path):
    """Phase E1: SCHEMA_VERSION is pinned at 6."""
    _clear()
    _seed("video", n_clusters=2)

    export_visualization_json(_settings(tmp_path), cases=("video",))

    clusters = json.loads((tmp_path / "runs" / "video" / "clusters.json").read_text())
    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert clusters["version"] == 6
    assert users["version"] == 6
    assert manifest["version"] == 6
