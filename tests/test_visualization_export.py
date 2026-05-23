"""Tests for modules.visualization.export.export_visualization_json."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.database import (
    Base,
    StageState,
    User,
    UserCluster,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_engine,
    get_session,
)
from modules.visualization.export import export_visualization_json
from modules.visualization.schema import SCHEMA_VERSION


def _clear() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (
            VisualizationUser,
            VisualizationCluster,
            Visualization,
            UserCluster,
            StageState,
            User,
        ):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()


def _seed_visualization(
    case: str, label: str, *, n_users: int = 6, n_clusters: int = 2
) -> None:
    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case=case,
                label=label,
                size=n_users,
                source_hash="hash-" + case,
            )
        )
        for uid in range(n_users):
            session.merge(User(id=uid))
            cid = uid % n_clusters
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


def _settings(tmp_path: Path, default_case: str = "video") -> SimpleNamespace:
    return SimpleNamespace(
        visualization=SimpleNamespace(
            export_dir=tmp_path,
            default_case=default_case,
            distinctiveness_z_min=0.5,
            distinctiveness_top_k=3,
            genre_top_k=5,
            instrument_top_k=3,
            languages_top_k=3,
            edge_percentile=66,
        )
    )


def test_export_writes_manifest_users_clusters_for_exposed_case(tmp_path):
    _clear()
    _seed_visualization("video", "Visual", n_users=6, n_clusters=2)
    export_visualization_json(_settings(tmp_path), cases=("video",))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == SCHEMA_VERSION
    assert manifest["default_run_id"] == "video"
    assert manifest["runs"] == [
        {
            "id": "video",
            "case": "video",
            "label": "Visual",
            "size": 6,
            "details_available": True,
        }
    ]

    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    assert users["version"] == SCHEMA_VERSION
    assert users["run_id"] == "video"
    assert users["bounds"] == {"minX": 0.0, "maxX": 5.0, "minY": -5.0, "maxY": 0.0}
    assert len(users["users"]) == 6
    # Tuple shape: [id, x, y, cluster_id, has_detail]
    assert all(len(u) == 5 for u in users["users"])
    # No MIR data seeded → has_detail must be False for every row.
    assert all(u[4] is False for u in users["users"])

    clusters = json.loads((tmp_path / "runs" / "video" / "clusters.json").read_text())
    assert clusters["version"] == SCHEMA_VERSION
    assert clusters["run_id"] == "video"
    assert len(clusters["clusters"]) == 2
    assert {c["label"] for c in clusters["clusters"]} == {"Cluster 1", "Cluster 2"}
    assert all("has_detail" in c for c in clusters["clusters"])


def test_export_filters_hidden_cases(tmp_path):
    _clear()
    _seed_visualization("video", "Visual")
    _seed_visualization("gemini", "Gemini")
    export_visualization_json(_settings(tmp_path), cases=("video", "gemini"))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["case"] for r in manifest["runs"]] == ["video"]
    assert not (tmp_path / "runs" / "gemini").exists()


def test_export_skips_cases_with_no_db_row(tmp_path):
    _clear()
    _seed_visualization("video", "Visual")
    # "sandwich" is in cases but has no Visualization row.
    export_visualization_json(_settings(tmp_path), cases=("video", "sandwich"))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["case"] for r in manifest["runs"]] == ["video"]
    assert not (tmp_path / "runs" / "sandwich").exists()


def test_export_no_visible_runs_skips_manifest(tmp_path):
    _clear()
    _seed_visualization("gemini", "Gemini")  # hidden-only
    export_visualization_json(_settings(tmp_path), cases=("gemini",))

    assert not (tmp_path / "manifest.json").exists()


def test_export_no_runs_removes_stale_manifest_and_run_dirs(tmp_path):
    _clear()
    stale_manifest = tmp_path / "manifest.json"
    stale_manifest.write_text(json.dumps({"version": "stale", "runs": [{"id": "old"}]}))
    stale_case = tmp_path / "runs" / "old"
    stale_case.mkdir(parents=True)
    (stale_case / "users.json").write_text("{}")

    export_visualization_json(_settings(tmp_path), cases=("video",))

    assert not stale_manifest.exists()
    assert not stale_case.exists()


def test_export_prunes_run_dirs_for_dropped_cases(tmp_path):
    _clear()
    _seed_visualization("video", "Visual")
    _seed_visualization("sandwich", "Sandwich")
    export_visualization_json(_settings(tmp_path), cases=("video", "sandwich"))
    assert (tmp_path / "runs" / "sandwich").exists()

    _clear()
    _seed_visualization("video", "Visual")
    export_visualization_json(_settings(tmp_path), cases=("video", "sandwich"))

    assert (tmp_path / "runs" / "video").exists()
    assert not (tmp_path / "runs" / "sandwich").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [r["case"] for r in manifest["runs"]] == ["video"]


def test_export_empty_case_writes_degenerate_bounds(tmp_path):
    _clear()
    session = get_session()
    try:
        session.merge(
            Visualization(
                embedding_case="video",
                label="Visual",
                size=0,
                source_hash="empty",
            )
        )
        session.commit()
    finally:
        session.close()

    export_visualization_json(_settings(tmp_path), cases=("video",))
    users = json.loads((tmp_path / "runs" / "video" / "users.json").read_text())
    assert users["users"] == []
    # Default safe box (non-zero extent) so the frontend's stretchRun
    # never divides by zero.
    assert users["bounds"] == {"minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0}
