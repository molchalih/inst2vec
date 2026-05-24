"""Tests for modules.export.writer and the pydantic schemas it uses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modules.export import (
    SCHEMA_VERSION,
    BoundsModel,
    ClusterModel,
    ClustersFile,
    Manifest,
    ManifestRun,
    RunPayload,
    UsersFile,
    write_dataset,
    write_manifest,
    write_run,
)


def _make_run(run_id: str = "video-1") -> RunPayload:
    return RunPayload(
        meta=ManifestRun(id=run_id, case="video", label="Visual", size=2),
        users=UsersFile(
            run_id=run_id,
            bounds=BoundsModel(minX=-1, maxX=1, minY=-1, maxY=1),
            users=[(0, -0.5, 0.0, 0), (1, 0.5, 0.0, 0)],
        ),
        clusters=ClustersFile(
            run_id=run_id,
            clusters=[
                ClusterModel(
                    id=0,
                    label="Cluster 1",
                    cx=0.0,
                    cy=0.0,
                    rx=0.5,
                    ry=0.2,
                    angle=0.0,
                    size=2,
                ),
            ],
        ),
    )


def test_schemas_import() -> None:
    assert SCHEMA_VERSION == 1


def test_write_run_creates_two_files(tmp_path: Path) -> None:
    run = _make_run("video-1")
    write_run(tmp_path, run)
    users_path = tmp_path / "runs" / "video-1" / "users.json"
    clusters_path = tmp_path / "runs" / "video-1" / "clusters.json"
    assert users_path.exists()
    assert clusters_path.exists()


def test_written_users_match_schema(tmp_path: Path) -> None:
    run = _make_run("video-1")
    write_run(tmp_path, run)
    raw = json.loads((tmp_path / "runs" / "video-1" / "users.json").read_text())
    parsed = UsersFile.model_validate(raw)
    assert parsed.run_id == "video-1"
    assert parsed.version == 1
    assert len(parsed.users) == 2


def test_written_clusters_match_schema(tmp_path: Path) -> None:
    run = _make_run("video-1")
    write_run(tmp_path, run)
    raw = json.loads((tmp_path / "runs" / "video-1" / "clusters.json").read_text())
    parsed = ClustersFile.model_validate(raw)
    assert parsed.run_id == "video-1"
    assert parsed.clusters[0].label == "Cluster 1"


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    manifest = Manifest(
        default_run_id="video-1",
        runs=[ManifestRun(id="video-1", case="video", label="Visual", size=2)],
    )
    write_manifest(tmp_path, manifest)
    raw = json.loads((tmp_path / "manifest.json").read_text())
    parsed = Manifest.model_validate(raw)
    assert parsed.default_run_id == "video-1"


def test_write_is_idempotent(tmp_path: Path) -> None:
    run = _make_run("video-1")
    write_run(tmp_path, run)
    first = (tmp_path / "runs" / "video-1" / "users.json").read_bytes()
    write_run(tmp_path, run)
    second = (tmp_path / "runs" / "video-1" / "users.json").read_bytes()
    assert first == second


def test_write_dataset_writes_everything(tmp_path: Path) -> None:
    run1 = _make_run("video-1")
    run2 = _make_run("sandwich-1")
    run2.meta = ManifestRun(id="sandwich-1", case="sandwich", label="Sandwich", size=2)
    run2.users.run_id = "sandwich-1"
    run2.clusters.run_id = "sandwich-1"

    write_dataset(
        tmp_path,
        default_run_id="video-1",
        runs=[run1, run2],
    )
    manifest_raw = json.loads((tmp_path / "manifest.json").read_text())
    manifest = Manifest.model_validate(manifest_raw)
    assert manifest.default_run_id == "video-1"
    assert {r.id for r in manifest.runs} == {"video-1", "sandwich-1"}
    assert (tmp_path / "runs" / "sandwich-1" / "users.json").exists()


def test_write_run_rejects_run_id_mismatch(tmp_path: Path) -> None:
    run = _make_run("video-1")
    run.users.run_id = "audio-1"  # diverges from meta.id
    with pytest.raises(ValueError, match="run_id"):
        write_run(tmp_path, run)
