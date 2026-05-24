"""Tests for the synthetic atlas fixture generator."""

from __future__ import annotations

import json
from pathlib import Path

from modules.export import ClustersFile, Manifest, UsersFile
from scripts.gen_atlas_fixture import CaseSpec, build_dataset


def _read_users(out_dir: Path, run_id: str) -> UsersFile:
    raw = json.loads((out_dir / "runs" / run_id / "users.json").read_text())
    return UsersFile.model_validate(raw)


def test_build_dataset_writes_all_three_cases(tmp_path: Path) -> None:
    build_dataset(tmp_path, seed=42)
    manifest_raw = json.loads((tmp_path / "manifest.json").read_text())
    manifest = Manifest.model_validate(manifest_raw)
    case_ids = {r.case for r in manifest.runs}
    assert case_ids == {"video", "sandwich", "audio"}
    assert manifest.default_run_id == "video-1"


def test_each_run_has_approximately_one_thousand_users(tmp_path: Path) -> None:
    build_dataset(tmp_path, seed=42)
    for case in ("video", "sandwich", "audio"):
        users = _read_users(tmp_path, f"{case}-1")
        assert 900 <= len(users.users) <= 1100


def test_runs_have_distinct_layouts(tmp_path: Path) -> None:
    """Different cases must produce visibly different layouts so case
    switching is meaningful when Phase 2 lands."""
    build_dataset(tmp_path, seed=42)
    video = _read_users(tmp_path, "video-1").users
    sandwich = _read_users(tmp_path, "sandwich-1").users
    audio = _read_users(tmp_path, "audio-1").users
    # Compare a few positions — at least one user with a given id should
    # be in a different place across cases.
    video_pos = {u[0]: (u[1], u[2]) for u in video}
    sandwich_pos = {u[0]: (u[1], u[2]) for u in sandwich}
    common = set(video_pos) & set(sandwich_pos)
    assert any(video_pos[i] != sandwich_pos[i] for i in common)
    _ = audio  # audio is also expected to differ; smoke-only


def test_generator_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_dataset(a, seed=42)
    build_dataset(b, seed=42)
    for relative in (
        Path("manifest.json"),
        Path("runs/video-1/users.json"),
        Path("runs/video-1/clusters.json"),
        Path("runs/sandwich-1/users.json"),
        Path("runs/sandwich-1/clusters.json"),
        Path("runs/audio-1/users.json"),
        Path("runs/audio-1/clusters.json"),
    ):
        assert (a / relative).read_bytes() == (b / relative).read_bytes(), relative


def test_clusters_have_size_matching_user_counts(tmp_path: Path) -> None:
    build_dataset(tmp_path, seed=42)
    for case in ("video", "sandwich", "audio"):
        run_id = f"{case}-1"
        users = _read_users(tmp_path, run_id).users
        clusters_raw = json.loads(
            (tmp_path / "runs" / run_id / "clusters.json").read_text()
        )
        clusters = ClustersFile.model_validate(clusters_raw).clusters
        for cluster in clusters:
            count = sum(1 for u in users if u[3] == cluster.id)
            assert cluster.size == count


def test_case_spec_dataclass_is_usable() -> None:
    spec = CaseSpec(
        run_id="custom-1",
        case="video",
        label="Custom",
        n_clusters=5,
        n_users=100,
        noise_fraction=0.05,
    )
    assert spec.run_id == "custom-1"
