"""Tests for scripts/publish_visualization.py.

The script copies the visualization JSON export tree from
``data/visualization/`` (or whatever ``settings.visualization.export_dir``
resolves to) into ``frontend/public/data/`` so the Pages workflow can
ship it. These tests pin the contract: errors when source is missing or
malformed, byte-for-byte copy on the happy path, stale-file pruning on
re-publish.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import publish_visualization


def test_publish_raises_when_source_missing(tmp_path):
    source = tmp_path / "absent"
    dest = tmp_path / "dest"
    with pytest.raises(FileNotFoundError, match="visualization export not found"):
        publish_visualization.publish(source, dest)


def test_publish_raises_when_manifest_missing(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    (source / "runs").mkdir()
    dest = tmp_path / "dest"
    with pytest.raises(FileNotFoundError, match=r"missing manifest\.json"):
        publish_visualization.publish(source, dest)


def _write_sample_export(root: Path) -> dict:
    """Build a fixture export tree that mirrors the real exporter's output."""
    manifest = {
        "version": 1,
        "default_run_id": "video",
        "runs": [
            {"id": "video", "case": "video", "label": "Video", "size": 42},
            {"id": "audio", "case": "audio", "label": "Audio", "size": 17},
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    for case in ("video", "audio"):
        case_dir = root / "runs" / case
        case_dir.mkdir(parents=True)
        (case_dir / "users.json").write_text(
            json.dumps(
                {"version": 1, "run_id": case, "bounds": [0, 0, 1, 1], "users": []}
            )
        )
        (case_dir / "clusters.json").write_text(
            json.dumps({"version": 1, "run_id": case, "clusters": []})
        )
    return manifest


def test_publish_copies_full_tree(tmp_path):
    source = tmp_path / "export"
    dest = tmp_path / "dest"
    expected_manifest = _write_sample_export(source)

    returned = publish_visualization.publish(source, dest)

    assert returned == expected_manifest
    assert json.loads((dest / "manifest.json").read_text()) == expected_manifest
    for case in ("video", "audio"):
        assert (dest / "runs" / case / "users.json").is_file()
        assert (dest / "runs" / case / "clusters.json").is_file()


def test_publish_prunes_stale_destination(tmp_path):
    source = tmp_path / "export"
    dest = tmp_path / "dest"
    _write_sample_export(source)

    # Seed dest with a prior, partially-overlapping snapshot
    (dest / "runs" / "obsolete_case").mkdir(parents=True)
    (dest / "runs" / "obsolete_case" / "users.json").write_text("{}")
    (dest / "manifest.json").write_text(json.dumps({"version": 0, "runs": []}))

    publish_visualization.publish(source, dest)

    assert not (dest / "runs" / "obsolete_case").exists()
    assert json.loads((dest / "manifest.json").read_text())["version"] == 1
