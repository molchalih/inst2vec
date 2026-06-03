"""Pure payload builders in modules.visualization.export.

The builders are the single producer of the version-7 payload shapes: the
file exporter writes their output, the offload decomposes it, and the API
reconstructs it. These tests pin the dict shapes the builders emit so the
offload/reconstruct round-trip has a stable contract to round-trip against.
"""

from __future__ import annotations

from core.contract import SCHEMA_VERSION
from core.database import get_session
from modules.visualization import export as export_mod
from tests.test_visualization_export import (
    _clear,
    _seed_case_with_mir,
    _settings,
)


def _case_payloads(tmp_path, case="video"):
    """Drive the builders for one seeded case via the public bundle helper."""
    session = get_session()
    try:
        return export_mod.build_case_payloads(
            session, settings_viz=_settings(tmp_path).visualization, case=case
        )
    finally:
        session.close()


def test_build_users_payload_returns_six_tuple(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    bundle = _case_payloads(tmp_path)
    users = bundle.users
    assert users["version"] == SCHEMA_VERSION
    assert users["run_id"] == "video"
    assert set(users["bounds"]) == {"minX", "maxX", "minY", "maxY"}
    assert users["users"]
    assert all(len(u) == 6 for u in users["users"])
    # has_detail True for MIR-backed users.
    assert all(u[4] is True for u in users["users"])


def test_build_clusters_payload_has_detail_flag(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    bundle = _case_payloads(tmp_path)
    clusters = bundle.clusters
    assert clusters["version"] == SCHEMA_VERSION
    assert clusters["run_id"] == "video"
    assert clusters["clusters"]
    assert all("has_detail" in c for c in clusters["clusters"])
    assert all(c["has_detail"] is True for c in clusters["clusters"])


def test_build_manifest_entry_shape(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    bundle = _case_payloads(tmp_path)
    entry = bundle.manifest_entry
    assert entry["id"] == "video"
    assert entry["case"] == "video"
    assert entry["details_available"] is True
    assert isinstance(entry["size"], int)
    assert isinstance(entry["label"], str)


def test_build_detail_payloads_validate_against_compute_shape(tmp_path):
    _clear()
    _seed_case_with_mir("video", n_users=4)
    bundle = _case_payloads(tmp_path)
    # creator + cluster detail dicts keyed by id.
    assert bundle.creator_details
    assert bundle.cluster_details
    sample_creator = next(iter(bundle.creator_details.values()))
    assert sample_creator["version"] == SCHEMA_VERSION
    assert "clips" in sample_creator
    assert "spatial" in sample_creator
    # Cluster main detail is version-less (the per-run bundle owns the version)
    # and label-less (the heavy label moved to its own per-cluster map); it
    # carries only the modality so the viewer can place the tag skeleton.
    sample_cluster = next(iter(bundle.cluster_details.values()))
    assert "version" not in sample_cluster
    assert "label" not in sample_cluster
    assert "spatial" in sample_cluster
    assert "label_modality" in sample_cluster


def test_export_still_writes_identical_files(tmp_path):
    """Refactor must not change on-disk bytes: builders feed the writer."""
    _clear()
    _seed_case_with_mir("video", n_users=4)
    export_mod.export_visualization_json(_settings(tmp_path), cases=("video",))
    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["runs"][0]["case"] == "video"
    detail = json.loads((tmp_path / "runs" / "video" / "users" / "0.json").read_text())
    assert detail["version"] == SCHEMA_VERSION
