"""Reconstruction layer: serving rows → version-6 payload dicts.

Round-trip inverse of the offload's decompose: for a seeded main DB, the
reconstructed dicts must deep-equal the builder payloads (the single shape
producer), with array order preserved via ``ord`` and optional/nullable
blocks gated on row-presence / NULL.
"""

from __future__ import annotations

import pytest

from core.database import get_serving_session, get_session, init_serving_db
from modules.visualization import export as export_mod
from services.atlas_api import reconstruct
from tests.test_visualization_export import (
    _clear,
    _seed_case_with_mir,
    _settings,
)


def _setup(tmp_path, case="video"):
    from scripts.offload_serving import offload

    _clear()
    _seed_case_with_mir(case, n_users=4)
    init_serving_db(f"sqlite:///{tmp_path / 'serving.db'}")
    offload(_settings(tmp_path), cases=(case,))


def _bundle(tmp_path, case="video"):
    session = get_session()
    try:
        return export_mod.build_case_payloads(
            session, settings_viz=_settings(tmp_path).visualization, case=case
        )
    finally:
        session.close()


def test_reconstruct_manifest_matches_builder(tmp_path):
    _setup(tmp_path)
    bundle = _bundle(tmp_path)
    expected = export_mod.build_manifest_payload("video", [bundle.manifest_entry])
    with get_serving_session() as s:
        assert reconstruct.reconstruct_manifest(s) == expected


def test_reconstruct_users_matches_builder(tmp_path):
    _setup(tmp_path)
    bundle = _bundle(tmp_path)
    with get_serving_session() as s:
        assert reconstruct.reconstruct_users(s, "video") == bundle.users


def test_reconstruct_clusters_matches_builder(tmp_path):
    _setup(tmp_path)
    bundle = _bundle(tmp_path)
    with get_serving_session() as s:
        assert reconstruct.reconstruct_clusters(s, "video") == bundle.clusters


def test_reconstruct_creator_detail_matches_builder(tmp_path):
    _setup(tmp_path)
    bundle = _bundle(tmp_path)
    with get_serving_session() as s:
        for uid, expected in bundle.creator_details.items():
            assert reconstruct.reconstruct_creator_detail(s, "video", uid) == expected


def test_reconstruct_cluster_detail_matches_builder(tmp_path):
    _setup(tmp_path)
    bundle = _bundle(tmp_path)
    with get_serving_session() as s:
        for cid, expected in bundle.cluster_details.items():
            assert reconstruct.reconstruct_cluster_detail(s, "video", cid) == expected


def test_reconstruct_missing_detail_returns_none(tmp_path):
    _setup(tmp_path)
    with get_serving_session() as s:
        assert reconstruct.reconstruct_creator_detail(s, "video", 999999) is None
        assert reconstruct.reconstruct_cluster_detail(s, "video", 999999) is None


def test_reconstruct_unknown_run_users_raises(tmp_path):
    _setup(tmp_path)
    with get_serving_session() as s, pytest.raises(KeyError):
        reconstruct.reconstruct_users(s, "nope")


# ── Shipped-fixture round-trip (rich label block + grounded clips + nearest) ──


def test_shipped_details_round_trip_byte_identical(tmp_path):
    """Decompose the actually-shipped detail JSON → rows → reconstruct, and
    assert byte equality. Exercises the rich paths the synthetic seed omits:
    the cluster ``label`` block, grounded clip tags with ``grounded_in``, and a
    non-null ``nearest_other_cluster``.
    """
    import glob
    import json
    from pathlib import Path

    from core.database import ServingRun
    from core.database.serving_decompose import (
        _cluster_detail_rows,
        _user_detail_rows,
    )
    from services.atlas_api.serialize import to_bytes

    data = Path(__file__).resolve().parents[1] / "frontend" / "public" / "data"
    if not (data / "runs" / "video").is_dir():
        pytest.skip(
            "static data bundle absent (API-only); round-trip covered by "
            "tests/test_offload_serving.py"
        )
    init_serving_db(f"sqlite:///{tmp_path / 'serving.db'}")

    cluster_files = sorted(
        glob.glob(str(data / "runs" / "video" / "clusters" / "*.json"))
    )
    user_files = sorted(glob.glob(str(data / "runs" / "video" / "users" / "*.json")))
    assert cluster_files and user_files

    with get_serving_session() as s:
        # A ServingRun row is required so _require_run-style lookups would pass;
        # detail reconstructors only need the detail rows, but add it for realism.
        s.add(
            ServingRun(
                run_id="video",
                case="video",
                label="Visual",
                size=len(user_files),
                details_available=True,
                manifest_ord=0,
                is_default=True,
                schema_version=6,
            )
        )
        cluster_expected: dict[int, bytes] = {}
        for f in cluster_files:
            d = json.loads(Path(f).read_text())
            cid = d["cluster_id"]
            cluster_expected[cid] = Path(f).read_bytes()
            s.add_all(_cluster_detail_rows("video", cid, d))
        user_expected: dict[int, bytes] = {}
        for f in user_files:
            d = json.loads(Path(f).read_text())
            uid = d["user_id"]
            user_expected[uid] = Path(f).read_bytes()
            s.add_all(_user_detail_rows("video", uid, d))
        s.commit()

    with get_serving_session() as s:
        for cid, expected in cluster_expected.items():
            got = reconstruct.reconstruct_cluster_detail(s, "video", cid)
            assert to_bytes(got) == expected, f"cluster {cid}"
        for uid, expected in user_expected.items():
            got = reconstruct.reconstruct_creator_detail(s, "video", uid)
            assert to_bytes(got) == expected, f"user {uid}"
