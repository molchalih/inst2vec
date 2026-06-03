"""Read-only atlas API over the serving DB.

The five endpoints mirror the static JSON paths 1:1. The GOLDEN-EQUALITY test
asserts each endpoint's raw response bytes equal what
``export_visualization_json`` writes to disk for the same DB state — the
strongest guard against normalise→reconstruct drift.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from core.database import get_serving_session, init_serving_db
from modules.visualization import export as export_mod
from services.atlas_api.app import build_app
from tests.test_visualization_export import (
    _clear,
    _seed_case_with_mir,
    _settings,
)


def _client(tmp_path, *, token: str = "", cors_origin: str = "") -> TestClient:
    _clear()
    _seed_case_with_mir("video", n_users=4)
    init_serving_db(f"sqlite:///{tmp_path / 'serving.db'}")
    from scripts.offload_serving import offload

    offload(_settings(tmp_path), cases=("video",))
    app = build_app(
        session_factory=get_serving_session, token=token, cors_origin=cors_origin
    )
    return TestClient(app)


def test_healthz(tmp_path):
    client = _client(tmp_path)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_manifest_endpoint(tmp_path):
    client = _client(tmp_path)
    r = client.get("/manifest.json")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 7
    assert body["runs"][0]["case"] == "video"


def test_users_and_clusters_endpoints(tmp_path):
    client = _client(tmp_path)
    assert client.get("/runs/video/users.json").status_code == 200
    assert client.get("/runs/video/clusters.json").status_code == 200


def test_unknown_run_404(tmp_path):
    client = _client(tmp_path)
    assert client.get("/runs/nope/users.json").status_code == 404
    assert client.get("/runs/nope/clusters.json").status_code == 404


def test_detail_endpoints(tmp_path):
    client = _client(tmp_path)
    assert client.get("/runs/video/users/0.json").status_code == 200
    # Cluster main detail ships as one per-run bundle.
    assert client.get("/runs/video/clusters-detail.json").status_code == 200


def test_missing_detail_404(tmp_path):
    client = _client(tmp_path)
    assert client.get("/runs/video/users/999999.json").status_code == 404
    # The synthetic seed has no cluster labels → every label endpoint 404s.
    assert client.get("/runs/video/clusters/999999.label.json").status_code == 404
    assert client.get("/runs/video/clusters/0.label.json").status_code == 404
    # The bundle 404s only for an unknown run.
    assert client.get("/runs/nope/clusters-detail.json").status_code == 404


def test_token_gates_endpoints(tmp_path):
    client = _client(tmp_path, token="secret")
    assert client.get("/manifest.json").status_code == 401
    ok = client.get("/manifest.json", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_cors_origin_allowed(tmp_path):
    origin = "https://example.github.io"
    client = _client(tmp_path, cors_origin=origin)
    r = client.get("/manifest.json", headers={"Origin": origin})
    assert r.headers.get("access-control-allow-origin") == origin


def test_entrypoint_requires_serving_url(monkeypatch):
    from services.atlas_api.__main__ import _build_from_env

    monkeypatch.delenv("SERVING_DATABASE_URL", raising=False)
    monkeypatch.setattr("services.atlas_api.__main__.load_dotenv", lambda: None)
    with __import__("pytest").raises(SystemExit):
        _build_from_env()


def test_entrypoint_boots_and_healthz(tmp_path, monkeypatch):
    from services.atlas_api.__main__ import _build_from_env

    monkeypatch.setenv("SERVING_DATABASE_URL", f"sqlite:///{tmp_path / 'serving.db'}")
    monkeypatch.setattr("services.atlas_api.__main__.load_dotenv", lambda: None)
    app = _build_from_env()
    assert TestClient(app).get("/healthz").status_code == 200


def test_golden_byte_equality_all_endpoints(tmp_path):
    """API bytes == the bytes the file exporter writes for the same DB."""
    _clear()
    _seed_case_with_mir("video", n_users=4)
    export_dir = tmp_path / "export"
    settings = _settings(export_dir)
    settings.visualization.export_dir = export_dir
    export_mod.export_visualization_json(settings, cases=("video",))

    init_serving_db(f"sqlite:///{tmp_path / 'serving.db'}")
    from scripts.offload_serving import offload

    offload(settings, cases=("video",))
    client = TestClient(build_app(session_factory=get_serving_session))

    def _file_bytes(*parts: str) -> bytes:
        return (export_dir.joinpath(*parts)).read_bytes()

    # Discover which ids the exporter actually wrote.
    with get_serving_session() as s:
        from core.database import ServingCluster, ServingUser

        user_ids = [
            u.user_id
            for u in s.query(ServingUser).filter_by(run_id="video", has_detail=True)
        ]
        cluster_ids = [
            c.cluster_id
            for c in s.query(ServingCluster).filter_by(run_id="video", has_detail=True)
        ]
    assert user_ids and cluster_ids

    assert client.get("/manifest.json").content == _file_bytes("manifest.json")
    assert client.get("/runs/video/users.json").content == _file_bytes(
        "runs", "video", "users.json"
    )
    assert client.get("/runs/video/clusters.json").content == _file_bytes(
        "runs", "video", "clusters.json"
    )
    for uid in user_ids:
        assert client.get(f"/runs/video/users/{uid}.json").content == _file_bytes(
            "runs", "video", "users", f"{uid}.json"
        )
    # Cluster main detail: one per-run bundle, byte-identical to the file.
    assert client.get("/runs/video/clusters-detail.json").content == _file_bytes(
        "runs", "video", "clusters-detail.json"
    )
    # Deferred label files, byte-identical to the API label endpoint (none for
    # the synthetic seed, which has no ClusterLabel rows).
    label_dir = export_dir / "runs" / "video" / "clusters"
    for f in sorted(label_dir.glob("*.label.json")):
        cid = int(f.name[: -len(".label.json")])
        assert (
            client.get(f"/runs/video/clusters/{cid}.label.json").content
            == f.read_bytes()
        )
    assert cluster_ids  # exporter wrote main detail for these

    # Sanity: the exporter's own files load as valid JSON (catch empty reads).
    assert json.loads(_file_bytes("manifest.json"))["version"] == 7
