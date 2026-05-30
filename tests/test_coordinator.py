from __future__ import annotations

from fastapi.testclient import TestClient

from modules.embeddings.broker import JobBroker, make_job
from modules.embeddings.coordinator import build_app


def _client(broker, token="t"):
    return TestClient(build_app(broker, token=token))


def _job(clip_id):
    return make_job(
        clip_id=clip_id,
        case="video",
        text=None,
        video_key=f"videos/{clip_id}.mp4",
        fps=2.0,
        max_frames=96,
        remote_eligible=True,
    )


def test_auth_required():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    c = _client(b)
    assert c.post("/lease", json={"served_only": True}).status_code == 401
    assert c.post(
        "/lease", json={"served_only": True}, headers={"Authorization": "Bearer t"}
    ).status_code in (200, 204, 410)


def test_lease_complete_flow_writes_completion():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(_job(1))
    b.producer_done()
    c = _client(b)
    h = {"Authorization": "Bearer t"}
    r = c.post("/lease", json={"served_only": True}, headers=h)
    assert r.status_code == 200
    lease_id = r.json()["lease_id"]
    assert r.json()["job"]["clip_id"] == 1
    ok = c.post(
        "/complete", json={"lease_id": lease_id, "embedding": [1.0, 2.0]}, headers=h
    )
    assert ok.status_code == 200
    item = b.completions.get_nowait()
    assert item.ok and item.clip_id == 1
    # bytes round-trip: 2 float32 = 8 bytes
    assert len(item.blob) == 8


def test_fail_endpoint_routes_to_broker():
    b = JobBroker(lease_ttl_s=600, max_attempts=1)
    b.add(_job(1))
    b.producer_done()
    c = _client(b)
    h = {"Authorization": "Bearer t"}
    lease_id = c.post("/lease", json={"served_only": True}, headers=h).json()[
        "lease_id"
    ]
    assert (
        c.post(
            "/fail", json={"lease_id": lease_id, "error": "boom"}, headers=h
        ).status_code
        == 200
    )
    assert b.case_failures("video") == 1


def test_healthz():
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    assert _client(b).get("/healthz").json()["status"] == "ok"
