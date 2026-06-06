"""End-to-end API contract test for the FastAPI backend (plan §2.1, §8.3).

All requests carry the per-user ``X-Access-Code`` header (the auth/identity gate).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from swipe_anchor.backend.app import build_app
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope
from swipe_anchor.db.models import AccessCode, Comparison

CODE = {"X-Access-Code": "tester-1"}


def _session_factory(engine):
    factory = make_session_factory(engine)

    @contextmanager
    def session_factory():
        with session_scope(factory) as s:
            yield s

    return session_factory


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    engine = create_app_engine(str(tmp_path / "app.db"))
    with Session(engine) as s:
        s.add(
            Comparison(
                comparison_id="c1", creator_a=10, creator_b=20, creator_c=30, target_k=5
            )
        )
        s.commit()
    app = build_app(session_factory=_session_factory(engine), token="")
    yield TestClient(app)


def test_next_batch_then_respond_full_loop(client: TestClient) -> None:
    r = client.post("/next-batch", json={"n": 3}, headers=CODE)
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    item = items[0]
    assert {c["creator_id"] for c in item["creators"]} == {10, 20, 30}

    r2 = client.post(
        "/respond",
        json={
            "assignment_id": item["assignment_id"],
            "odd_creator_id": 30,
            "confidence": 1.0,
        },
        headers=CODE,
    )
    assert r2.status_code == 200
    # With default Settings (min_overlap=5), a single response does not yet produce a
    # confident consensus; no triplets are materialized and the comparison stays open.
    body = r2.json()
    assert body["accepted"] is True
    assert body["n_triplets"] == 0
    assert body["retired"] is False


def test_missing_access_code_is_401(client: TestClient) -> None:
    assert client.post("/next-batch", json={"n": 1}).status_code == 401
    assert (
        client.post(
            "/respond", json={"assignment_id": "x", "odd_creator_id": 1}
        ).status_code
        == 401
    )


def test_unrecognised_code_is_403_when_allowlist_present(tmp_path) -> None:
    engine = create_app_engine(str(tmp_path / "app.db"))
    with Session(engine) as s:
        s.add(Comparison(comparison_id="c1", creator_a=10, creator_b=20, creator_c=30))
        s.add(AccessCode(code="known", note="my pal", is_active=True))
        s.commit()
    client = TestClient(build_app(session_factory=_session_factory(engine)))

    assert (
        client.post(
            "/next-batch", json={"n": 1}, headers={"X-Access-Code": "nope"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/next-batch", json={"n": 1}, headers={"X-Access-Code": "known"}
        ).status_code
        == 200
    )


def test_cannot_answer_another_codes_assignment(client: TestClient) -> None:
    items = client.post("/next-batch", json={"n": 1}, headers=CODE).json()["items"]
    assignment_id = items[0]["assignment_id"]

    r = client.post(
        "/respond",
        json={"assignment_id": assignment_id, "odd_creator_id": 30},
        headers={"X-Access-Code": "intruder"},
    )
    assert r.status_code == 403


def test_respond_unknown_assignment_is_404(client: TestClient) -> None:
    r = client.post(
        "/respond",
        json={"assignment_id": "nope", "odd_creator_id": 10},
        headers=CODE,
    )
    assert r.status_code == 404


def test_respond_odd_not_in_comparison_is_422(client: TestClient) -> None:
    items = client.post("/next-batch", json={"n": 1}, headers=CODE).json()["items"]
    r = client.post(
        "/respond",
        json={"assignment_id": items[0]["assignment_id"], "odd_creator_id": 999},
        headers=CODE,
    )
    assert r.status_code == 422


def test_serves_frontend_when_dist_configured(tmp_path) -> None:
    engine = create_app_engine(str(tmp_path / "app.db"))
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>triage</title>")
    app = build_app(session_factory=_session_factory(engine), frontend_dist=str(dist))
    client = TestClient(app)

    # API routes still win over the static mount...
    assert client.get("/health").json() == {"status": "ok"}
    # ...and the SPA is served at the root.
    root = client.get("/")
    assert root.status_code == 200
    assert "triage" in root.text


def test_bearer_token_still_enforced_alongside_code(tmp_path) -> None:
    engine = create_app_engine(str(tmp_path / "app.db"))
    with Session(engine) as s:
        s.add(Comparison(comparison_id="c1", creator_a=10, creator_b=20, creator_c=30))
        s.commit()
    client = TestClient(
        build_app(session_factory=_session_factory(engine), token="secret")
    )

    # Right code but missing bearer -> 401.
    assert client.post("/next-batch", json={"n": 1}, headers=CODE).status_code == 401
    ok = client.post(
        "/next-batch",
        json={"n": 1},
        headers={**CODE, "Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200
