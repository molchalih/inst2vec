"""/metrics surfaces headline agreement + counts (design §3.7)."""

from contextlib import contextmanager

from fastapi.testclient import TestClient

from swipe_anchor.backend.app import build_app
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope
from swipe_anchor.db.models import Annotator, Comparison, Consensus


def _client() -> TestClient:
    engine = create_app_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)

    @contextmanager
    def session_factory():
        with session_scope(factory) as s:
            yield s

    with session_factory() as s:
        s.add(Annotator(annotator_id="a", reliability=0.8))
        s.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3, status="retired"))
        s.add(Consensus(comparison_id="c1", consensus_odd=3, agreement=1.0, resolved=True))
    return TestClient(build_app(session_factory, settings=Settings()))


def test_metrics_reports_counts_and_agreement() -> None:
    r = _client().get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["n_annotators"] == 1
    assert body["comparisons"]["retired"] == 1
    assert "fleiss_kappa" in body["agreement"]
    assert 0.0 <= body["mean_reliability"] <= 1.0


def test_settings_propagate_to_request_path() -> None:
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    from swipe_anchor.backend.app import build_app
    from swipe_anchor.config import Settings
    from swipe_anchor.db import create_app_engine, make_session_factory, session_scope
    from swipe_anchor.db.models import Comparison

    engine = create_app_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)

    @contextmanager
    def session_factory():
        with session_scope(factory) as s:
            yield s

    with session_factory() as s:
        s.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3, target_k=5))

    # min_overlap=1 + threshold 0 → one answer must retire the comparison.
    client = TestClient(build_app(session_factory, settings=Settings(min_overlap=1, confidence_threshold=0.0)))
    batch = client.post("/next-batch", json={"n": 1}, headers={"X-Access-Code": "tester"}).json()
    item = batch["items"][0]
    odd = item["creators"][0]["creator_id"]
    resp = client.post("/respond", json={"assignment_id": item["assignment_id"], "odd_creator_id": odd}, headers={"X-Access-Code": "tester"}).json()
    assert resp["retired"] is True  # only true if low-threshold settings reached record_response
    # And confirm the metrics reflect the retirement.
    metrics = client.get("/metrics").json()
    assert metrics["comparisons"]["retired"] == 1
