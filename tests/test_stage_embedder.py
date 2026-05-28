"""StageEmbedder drains multiple cases over one persistent local worker.

Uses a fake provider (patched build_provider_router) and a fake session so
no Qwen model and no DB are needed. Token is blank → no coordinator server.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

import modules.embeddings.distributed as dist
from core import log as cl
from modules.embeddings.broker import make_job
from modules.embeddings.cases import CASE_REGISTRY
from modules.embeddings.distributed import StageEmbedder
from modules.embeddings.providers import ProviderRouter


@pytest.fixture(autouse=True)
def _reset_scope() -> Iterator[None]:
    """Ensure each test starts with no stale scope."""
    token = cl._scope_var.set(None)
    yield
    cl._scope_var.reset(token)


class _FakeSession:
    def __init__(self):
        self.merged = []

    def merge(self, obj):
        self.merged.append(obj)

    def commit(self):
        pass


class _StubProvider:
    def embed(self, payload):
        return [[float(payload["clip_id"]), 0.0]]


@dataclass
class _Emb:
    lease_ttl_s: int = 600
    max_attempts: int = 3
    inflight: int = 1
    coordinator_bind_host: str = "127.0.0.1"
    coordinator_bind_port: int = 0
    pod_drain_grace_s: float = 10.0
    drain_poll_s: float = 0.01


@dataclass
class _Paths:
    video_dir: str = "/v"
    audio_dir: str = "/a"


@dataclass
class _Settings:
    embeddings: _Emb
    paths: _Paths


@dataclass
class _Secrets:
    embedder_token: str = ""
    gemini_api_key: str | None = None


def _patch_router(monkeypatch):
    stub = _StubProvider()
    monkeypatch.setattr(
        dist,
        "build_provider_router",
        lambda *a, **kw: ProviderRouter({n: (lambda s=stub: s) for n in CASE_REGISTRY}),
    )


def _video_job(cid):
    return make_job(
        clip_id=cid,
        case="video",
        text=None,
        video_key=f"{cid}.mp4",
        fps=2.0,
        max_frames=32,
        remote_eligible=True,
    )


def _maest_job(cid):
    return make_job(
        clip_id=cid,
        case="maest",
        text=None,
        video_key=None,
        fps=None,
        max_frames=None,
        remote_eligible=False,
        audio_key=f"{cid}.mp3",
    )


def test_drains_single_case(monkeypatch):
    _patch_router(monkeypatch)
    settings = _Settings(_Emb(), _Paths())
    sess = _FakeSession()
    cl._scope_var.set("embed:video")
    with StageEmbedder(settings, _Secrets(), ["video"]) as stage:
        ok, fail = stage.drain_case(
            sess,
            CASE_REGISTRY["video"],
            [_video_job(1), _video_job(2)],
            {1: "h1", 2: "h2"},
            "embed:video",
        )
    assert (ok, fail) == (2, 0)
    assert {m.clip_id for m in sess.merged} == {1, 2}
    assert {m.embedding_case for m in sess.merged} == {"video"}


def test_one_worker_persists_across_two_cases(monkeypatch):
    _patch_router(monkeypatch)
    settings = _Settings(_Emb(), _Paths())
    sess = _FakeSession()
    with StageEmbedder(settings, _Secrets(), ["video", "maest"]) as stage:
        worker_before = stage._worker
        cl._scope_var.set("embed:video")
        ok1, _ = stage.drain_case(
            sess, CASE_REGISTRY["video"], [_video_job(1)], {1: "h"}, "embed:video"
        )
        cl._scope_var.set("embed:maest")
        ok2, _ = stage.drain_case(
            sess, CASE_REGISTRY["maest"], [_maest_job(3)], {3: "h"}, "embed:maest"
        )
        assert stage._worker is worker_before  # same long-lived worker
    assert ok1 == 1 and ok2 == 1
    assert {(m.clip_id, m.embedding_case) for m in sess.merged} == {
        (1, "video"),
        (3, "maest"),
    }


def test_close_holds_coordinator_open_for_drain_grace():
    # With a token set the coordinator serves remote pods. close() must keep it
    # up for pod_drain_grace_s after marking the broker drained so pods sleeping
    # between /lease polls observe HTTP 410 (and exit 0) rather than hitting a
    # connection error on a stopped server. Test close() in isolation with an
    # injected fake server so the local worker's poll timing can't confound it.
    grace = 0.3

    class _FakeServer:
        stopped_at: float | None = None

        def stop(self):
            self.stopped_at = time.monotonic()

    settings = _Settings(_Emb(pod_drain_grace_s=grace), _Paths())
    stage = StageEmbedder(settings, _Secrets(embedder_token="t"), ["video"])
    fake = _FakeServer()
    stage._server = fake  # no _start(): _worker/_reaper stay None
    t0 = time.monotonic()
    stage.close()
    assert fake.stopped_at is not None, "coordinator must be stopped on close"
    assert fake.stopped_at - t0 >= grace * 0.9


def test_close_stops_fleet_scaling_before_drain_grace():
    # close() must halt the fleet's background top-up before sleeping
    # pod_drain_grace_s: a pod launched during the grace would lease nothing,
    # get 410, and be billed for boot. stop_scaling must fire, and fire before
    # the grace-delayed server stop. Inject fakes so no real fleet/server is hit.
    grace = 0.3

    class _FakeServer:
        stopped_at: float | None = None

        def stop(self):
            self.stopped_at = time.monotonic()

    class _TimedFleet:
        stopped_at: float | None = None

        def ensure_started(self):
            pass

        def stop_scaling(self):
            self.stopped_at = time.monotonic()

    settings = _Settings(_Emb(pod_drain_grace_s=grace), _Paths())
    fleet = _TimedFleet()
    stage = StageEmbedder(
        settings, _Secrets(embedder_token="t"), ["video"], fleet=fleet
    )
    fake = _FakeServer()
    stage._server = fake  # no _start(): _worker/_reaper stay None
    t0 = time.monotonic()
    stage.close()
    assert fleet.stopped_at is not None, "fleet scaling must be stopped on close"
    # halted before the grace sleep even began (so well before the server stop)
    assert fleet.stopped_at - t0 < grace * 0.5
    assert fake.stopped_at is not None and fleet.stopped_at <= fake.stopped_at


def test_empty_case_is_a_noop(monkeypatch):
    _patch_router(monkeypatch)
    settings = _Settings(_Emb(), _Paths())
    sess = _FakeSession()
    with StageEmbedder(settings, _Secrets(), ["video"]) as stage:
        ok, fail = stage.drain_case(sess, CASE_REGISTRY["video"], [], {}, "embed:video")
    assert (ok, fail) == (0, 0) and sess.merged == []


class _SpyFleet:
    def __init__(self):
        self.started = 0
        self.scaling_stopped = 0

    def ensure_started(self):
        self.started += 1

    def stop_scaling(self):
        self.scaling_stopped += 1


def test_fleet_started_only_for_remote_leaseable_jobs(monkeypatch):
    """The fleet is deployed lazily: drain_case must trigger it only when a job
    a pod could actually lease is enqueued (served_remotely case + uploaded clip),
    never for a local-only case or an all-local batch."""
    _patch_router(monkeypatch)
    settings = _Settings(_Emb(), _Paths())
    sess = _FakeSession()
    fleet = _SpyFleet()
    cl._scope_var.set("embed:maest")
    with StageEmbedder(settings, _Secrets(), ["video", "maest"], fleet=fleet) as stage:
        # local-only case (maest is served_remotely=False) -> no deploy
        stage.drain_case(sess, CASE_REGISTRY["maest"], [_maest_job(1)], {1: "h"}, "t")
        assert fleet.started == 0
        # video case but the clip is not uploaded -> nothing a pod could lease
        local_video = make_job(
            clip_id=2,
            case="video",
            text=None,
            video_key="2.mp4",
            fps=2.0,
            max_frames=32,
            remote_eligible=False,
        )
        cl._scope_var.set("embed:video")
        stage.drain_case(sess, CASE_REGISTRY["video"], [local_video], {2: "h"}, "t")
        assert fleet.started == 0
        # video case with an uploaded clip -> a pod can lease it -> deploy
        stage.drain_case(sess, CASE_REGISTRY["video"], [_video_job(3)], {3: "h"}, "t")
        assert fleet.started == 1
