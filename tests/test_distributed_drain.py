"""Coordinator + one HttpJobSource worker drain to completion over loopback.

Also contains unit tests for drain_case log emission (T24).
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from core import log as cl
from modules.embeddings.broker import Completion, JobBroker, make_job
from modules.embeddings.coordinator import build_app, serve_in_thread
from modules.embeddings.worker import HttpJobSource, run_worker


class _StubProvider:
    def embed(self, payload):
        return [[float(payload["clip_id"]), 0.0]]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_http_worker_drains_all_jobs(tmp_path):
    for i in range(6):
        (tmp_path / f"{i}.mp4").write_bytes(b"x")
    broker = JobBroker(lease_ttl_s=600, max_attempts=3)
    for i in range(6):
        broker.add(
            make_job(
                clip_id=i,
                case="video",
                text=None,
                video_key=f"{i}.mp4",
                fps=2.0,
                max_frames=96,
                remote_eligible=True,
            )
        )
    broker.producer_done()

    port = _free_port()
    server = serve_in_thread(build_app(broker, token="t"), host="127.0.0.1", port=port)
    try:
        source = HttpJobSource(
            base_url=f"http://127.0.0.1:{port}",
            token="t",
            timeout_s=5,
            max_retries=2,
        )
        # Drain in a thread; main thread consumes completions.
        wt = threading.Thread(
            target=run_worker,
            args=(source,),
            kwargs=dict(
                provider=_StubProvider(),
                video_root=str(tmp_path),
                inflight=2,
                served_only=True,
                poll_idle_s=0.02,
            ),
            daemon=True,
        )
        wt.start()

        got = []
        import queue

        while not (broker.all_resolved() and broker.completions.empty()):
            try:
                got.append(broker.completions.get(timeout=0.5))
            except queue.Empty:
                continue
        wt.join(timeout=10)
    finally:
        server.stop()

    assert len(got) == 6
    assert all(c.ok for c in got)
    assert {c.clip_id for c in got} == set(range(6))


# ---------------------------------------------------------------------------
# T24: drain_case log emission — EXTRACT with time + dim, ERR on failure
# ---------------------------------------------------------------------------


@dataclass
class _LogEntry:
    scope: str
    verb: str
    target: str
    result: str
    stats: dict[str, Any]


@pytest.fixture
def captured_log(monkeypatch: pytest.MonkeyPatch) -> list[_LogEntry]:
    """Capture core.log._render calls as _LogEntry objects."""
    sink: list[_LogEntry] = []

    def fake_render(
        scope: str,
        verb: str,
        target: str,
        result: str = "ok",
        *,
        stats: dict[str, Any] | None = None,
    ) -> None:
        sink.append(_LogEntry(scope, verb, target, result, dict(stats or {})))

    monkeypatch.setattr(cl, "_render", fake_render)
    return sink


@pytest.fixture(autouse=True)
def _reset_scope_t24() -> Any:
    """Ensure drain tests start with no stale scope."""
    token = cl._scope_var.set(None)
    yield
    cl._scope_var.reset(token)


class _ImmediateBroker(JobBroker):
    """JobBroker that resolves each added job with a pre-configured Completion.

    When ``add(job)`` is called the broker increments its counters normally,
    then immediately pushes the configured completion and decrements the
    outstanding counter — mimicking a worker that finishes instantly.
    """

    def __init__(self, completion: Completion, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._completion = completion

    def add(self, job: dict) -> None:  # type: ignore[override]
        super().add(job)
        # Update counters as if the job resolved immediately.
        with self._lock:
            c = self._counters[job["case"]]
            if self._completion.ok:
                c.succeeded += 1
            else:
                c.failed += 1
            c.outstanding -= 1
            self.completions.put(self._completion)


class _FakeSession:
    def __init__(self) -> None:
        self.merged: list[Any] = []

    def merge(self, obj: Any) -> None:
        self.merged.append(obj)

    def commit(self) -> None:
        pass


@dataclass
class _Emb:
    lease_ttl_s: int = 600
    max_attempts: int = 3
    inflight: int = 1
    coordinator_bind_host: str = "127.0.0.1"
    coordinator_bind_port: int = 0
    pod_drain_grace_s: float = 0.0
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


def _make_drain_case(
    completion: Completion,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any]:
    """Return (stage_emb, session) ready to drain a single video job.

    The StageEmbedder is started with a broker that resolves every added job
    immediately with ``completion``, so drain_case exits after processing one
    item without needing a real embedding worker.
    """
    import modules.embeddings.distributed as dist
    from modules.embeddings.cases import CASE_REGISTRY
    from modules.embeddings.distributed import StageEmbedder
    from modules.embeddings.providers import ProviderRouter

    stub = _StubProvider()
    monkeypatch.setattr(
        dist,
        "build_provider_router",
        lambda *a, **kw: ProviderRouter({n: (lambda s=stub: s) for n in CASE_REGISTRY}),
    )
    settings = _Settings(_Emb(), _Paths())
    stage = StageEmbedder.__new__(StageEmbedder)
    stage._settings = settings
    stage._secrets = _Secrets()
    stage._cases = ["video"]
    stage._fleet = None
    stage._broker = _ImmediateBroker(completion, lease_ttl_s=600, max_attempts=3)
    stage._stop = threading.Event()
    stage._server = None
    stage._reaper = None
    stage._worker = None
    return stage, _FakeSession()


def _video_job(cid: int) -> dict:
    return make_job(
        clip_id=cid,
        case="video",
        text=None,
        video_key=f"{cid}.mp4",
        fps=2.0,
        max_frames=32,
        remote_eligible=False,
    )


def test_drain_loop_logs_extract_with_time_and_dim(
    captured_log: list[_LogEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from modules.embeddings.cases import CASE_REGISTRY

    blob = b"\x00" * 8192  # 8192 bytes / 4 = 2048 floats
    completion = Completion(clip_id=42, case="video", blob=blob, ok=True)
    stage, sess = _make_drain_case(completion, monkeypatch)

    cl._scope_var.set("embed:video")
    stage.drain_case(
        sess,
        CASE_REGISTRY["video"],
        [_video_job(42)],
        {42: "hash42"},
        "embed:video",
    )

    extract_entries = [
        e for e in captured_log if e.verb == "EXTRACT" and e.target == "clip_42"
    ]
    assert extract_entries, "no EXTRACT clip_42 line emitted"
    entry = extract_entries[0]
    assert entry.result == "ok", f"expected ok, got {entry.result}"
    assert "time" in entry.stats, "time= must appear on success lines"
    assert entry.stats.get("dim") == 2048, "dim= must equal blob_bytes // 4"


def test_drain_loop_logs_err_on_failed_job(
    captured_log: list[_LogEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from modules.embeddings.cases import CASE_REGISTRY

    completion = Completion(
        clip_id=42, case="video", blob=None, ok=False, err="CudaOOM"
    )
    stage, sess = _make_drain_case(completion, monkeypatch)

    cl._scope_var.set("embed:video")
    stage.drain_case(
        sess,
        CASE_REGISTRY["video"],
        [_video_job(42)],
        {42: "hash42"},
        "embed:video",
    )

    extract_entries = [
        e for e in captured_log if e.verb == "EXTRACT" and e.target == "clip_42"
    ]
    assert extract_entries, "no EXTRACT clip_42 line emitted for failure"
    entry = extract_entries[0]
    assert entry.result == "ERR", f"expected ERR, got {entry.result}"
    assert "err" in entry.stats, "err= must appear on failure lines"
