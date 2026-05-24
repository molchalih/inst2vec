"""Coordinator + one HttpJobSource worker drain to completion over loopback."""

from __future__ import annotations

import socket
import threading

from modules.embeddings.broker import JobBroker, make_job
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
                video_key=f"videos/{i}.mp4",
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
