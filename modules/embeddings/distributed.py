"""Drive one case's job set across the local worker + any connected pods.

Starts a JobBroker, a TTL reaper thread, and the in-process local worker
thread(s); when a bearer token is configured it also starts the coordinator
HTTP server so remote pods can lease jobs. Enqueues the jobs; then the
calling thread drains completions, writing one ClipEmbedding row per
success. The caller owns sealing.
"""

from __future__ import annotations

import queue
import threading
import time

from core.console import log
from core.database import ClipEmbedding
from modules.embeddings.broker import JobBroker
from modules.embeddings.worker import LocalJobSource, run_worker


def embed_jobs_distributed(
    settings, secrets, session, spec, jobs, per_clip, log_tag
) -> tuple[int, int]:
    emb = settings.embeddings
    broker = JobBroker(lease_ttl_s=emb.lease_ttl_s, max_attempts=emb.max_attempts)
    for job in jobs:
        broker.add(job)
    broker.producer_done()
    if not jobs:
        return 0, 0

    provider = spec.provider_factory(settings, secrets)
    # The coordinator only exists to feed remote pods over HTTP, and its auth
    # rejects every lease when the token is blank. A local-only run (no token)
    # drains entirely through the in-process worker's LocalJobSource, so skip
    # binding uvicorn — and defer importing it — to keep the optional embedder
    # deps and the bind port out of the local path.
    server = None
    if secrets.embedder_token:
        from modules.embeddings.coordinator import build_app, serve_in_thread

        app = build_app(broker, token=secrets.embedder_token)
        server = serve_in_thread(
            app, host=emb.coordinator_bind_host, port=emb.coordinator_bind_port
        )

    stop = threading.Event()

    def _reaper() -> None:
        while not stop.is_set():
            broker.reap_expired()
            time.sleep(min(emb.lease_ttl_s / 2 or 1, 30))

    reaper = threading.Thread(target=_reaper, daemon=True)
    reaper.start()

    worker = threading.Thread(
        target=run_worker,
        args=(LocalJobSource(broker),),
        kwargs=dict(
            provider=provider,
            video_root=settings.paths.video_dir,
            audio_root=settings.paths.audio_dir,
            inflight=emb.inflight,
            served_only=False,
            log_tag=log_tag,
        ),
        daemon=True,
    )
    worker.start()

    succeeded = 0
    failures = 0
    # If the in-process worker dies (it is the only leaser for served_remotely
    # =False cases and the sole leaser when no pod is connected), outstanding
    # jobs can never resolve and the drain loop would spin forever. Fail loudly
    # once the worker is gone and no completion has landed for a generous
    # window. Any progress (local or pod) resets the clock, so an active
    # pod-fed run never trips this.
    stall_timeout_s = max(emb.lease_ttl_s, 60)
    last_progress = time.monotonic()
    try:
        while not (broker.all_resolved() and broker.completions.empty()):
            try:
                item = broker.completions.get(timeout=0.5)
            except queue.Empty:
                if (
                    not worker.is_alive()
                    and broker.completions.empty()
                    and not broker.all_resolved()
                    and time.monotonic() - last_progress > stall_timeout_s
                ):
                    raise RuntimeError(
                        f"local embedding worker died with unresolved "
                        f"{spec.name} jobs; aborting stage (not sealed)"
                    ) from None
                continue
            last_progress = time.monotonic()
            if item.ok and item.blob is not None:
                blob = item.blob
                session.merge(
                    ClipEmbedding(
                        clip_id=item.clip_id,
                        embedding_case=item.case,
                        embedding=blob,
                        source_hash=per_clip[item.clip_id],
                    )
                )
                session.commit()
                succeeded += 1
                log(
                    log_tag,
                    "EMB",
                    f"clip_{item.clip_id}",
                    "ok",
                    stats={"dim": len(blob) // 4},
                )
            else:
                failures += 1
    finally:
        stop.set()
        worker.join(timeout=30)
        if server is not None:
            server.stop()
    return succeeded, failures
