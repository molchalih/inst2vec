"""Stage-scoped distributed clip embedding.

One JobBroker + (optional) coordinator HTTP server + TTL reaper + in-process
local worker live for the WHOLE clip-embedding stage. Each case enqueues its
jobs via ``drain_case`` and the calling thread drains that case's completions
(single writer → DB) until the case's outstanding count hits zero. The
broker's ``producer_done()`` — the only thing that makes the coordinator
answer HTTP 410 "drained" — is deferred to ``close()``, so connected pods see
a stable endpoint across cases and exit only when the stage finishes.

The local worker drains every case (served_only=False) through one
ProviderRouter, which shares a single Qwen instance across the Qwen-backbone
cases. Pods (served_only=True) lease only served_remotely cases.
"""

from __future__ import annotations

import queue
import threading
import time

from core.console import log
from core.database import ClipEmbedding
from modules.embeddings.broker import JobBroker
from modules.embeddings.cases import build_provider_router
from modules.embeddings.worker import LocalJobSource, run_worker


class StageEmbedder:
    """Open once per clip-embedding stage; drain each case via ``drain_case``."""

    def __init__(self, settings, secrets, cases: list[str], fleet=None) -> None:
        self._settings = settings
        self._secrets = secrets
        self._cases = list(cases)
        # The RunPod pod fleet (or None). Deployment is deferred until the first
        # remote-leaseable job is enqueued, so a sealed/all-local run never pays
        # for pods that would lease nothing — see drain_case.
        self._fleet = fleet
        emb = settings.embeddings
        self._broker = JobBroker(
            lease_ttl_s=emb.lease_ttl_s, max_attempts=emb.max_attempts
        )
        self._stop = threading.Event()
        self._server = None
        self._reaper: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    def __enter__(self) -> StageEmbedder:
        self._start()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def _start(self) -> None:
        try:
            emb = self._settings.embeddings
            # Coordinator only feeds remote pods, and its auth rejects every lease
            # when the token is blank. A local-only run drains entirely through the
            # in-process worker, so skip binding uvicorn (and defer importing it).
            if self._secrets.embedder_token:
                from modules.embeddings.coordinator import build_app, serve_in_thread

                app = build_app(self._broker, token=self._secrets.embedder_token)
                self._server = serve_in_thread(
                    app, host=emb.coordinator_bind_host, port=emb.coordinator_bind_port
                )

            interval = min(max(emb.lease_ttl_s / 2, 1), 30)

            def _reap() -> None:
                while not self._stop.is_set():
                    self._broker.reap_expired()
                    self._stop.wait(interval)

            self._reaper = threading.Thread(target=_reap, daemon=True)
            self._reaper.start()

            router = build_provider_router(self._settings, self._secrets, self._cases)
            self._worker = threading.Thread(
                target=run_worker,
                args=(LocalJobSource(self._broker),),
                kwargs=dict(
                    provider=router,
                    video_root=self._settings.paths.video_dir,
                    audio_root=self._settings.paths.audio_dir,
                    inflight=emb.inflight,
                    served_only=False,
                    log_tag="embed:local",
                ),
                daemon=True,
            )
            self._worker.start()
        except BaseException:
            # A failure after the server/reaper started would otherwise leak
            # the bound coordinator port; tear down what we started.
            self.close()
            raise

    def drain_case(
        self, session, spec, jobs, per_clip, log_tag: str
    ) -> tuple[int, int]:
        """Enqueue ``jobs`` for ``spec`` and write each success to ``session``.

        Returns (succeeded, failures). Blocks until the case's outstanding
        count is zero. Cases are drained strictly sequentially, so the shared
        completion queue only ever holds this case's items while we drain it.
        The caller must invoke ``drain_case`` from a single thread, never
        concurrently for two cases, since they share one completion queue."""
        if not jobs:
            return 0, 0
        # A pod can lease a job only when the case is served_remotely AND the clip
        # was uploaded; deploy the fleet lazily the moment such a job appears.
        if (
            self._fleet is not None
            and spec.served_remotely
            and any(job["remote_eligible"] for job in jobs)
        ):
            self._fleet.ensure_started()
        for job in jobs:
            self._broker.add(job)

        succeeded = 0
        failures = 0
        case = spec.name
        stall_timeout_s = max(self._settings.embeddings.lease_ttl_s, 60)
        last_progress = time.monotonic()
        while not (
            self._broker.case_outstanding(case) == 0
            and self._broker.completions.empty()
        ):
            try:
                item = self._broker.completions.get(timeout=0.5)
            except queue.Empty:
                if (
                    self._worker is not None
                    and not self._worker.is_alive()
                    and self._broker.completions.empty()
                    and self._broker.case_outstanding(case) > 0
                    and time.monotonic() - last_progress > stall_timeout_s
                ):
                    raise RuntimeError(
                        f"local embedding worker died with unresolved {case} "
                        f"jobs; aborting stage (not sealed)"
                    ) from None
                continue
            last_progress = time.monotonic()
            if item.ok and item.blob is not None:
                session.merge(
                    ClipEmbedding(
                        clip_id=item.clip_id,
                        embedding_case=item.case,
                        embedding=item.blob,
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
                    stats={"dim": len(item.blob) // 4},
                )
            else:
                failures += 1
        return succeeded, failures

    def close(self) -> None:
        # Mark the stage closed so the coordinator answers 410 and the local
        # worker's LocalJobSource sees DRAINED; then tear everything down.
        self._broker.producer_done()
        # No more jobs will be enqueued, so halt the fleet's background top-up
        # before the drain grace below — otherwise a poll firing (or a deploy
        # already in flight) during the grace launches a pod that leases nothing,
        # gets 410, and is billed for boot. pod_fleet still reaps on its own exit.
        if self._fleet is not None:
            self._fleet.stop_scaling()
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=30)
        if self._reaper is not None:
            self._reaper.join(timeout=5)
        if self._server is not None:
            # producer_done() above makes /lease answer HTTP 410. Pods poll
            # sub-second, so keep serving a grace window before stopping uvicorn:
            # a pod sleeping between polls then sees the 410 (and exits 0) rather
            # than a connection error that fails an otherwise-successful stage.
            grace = self._settings.embeddings.pod_drain_grace_s
            if grace > 0:
                time.sleep(grace)
            self._server.stop()
