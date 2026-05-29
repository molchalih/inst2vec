"""Run this process as an embedding pull-worker pod.

Loads pod-only config (no DB/ingest secrets), builds the local Qwen
provider on the pod's GPU, and drains the orchestrator's coordinator over
HTTP until it signals the queue is empty. Shared by ``main.py --pod`` and
the container entrypoint (``python -m services.embedder``); lives in the
embeddings package so the image — which copies modules/ but not main.py —
can import it.
"""

from __future__ import annotations

import os
import time

import httpx

from core.config import load_pod_config
from core.log import event, stage
from modules.embeddings.cases import build_provider_router, remote_served_cases
from modules.embeddings.worker import HttpJobSource, run_worker


def _coordinator_base_url(host: str) -> str:
    """Build the coordinator base URL from COORDINATOR_PUBLIC_HOST.

    A bare ``host:port`` (raw port-forward) defaults to ``http://``; a value
    that already carries a scheme (e.g. ``https://x.trycloudflare.com`` behind
    a TLS tunnel) is used verbatim so the pod speaks HTTPS to the tunnel."""
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}"


def wait_for_coordinator(
    base_url: str,
    *,
    timeout_s: int,
    poll_s: float = 3.0,
    _client: httpx.Client | None = None,
) -> None:
    """Block until the coordinator's /healthz answers 200, or raise SystemExit.

    Lets a pod be launched before the orchestrator reaches the embedding
    stage: it polls (no auth) until the endpoint appears, then returns so the
    model loads only once there is a coordinator to serve."""
    owns_client = _client is None
    client = _client if _client is not None else httpx.Client(timeout=5)
    url = base_url.rstrip("/") + "/healthz"
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"coordinator at {base_url} not reachable within {timeout_s}s"
                )
            time.sleep(poll_s)
    finally:
        if owns_client:
            client.close()


@stage("embed:pod")
def run_pod(host: str, video_root: str) -> None:
    settings = load_pod_config()
    token = os.environ.get("EMBEDDER_TOKEN", "")
    if not token:
        raise SystemExit("EMBEDDER_TOKEN env var is required for --pod")
    base_url = _coordinator_base_url(host)
    event("INIT", "coordinator", stats={"host": host})
    wait_for_coordinator(base_url, timeout_s=settings.embeddings.pod_connect_timeout_s)
    # One ProviderRouter over the cases a pod may be handed; the shared Qwen
    # backbone covers video/sandwich/audio with a single model instance.
    provider = build_provider_router(settings, None, remote_served_cases())
    source = HttpJobSource(
        base_url=base_url,
        token=token,
        timeout_s=settings.embeddings.worker_request_timeout_s,
        max_retries=settings.embeddings.worker_max_retries,
    )
    run_worker(
        source,
        provider=provider,
        video_root=video_root,
        inflight=settings.embeddings.inflight,
        served_only=True,
        unreachable_exit_s=settings.embeddings.pod_idle_ttl_s,
        batch_size=getattr(settings.embeddings, "embed_batch_size", 1),
        batch_fill_ms=getattr(settings.embeddings, "embed_batch_fill_ms", 0),
        log_tag="embed:pod",
    )
