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

from core.config import load_pod_config
from modules.embeddings.cases import qwen_provider
from modules.embeddings.worker import HttpJobSource, run_worker


def run_pod(host: str, video_root: str) -> None:
    settings = load_pod_config()
    token = os.environ.get("EMBEDDER_TOKEN", "")
    if not token:
        raise SystemExit("EMBEDDER_TOKEN env var is required for --pod")
    # Pod always uses the local Qwen model on its own GPU (video-frame variant
    # covers video/sandwich; audio case has no video and ignores frames).
    provider = qwen_provider(settings, None, with_frames=True)
    source = HttpJobSource(
        base_url=f"http://{host}",
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
        log_tag="embed:pod",
    )
