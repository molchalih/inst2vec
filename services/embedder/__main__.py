"""`python -m services.embedder` → run this container as an embedding pod.

Honored env vars:
  ORCHESTRATOR_HOST   host:port of the orchestrator coordinator (required)
  EMBEDDER_TOKEN      shared bearer token (required)
  MODEL_PATH          default /workspace/models/Qwen3-VL-Embedding-8B
  VIDEO_ROOT          default /workspace/videos
  HUGGINGFACE_TOKEN   optional; exported as HF_TOKEN for gated model pulls
"""

from __future__ import annotations

import os


def main() -> None:
    host = os.environ.get("ORCHESTRATOR_HOST", "")
    if not host:
        raise SystemExit("ORCHESTRATOR_HOST env var is required")
    if not os.environ.get("EMBEDDER_TOKEN"):
        raise SystemExit("EMBEDDER_TOKEN env var is required")
    from modules.embeddings.pod import run_pod

    run_pod(host, os.environ.get("VIDEO_ROOT", "/workspace/videos"))


if __name__ == "__main__":
    main()
