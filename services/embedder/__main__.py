"""`python -m services.embedder` → run the FastAPI app via uvicorn.

Honored env vars:
  EMBEDDER_HOST    (default 0.0.0.0)
  EMBEDDER_PORT    (default 8000)
  EMBEDDER_TOKEN   (required)
  MODEL_PATH       (default /workspace/models/Qwen3-VL-Embedding-8B)
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    if not os.environ.get("EMBEDDER_TOKEN"):
        raise SystemExit("EMBEDDER_TOKEN env var is required")
    uvicorn.run(
        "services.embedder.app:app",
        host=os.environ.get("EMBEDDER_HOST", "0.0.0.0"),
        port=int(os.environ.get("EMBEDDER_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
