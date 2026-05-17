"""`python -m services.embedder.smoke` — one-shot health + canned embed.

Reads ``EMBEDDER_REMOTE_URL`` and ``EMBEDDER_TOKEN`` from env, hits
``/healthz`` and ``/embed`` once with a tiny audio-case payload (no
video URL — easiest to test). Exits 0 on success, 1 on any failure.

Usage:
    EMBEDDER_REMOTE_URL=https://abc-8000.proxy.runpod.net \\
        EMBEDDER_TOKEN=... \\
        uv run python -m services.embedder.smoke
"""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    url = os.environ.get("EMBEDDER_REMOTE_URL")
    tok = os.environ.get("EMBEDDER_TOKEN")
    if not url or not tok:
        print("EMBEDDER_REMOTE_URL and EMBEDDER_TOKEN required", file=sys.stderr)
        return 1

    base = url.rstrip("/")
    with httpx.Client(timeout=30) as c:
        h = c.get(f"{base}/healthz")
        print(f"healthz: {h.status_code} {h.text}")
        if h.status_code != 200:
            return 1

        r = c.post(
            f"{base}/embed",
            headers={"Authorization": f"Bearer {tok}"},
            json={
                "case": "audio",
                "clip_id": -1,
                "text": "hello world",
                "instruction": "describe the audio character",
            },
        )
        if r.status_code != 200:
            print(f"embed FAILED: {r.status_code} {r.text}", file=sys.stderr)
            return 1
        body = r.json()
        print(f"embed: dim={body['dim']} took_ms={body['took_ms']}")
        print(f"  first 5 values: {body['embedding'][:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
