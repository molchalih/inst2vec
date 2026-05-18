"""Manual smoke test for the Gemini multimodal embedding case.

Picks one downloaded clip with an extracted audio file and runs a real
Gemini Embedding 2 call. Prints the first 10 dimensions and the elapsed
seconds. Requires:
    * GEMINI_API_KEY in env
    * embeddings.gemini_enabled = true in config.toml
    * uv sync --group gemini

Usage: uv run python scripts/smoke_gemini_embed.py [clip_id]
"""

from __future__ import annotations

import os
import sys
import time

from core.config import load_runtime_config
from core.database import Clip, get_session, init_db
from modules.embeddings.cases import EmbeddingSecrets, _gemini_factory


def main(argv: list[str]) -> int:
    settings, secrets = load_runtime_config()
    if not settings.embeddings.gemini_enabled:
        print("embeddings.gemini_enabled is false; aborting")
        return 2
    init_db(secrets.database_url, secrets.identity_db_url)

    session = get_session()
    clip_id = int(argv[1]) if len(argv) > 1 else None
    if clip_id is None:
        clip = session.query(Clip).filter(Clip.is_downloaded.is_(True)).first()
        if clip is None:
            print("no downloaded clip found")
            return 1
    else:
        clip = session.get(Clip, clip_id)
        if clip is None:
            print(f"no clip with id={clip_id}")
            return 1

    video_path = os.path.join(settings.paths.video_dir, f"{clip.id}.mp4")
    audio_path = os.path.join(settings.paths.audio_dir, f"{clip.id}.mp3")
    text = f"smoke test for clip {clip.id}"

    provider = _gemini_factory(
        settings, EmbeddingSecrets(gemini_api_key=secrets.gemini_api_key)
    )
    t0 = time.time()
    [vector] = provider.embed(
        {
            "video_path": video_path,
            "audio_path": audio_path,
            "text": text,
        }
    )
    print(f"clip_id={clip.id}  dim={len(vector)}  elapsed={time.time() - t0:.2f}s")
    print("head:", vector[:10] if hasattr(vector, "__getitem__") else vector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
