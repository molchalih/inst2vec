"""
One-off test: ffmpeg clip -> POST ReccoBeats /v1/analysis/audio-features.

Run from repo root or scripts/:
  python scripts/test_reccobeats_manual_extract.py [path/to/video.mp4]
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

from modules.audio_processor import (  # noqa: E402
    MANUAL_FEATURES_MAX_MB,
    MANUAL_FEATURES_MAX_SECONDS,
    MANUAL_FEATURES_MP3_BITRATE,
    MANUAL_FEATURES_SAMPLE_RATE,
    RECCOBEATS_ANALYSIS_FEATURES_URL,
    _extract_audio_sample,
)


def main():
    load_dotenv(PROJECT_ROOT / ".env")
    timeout = float(os.environ.get("MUSIC_HTTP_TIMEOUT", "60"))

    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        videos = sorted(Path("data/source/videos").glob("*.mp4"))
        if not videos:
            print("No mp4 under data/source/videos")
            return 1
        video = videos[0]

    print("video:", video.resolve())
    print(
        "settings:",
        f"max_s={MANUAL_FEATURES_MAX_SECONDS}",
        f"max_mb={MANUAL_FEATURES_MAX_MB}",
        f"sr={MANUAL_FEATURES_SAMPLE_RATE}",
        f"mp3_br={MANUAL_FEATURES_MP3_BITRATE}",
    )

    with tempfile.TemporaryDirectory(prefix="reccobeats-test-") as tmp:
        audio = _extract_audio_sample(video, Path(tmp))
        if not audio:
            print("ffmpeg failed or file too large; check stderr by running ffmpeg manually")
            return 1
        print("audio:", audio, "bytes:", audio.stat().st_size)

        suffix = audio.suffix.lower()
        mime = "audio/wav" if suffix == ".wav" else "audio/mpeg"
        with audio.open("rb") as fh:
            resp = httpx.post(
                RECCOBEATS_ANALYSIS_FEATURES_URL,
                files={"audioFile": (audio.name, fh, mime)},
                timeout=timeout,
            )

    print("status:", resp.status_code)
    print("body:", resp.text[:2000])
    try:
        data = resp.json()
        if isinstance(data, dict):
            print("keys:", sorted(data.keys()))
            print(json.dumps(data, indent=2)[:2000])
    except json.JSONDecodeError:
        pass
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
