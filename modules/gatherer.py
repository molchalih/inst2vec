import json
import os

import whisper
from acrcloud.recognizer import ACRCloudRecognizer

from modules.database import get_session, User, Clip

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", 5))
MIN_CONFIDENCE = float(os.environ.get("AUDIO_FINGERPRINT_CONFIDENCE", 0.8))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
VIDEO_DIR = "data/source/videos"


def _try_acrcloud(recognizer, path):
    raw = recognizer.recognize_by_file(path, 0)
    result = json.loads(raw)
    if result.get("status", {}).get("code") != 0:
        return None
    music = result.get("metadata", {}).get("music", [])
    if not music:
        return None
    best = music[0]
    score = best.get("score", 0) / 100.0
    if score < MIN_CONFIDENCE:
        return None
    artists = best.get("artists", [])
    artist = artists[0]["name"] if artists else None
    return artist, best.get("title"), score


def _try_whisper(model, path):
    result = model.transcribe(path)
    return result.get("text", "").strip()


def gather_info():
    acr = ACRCloudRecognizer({
        "host": os.environ["ARC_HOST"],
        "access_key": os.environ["ARC_ACCESS_KEY"],
        "access_secret": os.environ["ARC_SECRET_KEY"],
        "timeout": 10,
    })
    whisper_model = None

    session = get_session()
    users = session.query(User).limit(BATCH_SIZE).all()

    for user in users:
        for clip in user.clips[:MAX_CLIPS or None]:
            if clip.music_artist is not None or clip.speech_transcription is not None:
                continue

            path = os.path.join(VIDEO_DIR, f"{clip.pk}.mp4")
            if not os.path.exists(path):
                continue

            print(f"[gather] {user.username}/{clip.pk} — ", end="", flush=True)

            try:
                result = _try_acrcloud(acr, path)
            except Exception:
                result = None

            if result:
                clip.music_artist, clip.music_track, clip.music_confidence = result
                print(f"music: {clip.music_artist} - {clip.music_track} ({clip.music_confidence:.0%})")
            else:
                if whisper_model is None:
                    print("loading whisper... ", end="", flush=True)
                    whisper_model = whisper.load_model(WHISPER_MODEL)
                try:
                    clip.speech_transcription = _try_whisper(whisper_model, path) or ""
                except Exception:
                    clip.speech_transcription = ""
                print(f"speech: {clip.speech_transcription[:80]}")

        session.commit()
    session.close()
