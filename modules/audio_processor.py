import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
import whisper
from acrcloud.recognizer import ACRCloudRecognizer
from deepl import Translator

from modules.database import Clip, Music, get_session

# endpoints
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
RECCOBEATS_TRACK_URL = "https://api.reccobeats.com/v1/track"
RECCOBEATS_FEATURES_URL = "https://api.reccobeats.com/v1/audio-features"
RECCOBEATS_ANALYSIS_FEATURES_URL = "https://api.reccobeats.com/v1/analysis/audio-features"

# fields
FEATURE_FIELDS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "valence",
]

# config
RECCOBEATS_BATCH_SIZE = int(os.environ["RECCOBEATS_BATCH_SIZE"])
RECCOBEATS_DELAY_MIN = float(os.environ["RECCOBEATS_DELAY_MIN"])
RECCOBEATS_DELAY_MAX = float(os.environ["RECCOBEATS_DELAY_MAX"])
HTTP_TIMEOUT = float(os.environ["MUSIC_HTTP_TIMEOUT"])
SPOTIFY_SEARCH_LIMIT = int(os.environ["SPOTIFY_SEARCH_LIMIT"])
SPOTIFY_TOKEN_SKEW_SECONDS = int(os.environ["SPOTIFY_TOKEN_SKEW_SECONDS"])
SPOTIFY_COMMIT_EVERY = int(os.environ["SPOTIFY_COMMIT_EVERY"])
SPOTIFY_PROGRESS_EVERY = int(os.environ.get("SPOTIFY_PROGRESS_EVERY", "10"))
SPOTIFY_REQUEST_TIMEOUT = float(os.environ.get("SPOTIFY_REQUEST_TIMEOUT", "8"))
SPOTIFY_NO_MATCH_SENTINEL = "none"
RECCOBEATS_NO_MATCH_SENTINEL = "none"
DEEPL_API_KEY = os.environ["DEEPL_API_KEY"]
DEEPL_TARGET_LANG = os.environ.get("DEEPL_TARGET_LANG", "EN")
DEEPL_COMMIT_EVERY = int(os.environ.get("DEEPL_COMMIT_EVERY", "50"))
SPEECH_TRANSLATION_MAX_CHARS = 1000
MANUAL_FEATURES_MAX_SECONDS = int(os.environ.get("MANUAL_FEATURES_MAX_SECONDS", "20"))
MANUAL_FEATURES_MAX_MB = float(os.environ.get("MANUAL_FEATURES_MAX_MB", "5"))
MANUAL_FEATURES_SAMPLE_RATE = int(os.environ.get("MANUAL_FEATURES_SAMPLE_RATE", "44100"))
MANUAL_FEATURES_MP3_BITRATE = os.environ.get("MANUAL_FEATURES_MP3_BITRATE", "128k")
MANUAL_FEATURES_COMMIT_EVERY = int(os.environ.get("MANUAL_FEATURES_COMMIT_EVERY", "20"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", 5))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
MIN_CONFIDENCE = float(os.environ.get("AUDIO_FINGERPRINT_CONFIDENCE", 0.8))
VIDEO_DIR = "data/source/videos"

FEATURES_BY_UPLOAD_FIELDS = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


def _normalize_music_value(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip()


def _fmt_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _try_whisper(model, path):
    result = model.transcribe(path)
    return result.get("text", "").strip()


def _get_or_create_music(session, artist, track):
    artist_norm = _normalize_music_value(artist)
    track_norm = _normalize_music_value(track)
    row = session.query(Music).filter_by(artist=artist_norm, track=track_norm).first()
    if row:
        return row
    row = Music(artist=artist_norm, track=track_norm)
    session.add(row)
    session.flush()
    return row


def _try_acrcloud(recognizer, path):
    raw = recognizer.recognize_by_file(path, 0)
    if not raw:
        return None

    try:
        result = json.loads(raw)
    except Exception:
        return None

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
    artist = artists[0].get("name") if artists else ""
    track = best.get("title")
    if not artist and not track:
        return None
    return artist, track, score


def process_clip_audio():
    acr = ACRCloudRecognizer(
        {
            "host": os.environ["ARC_HOST"],
            "access_key": os.environ["ARC_ACCESS_KEY"],
            "access_secret": os.environ["ARC_SECRET_KEY"],
            "timeout": 10,
        }
    )
    whisper_model = None

    session = get_session()
    clips = (
        session.query(Clip)
        .filter(Clip.audio_type.is_(None))
        .order_by(Clip.pk.desc())
        .limit(BATCH_SIZE * max(1, MAX_CLIPS))
        .all()
    )
    print(f"[clip_audio] pending (audio_type IS NULL): {len(clips)}")
    if not clips:
        print("[clip_audio] nothing to process")
        session.close()
        return

    missing_files = 0
    processed = 0

    for clip in clips:
        print(f"[clip_audio] {clip.pk} — ", end="", flush=True)
        path = os.path.join(VIDEO_DIR, f"{clip.pk}.mp4")
        if not os.path.exists(path):
            missing_files += 1
            print("skip: missing video file")
            continue

        started_at = time.perf_counter()
        events = []

        if clip.music_id is None:
            fp_started_at = time.perf_counter()
            try:
                recognized = _try_acrcloud(acr, path)
                fp_elapsed = time.perf_counter() - fp_started_at
                events.append(f"fingerprint={_fmt_seconds(fp_elapsed)}")
            except Exception as error:
                recognized = None
                fp_elapsed = time.perf_counter() - fp_started_at
                events.append(f"fingerprint_error={error.__class__.__name__}({_fmt_seconds(fp_elapsed)})")

            if recognized:
                artist, track, confidence = recognized
                music_row = _get_or_create_music(session, artist, track)
                clip.music_id = music_row.id
                clip.music_confidence = confidence
                clip.audio_type = "music"
                elapsed = time.perf_counter() - started_at
                print(
                    f"audio_type=music; {events[-1]}; track={music_row.artist} - {music_row.track}; "
                    f"confidence={clip.music_confidence:.0%}; total={_fmt_seconds(elapsed)}"
                )
                continue
            events.append("fingerprint=no_match")
        else:
            events.append("fingerprint=skip(has_music)")

        if clip.speech_transcription is None:
            if whisper_model is None:
                load_started_at = time.perf_counter()
                whisper_model = whisper.load_model(WHISPER_MODEL)
                events.append(f"whisper_load={_fmt_seconds(time.perf_counter() - load_started_at)}")
            whisper_started_at = time.perf_counter()
            try:
                clip.speech_transcription = _try_whisper(whisper_model, path) or ""
            except Exception:
                clip.speech_transcription = ""
            events.append(f"whisper={_fmt_seconds(time.perf_counter() - whisper_started_at)}")
        else:
            events.append("whisper=skip(existing_transcription)")

        speech_text = (clip.speech_transcription or "").strip()
        if clip.music_id is not None:
            clip.audio_type = "music"
        elif speech_text:
            clip.audio_type = "speech"
        else:
            clip.audio_type = "none"

        elapsed = time.perf_counter() - started_at
        print(f"audio_type={clip.audio_type}; {'; '.join(events)}; total={_fmt_seconds(elapsed)}")
        processed += 1

    session.commit()
    session.close()
    print(f"[clip_audio] done; processed={processed}; missing_files={missing_files}")


def _sleep_reccobeats_delay():
    time.sleep(random.uniform(RECCOBEATS_DELAY_MIN, RECCOBEATS_DELAY_MAX))


def _chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _extract_spotify_id_from_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    try:
        path = urlparse(href).path
    except Exception:
        return None
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 2 and parts[-2] == "track":
        return parts[-1]
    return None


class _SpotifyClient:
    def __init__(self, client: httpx.Client):
        self.client = client
        self.token = None
        self.token_expires_at = 0.0
        self.client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
        self.client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")

    def _ensure_token(self) -> bool:
        if not self.client_id or not self.client_secret:
            return False

        if self.token and time.time() < self.token_expires_at:
            return True

        try:
            response = self.client.post(
                SPOTIFY_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=SPOTIFY_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return False

        access_token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))
        if not access_token:
            return False

        self.token = access_token
        self.token_expires_at = time.time() + max(0, expires_in - SPOTIFY_TOKEN_SKEW_SECONDS)
        return True

    def search_track_id(self, artist: str, track: str) -> Optional[str]:
        artist = _normalize_music_value(artist)
        track = _normalize_music_value(track)
        if not track:
            return None
        if not self._ensure_token():
            return None

        query = f"track:{track}"
        if artist:
            query = f"{query} artist:{artist}"

        try:
            response = self.client.get(
                SPOTIFY_SEARCH_URL,
                params={"q": query, "type": "track", "limit": SPOTIFY_SEARCH_LIMIT},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=SPOTIFY_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        items = payload.get("tracks", {}).get("items", [])
        if not items:
            return None
        return items[0].get("id")


def _fill_spotify_ids(session, client: httpx.Client) -> int:
    spotify = _SpotifyClient(client)
    rows = session.query(Music).filter(Music.spotify_id.is_(None)).order_by(Music.id).all()
    updated = 0
    marked_none = 0
    attempted = 0
    total = len(rows)
    print(f"[music][spotify] pending={total}")

    for row in rows:
        attempted += 1
        if attempted == 1 or (SPOTIFY_PROGRESS_EVERY > 0 and attempted % SPOTIFY_PROGRESS_EVERY == 0):
            print(f"[music][spotify] querying {attempted}/{total}")
        spotify_id = spotify.search_track_id(row.artist, row.track)
        if not spotify_id:
            row.spotify_id = SPOTIFY_NO_MATCH_SENTINEL
            marked_none += 1
            if SPOTIFY_COMMIT_EVERY > 0 and (updated + marked_none) % SPOTIFY_COMMIT_EVERY == 0:
                session.commit()
                print(
                    f"[music][spotify] committed updated={updated}/{total}; "
                    f"marked_none={marked_none}/{total}"
                )
            if attempted % 100 == 0:
                print(
                    f"[music][spotify] progress {attempted}/{total}; "
                    f"updated={updated}; marked_none={marked_none}"
                )
            continue
        row.spotify_id = spotify_id
        updated += 1
        if SPOTIFY_COMMIT_EVERY > 0 and (updated + marked_none) % SPOTIFY_COMMIT_EVERY == 0:
            session.commit()
            print(
                f"[music][spotify] committed updated={updated}/{total}; "
                f"marked_none={marked_none}/{total}"
            )
        elif attempted % 100 == 0:
            print(
                f"[music][spotify] progress {attempted}/{total}; "
                f"updated={updated}; marked_none={marked_none}"
            )

    if updated or marked_none:
        session.commit()
    print(f"[music][spotify] done updated={updated}/{total}; marked_none={marked_none}/{total}")
    return updated


def _fill_reccobeats_ids(session, client: httpx.Client) -> int:
    todo = (
        session.query(Music)
        .filter(
            Music.spotify_id.is_not(None),
            Music.spotify_id != SPOTIFY_NO_MATCH_SENTINEL,
            Music.reccobeats_id.is_(None),
        )
        .order_by(Music.id)
        .all()
    )
    if not todo:
        print("[music][reccobeats_id] pending=0")
        return 0

    by_spotify: Dict[str, Music] = {}
    for row in todo:
        if row.spotify_id:
            by_spotify[row.spotify_id] = row

    updated = 0
    marked_none = 0
    spotify_ids = list(by_spotify.keys())
    batches = list(_chunked(spotify_ids, RECCOBEATS_BATCH_SIZE))
    print(f"[music][reccobeats_id] pending={len(spotify_ids)} in {len(batches)} batches")
    for index, batch in enumerate(batches, 1):
        try:
            response = client.get(RECCOBEATS_TRACK_URL, params={"ids": ",".join(batch)})
            response.raise_for_status()
            payload = response.json()
        except Exception:
            print(f"[music][reccobeats_id] batch {index}/{len(batches)} failed")
            _sleep_reccobeats_delay()
            continue

        matched_spotify_ids = set()
        for item in payload.get("content", []):
            spotify_id = _extract_spotify_id_from_href(item.get("href"))
            if not spotify_id:
                continue
            row = by_spotify.get(spotify_id)
            if not row:
                continue
            reccobeats_id = item.get("id")
            if not reccobeats_id:
                continue
            row.reccobeats_id = reccobeats_id
            matched_spotify_ids.add(spotify_id)
            updated += 1

        for spotify_id in batch:
            if spotify_id in matched_spotify_ids:
                continue
            row = by_spotify.get(spotify_id)
            if not row:
                continue
            if row.reccobeats_id is None:
                row.reccobeats_id = RECCOBEATS_NO_MATCH_SENTINEL
                marked_none += 1

        session.commit()
        print(
            f"[music][reccobeats_id] batch {index}/{len(batches)} done; "
            f"updated={updated}; marked_none={marked_none}"
        )
        _sleep_reccobeats_delay()

    print(
        f"[music][reccobeats_id] done updated={updated}/{len(spotify_ids)}; "
        f"marked_none={marked_none}/{len(spotify_ids)}"
    )
    return updated


def _fill_reccobeats_features(session, client: httpx.Client) -> int:
    todo = (
        session.query(Music)
        .filter(
            Music.reccobeats_id.is_not(None),
            Music.reccobeats_id != RECCOBEATS_NO_MATCH_SENTINEL,
        )
        .order_by(Music.id)
        .all()
    )
    if not todo:
        print("[music][features] pending=0")
        return 0

    by_reccobeats: Dict[str, Music] = {}
    for row in todo:
        if not row.reccobeats_id:
            continue
        if any(getattr(row, field) is None for field in FEATURE_FIELDS):
            by_reccobeats[row.reccobeats_id] = row

    updated = 0
    reccobeats_ids = list(by_reccobeats.keys())
    batches = list(_chunked(reccobeats_ids, RECCOBEATS_BATCH_SIZE))
    print(f"[music][features] pending={len(reccobeats_ids)} in {len(batches)} batches")
    for index, batch in enumerate(batches, 1):
        try:
            response = client.get(RECCOBEATS_FEATURES_URL, params={"ids": ",".join(batch)})
            response.raise_for_status()
            payload = response.json()
        except Exception:
            print(f"[music][features] batch {index}/{len(batches)} failed")
            _sleep_reccobeats_delay()
            continue

        for item in payload.get("content", []):
            reccobeats_id = item.get("id")
            if not reccobeats_id:
                continue
            row = by_reccobeats.get(reccobeats_id)
            if not row:
                continue

            for field in FEATURE_FIELDS:
                if field in item:
                    setattr(row, field, item.get(field))
            if all(getattr(row, field) is not None for field in FEATURE_FIELDS):
                row.has_features = "yes"
            updated += 1

        session.commit()
        print(f"[music][features] batch {index}/{len(batches)} done; updated={updated}")
        _sleep_reccobeats_delay()

    print(f"[music][features] done updated={updated}/{len(reccobeats_ids)}")
    return updated


def _pick_video_path_for_music(session, music_id: int) -> Optional[Path]:
    clips = (
        session.query(Clip.pk)
        .filter(Clip.music_id == music_id)
        .order_by(Clip.play_count.desc(), Clip.pk.desc())
        .all()
    )
    for clip_pk, in clips:
        path = Path(VIDEO_DIR) / f"{clip_pk}.mp4"
        if path.exists():
            return path
    return None


def _extract_audio_sample(video_path: Path, output_dir: Path) -> Optional[Path]:
    """ReccoBeats analysis rejects low-rate mono MP3 in practice; use 44.1kHz stereo WAV (docs)."""
    max_bytes = int(MANUAL_FEATURES_MAX_MB * 1024 * 1024)
    wav_path = output_dir / f"{video_path.stem}.wav"
    wav_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-t",
        str(MANUAL_FEATURES_MAX_SECONDS),
        "-ac",
        "2",
        "-ar",
        str(MANUAL_FEATURES_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    result = subprocess.run(wav_cmd, capture_output=True, text=True)
    if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size <= max_bytes:
        return wav_path

    mp3_path = output_dir / f"{video_path.stem}.mp3"
    mp3_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-t",
        str(MANUAL_FEATURES_MAX_SECONDS),
        "-ac",
        "2",
        "-ar",
        str(MANUAL_FEATURES_SAMPLE_RATE),
        "-b:a",
        MANUAL_FEATURES_MP3_BITRATE,
        str(mp3_path),
    ]
    result = subprocess.run(mp3_cmd, capture_output=True, text=True)
    if result.returncode != 0 or not mp3_path.exists():
        return None
    if mp3_path.stat().st_size > max_bytes:
        return None
    return mp3_path


def _extract_features_by_upload(client: httpx.Client, audio_path: Path) -> Optional[dict]:
    suffix = audio_path.suffix.lower()
    mime = "audio/wav" if suffix == ".wav" else "audio/mpeg"
    with audio_path.open("rb") as fh:
        try:
            response = client.post(
                RECCOBEATS_ANALYSIS_FEATURES_URL,
                files={"audioFile": (audio_path.name, fh, mime)},
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    if not any(field in payload for field in FEATURES_BY_UPLOAD_FIELDS):
        return None
    return payload


def _fill_reccobeats_features_by_upload(session, client: httpx.Client) -> int:
    todo = (
        session.query(Music)
        .join(Clip, Clip.music_id == Music.id)
        .filter(
            (Music.reccobeats_id.is_(None)) | (Music.reccobeats_id == RECCOBEATS_NO_MATCH_SENTINEL)
        )
        .filter((Music.has_features.is_(None)) | (Music.has_features != "yes"))
        .distinct()
        .order_by(Music.id)
        .all()
    )
    total = len(todo)
    print(f"[music][manual_features] pending={total}")
    if total == 0:
        return 0

    updated = 0
    for index, row in enumerate(todo, 1):
        video_path = _pick_video_path_for_music(session, row.id)
        if not video_path:
            row.has_features = "none"
            if index % 25 == 0:
                print(f"[music][manual_features] progress {index}/{total}; updated={updated}")
            continue

        with tempfile.TemporaryDirectory(prefix="reccobeats-audio-") as tmp:
            audio_path = _extract_audio_sample(video_path, Path(tmp))
            if not audio_path:
                row.has_features = "none"
                if index % 25 == 0:
                    print(f"[music][manual_features] progress {index}/{total}; updated={updated}")
                continue

            payload = _extract_features_by_upload(client, audio_path)
            if not payload:
                row.has_features = "none"
                if index % 25 == 0:
                    print(f"[music][manual_features] progress {index}/{total}; updated={updated}")
                continue

        for field in FEATURES_BY_UPLOAD_FIELDS:
            if field in payload:
                setattr(row, field, payload.get(field))
        row.has_features = "yes"
        updated += 1

        if MANUAL_FEATURES_COMMIT_EVERY > 0 and index % MANUAL_FEATURES_COMMIT_EVERY == 0:
            session.commit()
            print(f"[music][manual_features] committed index={index}/{total}; updated={updated}")
        elif index % 25 == 0:
            print(f"[music][manual_features] progress {index}/{total}; updated={updated}")

    session.commit()
    print(f"[music][manual_features] done updated={updated}/{total}")
    return updated


def _fill_speech_translations(session) -> int:
    translator = Translator(DEEPL_API_KEY)
    clips = (
        session.query(Clip)
        .filter(Clip.speech_transcription.is_not(None))
        .filter(Clip.speech_transcription != "")
        .filter((Clip.speech_translation.is_(None)) | (Clip.speech_translation == ""))
        .order_by(Clip.pk)
        .all()
    )
    updated = 0
    total = len(clips)
    print(f"[speech][deepl] pending={total}")

    for index, clip in enumerate(clips, 1):
        source = (clip.speech_transcription or "").strip()
        if not source:
            if index % 100 == 0:
                print(f"[speech][deepl] progress {index}/{total}; updated={updated}")
            continue
        source = source[:SPEECH_TRANSLATION_MAX_CHARS]
        try:
            result = translator.translate_text(source, target_lang=DEEPL_TARGET_LANG)
        except Exception:
            if index % 100 == 0:
                print(f"[speech][deepl] progress {index}/{total}; updated={updated}")
            continue
        clip.speech_translation = result.text
        updated += 1
        if DEEPL_COMMIT_EVERY > 0 and updated % DEEPL_COMMIT_EVERY == 0:
            session.commit()
            print(f"[speech][deepl] committed updated={updated}/{total}")
        elif index % 100 == 0:
            print(f"[speech][deepl] progress {index}/{total}; updated={updated}")

    if updated:
        session.commit()
    print(f"[speech][deepl] done updated={updated}/{total}")
    return updated


def process_music_metadata():
    session = get_session()
    print("[music] processing started")
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        spotify_updates = _fill_spotify_ids(session, client)
        reccobeats_id_updates = _fill_reccobeats_ids(session, client)
        feature_updates = _fill_reccobeats_features(session, client)
        manual_feature_updates = _fill_reccobeats_features_by_upload(session, client)
    speech_translation_updates = _fill_speech_translations(session)
    session.close()

    print(
        "[music] done — "
        f"spotify_id: {spotify_updates}, "
        f"reccobeats_id: {reccobeats_id_updates}, "
        f"features: {feature_updates}, "
        f"manual_features: {manual_feature_updates}, "
        f"speech_translation: {speech_translation_updates}"
    )
