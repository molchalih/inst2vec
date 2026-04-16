"""Phase 1 – music pipeline.

  classify_music()         ACRCloud fingerprint every clip → link to Music rows.
  extract_music_features() Spotify → ReccoBeats IDs → audio features (upload fallback).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from acrcloud.recognizer import ACRCloudRecognizer
from sqlalchemy import or_

from modules.database import Clip, Music, get_session
from modules.services import ReccoBeatsClient, SpotifyClient, log

VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "data/source/videos"))
MIN_CONFIDENCE = float(os.environ.get("AUDIO_FINGERPRINT_CONFIDENCE", 0.8))
COMMIT_EVERY = int(os.environ.get("MUSIC_COMMIT_EVERY", 50))

FEATURE_FIELDS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence",
]
UPLOAD_FIELDS = [f for f in FEATURE_FIELDS if f not in ("key", "mode")]
_NO_MATCH = "none"

_SAMPLE_SECS = int(os.environ.get("MANUAL_FEATURES_MAX_SECONDS", 20))
_SAMPLE_RATE = int(os.environ.get("MANUAL_FEATURES_SAMPLE_RATE", 44100))
_SAMPLE_MAX_BYTES = int(float(os.environ.get("MANUAL_FEATURES_MAX_MB", 5)) * 1024 * 1024)
_SAMPLE_BITRATE = os.environ.get("MANUAL_FEATURES_MP3_BITRATE", "128k")

SCOPE_CLASSIFY = "classify_music"
SCOPE_FEATURES = "extract_features"


# ── helpers ────────────────────────────────────────────────────────────────────

def _fingerprint(acr: ACRCloudRecognizer, path: str) -> Optional[tuple[str, str, float]]:
    try:
        data = json.loads(acr.recognize_by_file(path, 0) or "")
    except Exception:
        return None
    if data.get("status", {}).get("code") != 0:
        return None
    music = data.get("metadata", {}).get("music", [])
    if not music:
        return None
    best = music[0]
    score = best.get("score", 0) / 100.0
    if score < MIN_CONFIDENCE:
        return None
    artists = best.get("artists", [])
    artist = (artists[0].get("name") or "").strip() if artists else ""
    track = (best.get("title") or "").strip()
    return (artist, track, score) if (artist or track) else None


def _get_or_create_music(session, artist: str, track: str) -> Music:
    artist, track = artist.strip(), track.strip()
    row = session.query(Music).filter_by(artist=artist, track=track).first()
    if not row:
        row = Music(artist=artist, track=track)
        session.add(row)
        session.flush()
    return row


def _pick_video(session, music_id: int) -> Optional[Path]:
    for (pk,) in (
        session.query(Clip.pk)
        .filter(Clip.music_id == music_id, or_(Clip.disqualified.is_(None), Clip.disqualified == 0))
        .order_by(Clip.play_count.desc(), Clip.pk.desc())
    ):
        p = VIDEO_DIR / f"{pk}.mp4"
        if p.exists():
            return p
    return None


def _extract_audio_sample(video: Path, out_dir: Path) -> Optional[Path]:
    """ffmpeg: extract a short stereo clip. WAV preferred; MP3 as size fallback.

    ReccoBeats analysis rejects low-rate mono input, so we always use 44.1 kHz stereo.
    """
    base = ["ffmpeg", "-y", "-i", str(video), "-vn",
            "-t", str(_SAMPLE_SECS), "-ac", "2", "-ar", str(_SAMPLE_RATE)]
    wav = out_dir / f"{video.stem}.wav"
    if (subprocess.run(base + ["-c:a", "pcm_s16le", str(wav)], capture_output=True).returncode == 0
            and wav.exists() and wav.stat().st_size <= _SAMPLE_MAX_BYTES):
        return wav
    mp3 = out_dir / f"{video.stem}.mp3"
    if (subprocess.run(base + ["-b:a", _SAMPLE_BITRATE, str(mp3)], capture_output=True).returncode == 0
            and mp3.exists() and mp3.stat().st_size <= _SAMPLE_MAX_BYTES):
        return mp3
    return None


# ── public API ─────────────────────────────────────────────────────────────────

def classify_music() -> None:
    """Fingerprint all unresolved clips with ACRCloud and link them to Music rows.

    Sets has_music=1 (match found) or has_music=0 (no match). Clips with missing
    video files are skipped and retried on the next run.
    """
    session = get_session()

    clips = (
        session.query(Clip)
        .filter(Clip.has_music.is_(None), or_(Clip.disqualified.is_(None), Clip.disqualified == 0))
        .order_by(Clip.pk.desc())
        .all()
    )
    if not clips:
        session.close()
        return

    acr = ACRCloudRecognizer({
        "host": os.environ["ARC_HOST"],
        "access_key": os.environ["ARC_ACCESS_KEY"],
        "access_secret": os.environ["ARC_SECRET_KEY"],
        "timeout": 10,
    })
    log(SCOPE_CLASSIFY, f"{len(clips)} clips to fingerprint")
    matched = no_match = missing = 0

    for i, clip in enumerate(clips, 1):
        path = VIDEO_DIR / f"{clip.pk}.mp4"
        if not path.exists():
            missing += 1
            continue

        # Resolve from fresh fingerprinting for every unresolved clip.
        clip.music_id = None
        clip.music_confidence = None
        result = _fingerprint(acr, str(path))
        if result:
            artist, track, confidence = result
            music = _get_or_create_music(session, artist, track)
            clip.music_id = music.id
            clip.music_confidence = confidence
            clip.has_music = 1
            matched += 1
            log(SCOPE_CLASSIFY, f"{i}/{len(clips)} — {clip.pk}: {artist} – {track} ({confidence:.0%})")
        else:
            clip.has_music = 0
            no_match += 1

        if i % COMMIT_EVERY == 0:
            session.commit()

    session.commit()
    session.close()
    parts = [f"{matched} matched", f"{no_match} no match"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}")


def extract_music_features() -> None:
    """Fill Spotify IDs, ReccoBeats IDs, and audio features for all linked Music rows.

    Steps run in order:
      1. Spotify search    → Music.spotify_id
      2. ReccoBeats lookup → Music.reccobeats_id  (batched)
      3. Catalog features  → Music feature columns (batched)
      4. Upload fallback   → same columns, for tracks not in the ReccoBeats catalog
    """
    session = get_session()
    with httpx.Client(timeout=float(os.environ.get("MUSIC_HTTP_TIMEOUT", 20))) as http:
        spotify = SpotifyClient(http)
        rb = ReccoBeatsClient(http)

        # 1. Spotify IDs — one request per track
        rows = session.query(Music).filter(Music.spotify_id.is_(None)).all()
        if rows:
            total = len(rows)
            log(SCOPE_FEATURES, f"spotify: resolving {total} tracks")
            found = 0
            for i, row in enumerate(rows, 1):
                sid = spotify.search_id(row.artist, row.track)
                row.spotify_id = sid or _NO_MATCH
                if sid:
                    found += 1
                if i % COMMIT_EVERY == 0:
                    session.commit()
                    log(SCOPE_FEATURES, f"spotify: {i}/{total} done")
            session.commit()
            log(SCOPE_FEATURES, f"spotify: done — {found} found, {total - found} no match")

        # 2. ReccoBeats track IDs — batched lookup by Spotify ID
        rows = session.query(Music).filter(
            Music.spotify_id.is_not(None),
            Music.spotify_id != _NO_MATCH,
            Music.reccobeats_id.is_(None),
        ).all()
        if rows:
            log(SCOPE_FEATURES, f"reccobeats_id: {len(rows)} tracks to resolve")
            rb_id_map = rb.get_ids(
                [r.spotify_id for r in rows if r.spotify_id],
                on_batch=lambda i, n, m: log(SCOPE_FEATURES, f"reccobeats_id: batch {i}/{n} — {m} matched"),
            )
            for row in rows:
                row.reccobeats_id = rb_id_map.get(row.spotify_id, _NO_MATCH)
            session.commit()
            matched = sum(1 for r in rows if r.reccobeats_id != _NO_MATCH)
            log(SCOPE_FEATURES, f"reccobeats_id: done — {matched} matched, {len(rows) - matched} no match")

        # 3. Catalog audio features — batched by ReccoBeats ID
        rows = session.query(Music).filter(
            Music.reccobeats_id.is_not(None),
            Music.reccobeats_id != _NO_MATCH,
            Music.has_features.is_(None),
        ).all()
        if rows:
            log(SCOPE_FEATURES, f"catalog features: {len(rows)} tracks to enrich")
            feat_map = rb.get_features(
                [r.reccobeats_id for r in rows if r.reccobeats_id],
                on_batch=lambda i, n, m: log(SCOPE_FEATURES, f"catalog features: batch {i}/{n} — {m} enriched"),
            )
            enriched = 0
            for row in rows:
                feats = feat_map.get(row.reccobeats_id)
                if feats:
                    for f in FEATURE_FIELDS:
                        if f in feats:
                            setattr(row, f, feats[f])
                    if all(getattr(row, f) is not None for f in FEATURE_FIELDS):
                        row.has_features = "yes"
                        enriched += 1
            session.commit()
            log(SCOPE_FEATURES, f"catalog features: done — {enriched} enriched")

        # 4. Upload fallback — extract audio from clip and POST to ReccoBeats analysis
        rows = (
            session.query(Music)
            .join(Clip, Clip.music_id == Music.id)
            .filter(
                or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
                (Music.reccobeats_id.is_(None)) | (Music.reccobeats_id == _NO_MATCH),
                (Music.has_features.is_(None)) | (Music.has_features != "yes"),
            )
            .distinct().order_by(Music.id).all()
        )
        if rows:
            total = len(rows)
            log(SCOPE_FEATURES, f"upload fallback: {total} tracks without catalog coverage")
            enriched = 0
            for i, row in enumerate(rows, 1):
                video = _pick_video(session, row.id)
                if not video:
                    row.has_features = "none"
                    log(SCOPE_FEATURES, f"upload fallback: {i}/{total} — {row.artist} – {row.track} → no video")
                    continue

                with tempfile.TemporaryDirectory(prefix="rb-audio-") as tmp:
                    audio = _extract_audio_sample(video, Path(tmp))
                    if not audio:
                        row.has_features = "none"
                        log(SCOPE_FEATURES, f"upload fallback: {i}/{total} — {row.artist} – {row.track} → audio extraction failed")
                        continue
                    feats = rb.upload_features(audio)

                if feats and any(f in feats for f in UPLOAD_FIELDS):
                    for f in UPLOAD_FIELDS:
                        if f in feats:
                            setattr(row, f, feats[f])
                    row.has_features = "yes"
                    enriched += 1
                    log(SCOPE_FEATURES, f"upload fallback: {i}/{total} — {row.artist} – {row.track} → ok")
                else:
                    row.has_features = "none"
                    log(SCOPE_FEATURES, f"upload fallback: {i}/{total} — {row.artist} – {row.track} → no features returned")

                if i % COMMIT_EVERY == 0:
                    session.commit()

            session.commit()
            log(SCOPE_FEATURES, f"upload fallback: done — {enriched}/{total} enriched")

    session.close()
