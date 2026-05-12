"""Phase 1 – music pipeline.

classify_music()         ACRCloud fingerprint every clip → link to Music rows.
extract_music_features() Spotify → ReccoBeats IDs → audio features (upload fallback).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import httpx
from acrcloud.recognizer import ACRCloudRecognizer
from sqlalchemy import or_

from modules.console import log, progress
from modules.database import Clip, Music, get_session
from modules.services import ReccoBeatsClient, SpotifyClient

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
UPLOAD_FIELDS = [f for f in FEATURE_FIELDS if f not in ("key", "mode")]
_NO_MATCH = "none"

SCOPE_CLASSIFY = "classify_music"
SCOPE_FEATURES = "extract_features"


# ── helpers ────────────────────────────────────────────────────────────────────


def _fingerprint(
    acr: ACRCloudRecognizer, path: str, min_confidence: float
) -> tuple[str, str, float] | None:
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
    if score < min_confidence:
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


def _pick_video(session, music_id: int, video_dir: str) -> Path | None:
    video_dir_path = Path(video_dir)
    for (clip_id,) in (
        session.query(Clip.id)
        .filter(
            Clip.music_id == music_id,
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
        )
        .order_by(Clip.play_count.desc(), Clip.id.desc())
    ):
        p = video_dir_path / f"{clip_id}.mp4"
        if p.exists():
            return p
    return None


def _extract_audio_sample(
    video: Path,
    out_dir: Path,
    sample_secs: int,
    sample_rate: int,
    sample_max_bytes: int,
    sample_bitrate: str,
) -> Path | None:
    """ffmpeg: extract a short stereo clip. WAV preferred; MP3 as size fallback.

    ReccoBeats analysis rejects low-rate mono input, so we always use 44.1 kHz stereo.
    """
    base = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-t",
        str(sample_secs),
        "-ac",
        "2",
        "-ar",
        str(sample_rate),
    ]
    wav = out_dir / f"{video.stem}.wav"
    if (
        subprocess.run(
            [*base, "-c:a", "pcm_s16le", str(wav)], capture_output=True
        ).returncode
        == 0
        and wav.exists()
        and wav.stat().st_size <= sample_max_bytes
    ):
        return wav
    mp3 = out_dir / f"{video.stem}.mp3"
    if (
        subprocess.run(
            [*base, "-b:a", sample_bitrate, str(mp3)], capture_output=True
        ).returncode
        == 0
        and mp3.exists()
        and mp3.stat().st_size <= sample_max_bytes
    ):
        return mp3
    return None


# ── public API ─────────────────────────────────────────────────────────────────


def classify_music(
    video_dir: str,
    min_confidence: float,
    commit_every: int,
    arc_host: str,
    arc_access_key: str,
    arc_secret_key: str,
) -> None:
    """Fingerprint all unresolved clips with ACRCloud and link them to Music rows.

    Sets has_music=1 (match found) or has_music=0 (no match). Clips with missing
    video files are skipped and retried on the next run.
    """
    session = get_session()
    video_dir_path = Path(video_dir)

    clips = (
        session.query(Clip)
        .filter(
            Clip.has_music.is_(None),
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
        )
        .order_by(Clip.id.desc())
        .all()
    )
    if not clips:
        session.close()
        return

    acr = ACRCloudRecognizer(
        {
            "host": arc_host,
            "access_key": arc_access_key,
            "access_secret": arc_secret_key,
            "timeout": 10,
        }
    )
    log(SCOPE_CLASSIFY, f"{len(clips)} clips to fingerprint")
    matched = no_match = missing = 0

    with progress(len(clips), "Fingerprinting") as advance:
        for i, clip in enumerate(clips, 1):
            path = video_dir_path / f"{clip.id}.mp4"
            if not path.exists():
                missing += 1
                advance()
                continue

            clip.music_id = None
            clip.music_confidence = None
            result = _fingerprint(acr, str(path), min_confidence)
            if result:
                artist, track, confidence = result
                music = _get_or_create_music(session, artist, track)
                clip.music_id = music.id
                clip.music_confidence = confidence
                clip.has_music = 1
                matched += 1
                advance(detail=f"{clip.id}: {artist} – {track} ({confidence:.0%})")
            else:
                clip.has_music = 0
                no_match += 1
                advance()

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    parts = [f"{matched} matched", f"{no_match} no match"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}", level="ok")


def extract_music_features(
    video_dir: str,
    http_timeout: float,
    commit_every: int,
    spotify_client_id: str,
    spotify_client_secret: str,
    spotify_token_skew_seconds: int,
    spotify_search_limit: int,
    spotify_request_timeout: float,
    reccobeats_batch_size: int,
    reccobeats_delay_min: float,
    reccobeats_delay_max: float,
    manual_features_max_seconds: int,
    manual_features_sample_rate: int,
    manual_features_max_mb: float,
    manual_features_mp3_bitrate: str,
) -> None:
    """Fill Spotify IDs, ReccoBeats IDs, and audio features for all linked Music rows.

    Steps run in order:
      1. Spotify search    → Music.spotify_id
      2. ReccoBeats lookup → Music.reccobeats_id  (batched)
      3. Catalog features  → Music feature columns (batched)
      4. Upload fallback   → same columns, for tracks not in the ReccoBeats catalog
    """
    session = get_session()
    sample_max_bytes = int(manual_features_max_mb * 1024 * 1024)
    with httpx.Client(timeout=http_timeout) as http:
        spotify = SpotifyClient(
            http,
            client_id=spotify_client_id,
            client_secret=spotify_client_secret,
            token_skew=spotify_token_skew_seconds,
            search_limit=spotify_search_limit,
            search_timeout=spotify_request_timeout,
        )
        rb = ReccoBeatsClient(
            http,
            batch=reccobeats_batch_size,
            delay_min=reccobeats_delay_min,
            delay_max=reccobeats_delay_max,
            timeout=http_timeout,
        )

        # 1. Spotify IDs — one request per track
        rows = session.query(Music).filter(Music.spotify_id.is_(None)).all()
        if rows:
            total = len(rows)
            log(SCOPE_FEATURES, f"spotify: resolving {total} tracks")
            found = 0
            with progress(total, "Spotify lookup") as advance:
                for i, row in enumerate(rows, 1):
                    sid = spotify.search_id(row.artist, row.track)
                    row.spotify_id = sid or _NO_MATCH
                    if sid:
                        found += 1
                    if i % commit_every == 0:
                        session.commit()
                    advance(detail=f"{row.artist} – {row.track}")
            session.commit()
            log(
                SCOPE_FEATURES,
                f"spotify: done — {found} found, {total - found} no match",
                level="ok",
            )

        # 2. ReccoBeats track IDs — batched lookup by Spotify ID
        rows = (
            session.query(Music)
            .filter(
                Music.spotify_id.is_not(None),
                Music.spotify_id != _NO_MATCH,
                Music.reccobeats_id.is_(None),
            )
            .all()
        )
        if rows:
            log(SCOPE_FEATURES, f"reccobeats_id: {len(rows)} tracks to resolve")
            rb_id_map = rb.get_ids(
                [r.spotify_id for r in rows if r.spotify_id],
                on_batch=lambda i, n, m: log(
                    SCOPE_FEATURES, f"reccobeats_id: batch {i}/{n} — {m} matched"
                ),
            )
            for row in rows:
                row.reccobeats_id = rb_id_map.get(row.spotify_id, _NO_MATCH)
            session.commit()
            matched = sum(1 for r in rows if r.reccobeats_id != _NO_MATCH)
            log(
                SCOPE_FEATURES,
                f"reccobeats_id: done — {matched} matched, {len(rows) - matched} no match",
            )

        # 3. Catalog audio features — batched by ReccoBeats ID
        rows = (
            session.query(Music)
            .filter(
                Music.reccobeats_id.is_not(None),
                Music.reccobeats_id != _NO_MATCH,
                Music.has_features.is_(None),
            )
            .all()
        )
        if rows:
            log(SCOPE_FEATURES, f"catalog features: {len(rows)} tracks to enrich")
            feat_map = rb.get_features(
                [r.reccobeats_id for r in rows if r.reccobeats_id],
                on_batch=lambda i, n, m: log(
                    SCOPE_FEATURES, f"catalog features: batch {i}/{n} — {m} enriched"
                ),
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
            .distinct()
            .order_by(Music.id)
            .all()
        )
        if rows:
            total = len(rows)
            log(
                SCOPE_FEATURES,
                f"upload fallback: {total} tracks without catalog coverage",
            )
            enriched = 0
            with progress(total, "Upload fallback") as advance:
                for i, row in enumerate(rows, 1):
                    video = _pick_video(session, row.id, video_dir)
                    if not video:
                        row.has_features = "none"
                        advance(detail=f"{row.artist} – {row.track} (no video)")
                        continue

                    with tempfile.TemporaryDirectory(prefix="rb-audio-") as tmp:
                        audio = _extract_audio_sample(
                            video,
                            Path(tmp),
                            manual_features_max_seconds,
                            manual_features_sample_rate,
                            sample_max_bytes,
                            manual_features_mp3_bitrate,
                        )
                        if not audio:
                            row.has_features = "none"
                            advance(
                                detail=f"{row.artist} – {row.track} (audio extract failed)"
                            )
                            continue
                        feats = rb.upload_features(audio)

                    if feats and any(f in feats for f in UPLOAD_FIELDS):
                        for f in UPLOAD_FIELDS:
                            if f in feats:
                                setattr(row, f, feats[f])
                        row.has_features = "yes"
                        enriched += 1
                        advance(detail=f"{row.artist} – {row.track} (ok)")
                    else:
                        row.has_features = "none"
                        advance(detail=f"{row.artist} – {row.track} (no features)")

                    if i % commit_every == 0:
                        session.commit()

            session.commit()
            log(
                SCOPE_FEATURES,
                f"upload fallback: done — {enriched}/{total} enriched",
                level="ok",
            )

    session.close()
