"""Spotify → ReccoBeats → audio features pipeline.

Four sub-stages, each idempotent:
    1. _resolve_spotify_ids   — fill Music.spotify_id (None on no match; _NO_MATCH on terminal-fail)
    2. _resolve_reccobeats_ids — fill Music.reccobeats_id (batched)
    3. _enrich_catalog_features — fill audio feature columns (batched); writes True on success
    4. _enrich_upload_fallback  — ffmpeg + POST for rows Stage 3 missed; writes True/False;
                                 sweeps remaining NULL → False at the end (invariant)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy import func

from modules.config import MusicSettings, PathsSettings
from modules.console import log, progress
from modules.database import Clip, Music, clip_used_in_analysis, get_session
from modules.music.audio_sample import extract_audio_sample
from modules.music.clients import ReccoBeatsClient, SpotifyClient, TransientError
from modules.music.state import (
    _NO_MATCH,
    FEATURE_FIELDS,
    SCOPE_FEATURES,
    UPLOAD_FIELDS,
    music_has_features,
)


@dataclass(frozen=True)
class MusicSecrets:
    spotify_client_id: str
    spotify_client_secret: str


def _make_spotify(
    http: httpx.Client, music: MusicSettings, secrets: MusicSecrets
) -> SpotifyClient:
    return SpotifyClient(
        http,
        client_id=secrets.spotify_client_id,
        client_secret=secrets.spotify_client_secret,
        token_skew=music.spotify_token_skew_seconds,
        search_limit=music.spotify_search_limit,
        search_timeout=music.spotify_request_timeout,
        max_attempts=music.api_max_attempts,
        retry_delay=music.api_retry_delay,
        retry_jitter=music.api_retry_jitter,
    )


def _make_reccobeats(http: httpx.Client, music: MusicSettings) -> ReccoBeatsClient:
    return ReccoBeatsClient(
        http,
        batch=music.reccobeats_batch_size,
        delay_min=music.reccobeats_delay_min,
        delay_max=music.reccobeats_delay_max,
        timeout=music.http_timeout,
        max_attempts=music.api_max_attempts,
        retry_delay=music.api_retry_delay,
        retry_jitter=music.api_retry_jitter,
    )


def _log_batch(stage: str):
    def cb(i: int, n: int, m: int) -> None:
        log(SCOPE_FEATURES, f"{stage}: batch {i}/{n} — {m} matched")

    return cb


# ── Stage 1 ────────────────────────────────────────────────────────────────────


def _resolve_spotify_ids(session, spotify: SpotifyClient, music: MusicSettings) -> None:
    rows = session.query(Music).filter(Music.spotify_id.is_(None)).all()
    if not rows:
        return

    total = len(rows)
    log(SCOPE_FEATURES, f"spotify: resolving {total} tracks")
    found = 0
    with progress(total, "Spotify lookup") as advance:
        for i, row in enumerate(rows, 1):
            try:
                sid = spotify.search_id(row.artist, row.track)
            except TransientError:
                row.spotify_id = _NO_MATCH
                advance(detail=f"{row.artist} – {row.track} (transient)")
                if i % music.commit_every == 0:
                    session.commit()
                continue
            row.spotify_id = sid or _NO_MATCH
            if sid:
                found += 1
            advance(detail=f"{row.artist} – {row.track}")
            if i % music.commit_every == 0:
                session.commit()
    session.commit()
    log(
        SCOPE_FEATURES,
        f"spotify: done — {found} found, {total - found} no match",
        level="ok",
    )


# ── Stage 2 ────────────────────────────────────────────────────────────────────


def _resolve_reccobeats_ids(session, rb: ReccoBeatsClient) -> None:
    rows = (
        session.query(Music)
        .filter(
            Music.spotify_id.is_not(None),
            Music.spotify_id != _NO_MATCH,
            Music.reccobeats_id.is_(None),
        )
        .all()
    )
    if not rows:
        return

    log(SCOPE_FEATURES, f"reccobeats_id: {len(rows)} tracks to resolve")
    rb_id_map = rb.get_ids(
        [r.spotify_id for r in rows if r.spotify_id],
        on_batch=_log_batch("reccobeats_id"),
    )
    for row in rows:
        row.reccobeats_id = rb_id_map.get(row.spotify_id, _NO_MATCH)
    session.commit()
    matched = sum(1 for r in rows if r.reccobeats_id != _NO_MATCH)
    log(
        SCOPE_FEATURES,
        f"reccobeats_id: done — {matched} matched, {len(rows) - matched} no match",
    )


# ── Stage 3 ────────────────────────────────────────────────────────────────────


def _enrich_catalog_features(session, rb: ReccoBeatsClient) -> None:
    rows = (
        session.query(Music)
        .filter(
            Music.reccobeats_id.is_not(None),
            Music.reccobeats_id != _NO_MATCH,
            Music.is_audio_features_extracted.is_(None),
        )
        .all()
    )
    if not rows:
        return

    log(SCOPE_FEATURES, f"catalog features: {len(rows)} tracks to enrich")
    feat_map = rb.get_features(
        [r.reccobeats_id for r in rows if r.reccobeats_id],
        on_batch=_log_batch("catalog features"),
    )
    enriched = 0
    for row in rows:
        feats = feat_map.get(row.reccobeats_id)
        if feats:
            for f in FEATURE_FIELDS:
                if f in feats:
                    setattr(row, f, feats[f])
            if music_has_features(row):
                row.is_audio_features_extracted = True
                enriched += 1
    session.commit()
    log(SCOPE_FEATURES, f"catalog features: done — {enriched} enriched")


# ── Stage 4 ────────────────────────────────────────────────────────────────────


def _enrich_upload_fallback(
    session,
    rb: ReccoBeatsClient,
    video_dir: str,
    music: MusicSettings,
) -> None:
    rows = (
        session.query(Music, func.min(Clip.id).label("clip_id"))
        .join(Clip, Clip.music_id == Music.id)
        .filter(
            *clip_used_in_analysis(),
            Music.is_audio_features_extracted.is_(None),
        )
        .group_by(Music.id)
        .order_by(Music.id)
        .all()
    )

    total = len(rows)
    if total:
        log(SCOPE_FEATURES, f"upload fallback: {total} tracks without catalog coverage")
        enriched = 0
        video_dir_path = Path(video_dir)
        with progress(total, "Upload fallback") as advance:
            for i, (row, clip_id) in enumerate(rows, 1):
                video = video_dir_path / f"{clip_id}.mp4"
                if not video.exists():
                    row.is_audio_features_extracted = False
                    advance(
                        detail=f"{row.artist} – {row.track} (video missing on disk)"
                    )
                    if i % music.commit_every == 0:
                        session.commit()
                    continue

                with tempfile.TemporaryDirectory(prefix="rb-audio-") as tmp:
                    audio = extract_audio_sample(video, Path(tmp), music)
                    if not audio:
                        row.is_audio_features_extracted = False
                        advance(detail=f"{row.artist} – {row.track} (ffmpeg failed)")
                        if i % music.commit_every == 0:
                            session.commit()
                        continue
                    try:
                        feats = rb.upload_features(audio)
                    except TransientError:
                        row.is_audio_features_extracted = False
                        advance(detail=f"{row.artist} – {row.track} (RB transient)")
                        if i % music.commit_every == 0:
                            session.commit()
                        continue

                if feats and any(f in feats for f in UPLOAD_FIELDS):
                    for f in UPLOAD_FIELDS:
                        if f in feats:
                            setattr(row, f, feats[f])
                    row.is_audio_features_extracted = True
                    enriched += 1
                    advance(detail=f"{row.artist} – {row.track} (ok)")
                else:
                    row.is_audio_features_extracted = False
                    advance(detail=f"{row.artist} – {row.track} (no features)")

                if i % music.commit_every == 0:
                    session.commit()
        session.commit()
        log(
            SCOPE_FEATURES,
            f"upload fallback: done — {enriched}/{total} enriched",
            level="ok",
        )

    swept = (
        session.query(Music)
        .filter(
            Music.is_audio_features_extracted.is_(None),
        )
        .update(
            {Music.is_audio_features_extracted: False},
            synchronize_session=False,
        )
    )
    if swept:
        session.commit()
        log(SCOPE_FEATURES, f"sweep: {swept} rows terminal-marked as False")


# ── Public entry ───────────────────────────────────────────────────────────────


def extract_music_features(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: MusicSecrets,
) -> None:
    """Fill Spotify IDs, ReccoBeats IDs, and audio features for all linked Music rows."""
    session = get_session()
    try:
        with httpx.Client(timeout=music.http_timeout) as http:
            spotify = _make_spotify(http, music, secrets)
            rb = _make_reccobeats(http, music)
            _resolve_spotify_ids(session, spotify, music)
            _resolve_reccobeats_ids(session, rb)
            _enrich_catalog_features(session, rb)
            _enrich_upload_fallback(session, rb, paths.video_dir, music)
    finally:
        session.close()
