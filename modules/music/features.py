"""Spotify → ReccoBeats → audio features pipeline.

Four sub-stages, each idempotent:
    1. _resolve_spotify_ids   — fill Music.spotify_id and Music.recognition_status
                                 ("matched" with id, "no_match" without; transient
                                 failures leave both NULL/"pending" for retry)
    2. _resolve_reccobeats_ids — fill Music.reccobeats_id (batched)
    3. _enrich_catalog_features — fill audio feature columns (batched); writes True on success
    4. _enrich_upload_fallback  — ffmpeg + POST for rows Stage 3 missed; writes True/False;
                                 sweeps remaining NULL → False at the end (only for rows
                                 with Stage 1 terminated, i.e. recognition_status != "pending";
                                 transient Stage-1 failures stay NULL so the next run retries
                                 them)
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from core import fingerprint as fp
from core.config import MusicSettings, PathsSettings
from core.console import log, progress
from core.database import Clip, Music, clip_used_in_analysis, get_session
from modules.music.audio_sample import extract_audio_sample
from modules.music.clients import ReccoBeatsClient, SpotifyClient, TransientError
from modules.music.state import (
    FEATURE_FIELDS,
    SCOPE_MUSIC,
    STAGE_MUSIC_FEATURES,
    UPLOAD_FIELDS,
    features_config_payload,
    music_has_features,
    reset_music_features,
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


def _log_batch(scope: str):
    def cb(i: int, n: int, m: int) -> None:
        log(scope, "SCAN", f"batch/{i}", "ok", stats={"of": n, "matched": m})

    return cb


# ── Stage 1 ────────────────────────────────────────────────────────────────────


def _resolve_spotify_ids(session, spotify: SpotifyClient, music: MusicSettings) -> None:
    rows = session.query(Music).filter(Music.recognition_status == "pending").all()
    if not rows:
        return

    total = len(rows)
    log("features:spotify", "SCAN", "tracks", "ok", stats={"todo": total})
    found = 0
    t_pass = time.perf_counter()
    with progress(total, "Spotify lookup") as advance:
        for i, row in enumerate(rows, 1):
            t0 = time.perf_counter()
            try:
                sid = spotify.search_id(row.artist, row.track)
            except TransientError as exc:
                log(
                    "features:spotify",
                    "GET",
                    f"music_{row.id}",
                    "ERR",
                    stats={
                        "time": time.perf_counter() - t0,
                        "err": f"spotify transient: {exc}",
                    },
                )
                advance(detail=f"{row.artist} – {row.track} (transient, retryable)")
                if i % music.commit_every == 0:
                    session.commit()
                continue
            if sid:
                row.spotify_id = sid
                row.recognition_status = "matched"
                found += 1
                log(
                    "features:spotify",
                    "GET",
                    f"music_{row.id}",
                    "ok",
                    stats={"time": time.perf_counter() - t0, "spotify_id": sid},
                )
            else:
                row.recognition_status = "no_match"
                log(
                    "features:spotify",
                    "GET",
                    f"music_{row.id}",
                    "none",
                    stats={"time": time.perf_counter() - t0},
                )
            advance(detail=f"{row.artist} – {row.track}")
            if i % music.commit_every == 0:
                session.commit()
    session.commit()
    log(
        "features:spotify",
        "SEAL",
        "spotify",
        "ok",
        stats={
            "matched": found,
            "no_match": total - found,
            "time": time.perf_counter() - t_pass,
        },
    )


# ── Stage 2 ────────────────────────────────────────────────────────────────────


def _resolve_reccobeats_ids(session, rb: ReccoBeatsClient) -> set[int]:
    """Returns the set of music_ids whose RB id lookup landed in a batch
    that exhausted transient retries — those rows must stay retryable
    (excluded from the upload-fallback sweep)."""
    rows = (
        session.query(Music)
        .filter(
            Music.recognition_status == "matched",
            Music.reccobeats_id.is_(None),
        )
        .all()
    )
    if not rows:
        return set()

    t_pass = time.perf_counter()
    log("features:reccobeats", "SCAN", "tracks", "ok", stats={"todo": len(rows)})
    rb_id_map, exhausted_spotify_ids = rb.get_ids(
        [r.spotify_id for r in rows if r.spotify_id],
        on_batch=_log_batch("features:reccobeats"),
    )
    transient: set[int] = set()
    for row in rows:
        if row.spotify_id in exhausted_spotify_ids:
            transient.add(row.id)
            continue
        row.reccobeats_id = rb_id_map.get(row.spotify_id)
    session.commit()
    matched = sum(1 for r in rows if r.reccobeats_id is not None)
    log(
        "features:reccobeats",
        "SEAL",
        "reccobeats",
        "ok",
        stats={
            "matched": matched,
            "no_match": len(rows) - matched - len(transient),
            "transient": len(transient),
            "time": time.perf_counter() - t_pass,
        },
    )
    return transient


# ── Stage 3 ────────────────────────────────────────────────────────────────────


def _enrich_catalog_features(session, rb: ReccoBeatsClient) -> set[int]:
    """Returns the set of music_ids whose catalog-features lookup landed
    in a batch that exhausted transient retries — those rows must stay
    retryable (excluded from the upload-fallback sweep)."""
    rows = (
        session.query(Music)
        .filter(
            Music.reccobeats_id.is_not(None),
            Music.is_audio_features_extracted.is_(None),
        )
        .all()
    )
    if not rows:
        return set()

    t_pass = time.perf_counter()
    log("features:catalog", "SCAN", "tracks", "ok", stats={"todo": len(rows)})
    feat_map, exhausted_rb_ids = rb.get_features(
        [r.reccobeats_id for r in rows if r.reccobeats_id],
        on_batch=_log_batch("features:catalog"),
    )
    transient: set[int] = set()
    enriched = 0
    for row in rows:
        if row.reccobeats_id in exhausted_rb_ids:
            transient.add(row.id)
            continue
        feats = feat_map.get(row.reccobeats_id)
        if feats:
            for f in FEATURE_FIELDS:
                if f in feats:
                    setattr(row, f, feats[f])
            if music_has_features(row):
                row.is_audio_features_extracted = True
                enriched += 1
    session.commit()
    log(
        "features:catalog",
        "SEAL",
        "catalog",
        "ok",
        stats={
            "enriched": enriched,
            "transient": len(transient),
            "time": time.perf_counter() - t_pass,
        },
    )
    return transient


# ── Stage 4 ────────────────────────────────────────────────────────────────────


def _enrich_upload_fallback(
    session,
    rb: ReccoBeatsClient,
    video_dir: str,
    music: MusicSettings,
    transient_music_ids: set[int] | None = None,
) -> None:
    # Require Stage 1 to have terminated for the row (matched or no_match) —
    # rows still at recognition_status="pending" are Stage 1 transient
    # failures that must stay pending so the next run retries them.
    # ``transient_music_ids`` carries music_ids whose Stage-2 or Stage-3 RB
    # batch exhausted transient retries; those rows are skipped both as
    # upload-fallback candidates and by the final sweep.
    external_transient: set[int] = (
        set(transient_music_ids) if transient_music_ids else set()
    )
    pairs = (
        session.query(Music.id.label("music_id"), Clip.id.label("clip_id"))
        .join(Clip, Clip.music_id == Music.id)
        .filter(
            *clip_used_in_analysis(),
            Music.recognition_status != "pending",
            Music.is_audio_features_extracted.is_(None),
        )
        .order_by(Music.id, Clip.id)
        .all()
    )

    candidates_by_music: dict[int, list[int]] = {}
    for music_id, clip_id in pairs:
        if music_id in external_transient:
            continue
        candidates_by_music.setdefault(music_id, []).append(clip_id)

    transient_music_ids_local: set[int] = set(external_transient)
    total = len(candidates_by_music)

    if total:
        log("features:upload_fb", "SCAN", "tracks", "ok", stats={"todo": total})
        enriched = 0
        t_pass = time.perf_counter()
        video_dir_path = Path(video_dir)
        with progress(total, "Upload fallback") as advance:
            for i, (music_id, clip_ids) in enumerate(candidates_by_music.items(), 1):
                row = session.get(Music, music_id)
                label = f"{row.artist} – {row.track}"
                t0 = time.perf_counter()
                outcome = _try_candidates_for_music(
                    rb, row, clip_ids, video_dir_path, music
                )
                if outcome == "ok":
                    enriched += 1
                    log(
                        "features:upload_fb",
                        "PUT",
                        f"music_{music_id}",
                        "ok",
                        stats={"time": time.perf_counter() - t0},
                    )
                    advance(detail=f"{label} (ok)")
                elif outcome == "transient":
                    transient_music_ids_local.add(music_id)
                    log(
                        "features:upload_fb",
                        "PUT",
                        f"music_{music_id}",
                        "ERR",
                        stats={
                            "time": time.perf_counter() - t0,
                            "err": "RB transient (retryable)",
                        },
                    )
                    advance(detail=f"{label} (RB transient — left retryable)")
                else:
                    log(
                        "features:upload_fb",
                        "PUT",
                        f"music_{music_id}",
                        "none",
                        stats={
                            "time": time.perf_counter() - t0,
                            "reason": outcome,
                        },
                    )
                    advance(detail=f"{label} ({outcome})")

                if i % music.commit_every == 0:
                    session.commit()
        session.commit()
        log(
            "features:upload_fb",
            "SEAL",
            "upload_fb",
            "ok",
            stats={
                "enriched": enriched,
                "of": total,
                "time": time.perf_counter() - t_pass,
            },
        )

    sweep_q = session.query(Music).filter(
        Music.recognition_status != "pending",
        Music.is_audio_features_extracted.is_(None),
    )
    if transient_music_ids_local:
        sweep_q = sweep_q.filter(~Music.id.in_(transient_music_ids_local))
    swept = sweep_q.update(
        {Music.is_audio_features_extracted: False},
        synchronize_session=False,
    )
    if swept:
        session.commit()
        log("features", "WRITE", "music_features", "ok", stats={"rows": swept})


def _try_candidates_for_music(
    rb: ReccoBeatsClient,
    row: Music,
    clip_ids: list[int],
    video_dir_path: Path,
    music: MusicSettings,
) -> str:
    """Try clip candidates in order; return one of:
    "ok"             — features written, row marked True
    "no video"       — all candidates lacked a video on disk
    "ffmpeg failed"  — all candidates failed ffmpeg
    "no features"    — RB returned an empty payload for every candidate
    "transient"      — RB raised TransientError; row left at NULL
    """
    last_perm_reason = "no video"
    for clip_id in clip_ids:
        video = video_dir_path / f"{clip_id}.mp4"
        if not video.exists():
            last_perm_reason = "no video"
            continue
        with tempfile.TemporaryDirectory(prefix="rb-audio-") as tmp:
            audio = extract_audio_sample(video, Path(tmp), music)
            if not audio:
                last_perm_reason = "ffmpeg failed"
                continue
            try:
                feats = rb.upload_features(audio)
            except TransientError:
                return "transient"
        if feats and any(f in feats for f in UPLOAD_FIELDS):
            for f in UPLOAD_FIELDS:
                if f in feats:
                    setattr(row, f, feats[f])
            row.is_audio_features_extracted = True
            return "ok"
        last_perm_reason = "no features"
    row.is_audio_features_extracted = False
    return last_perm_reason


# ── Public entry ───────────────────────────────────────────────────────────────


def extract_music_features(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: MusicSecrets,
) -> None:
    """Fill Spotify IDs, ReccoBeats IDs, and audio features for all linked Music rows."""
    session = get_session()
    t_stage = time.perf_counter()
    try:
        current = fp.Fingerprint(
            data=fp.hash_text(""),
            config=fp.hash_text(features_config_payload(music)),
            dependency=fp.hash_text(""),
        )
        fp.gate(
            session,
            STAGE_MUSIC_FEATURES,
            SCOPE_MUSIC,
            current,
            reset_music_features,
            log_scope="features",
            drift_msg="resetting feature columns",
        )

        with httpx.Client(timeout=music.http_timeout) as http:
            spotify = _make_spotify(http, music, secrets)
            rb = _make_reccobeats(http, music)
            _resolve_spotify_ids(session, spotify, music)
            transient = _resolve_reccobeats_ids(session, rb)
            transient |= _enrich_catalog_features(session, rb)
            _enrich_upload_fallback(
                session,
                rb,
                paths.video_dir,
                music,
                transient_music_ids=transient,
            )

        fp.mark_complete(session, STAGE_MUSIC_FEATURES, SCOPE_MUSIC, current)
        session.commit()
        log(
            "features",
            "SEAL",
            "features",
            "ok",
            stats={"time": time.perf_counter() - t_stage},
        )
    finally:
        session.close()
