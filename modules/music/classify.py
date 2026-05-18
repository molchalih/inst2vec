"""ACR fingerprinting stage: link clips to Music rows."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass

from acrcloud.recognizer import ACRCloudRecognizer

from core import fingerprint as fp
from core.config import MusicSettings, PathsSettings
from core.console import log, progress
from core.database import (
    Clip,
    Music,
    clip_used_in_analysis,
    get_session,
)
from modules.music.clients import TransientError
from modules.music.state import (
    SCOPE_CLASSIFY,
    SCOPE_MUSIC,
    STAGE_MUSIC_CLASSIFY,
    classify_config_payload,
    reset_music_classify,
)


@dataclass(frozen=True)
class AcrSecrets:
    host: str
    access_key: str
    access_secret: str


def _fingerprint(
    acr: ACRCloudRecognizer,
    path: str,
    min_confidence: float,
    max_attempts: int = 2,
    retry_delay: float = 1.0,
    retry_jitter: float = 1.5,
) -> tuple[str, str, float] | None:
    """Try ACR fingerprinting up to max_attempts on transient failures.

    Returns (artist, track, score) on match, None on clean no-match.
    Raises modules.music.clients.TransientError after exhausted retries.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            raw = acr.recognize_by_file(path, 0) or ""
            data = json.loads(raw)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(retry_delay + random.uniform(0, retry_jitter))
            continue

        status_code = data.get("status", {}).get("code")
        if status_code == 0:
            music = data.get("metadata", {}).get("music", [])
            if not music:
                return None  # clean no-match
            best = music[0]
            score = best.get("score", 0) / 100.0
            if score < min_confidence:
                return None
            artists = best.get("artists", [])
            artist = (artists[0].get("name") or "").strip() if artists else ""
            track = (best.get("title") or "").strip()
            return (artist, track, score) if (artist or track) else None

        # Non-zero ACR status codes:
        #   1001: no result (clean no-match, terminal)
        #   3xxx: HTTP/network error per ACR docs (transient)
        #   anything else: treat as permanent (corrupt signature, etc.)
        if status_code == 1001:
            return None
        if isinstance(status_code, int) and 3000 <= status_code < 4000:
            last_exc = RuntimeError(f"acr status {status_code}")
            if attempt < max_attempts - 1:
                time.sleep(retry_delay + random.uniform(0, retry_jitter))
            continue
        return None

    raise TransientError(f"acr exhausted: {last_exc!r}")


def _get_or_create_music(session, artist: str, track: str) -> Music:
    artist, track = artist.strip(), track.strip()
    row = session.query(Music).filter_by(artist=artist, track=track).first()
    if not row:
        row = Music(artist=artist, track=track)
        session.add(row)
        session.flush()
    return row


def classify_music(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: AcrSecrets,
) -> None:
    """Fingerprint unresolved clips with ACR; link them to Music rows.

    Sets is_music_recognized=True (match) or False (clean no-match / terminal-failed).
    """
    session = get_session()

    current = fp.Fingerprint(
        data=fp.hash_text(""),
        config=fp.hash_text(classify_config_payload(music)),
        dependency=fp.hash_text(""),
    )
    fp.gate(
        session,
        STAGE_MUSIC_CLASSIFY,
        SCOPE_MUSIC,
        current,
        reset_music_classify,
        log_scope=SCOPE_CLASSIFY,
        drift_msg="resetting music classify outputs",
    )

    clips = (
        session.query(Clip)
        .filter(
            Clip.is_music_recognized.is_(None),
            *clip_used_in_analysis(),
        )
        .order_by(Clip.id.desc())
        .all()
    )
    if not clips:
        fp.mark_complete(session, STAGE_MUSIC_CLASSIFY, SCOPE_MUSIC, current)
        session.commit()
        session.close()
        return

    acr = ACRCloudRecognizer(
        {
            "host": secrets.host,
            "access_key": secrets.access_key,
            "access_secret": secrets.access_secret,
            "timeout": 10,
        }
    )
    log(SCOPE_CLASSIFY, f"{len(clips)} clips to fingerprint")
    matched = no_match = missing = 0

    with progress(len(clips), "Fingerprinting") as advance:
        for i, clip in enumerate(clips, 1):
            path = paths.video_for(clip.id)
            if not path.exists():
                missing += 1
                advance()
                continue

            try:
                result = _fingerprint(
                    acr,
                    str(path),
                    music.audio_fingerprint_confidence,
                    max_attempts=music.acr_max_attempts,
                    retry_delay=music.api_retry_delay,
                    retry_jitter=music.api_retry_jitter,
                )
            except TransientError:
                # Leave is_music_recognized=None so the next run retries.
                # Counter is bumped under the "missing/transient" label so the
                # summary doesn't conflate transient errors with clean no-match.
                missing += 1
                advance(detail=f"{clip.id}: ACR transient (left retryable)")
                if i % music.commit_every == 0:
                    session.commit()
                continue

            if result:
                artist, track, confidence = result
                music_row = _get_or_create_music(session, artist, track)
                clip.music_id = music_row.id
                clip.music_confidence = confidence
                clip.is_music_recognized = True
                matched += 1
                advance(detail=f"{clip.id}: {artist} – {track} ({confidence:.0%})")
            else:
                clip.is_music_recognized = False
                no_match += 1
                advance()

            if i % music.commit_every == 0:
                session.commit()

    session.commit()
    fp.mark_complete(session, STAGE_MUSIC_CLASSIFY, SCOPE_MUSIC, current)
    session.commit()
    session.close()
    parts = [f"{matched} matched", f"{no_match} no match"]
    if missing:
        parts.append(f"{missing} skipped (video missing or ACR transient)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}", level="ok")
