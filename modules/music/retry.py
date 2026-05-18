"""Manual recovery for failed Music rows.

Functions here are also invoked by ``scripts/retry_failed_music_*.py``
CLI wrappers. Tests import from this module directly.
"""

from __future__ import annotations

from core.config import MusicSettings, PathsSettings
from core.console import log
from core.database import Clip, Music, clip_used_in_analysis, get_session
from modules.music.classify import AcrSecrets, classify_music
from modules.music.features import MusicSecrets, extract_music_features

_SCOPE_FEATURES = "retry-features"
_SCOPE_RECOGNITION = "retry-music"


def retry_failed_features(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: MusicSecrets,
) -> None:
    """Re-attempt Music rows marked is_audio_features_extracted=False.

    Resets terminal markers on target rows (is_audio_features_extracted=NULL;
    recognition_status from "no_match" back to "pending") and re-runs
    extract_music_features with api_max_attempts=1.
    """
    session = get_session()
    try:
        failed = (
            session.query(Music)
            .filter(Music.is_audio_features_extracted.is_(False))
            .all()
        )
        if not failed:
            log(_SCOPE_FEATURES, "no failed music rows to retry")
            return

        log(_SCOPE_FEATURES, f"resetting {len(failed)} failed music rows")
        for row in failed:
            row.is_audio_features_extracted = None
            if row.recognition_status == "no_match":
                row.recognition_status = "pending"
        session.commit()
    finally:
        session.close()

    retry_music = music.model_copy(
        update={
            "api_max_attempts": 1,
            "api_retry_delay": 0.0,
            "api_retry_jitter": 0.0,
        }
    )
    log(_SCOPE_FEATURES, "re-running extract_music_features with api_max_attempts=1")
    extract_music_features(music=retry_music, paths=paths, secrets=secrets)


def retry_failed_recognition(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: AcrSecrets,
) -> None:
    """Re-attempt Clip rows stuck in failed recognition states.

    Flips is_music_recognized=False rows back to NULL, then calls
    classify_music with the unmodified MusicSettings so the classify seal
    is preserved (overriding retry knobs would change the config hash and
    trigger reset_music_classify, wiping every Music row).
    """
    session = get_session()
    try:
        failed = (
            session.query(Clip)
            .filter(Clip.is_music_recognized.is_(False), *clip_used_in_analysis())
            .all()
        )
        if not failed:
            log(_SCOPE_RECOGNITION, "no failed clips to retry")
            return

        log(
            _SCOPE_RECOGNITION,
            f"resetting {len(failed)} failed clips to retryable state",
        )
        for clip in failed:
            clip.is_music_recognized = None
        session.commit()
    finally:
        session.close()

    log(_SCOPE_RECOGNITION, "re-running classify_music with configured MusicSettings")
    classify_music(music=music, paths=paths, secrets=secrets)
