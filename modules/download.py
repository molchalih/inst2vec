"""Audio extraction stage."""

from __future__ import annotations

import contextlib
import os

from modules import fingerprint as fp
from modules.console import log, progress
from modules.database import Base, Clip, get_engine, get_session
from modules.ffmpeg import run_ffmpeg

AUDIO_EXTRACT_STAGE = "audio_extract"
AUDIO_EXTRACT_SCOPE = "default"


def extract_audio(
    video_path: str,
    audio_path: str,
    *,
    bitrate_kbps: int,
    sample_rate_hz: int,
    timeout_s: int,
) -> bool:
    """Extract mp3 audio from ``video_path`` to ``audio_path``.

    Idempotent: returns True without invoking ffmpeg when ``audio_path``
    exists and is at least as new as ``video_path``.
    """
    if (
        os.path.exists(audio_path)
        and os.path.exists(video_path)
        and os.path.getmtime(audio_path) >= os.path.getmtime(video_path)
    ):
        return True
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    # Write to a temp path and only replace ``audio_path`` on success so a
    # timeout / non-zero ffmpeg leaves no truncated mp3 behind. Without this,
    # the mtime short-circuit above would treat a partial file as fresh and
    # extract_audio_stage would seal over corrupt input.
    tmp = audio_path + ".part"
    # Pass ``-f mp3`` because the temp filename doesn't end in .mp3 and ffmpeg
    # otherwise infers the container from the extension.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_kbps}k",
        "-ar",
        str(sample_rate_hz),
        "-f",
        "mp3",
        tmp,
    ]
    ok = run_ffmpeg(cmd, timeout=timeout_s)
    if ok:
        os.replace(tmp, audio_path)
        return True
    if os.path.exists(tmp):
        with contextlib.suppress(OSError):
            os.remove(tmp)
    return False


def _video_stat(video_dir: str, clip_id: int) -> tuple[int, int]:
    p = os.path.join(video_dir, f"{clip_id}.mp4")
    if not os.path.exists(p):
        return (-1, -1)
    st = os.stat(p)
    return (st.st_size, st.st_mtime_ns)


def extract_audio_stage(settings) -> None:
    """Extract mp3 audio for every downloaded clip into ``paths.audio_dir``.

    No-op when ``embeddings.gemini_enabled`` is False — gemini_mm is the
    only consumer today.
    """
    if not settings.embeddings.gemini_enabled:
        log(AUDIO_EXTRACT_STAGE, "disabled — skipping")
        return

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        clips = (
            session.query(Clip)
            .filter(Clip.is_downloaded.is_(True))
            .order_by(Clip.id)
            .all()
        )
        if not clips:
            log(AUDIO_EXTRACT_STAGE, "no downloaded clips — nothing to do")
            return

        ids = [c.id for c in clips]
        video_dir = settings.paths.video_dir
        audio_dir = settings.paths.audio_dir
        os.makedirs(audio_dir, exist_ok=True)

        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in ids),
            config=fp.hash_text(
                f"bitrate={settings.embeddings.audio_bitrate_kbps}"
                f"|sr={settings.embeddings.audio_sample_rate_hz}"
                f"|codec=libmp3lame"
            ),
            dependency=fp.hash_rows(_video_stat(video_dir, cid) for cid in ids),
        )
        if not fp.is_stale(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current):
            # The fingerprint only hashes video stats, so deleting / truncating
            # an mp3 after a seal would not flip is_stale. Verify outputs exist
            # before trusting the seal; if anything is missing fall through and
            # re-extract (extract_audio is idempotent on intact outputs).
            missing = [
                c.id
                for c in clips
                if not os.path.exists(os.path.join(audio_dir, f"{c.id}.mp3"))
            ]
            if not missing:
                log(AUDIO_EXTRACT_STAGE, "fingerprint match — skipping")
                return
            log(
                AUDIO_EXTRACT_STAGE,
                f"fingerprint match but {len(missing)} mp3 output(s) missing "
                "— re-extracting",
                level="warn",
            )

        failures = 0
        with progress(len(clips), "Extracting audio") as advance:
            for clip in clips:
                video_path = os.path.join(video_dir, f"{clip.id}.mp4")
                audio_path = os.path.join(audio_dir, f"{clip.id}.mp3")
                if not os.path.exists(video_path):
                    failures += 1
                    advance(detail=f"✗ {clip.id} (no video)")
                    continue
                ok = extract_audio(
                    video_path,
                    audio_path,
                    bitrate_kbps=settings.embeddings.audio_bitrate_kbps,
                    sample_rate_hz=settings.embeddings.audio_sample_rate_hz,
                    timeout_s=settings.embeddings.audio_extract_timeout_s,
                )
                if ok:
                    advance(detail=f"✓ {clip.id}")
                else:
                    failures += 1
                    advance(detail=f"✗ {clip.id}")

        if failures == 0:
            fp.mark_complete(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current)
            session.commit()
            log(AUDIO_EXTRACT_STAGE, "done", level="ok")
        else:
            log(
                AUDIO_EXTRACT_STAGE,
                f"{failures}/{len(clips)} failed — leaving stage stale for retry",
                level="warn",
            )
    finally:
        session.close()
