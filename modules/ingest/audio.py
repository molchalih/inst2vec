"""Audio extraction stage: extract mp3 audio from downloaded videos."""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from core import fingerprint as fp
from core.console import log, progress
from core.database import Clip, get_session
from core.ffmpeg import run_ffmpeg
from core.pipeline import Stage

AUDIO_EXTRACT_STAGE = Stage.AUDIO_EXTRACT
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


def extract_audio_stage(settings) -> None:
    """Extract mp3 audio for every downloaded clip into ``paths.audio_dir``.

    Always runs. Idempotent via fingerprint seal + per-file mtime check.
    """
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
        paths = settings.paths
        os.makedirs(paths.audio_dir, exist_ok=True)

        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in ids),
            config=fp.hash_text(
                f"bitrate={settings.audio_extraction.audio_bitrate_kbps}"
                f"|sr={settings.audio_extraction.audio_sample_rate_hz}"
                f"|codec=libmp3lame"
            ),
            dependency=fp.hash_rows(
                fp.file_stat_for_hash(paths.video_for(cid)) for cid in ids
            ),
        )
        if not fp.is_stale(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current):
            # The fingerprint only hashes video stats, so deleting / truncating
            # an mp3 after a seal would not flip is_stale. Verify outputs exist
            # before trusting the seal; if anything is missing fall through and
            # re-extract (extract_audio is idempotent on intact outputs).
            missing = [c.id for c in clips if not paths.audio_for(c.id).exists()]
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
        bitrate = settings.audio_extraction.audio_bitrate_kbps
        sr = settings.audio_extraction.audio_sample_rate_hz
        timeout_s = settings.audio_extraction.audio_extract_timeout_s
        with (
            progress(len(clips), "Extracting audio") as advance,
            ThreadPoolExecutor(max_workers=settings.download.concurrency) as pool,
        ):
            future_to_id: dict = {}
            for clip in clips:
                video_path = str(paths.video_for(clip.id))
                audio_path = str(paths.audio_for(clip.id))
                if not os.path.exists(video_path):
                    failures += 1
                    advance(detail=f"✗ {clip.id} (no video)")
                    continue
                fut = pool.submit(
                    extract_audio,
                    video_path,
                    audio_path,
                    bitrate_kbps=bitrate,
                    sample_rate_hz=sr,
                    timeout_s=timeout_s,
                )
                future_to_id[fut] = clip.id

            for fut in as_completed(future_to_id):
                cid = future_to_id[fut]
                ok = fut.result()
                if ok:
                    advance(detail=f"✓ {cid}")
                else:
                    failures += 1
                    advance(detail=f"✗ {cid}")

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
