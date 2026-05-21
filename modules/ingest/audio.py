"""Audio extraction stage: extract mp3 audio from downloaded videos."""

from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

from core import fingerprint as fp
from core.console import log, progress
from core.database import Clip, StageState, get_session
from core.ffmpeg import has_audio_stream, run_ffmpeg
from core.pipeline import Stage

AUDIO_EXTRACT_STAGE = Stage.AUDIO_EXTRACT
AUDIO_EXTRACT_SCOPE = "default"
SCOPE = "audio"


class ExtractResult(NamedTuple):
    ok: bool
    duration: float
    size: int | None
    err: str | None


def extract_audio(
    video_path: str,
    audio_path: str,
    *,
    bitrate_kbps: int,
    sample_rate_hz: int,
    timeout_s: int,
    codec: str = "libmp3lame",
    extension: str = "mp3",
    channels: int | None = None,
    force: bool = False,
) -> ExtractResult:
    """Extract audio from ``video_path`` to ``audio_path``.

    ``codec`` selects the ffmpeg audio encoder. ``extension`` selects the
    container (``-f``). ``bitrate_kbps`` is ignored when the encoder is a
    PCM codec (``pcm_s16le``, etc.). ``channels`` forces ``-ac`` when set.
    ``force=True`` bypasses the mtime shortcut so the file is rewritten
    even if it already exists (used when encoding config has drifted).
    """
    t0 = time.perf_counter()
    if (
        not force
        and os.path.exists(audio_path)
        and os.path.exists(video_path)
        and os.path.getmtime(audio_path) >= os.path.getmtime(video_path)
    ):
        return ExtractResult(
            ok=True,
            duration=time.perf_counter() - t0,
            size=os.path.getsize(audio_path),
            err=None,
        )
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    tmp = audio_path + ".part"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-c:a", codec]
    if not codec.startswith("pcm_"):
        cmd += ["-b:a", f"{bitrate_kbps}k"]
    cmd += ["-ar", str(sample_rate_hz)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    cmd += ["-f", extension, tmp]
    ok = run_ffmpeg(cmd, timeout=timeout_s)
    if ok:
        os.replace(tmp, audio_path)
        return ExtractResult(
            ok=True,
            duration=time.perf_counter() - t0,
            size=os.path.getsize(audio_path),
            err=None,
        )
    if os.path.exists(tmp):
        with contextlib.suppress(OSError):
            os.remove(tmp)
    return ExtractResult(
        ok=False,
        duration=time.perf_counter() - t0,
        size=None,
        err="ffmpeg failed or timed out",
    )


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
            log(SCOPE, "SCAN", "clips", "none")
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
            log(SCOPE, "SKIP", "fingerprint", "ok")
            return

        failures = 0
        skipped = 0
        bitrate = settings.audio_extraction.audio_bitrate_kbps
        sr = settings.audio_extraction.audio_sample_rate_hz
        timeout_s = settings.audio_extraction.audio_extract_timeout_s
        t_stage = time.perf_counter()
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
                    log(
                        SCOPE,
                        "EXTRACT",
                        f"clip_{clip.id}.mp3",
                        "ERR",
                        stats={"err": "no video on disk"},
                    )
                    advance(detail=f"✗ {clip.id} (no video)")
                    continue
                if not has_audio_stream(video_path):
                    skipped += 1
                    log(
                        SCOPE,
                        "EXTRACT",
                        f"clip_{clip.id}.mp3",
                        "none",
                        stats={"reason": "no audio stream"},
                    )
                    advance(detail=f"⊘ {clip.id} (no audio)")
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
                result = fut.result()
                if result.ok:
                    log(
                        SCOPE,
                        "EXTRACT",
                        f"clip_{cid}.mp3",
                        "ok",
                        stats={
                            "time": result.duration,
                            "size": result.size or 0,
                        },
                    )
                    advance(detail=f"✓ {cid}")
                else:
                    failures += 1
                    log(
                        SCOPE,
                        "EXTRACT",
                        f"clip_{cid}.mp3",
                        "ERR",
                        stats={
                            "time": result.duration,
                            "err": result.err or "unknown",
                        },
                    )
                    advance(detail=f"✗ {cid}")

        if failures == 0:
            fp.mark_complete(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current)
            session.commit()
            log(
                SCOPE,
                "SEAL",
                "audio",
                "ok",
                stats={
                    "done": len(clips) - skipped,
                    "skipped": skipped,
                    "time": time.perf_counter() - t_stage,
                },
            )
        else:
            log(
                SCOPE,
                "SEAL",
                "audio",
                "stale",
                stats={
                    "done": len(clips) - failures - skipped,
                    "skipped": skipped,
                    "err": failures,
                    "time": time.perf_counter() - t_stage,
                },
            )
    finally:
        session.close()


AUDIO_EXTRACT_MIR_STAGE = Stage.AUDIO_EXTRACT_MIR
AUDIO_EXTRACT_MIR_SCOPE = "default"
SCOPE_MIR = "audio_mir"


def extract_audio_mir_stage(settings) -> None:
    """Extract high-quality WAV audio for every downloaded clip into ``paths.audio_mir_dir``.

    Sibling of ``extract_audio_stage``; runs after it in ``main.py``.
    Idempotent via fingerprint seal + per-file mtime check.
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
            log(SCOPE_MIR, "SCAN", "clips", "none")
            return

        ids = [c.id for c in clips]
        paths = settings.paths
        os.makedirs(paths.audio_mir_dir, exist_ok=True)

        ae = settings.audio_extraction
        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in ids),
            config=fp.hash_text(
                f"codec={ae.mir_codec}"
                f"|ext={ae.mir_extension}"
                f"|sr={ae.mir_sample_rate_hz}"
                f"|ch={ae.mir_channels}"
            ),
            dependency=fp.hash_rows(
                fp.file_stat_for_hash(paths.video_for(cid)) for cid in ids
            ),
        )
        if not fp.is_stale(
            session, AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE, current
        ):
            log(SCOPE_MIR, "SKIP", "fingerprint", "ok")
            return

        stored = session.get(
            StageState, (AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE)
        )
        force_reencode = stored is not None and stored.config_hash != current.config

        failures = 0
        skipped = 0
        timeout_s = ae.mir_extract_timeout_s
        t_stage = time.perf_counter()
        with (
            progress(len(clips), "Extracting MIR audio") as advance,
            ThreadPoolExecutor(max_workers=settings.download.concurrency) as pool,
        ):
            future_to_id: dict = {}
            for clip in clips:
                video_path = str(paths.video_for(clip.id))
                audio_path = str(paths.audio_mir_for(clip.id))
                if not os.path.exists(video_path):
                    failures += 1
                    log(
                        SCOPE_MIR,
                        "EXTRACT",
                        f"clip_{clip.id}.wav",
                        "ERR",
                        stats={"err": "no video on disk"},
                    )
                    advance(detail=f"✗ {clip.id} (no video)")
                    continue
                if not has_audio_stream(video_path):
                    skipped += 1
                    log(
                        SCOPE_MIR,
                        "EXTRACT",
                        f"clip_{clip.id}.wav",
                        "none",
                        stats={"reason": "no audio stream"},
                    )
                    advance(detail=f"⊘ {clip.id} (no audio)")
                    continue
                fut = pool.submit(
                    extract_audio,
                    video_path,
                    audio_path,
                    bitrate_kbps=0,
                    sample_rate_hz=ae.mir_sample_rate_hz,
                    timeout_s=timeout_s,
                    codec=ae.mir_codec,
                    extension=ae.mir_extension,
                    channels=ae.mir_channels,
                    force=force_reencode,
                )
                future_to_id[fut] = clip.id

            for fut in as_completed(future_to_id):
                cid = future_to_id[fut]
                result = fut.result()
                if result.ok:
                    log(
                        SCOPE_MIR,
                        "EXTRACT",
                        f"clip_{cid}.wav",
                        "ok",
                        stats={
                            "time": result.duration,
                            "size": result.size or 0,
                        },
                    )
                    advance(detail=f"✓ {cid}")
                else:
                    failures += 1
                    log(
                        SCOPE_MIR,
                        "EXTRACT",
                        f"clip_{cid}.wav",
                        "ERR",
                        stats={
                            "time": result.duration,
                            "err": result.err or "unknown",
                        },
                    )
                    advance(detail=f"✗ {cid}")

        if failures == 0:
            fp.mark_complete(
                session, AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE, current
            )
            session.commit()
            log(
                SCOPE_MIR,
                "SEAL",
                "audio_mir",
                "ok",
                stats={
                    "done": len(clips) - skipped,
                    "skipped": skipped,
                    "time": time.perf_counter() - t_stage,
                },
            )
        else:
            log(
                SCOPE_MIR,
                "SEAL",
                "audio_mir",
                "stale",
                stats={
                    "done": len(clips) - failures - skipped,
                    "skipped": skipped,
                    "err": failures,
                    "time": time.perf_counter() - t_stage,
                },
            )
    finally:
        session.close()
