"""Audio extraction stage: extract mp3 audio from downloaded videos."""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

from core import fingerprint as fp
from core.config import AudioExtractionSettings
from core.console import progress
from core.database import Clip, StageState, clip_used_in_analysis, get_session
from core.ffmpeg import has_audio_stream, run_ffmpeg
from core.fingerprint import stable_subset_payload
from core.log import StageResult, event, stage
from core.pipeline import (
    AUDIO_EXTRACT_MIR_SCOPE,
    AUDIO_EXTRACT_MIR_STAGE,
    Stage,
)

AUDIO_EXTRACT_STAGE = Stage.AUDIO_EXTRACT
AUDIO_EXTRACT_SCOPE = "default"


class ExtractResult(NamedTuple):
    ok: bool
    duration: float
    size: int | None
    err: str | None


def _output_is_fresh(audio_path: str, video_path: str) -> bool:
    """True when the extracted audio exists and is at least as new as its source.

    The cheap, file-only signal that a clip is already done: no ffprobe, no
    ffmpeg. Lets a stale-fingerprint re-walk skip every previously-extracted
    clip without re-probing it.
    """
    return (
        os.path.exists(audio_path)
        and os.path.exists(video_path)
        and os.path.getmtime(audio_path) >= os.path.getmtime(video_path)
    )


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
    if not force and _output_is_fresh(audio_path, video_path):
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


def _consume_extract_results(future_to_id: dict, advance) -> int:
    """Drain completed extraction futures, logging each. Returns failure count."""
    failures = 0
    for fut in as_completed(future_to_id):
        cid = future_to_id[fut]
        result = fut.result()
        if result.ok:
            event(
                "EXTRACT",
                f"clip_{cid}",
                stats={
                    "time": result.duration,
                    "size": result.size or 0,
                },
            )
            advance(detail=f"✓ {cid}")
        else:
            failures += 1
            event(
                "EXTRACT",
                f"clip_{cid}",
                result="ERR",
                stats={
                    "time": result.duration,
                    "err": result.err or "unknown",
                },
            )
            advance(detail=f"✗ {cid}")
    return failures


class _ClipExtractDecision(NamedTuple):
    """Outcome of inspecting one clip before submitting an extraction job."""

    action: str  # "fail" | "fresh" | "skip" | "submit"
    video_path: str
    audio_path: str


def _decide_clip_extract(
    clip,
    paths,
    *,
    audio_for,
    force: bool,
    advance,
) -> _ClipExtractDecision:
    """Resolve paths and emit any terminal/cached/skip event for one clip.

    ``audio_for`` is the path-builder (``paths.audio_for`` or
    ``paths.audio_mir_for``). ``force`` bypasses the freshness shortcut.
    Logging and progress side effects match the original inline loop.
    """
    video_path = str(paths.video_for(clip.id))
    audio_path = str(audio_for(clip.id))
    if not os.path.exists(video_path):
        event(
            "EXTRACT",
            f"clip_{clip.id}",
            result="ERR",
            stats={"err": "no video on disk"},
        )
        advance(detail=f"✗ {clip.id} (no video)")
        return _ClipExtractDecision("fail", video_path, audio_path)
    if not force and _output_is_fresh(audio_path, video_path):
        advance(detail=f"• {clip.id} (cached)")
        return _ClipExtractDecision("fresh", video_path, audio_path)
    if not has_audio_stream(video_path):
        event(
            "EXTRACT",
            f"clip_{clip.id}",
            stats={"reason": "no audio stream"},
        )
        advance(detail=f"⊘ {clip.id} (no audio)")
        return _ClipExtractDecision("skip", video_path, audio_path)
    return _ClipExtractDecision("submit", video_path, audio_path)


@stage("ingest:audio")
def extract_audio_stage(settings) -> StageResult:
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
            return StageResult(done=0)

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
            event("SKIP", "fingerprint")
            return StageResult(done=0)

        failures = 0
        skipped = 0
        fresh = 0
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
                decision = _decide_clip_extract(
                    clip, paths, audio_for=paths.audio_for, force=False, advance=advance
                )
                if decision.action == "fail":
                    failures += 1
                    continue
                if decision.action == "fresh":
                    fresh += 1
                    continue
                if decision.action == "skip":
                    skipped += 1
                    continue
                fut = pool.submit(
                    extract_audio,
                    decision.video_path,
                    decision.audio_path,
                    bitrate_kbps=bitrate,
                    sample_rate_hz=sr,
                    timeout_s=timeout_s,
                )
                future_to_id[fut] = clip.id

            failures += _consume_extract_results(future_to_id, advance)

        if failures == 0:
            fp.mark_complete(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current)
            session.commit()

        done = len(clips) - skipped - fresh - failures
        return StageResult(
            done=done,
            cached=fresh,
            skipped=skipped,
            failed=failures,
            time=time.perf_counter() - t_stage,
        )
    finally:
        session.close()


_MIR_EXTRACT_CONFIG_FIELDS: tuple[str, ...] = (
    "mir_codec",
    "mir_extension",
    "mir_sample_rate_hz",
    "mir_channels",
)


def _mir_config_payload(audio_extraction: AudioExtractionSettings) -> str:
    """Stable JSON of AudioExtractionSettings fields that affect the MIR WAV."""
    return stable_subset_payload(audio_extraction, _MIR_EXTRACT_CONFIG_FIELDS)


_WAV_FILENAME_RE = re.compile(r"^(\d+)\.wav$")


def _probe_wav_format(path: str, *, timeout: int = 5) -> tuple[int, int] | None:
    """Return ``(sample_rate_hz, channels)`` for a WAV file, or None on probe failure."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        streams = json.loads(result.stdout or "{}").get("streams") or []
        if not streams:
            return None
        sr = int(streams[0].get("sample_rate"))
        ch = int(streams[0].get("channels"))
    except (KeyError, TypeError, ValueError):
        return None
    return (sr, ch)


def sweep_orphan_mir_wavs(
    *,
    session,
    audio_mir_dir: str,
    expected_sample_rate_hz: int,
    expected_channels: int,
) -> int:
    """Delete WAVs whose clip_id is not selected or whose probed format drifted.

    Returns the count of files removed. Idempotent: a clean tree deletes zero
    files. WAVs that ffprobe cannot read are conservatively kept (a probe
    failure is not a verdict of format drift).
    """
    if not os.path.isdir(audio_mir_dir):
        return 0

    selected_ids = {
        cid for (cid,) in session.query(Clip.id).filter(*clip_used_in_analysis()).all()
    }

    removed = 0
    for entry in os.listdir(audio_mir_dir):
        match = _WAV_FILENAME_RE.match(entry)
        if not match:
            continue
        clip_id = int(match.group(1))
        path = os.path.join(audio_mir_dir, entry)
        if clip_id not in selected_ids:
            with contextlib.suppress(OSError):
                os.remove(path)
                removed += 1
            continue
        probed = _probe_wav_format(path)
        if probed is None:
            continue  # conservative: keep unprobeable WAVs
        if probed != (expected_sample_rate_hz, expected_channels):
            with contextlib.suppress(OSError):
                os.remove(path)
                removed += 1
    return removed


@stage("ingest:audio_mir")
def extract_audio_mir_stage(settings) -> StageResult:
    """Extract high-quality WAV audio for every downloaded clip into ``paths.audio_mir_dir``.

    Sibling of ``extract_audio_stage``; runs after it in ``main.py``.
    Idempotent via fingerprint seal + per-file mtime check.
    """
    session = get_session()
    try:
        clips = (
            session.query(Clip).filter(*clip_used_in_analysis()).order_by(Clip.id).all()
        )
        if not clips:
            return StageResult(done=0)

        ids = [c.id for c in clips]
        paths = settings.paths
        os.makedirs(paths.audio_mir_dir, exist_ok=True)

        ae = settings.audio_extraction
        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in ids),
            config=fp.hash_text(_mir_config_payload(ae)),
            dependency=fp.hash_rows(
                fp.file_stat_for_hash(paths.video_for(cid)) for cid in ids
            ),
        )
        if not fp.is_stale(
            session, AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE, current
        ):
            event("SKIP", "fingerprint")
            return StageResult(done=0)

        stored = session.get(
            StageState, (AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE)
        )
        # Re-encode every existing WAV only when the encode *config* drifts
        # (sample rate / codec / channels). Dependency drift just means the
        # clip set grew or a video was re-downloaded — a re-download is already
        # caught per-file by extract_audio's mtime check (video newer than WAV),
        # so forcing a full re-encode on dependency drift needlessly rewrites
        # every WAV whenever new clips are added.
        force_reencode = stored is not None and stored.config_hash != current.config

        n_removed = sweep_orphan_mir_wavs(
            session=session,
            audio_mir_dir=paths.audio_mir_dir,
            expected_sample_rate_hz=ae.mir_sample_rate_hz,
            expected_channels=ae.mir_channels,
        )
        if n_removed:
            event("DELETE", "audio_mir", stats={"removed": n_removed})

        failures = 0
        skipped = 0
        fresh = 0
        timeout_s = ae.mir_extract_timeout_s
        t_stage = time.perf_counter()
        with (
            progress(len(clips), "Extracting MIR audio") as advance,
            ThreadPoolExecutor(max_workers=settings.download.concurrency) as pool,
        ):
            future_to_id: dict = {}
            for clip in clips:
                decision = _decide_clip_extract(
                    clip,
                    paths,
                    audio_for=paths.audio_mir_for,
                    force=force_reencode,
                    advance=advance,
                )
                if decision.action == "fail":
                    failures += 1
                    continue
                if decision.action == "fresh":
                    fresh += 1
                    continue
                if decision.action == "skip":
                    skipped += 1
                    continue
                fut = pool.submit(
                    extract_audio,
                    decision.video_path,
                    decision.audio_path,
                    bitrate_kbps=0,
                    sample_rate_hz=ae.mir_sample_rate_hz,
                    timeout_s=timeout_s,
                    codec=ae.mir_codec,
                    extension=ae.mir_extension,
                    channels=ae.mir_channels,
                    force=force_reencode,
                )
                future_to_id[fut] = clip.id

            failures += _consume_extract_results(future_to_id, advance)

        if failures == 0:
            fp.mark_complete(
                session, AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE, current
            )
            session.commit()

        done = len(clips) - skipped - fresh - failures
        return StageResult(
            done=done,
            cached=fresh,
            skipped=skipped,
            failed=failures,
            time=time.perf_counter() - t_stage,
        )
    finally:
        session.close()
