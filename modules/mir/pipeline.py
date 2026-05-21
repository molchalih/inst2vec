"""run_mir: per-clip inference loop over MAEST + EffNet heads."""

from __future__ import annotations

import time
from pathlib import Path
from queue import Queue
from threading import Thread
from time import perf_counter

import numpy as np
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import MirSettings, Settings
from core.console import log, progress
from core.database import AudioMIR, Clip, clip_used_in_analysis, get_session
from core.pipeline import AUDIO_EXTRACT_MIR_SCOPE, AUDIO_EXTRACT_MIR_STAGE
from modules.mir.checkpoints import ensure_checkpoints, validate_checkpoint_sidecars
from modules.mir.descriptors import load_labels, topk_csv
from modules.mir.models import build_effnet, build_maest
from modules.mir.state import (
    _RESET_COLUMNS,
    POS,
    SCOPE_MIR,
    STAGE_MIR,
    mir_config_payload,
    reset_audio_mir,
)

_LABELS_DIR = Path(__file__).resolve().parent / "labels"
_SENTINEL = object()
_LOG = "mir"
_LOG_LOAD = "mir:load"
_LOG_INFER = "mir:infer"


def _load_audio(path: str, sr: int) -> np.ndarray:
    from essentia.standard import MonoLoader  # ty: ignore[unresolved-import]

    return MonoLoader(filename=path, sampleRate=sr, resampleQuality=4)()


def _terminal_failure(clip_id: int, err: str) -> AudioMIR:
    return AudioMIR(clip_id=clip_id, is_mir_extracted=False, mir_error=err)


def _infer_with_error_attribution(
    clip_id: int,
    audio: np.ndarray,
    maest,
    effnet,
    mir: MirSettings,
    labels_genre: list[str],
    labels_moodtheme: list[str],
    labels_instrument: list[str],
) -> AudioMIR:
    """Run inference with explicit phase attribution.

    Errors during ``maest.predict`` produce ``mir_error="maest"``;
    errors during EffNet embed/predict produce ``mir_error="effnet"``.
    """
    t0 = perf_counter()
    try:
        genre_probs = maest.predict(audio)
    except Exception:  # per-clip safety net
        return _terminal_failure(clip_id, "maest")
    try:
        eff_embed = effnet.embed(audio)
        heads = effnet.predict_all(eff_embed)
    except Exception:  # per-clip safety net
        return _terminal_failure(clip_id, "effnet")

    g_lab, g_sc = topk_csv(genre_probs, labels_genre, mir.topk_genre)
    m_lab, m_sc = topk_csv(heads["moodtheme"], labels_moodtheme, mir.topk_moodtheme)
    i_lab, i_sc = topk_csv(heads["instrument"], labels_instrument, mir.topk_instrument)
    th = mir.binary_threshold

    def b(name: str) -> bool:
        return bool(float(heads[name][POS]) >= th)

    return AudioMIR(
        clip_id=clip_id,
        approachability=float(np.asarray(heads["approachability"]).reshape(-1)[0]),
        engagement=float(np.asarray(heads["engagement"]).reshape(-1)[0]),
        danceability=float(heads["danceability"][POS]),
        is_aggressive=b("mood_aggressive"),
        is_happy=b("mood_happy"),
        is_party=b("mood_party"),
        is_relaxed=b("mood_relaxed"),
        is_sad=b("mood_sad"),
        is_acoustic=b("mood_acoustic"),
        is_electronic=b("mood_electronic"),
        is_instrumental=b("voice_instrumental"),
        is_female_voice=b("gender"),
        is_bright_timbre=b("timbre"),
        is_tonal=b("tonal_atonal"),
        genre_labels=g_lab,
        genre_scores=g_sc,
        moodtheme_labels=m_lab,
        moodtheme_scores=m_sc,
        instrument_labels=i_lab,
        instrument_scores=i_sc,
        audio_duration_s=float(len(audio)) / float(mir.inference_sample_rate),
        inference_time_ms=(perf_counter() - t0) * 1000.0,
        is_mir_extracted=True,
    )


def _eligible_clip_ids(session: Session) -> list[int]:
    rows = (
        session.query(Clip.id)
        .outerjoin(AudioMIR, AudioMIR.clip_id == Clip.id)
        .filter(*clip_used_in_analysis(), AudioMIR.is_mir_extracted.is_(None))
        .order_by(Clip.id)
        .all()
    )
    return [r[0] for r in rows]


def _upsert(session: Session, row: AudioMIR) -> None:
    existing = session.get(AudioMIR, row.clip_id)
    if existing is None:
        session.add(row)
        return
    for col in _RESET_COLUMNS:
        setattr(existing, col, getattr(row, col))


def run_mir(settings: Settings, secrets=None) -> None:
    """Per-clip MIR descriptors via MAEST + EffNet-Discogs."""
    mir = settings.mir
    paths = settings.paths
    session = get_session()
    try:
        validate_checkpoint_sidecars(mir)

        def _current() -> fp.Fingerprint:
            return fp.Fingerprint(
                data=fp.hash_text(""),
                config=fp.hash_text(mir_config_payload(mir)),
                dependency=fp.stage_dependency_hash(
                    session, AUDIO_EXTRACT_MIR_STAGE, AUDIO_EXTRACT_MIR_SCOPE
                ),
            )

        current = _current()
        fp.gate(
            session,
            STAGE_MIR,
            SCOPE_MIR,
            current,
            reset_audio_mir,
            log_scope=_LOG,
            drift_msg="resetting MIR outputs",
            check_dependency=True,
        )

        eligible = _eligible_clip_ids(session)
        if not eligible:
            log(_LOG, "SCAN", "clips", "none")
            fp.mark_complete(session, STAGE_MIR, SCOPE_MIR, current)
            session.commit()
            return

        ensure_checkpoints(mir)
        # Recompute now that sidecars are guaranteed fresh post-download.
        current = _current()

        labels_genre = load_labels(_LABELS_DIR / "genre_discogs519.json")
        labels_moodtheme = load_labels(_LABELS_DIR / "mtg_jamendo_moodtheme.json")
        labels_instrument = load_labels(_LABELS_DIR / "mtg_jamendo_instrument.json")

        log(_LOG_LOAD, "GET", "maest+effnet", "ok", stats={"clips": len(eligible)})
        t_stage = time.perf_counter()
        with build_maest(mir) as maest, build_effnet(mir) as effnet:
            queue: Queue = Queue(maxsize=mir.prefetch_queue_size)

            def prefetch() -> None:
                try:
                    for cid in eligible:
                        path = paths.audio_mir_for(cid)
                        if not path.exists():
                            queue.put((cid, None, "no_audio_file"))
                            continue
                        try:
                            audio = _load_audio(str(path), mir.inference_sample_rate)
                        except Exception:  # per-clip safety net
                            queue.put((cid, None, "audio_load"))
                            continue
                        queue.put((cid, audio, None))
                finally:
                    queue.put(_SENTINEL)

            t = Thread(target=prefetch, daemon=True)
            t.start()

            with progress(len(eligible), "MIR inference") as advance:
                processed = 0
                while True:
                    item = queue.get()
                    if item is _SENTINEL:
                        break
                    cid, audio, prefetch_err = item
                    if prefetch_err is not None:
                        row = _terminal_failure(cid, prefetch_err)
                        log(
                            _LOG_INFER,
                            "PUT",
                            f"clip_{cid}",
                            "ERR",
                            stats={"err": prefetch_err},
                        )
                    else:
                        row = _infer_with_error_attribution(
                            cid,
                            audio,
                            maest,
                            effnet,
                            mir,
                            labels_genre,
                            labels_moodtheme,
                            labels_instrument,
                        )
                        if row.is_mir_extracted:
                            log(
                                _LOG_INFER,
                                "PUT",
                                f"clip_{cid}",
                                "ok",
                                stats={"time": row.inference_time_ms / 1000.0},
                            )
                        else:
                            log(
                                _LOG_INFER,
                                "PUT",
                                f"clip_{cid}",
                                "ERR",
                                stats={"err": row.mir_error},
                            )
                    _upsert(session, row)
                    processed += 1
                    advance(detail=f"clip_{cid}")
                    if processed % mir.commit_every == 0:
                        session.commit()
            t.join(timeout=1.0)
            if t.is_alive():
                log(_LOG, "GET", "prefetch", "timeout")

        session.commit()
        fp.mark_complete(session, STAGE_MIR, SCOPE_MIR, current)
        session.commit()
        log(
            _LOG,
            "SEAL",
            "mir",
            "ok",
            stats={"done": len(eligible), "time": time.perf_counter() - t_stage},
        )
    finally:
        session.close()
