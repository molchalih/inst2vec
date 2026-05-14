import subprocess
from collections import defaultdict

import numpy as np

from modules.console import log, progress
from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    User,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.eligibility import is_eligible
from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def verbalize_music(music) -> str:
    descriptors = []

    if music.energy is not None:
        if music.energy >= 0.80:
            descriptors.append("very high energy")
        elif music.energy >= 0.55:
            descriptors.append("high energy")
        elif music.energy >= 0.30:
            descriptors.append("moderate energy")
        else:
            descriptors.append("low energy")

    if music.valence is not None:
        if music.valence >= 0.75:
            descriptors.append("very upbeat")
        elif music.valence >= 0.50:
            descriptors.append("positive")
        elif music.valence >= 0.25:
            descriptors.append("bittersweet")
        else:
            descriptors.append("dark and melancholic")

    if music.acousticness is not None:
        if music.acousticness >= 0.75:
            descriptors.append("acoustic")
        elif music.acousticness <= 0.20:
            descriptors.append("electronic")

    if music.instrumentalness is not None:
        if music.instrumentalness >= 0.50:
            descriptors.append("instrumental")
        else:
            descriptors.append("vocal")

    if music.danceability is not None:
        if music.danceability >= 0.75:
            descriptors.append("highly danceable")
        elif music.danceability <= 0.25:
            descriptors.append("not danceable")

    if music.speechiness is not None and music.speechiness >= 0.33:
        if music.speechiness >= 0.66:
            descriptors.append("spoken word")
        else:
            descriptors.append("rap or speech-heavy")

    if music.tempo is not None:
        bpm = round(music.tempo)
        if music.tempo >= 150:
            descriptors.append(f"fast ({bpm} BPM)")
        elif music.tempo >= 110:
            descriptors.append(f"moderate tempo ({bpm} BPM)")
        elif music.tempo >= 75:
            descriptors.append(f"slow ({bpm} BPM)")
        else:
            descriptors.append(f"very slow ({bpm} BPM)")

    if music.mode is not None and music.key is not None and 0 <= int(music.key) <= 11:
        mode_str = "major" if music.mode == 1 else "minor"
        key_str = _KEY_NAMES[int(music.key)]
        descriptors.append(f"{key_str} {mode_str}")

    desc = ", ".join(descriptors)
    track = music.track or "Unknown Track"
    artist = music.artist or "Unknown Artist"
    return f'Music: "{track}" by {artist} — {desc}'


def _to_bytes(tensor):
    return tensor.cpu().float().numpy().tobytes()


def _bytes_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _aggregate_user_embeddings(rows: list[tuple[bytes, int]]) -> dict[int, bytes]:
    """Mean-pool clip embedding blobs by user. Returns {user_id: mean_blob}."""
    user_arrays: dict[int, list[np.ndarray]] = defaultdict(list)
    for blob, user_id in rows:
        user_arrays[user_id].append(_bytes_to_array(blob))
    return {
        user_id: np.stack(arrays).mean(axis=0).astype(np.float32).tobytes()
        for user_id, arrays in user_arrays.items()
    }


def _eligible_clips(session, exclude_disqualified_users: bool):
    clips_q = session.query(Clip).filter(is_eligible(Clip.eligibility))
    if exclude_disqualified_users:
        clips_q = clips_q.join(User, Clip.user_id == User.id).filter(
            is_eligible(User.eligibility),
        )
    return clips_q.all()


def _video_path(clip_id: int, video_dir: str) -> str:
    import os

    return os.path.abspath(os.path.join(video_dir, f"{clip_id}.mp4"))


def _build_text(clip, music_map: dict) -> str | None:
    parts = []

    cap = (
        clip.caption_translation
        if clip.caption_language not in ("en", None)
        and clip.caption_translation
        and clip.caption_translation.strip()
        else (clip.caption_text or "")
    )
    if cap.strip():
        parts.append(cap.strip())

    speech = (
        clip.speech_translation
        if clip.speech_language not in ("en", None)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )
    if speech.strip():
        parts.append(speech.strip())

    if clip.music_id is not None and clip.music_id in music_map:
        parts.append(verbalize_music(music_map[clip.music_id]))

    return " | ".join(parts) if parts else None


def _build_audio_text(clip, music_map: dict) -> str | None:
    # Order: speech first, music second — matches the audio embedding instruction priority.
    parts = []

    speech = (
        clip.speech_translation
        if clip.speech_language not in ("en", None)
        and clip.speech_translation
        and clip.speech_translation.strip()
        else (clip.speech_transcription or "")
    )
    if speech.strip():
        parts.append(speech.strip())

    if clip.music_id is not None and clip.music_id in music_map:
        parts.append(verbalize_music(music_map[clip.music_id]))

    return " | ".join(parts) if parts else None


def _probe_duration_seconds(path: str) -> float | None:
    """Return video duration in seconds via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    try:
        duration = float(raw)
    except ValueError:
        return None
    return duration if duration > 0 else None


def _adaptive_video_sampling(
    path: str, adaptive_max_frames: int, adaptive_default_fps: float
) -> tuple[float, int, float | None]:
    """Choose fps/max_frames from clip duration."""
    duration = _probe_duration_seconds(path)
    if duration is None:
        return adaptive_default_fps, adaptive_max_frames, None
    if duration < 15:
        return 3.0, adaptive_max_frames, duration
    if duration <= 45:
        return 2.0, adaptive_max_frames, duration
    return 1.0, adaptive_max_frames, duration


def _frame_retry_schedule(initial_max_frames: int) -> list[int]:
    """Return descending retry caps for frame count."""
    caps = [initial_max_frames, 64, 48, 32, 24, 16]
    unique = []
    seen = set()
    for c in caps:
        if c <= initial_max_frames and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _is_token_mismatch_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "Mismatch in `video` token count" in msg
        or "Likely due to `truncation='max_length'" in msg
    )


def embed_video_clips(
    model_path: str,
    video_dir: str,
    embed_max_length: int,
    adaptive_max_frames: int,
    adaptive_default_fps: float,
    exclude_disqualified_users: bool,
) -> None:
    import os

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        done_video = {
            r.clip_id
            for r in session.query(ClipEmbedding.clip_id)
            .filter(ClipEmbedding.embedding_case == "video")
            .all()
        }

        clips = _eligible_clips(session, exclude_disqualified_users)
        todo = []
        for clip in clips:
            if clip.id in done_video:
                continue
            path = _video_path(clip.id, video_dir)
            if not os.path.exists(path):
                continue
            todo.append(clip)

        if not todo:
            log("embed:video", "nothing to do")
            return

        log(
            "embed:video",
            f"{len(todo)} clips to embed ({len(done_video)} already done)",
        )
        from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=embed_max_length,
            max_frames=adaptive_max_frames,
            fps=adaptive_default_fps,
        )

        with progress(len(todo), "Embedding video") as advance:
            for _, clip in enumerate(todo, 1):
                path = _video_path(clip.id, video_dir)
                fps, max_frames, _duration = _adaptive_video_sampling(
                    path, adaptive_max_frames, adaptive_default_fps
                )
                frame_caps = _frame_retry_schedule(max_frames)
                embeddings = None
                for attempt_idx, frame_cap in enumerate(frame_caps):
                    try:
                        embeddings = model.process(
                            [{"video": path, "fps": fps, "max_frames": frame_cap}]
                        )
                        break
                    except Exception as e:
                        if (
                            _is_token_mismatch_error(e)
                            and attempt_idx < len(frame_caps) - 1
                        ):
                            continue
                        break
                if embeddings is None:
                    advance(detail=f"✗ {clip.id}")
                    continue

                video_row = ClipEmbedding(
                    clip_id=clip.id,
                    embedding_case="video",
                    embedding=_to_bytes(embeddings[0]),
                )
                session.merge(video_row)
                session.commit()
                advance(detail=f"✓ {clip.id}")

        log("embed:video", "done", level="ok")
    finally:
        session.close()


def embed_sandwich_clips(
    model_path: str,
    video_dir: str,
    embed_max_length: int,
    adaptive_max_frames: int,
    adaptive_default_fps: float,
    exclude_disqualified_users: bool,
) -> None:
    import os

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        done_sandwich = {
            r.clip_id
            for r in session.query(ClipEmbedding.clip_id)
            .filter(ClipEmbedding.embedding_case == "sandwich")
            .all()
        }

        music_map = {m.id: m for m in session.query(Music).all()}

        clips = _eligible_clips(session, exclude_disqualified_users)
        todo = []
        for clip in clips:
            if clip.id in done_sandwich:
                continue
            path = _video_path(clip.id, video_dir)
            if not os.path.exists(path):
                continue
            text = _build_text(clip, music_map)
            if text is None:
                continue
            todo.append((clip, text))

        if not todo:
            log("embed:sandwich", "nothing to do")
            return

        log(
            "embed:sandwich",
            f"{len(todo)} clips to embed ({len(done_sandwich)} already done)",
        )
        from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=embed_max_length,
            max_frames=adaptive_max_frames,
            fps=adaptive_default_fps,
        )

        with progress(len(todo), "Embedding sandwich") as advance:
            for _, (clip, text) in enumerate(todo, 1):
                path = _video_path(clip.id, video_dir)
                fps, max_frames, _duration = _adaptive_video_sampling(
                    path, adaptive_max_frames, adaptive_default_fps
                )
                frame_caps = _frame_retry_schedule(max_frames)
                embedding = None
                for attempt_idx, frame_cap in enumerate(frame_caps):
                    try:
                        embeddings = model.process(
                            [
                                {
                                    "video": path,
                                    "fps": fps,
                                    "max_frames": frame_cap,
                                    "text": text,
                                }
                            ]
                        )
                        embedding = embeddings[0]
                        break
                    except Exception as e:
                        if (
                            _is_token_mismatch_error(e)
                            and attempt_idx < len(frame_caps) - 1
                        ):
                            continue
                        break

                if embedding is None:
                    advance(detail=f"✗ {clip.id}")
                    continue

                sandwich_row = ClipEmbedding(
                    clip_id=clip.id,
                    embedding_case="sandwich",
                    embedding=_to_bytes(embedding),
                )
                session.merge(sandwich_row)
                session.commit()
                advance(detail=f"✓ {clip.id}")

        log("embed:sandwich", "done", level="ok")
    finally:
        session.close()


AUDIO_INSTRUCTION = (
    "Represent the audio character of this video: its musical mood, energy, "
    "and any spoken content."
)


def embed_audio_clips(
    model_path: str,
    video_dir: str,
    embed_max_length: int,
    adaptive_max_frames: int,
    adaptive_default_fps: float,
    exclude_disqualified_users: bool,
) -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        done_audio = {
            r.clip_id
            for r in session.query(ClipEmbedding.clip_id)
            .filter(ClipEmbedding.embedding_case == "audio")
            .all()
        }

        music_map = {m.id: m for m in session.query(Music).all()}

        clips = _eligible_clips(session, exclude_disqualified_users)
        todo = []
        for clip in clips:
            if clip.id in done_audio:
                continue
            text = _build_audio_text(clip, music_map)
            if text is None:
                continue
            todo.append((clip, text))

        if not todo:
            log("embed:audio", "nothing to do")
            return

        log(
            "embed:audio",
            f"{len(todo)} clips to embed ({len(done_audio)} already done)",
        )
        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            max_length=embed_max_length,
        )

        with progress(len(todo), "Embedding audio") as advance:
            for _, (clip, text) in enumerate(todo, 1):
                try:
                    embeddings = model.process(
                        [{"text": text, "instruction": AUDIO_INSTRUCTION}]
                    )
                    embedding = embeddings[0]
                except Exception:
                    advance(detail=f"✗ {clip.id}")
                    continue

                audio_row = ClipEmbedding(
                    clip_id=clip.id,
                    embedding_case="audio",
                    embedding=_to_bytes(embedding),
                )
                session.merge(audio_row)
                session.commit()
                advance(detail=f"✓ {clip.id}")

        log("embed:audio", "done", level="ok")
    finally:
        session.close()


def embed_user_clips(cases: list[str] | None = None):
    if cases is None:
        cases = ["video", "sandwich", "audio"]
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for case in cases:
            rows = (
                session.query(ClipEmbedding.embedding, Clip.user_id)
                .join(Clip, ClipEmbedding.clip_id == Clip.id)
                .filter(ClipEmbedding.embedding_case == case)
                .all()
            )

            if not rows:
                log(f"embed:user:{case}", "nothing to do")
                continue

            aggregated = _aggregate_user_embeddings(rows)
            log(f"embed:user:{case}", f"{len(aggregated)} users to embed")

            for user_id, mean_blob in aggregated.items():
                row = UserEmbedding(
                    user_id=user_id,
                    embedding_case=case,
                    embedding=mean_blob,
                )
                session.merge(row)
                session.commit()

            log(f"embed:user:{case}", "done", level="ok")
    finally:
        session.close()
