import os
import subprocess
from sqlalchemy import or_

from modules.database import Base, engine, get_session, Clip, ClipEmbedding, User
from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

MODEL_PATH = "./models/Qwen3-VL-Embedding-8B"
VIDEO_DIR = "data/source/videos"
EXCLUDE_DISQUALIFIED_USERS = os.environ.get("EMBEDDINGS_EXCLUDE_DISQUALIFIED_USERS", "1") == "1"
ADAPTIVE_MAX_FRAMES = 96
ADAPTIVE_DEFAULT_FPS = 2.0


def _to_bytes(tensor):
    return tensor.cpu().float().numpy().tobytes()


def _eligible_clips(session):
    clips_q = session.query(Clip).filter(or_(Clip.disqualified.is_(None), Clip.disqualified == 0))
    if EXCLUDE_DISQUALIFIED_USERS:
        clips_q = (
            clips_q.join(User, Clip.user_pk == User.pk)
            .filter(
                or_(User.user_disqualified.is_(None), User.user_disqualified == 0),
            )
        )
    return clips_q.all()


def _video_path(clip_pk: int) -> str:
    return os.path.abspath(os.path.join(VIDEO_DIR, f"{clip_pk}.mp4"))


def _text_parts(clip: Clip) -> list[str]:
    parts = []
    if clip.caption_text:
        parts.append(clip.caption_text)
    if clip.speech_transcription:
        parts.append(clip.speech_transcription)
    return parts


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


def _adaptive_video_sampling(path: str) -> tuple[float, int, float | None]:
    """Choose fps/max_frames from clip duration."""
    duration = _probe_duration_seconds(path)
    if duration is None:
        return ADAPTIVE_DEFAULT_FPS, ADAPTIVE_MAX_FRAMES, None
    if duration < 15:
        return 3.0, ADAPTIVE_MAX_FRAMES, duration
    if duration <= 45:
        return 2.0, ADAPTIVE_MAX_FRAMES, duration
    return 1.0, ADAPTIVE_MAX_FRAMES, duration


def embed_video_clips():
    Base.metadata.create_all(engine)
    session = get_session()
    try:
        done_video = {
            r.clip_pk
            for r in session.query(ClipEmbedding.clip_pk).filter(ClipEmbedding.embedding_case == "video").all()
        }

        clips = _eligible_clips(session)
        todo = []
        for clip in clips:
            if clip.pk in done_video:
                continue
            path = _video_path(clip.pk)
            if not os.path.exists(path):
                continue
            todo.append(clip)

        if not todo:
            print("[embed:video] nothing to do")
            return

        print(f"[embed:video] {len(todo)} clips to embed ({len(done_video)} already done)")
        model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_PATH,
            max_frames=ADAPTIVE_MAX_FRAMES,
            fps=ADAPTIVE_DEFAULT_FPS,
        )

        for i, clip in enumerate(todo, 1):
            path = _video_path(clip.pk)
            fps, max_frames, duration = _adaptive_video_sampling(path)
            dur_str = f"{duration:.1f}s" if duration is not None else "na"
            print(
                f"[embed:video] ({i}/{len(todo)}) {clip.pk} (fps={fps:g}, max={max_frames}, dur={dur_str})",
                end="",
                flush=True,
            )
            try:
                embeddings = model.process([{"video": path, "fps": fps, "max_frames": max_frames}])
            except Exception as e:
                print(f" ✗ {e}")
                continue

            video_row = ClipEmbedding(
                clip_pk=clip.pk,
                embedding_case="video",
                embedding=_to_bytes(embeddings[0]),
            )
            session.merge(video_row)
            session.commit()
            print(" ✓")

        print("[embed:video] done")
    finally:
        session.close()


def embed_sandwich_clips():
    Base.metadata.create_all(engine)
    session = get_session()
    try:
        done_sandwich = {
            r.clip_pk
            for r in session.query(ClipEmbedding.clip_pk).filter(ClipEmbedding.embedding_case == "sandwich").all()
        }

        clips = _eligible_clips(session)
        todo = []
        for clip in clips:
            if clip.pk in done_sandwich:
                continue
            path = _video_path(clip.pk)
            if not os.path.exists(path):
                continue
            if not _text_parts(clip):
                continue
            todo.append(clip)

        if not todo:
            print("[embed:sandwich] nothing to do")
            return

        print(f"[embed:sandwich] {len(todo)} clips to embed ({len(done_sandwich)} already done)")
        model = Qwen3VLEmbedder(
            model_name_or_path=MODEL_PATH,
            max_frames=ADAPTIVE_MAX_FRAMES,
            fps=ADAPTIVE_DEFAULT_FPS,
        )

        for i, clip in enumerate(todo, 1):
            path = _video_path(clip.pk)
            text = " | ".join(_text_parts(clip))
            fps, max_frames, duration = _adaptive_video_sampling(path)
            dur_str = f"{duration:.1f}s" if duration is not None else "na"
            print(
                f"[embed:sandwich] ({i}/{len(todo)}) {clip.pk} (fps={fps:g}, max={max_frames}, dur={dur_str})",
                end="",
                flush=True,
            )
            try:
                embeddings = model.process(
                    [{"video": path, "fps": fps, "max_frames": max_frames}, {"text": text}]
                )
            except Exception as e:
                print(f" ✗ {e}")
                continue
            if len(embeddings) < 2:
                print(" ✗ missing sandwich embedding")
                continue

            sandwich_row = ClipEmbedding(
                clip_pk=clip.pk,
                embedding_case="sandwich",
                embedding=_to_bytes(embeddings[1]),
            )
            session.merge(sandwich_row)
            session.commit()
            print(" ✓")

        print("[embed:sandwich] done")
    finally:
        session.close()


def embed_audio_clips():
    """Placeholder for future audio embedding case."""
    print("[embed:audio] not implemented yet")


def embed_clips():
    """Backwards-compatible entrypoint."""
    embed_video_clips()
