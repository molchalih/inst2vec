import os
from sqlalchemy import or_

from modules.database import Base, engine, get_session, Clip, ClipEmbedding, User
from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

MODEL_PATH = "./models/Qwen3-VL-Embedding-8B"
VIDEO_DIR = "data/source/videos"
EXCLUDE_DISQUALIFIED_USERS = os.environ.get("EMBEDDINGS_EXCLUDE_DISQUALIFIED_USERS", "1") == "1"


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
        model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, max_frames=16, fps=1)

        for i, clip in enumerate(todo, 1):
            path = _video_path(clip.pk)
            print(f"[embed:video] ({i}/{len(todo)}) {clip.pk}", end="", flush=True)
            try:
                embeddings = model.process([{"video": path}])
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
        model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, max_frames=16, fps=1)

        for i, clip in enumerate(todo, 1):
            path = _video_path(clip.pk)
            text = " | ".join(_text_parts(clip))
            print(f"[embed:sandwich] ({i}/{len(todo)}) {clip.pk}", end="", flush=True)
            try:
                embeddings = model.process([{"video": path}, {"text": text}])
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
