import os
import torch
from sqlalchemy import Column, BigInteger, LargeBinary

from modules.database import Base, engine, get_session, Clip
from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

MODEL_PATH = "./models/Qwen3-VL-Embedding-8B"
VIDEO_DIR = "data/source/videos"


class Embedding(Base):
    __tablename__ = "embeddings"
    clip_pk = Column(BigInteger, primary_key=True)
    video = Column(LargeBinary)
    text = Column(LargeBinary)


def _to_bytes(tensor):
    return tensor.cpu().float().numpy().tobytes()


def embed_clips():
    Base.metadata.create_all(engine)
    session = get_session()

    done = {r.clip_pk for r in session.query(Embedding.clip_pk).all()}

    clips = session.query(Clip).all()
    todo = []
    for clip in clips:
        if clip.pk in done:
            continue
        path = os.path.join(VIDEO_DIR, f"{clip.pk}.mp4")
        if not os.path.exists(path):
            continue
        todo.append(clip)

    if not todo:
        print("[embed] nothing to do")
        session.close()
        return

    print(f"[embed] {len(todo)} clips to embed ({len(done)} already done)")
    model = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, max_frames=16, fps=1)

    for i, clip in enumerate(todo):
        path = os.path.abspath(os.path.join(VIDEO_DIR, f"{clip.pk}.mp4"))

        text_parts = []
        if clip.caption_text:
            text_parts.append(clip.caption_text)
        if clip.speech_transcription:
            text_parts.append(clip.speech_transcription)

        inputs = [{"video": path}]
        if text_parts:
            inputs.append({"text": " | ".join(text_parts)})

        print(f"[embed] ({i+1}/{len(todo)}) {clip.pk}", end="", flush=True)
        try:
            embeddings = model.process(inputs)
        except Exception as e:
            print(f" ✗ {e}")
            continue

        row = Embedding(
            clip_pk=clip.pk,
            video=_to_bytes(embeddings[0]),
            text=_to_bytes(embeddings[1]) if len(embeddings) > 1 else None,
        )
        session.merge(row)
        session.commit()
        print(f" ✓")

    session.close()
    print("[embed] done")
