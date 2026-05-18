from types import SimpleNamespace

import numpy as np

from core import fingerprint as fp
from core.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    User,
    get_engine,
    get_session,
)
from modules.embeddings.cases import CASE_REGISTRY
from modules.embeddings.state import (
    dependency_rows_for_case,
    get_clip_embedding_rows_for_user_aggregation,
    get_embedded_source_hashes,
    per_clip_source_hashes_and_aggregate,
)


def _paths_stub(tmp_path):
    video_dir = tmp_path / "videos"
    audio_dir = tmp_path / "audios"
    return SimpleNamespace(
        video_dir=str(video_dir),
        audio_dir=str(audio_dir),
        video_for=lambda cid, vd=video_dir: vd / f"{cid}.mp4",
        audio_for=lambda cid, ad=audio_dir: ad / f"{cid}.mp3",
    )


def _settings_stub(tmp_path):
    return SimpleNamespace(paths=_paths_stub(tmp_path))


def _make_blob(values: list[float]) -> bytes:
    """Helper to create embedding blob from float array."""
    return np.array(values, dtype=np.float32).tobytes()


def test_aggregation_excludes_orphan_rows():
    """Verify that clips deselected after embedding don't contaminate user means."""
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=False, is_downloaded=True))
    session.merge(
        ClipEmbedding(
            clip_id=10, embedding_case="video", embedding=_make_blob([1.0, 2.0, 3.0])
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=11, embedding_case="video", embedding=_make_blob([4.0, 5.0, 6.0])
        )
    )
    session.commit()

    rows = get_clip_embedding_rows_for_user_aggregation(
        session, "video", exclude_disqualified_users=False
    )
    user_ids_seen = {user_id for _, _, user_id in rows}
    clip_ids_seen = {
        ce.clip_id
        for ce in session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert user_ids_seen == {1}, "user 1 should contribute"
    assert clip_ids_seen == {10, 11}, "both embedding rows should still exist on disk"
    assert len(rows) == 1, "only clip 10 should be included in aggregation"
    assert rows[0][0] == 10, "row must carry clip_id for fingerprint use"
    session.close()


def test_get_embedded_source_hashes_returns_clip_id_to_hash_map():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(
        ClipEmbedding(
            clip_id=10,
            embedding_case="video",
            embedding=b"\x00" * 4,
            source_hash="abc",
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=11,
            embedding_case="video",
            embedding=b"\x00" * 4,
            source_hash=None,
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=10,
            embedding_case="audio",
            embedding=b"\x00" * 4,
            source_hash="zzz",
        )
    )
    session.commit()

    out = get_embedded_source_hashes(session, "video")
    assert out == {10: "abc", 11: None}
    session.close()


def test_per_clip_source_hashes_match_dependency_rows(tmp_path):
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
    session.commit()

    settings = _settings_stub(tmp_path)
    per_clip, aggregate = per_clip_source_hashes_and_aggregate(
        session, "video", [10, 11], settings=settings
    )

    # Per-clip hash must equal hash_rows of the per-clip dependency rows.
    clips = session.query(Clip).filter(Clip.id.in_([10, 11])).order_by(Clip.id).all()
    expected_per_clip = {}
    expected_all: list[tuple] = []
    for clip in clips:
        rows = dependency_rows_for_case("video", clip, paths=settings.paths)
        expected_per_clip[clip.id] = fp.hash_rows(rows)
        expected_all.extend(rows)
    assert per_clip == expected_per_clip
    # Aggregate must equal hash_rows over the full ordered row list.
    assert aggregate == fp.hash_rows(expected_all)
    session.close()


def test_per_clip_source_hashes_with_no_candidates(tmp_path):
    Base.metadata.create_all(get_engine())
    session = get_session()
    per_clip, aggregate = per_clip_source_hashes_and_aggregate(
        session, "video", [], settings=_settings_stub(tmp_path)
    )
    assert per_clip == {}
    assert aggregate == fp.hash_rows([])
    session.close()


def test_dependency_rows_uses_case_spec_columns(tmp_path):
    """For every case in CASE_REGISTRY, dependency_rows_for_case returns
    exactly the columns the case spec declares — no more, no less."""
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(
        Clip(
            id=42,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_clean="cap",
            caption_language="en",
            caption_translation="cap-en",
            speech_transcription="hi",
            speech_language="en",
            speech_translation="hi-en",
            music_id=None,
        )
    )
    session.commit()
    clip = session.query(Clip).filter_by(id=42).one()

    paths = _paths_stub(tmp_path)
    for name, spec in CASE_REGISTRY.items():
        music_map = {} if "_music_row" in spec.dependency_columns else None
        rows = dependency_rows_for_case(name, clip, paths=paths, music_map=music_map)
        keys = [k for k, _ in rows]
        assert keys == list(spec.dependency_columns), (
            f"case {name}: rows={keys} expected={list(spec.dependency_columns)}"
        )
    session.close()


def test_music_row_sentinel_flips_when_features_arrive(tmp_path):
    """An audio/sandwich case sealed before Music features were filled in
    must observe a hash change once Spotify/ReccoBeats backfill the row."""
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()

    session.merge(Music(id=7, artist="Artist", track="Song"))
    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(
        Clip(
            id=42,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            speech_transcription="hi",
            speech_language="en",
            speech_translation=None,
            music_id=7,
        )
    )
    session.commit()
    clip = session.query(Clip).filter_by(id=42).one()
    paths = _paths_stub(tmp_path)

    before = dependency_rows_for_case(
        "audio", clip, paths=paths, music_map={7: session.get(Music, 7)}
    )

    music = session.get(Music, 7)
    music.energy = 0.8
    music.tempo = 128.0
    session.commit()

    after = dependency_rows_for_case(
        "audio", clip, paths=paths, music_map={7: session.get(Music, 7)}
    )
    assert fp.hash_rows(before) != fp.hash_rows(after), (
        "filling Music features for the linked row must change the audio fingerprint"
    )
    session.close()


def test_music_row_sentinel_flips_when_music_match_arrives(tmp_path):
    """A speechless clip with no music_id (skipped in audio) must observe
    a hash change once a music match is recorded."""
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()

    session.merge(Music(id=9, artist="A", track="T", energy=0.5))
    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(
        Clip(
            id=99,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            speech_transcription=None,
            speech_language=None,
            speech_translation=None,
            music_id=None,
        )
    )
    session.commit()
    clip = session.query(Clip).filter_by(id=99).one()
    paths = _paths_stub(tmp_path)
    music_map = {9: session.get(Music, 9)}

    before = dependency_rows_for_case("audio", clip, paths=paths, music_map=music_map)
    clip.music_id = 9
    session.commit()
    after = dependency_rows_for_case("audio", clip, paths=paths, music_map=music_map)
    assert fp.hash_rows(before) != fp.hash_rows(after)
    session.close()


def test_get_stored_user_hashes_returns_user_id_to_hash_map():
    from core.database import UserEmbedding
    from modules.embeddings.state import get_stored_user_hashes

    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (UserEmbedding, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(User(id=2, is_selected=True, is_eligible=True))
    session.merge(
        UserEmbedding(
            user_id=1, embedding_case="video", embedding=b"\x00" * 4, source_hash="h1"
        )
    )
    session.merge(
        UserEmbedding(
            user_id=2, embedding_case="video", embedding=b"\x00" * 4, source_hash=None
        )
    )
    session.merge(
        UserEmbedding(
            user_id=1, embedding_case="audio", embedding=b"\x00" * 4, source_hash="aud"
        )
    )
    session.commit()

    assert get_stored_user_hashes(session, "video") == {1: "h1", 2: None}
    session.close()


def test_per_user_source_hashes_groups_clip_blob_pairs_by_user():
    """Per-user hash digests the same (clip_id, sha256(blob)) pairs the
    user's clips contribute, in the row order returned by the aggregation
    query. This keeps users.py's per-user hash byte-identical with the
    sub-slice of _compute_fingerprint's dependency hash.
    """
    import hashlib

    from core import fingerprint as fp
    from modules.embeddings.state import per_user_source_hashes

    blob1 = _make_blob([1.0])
    blob2 = _make_blob([2.0])
    blob3 = _make_blob([3.0])
    rows = [
        (10, blob1, 1),
        (11, blob2, 1),
        (20, blob3, 2),
    ]

    out = per_user_source_hashes(rows)
    expected_user1 = fp.hash_rows(
        [
            (10, hashlib.sha256(blob1).hexdigest()),
            (11, hashlib.sha256(blob2).hexdigest()),
        ]
    )
    expected_user2 = fp.hash_rows([(20, hashlib.sha256(blob3).hexdigest())])
    assert out == {1: expected_user1, 2: expected_user2}


def test_per_user_source_hashes_empty_rows_returns_empty():
    from modules.embeddings.state import per_user_source_hashes

    assert per_user_source_hashes([]) == {}
