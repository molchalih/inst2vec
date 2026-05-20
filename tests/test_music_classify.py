"""Behavior tests for classify_music."""

from unittest.mock import MagicMock

import pytest

from core.config import MusicSettings, PathsSettings
from core.database import (
    Base,
    Clip,
    Music,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.music.classify import AcrSecrets, classify_music


def _music_settings(**overrides) -> MusicSettings:
    base = dict(
        audio_fingerprint_confidence=0.5,
        commit_every=50,
        http_timeout=20.0,
        spotify_search_limit=5,
        spotify_token_skew_seconds=30,
        spotify_request_timeout=8.0,
        reccobeats_batch_size=20,
        reccobeats_delay_min=0.0,
        reccobeats_delay_max=0.0,
        manual_features_max_seconds=20,
        manual_features_sample_rate=44100,
        manual_features_max_mb=5.0,
        manual_features_mp3_bitrate="128k",
        api_max_attempts=3,
        api_retry_delay=0.0,
        api_retry_jitter=0.0,
        acr_max_attempts=2,
        ffmpeg_timeout_seconds=60,
        reccobeats_upstream_fail_threshold=3,
    )
    base.update(overrides)
    return MusicSettings(**base)


def _paths(video_dir: str) -> PathsSettings:
    return PathsSettings(
        video_dir=video_dir,
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        speech_audio_dir="/tmp/audio",
        audio_dir="/tmp/audio",
        data_csv_path="/tmp/data.csv",
    )


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, Clip, Music, User):
        session.query(model).delete()
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        for model in (StageState, Clip, Music, User):
            session.query(model).delete()
        session.commit()
        session.close()


@pytest.fixture
def seeded_db(db_session, tmp_path):
    db_session.add(User(id=1, parse_status="success", is_selected=True))
    db_session.add(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            is_music_recognized=None,
        )
    )
    db_session.commit()
    video_file = tmp_path / "10.mp4"
    video_file.write_bytes(b"fake")
    yield db_session, tmp_path


def test_classify_match_sets_is_music_recognized_true(seeded_db, monkeypatch):
    s, tmp_path = seeded_db

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[{'
        '"title":"Song","artists":[{"name":"Artist"}],"score":90}]}}'
    )
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is True
    assert clip.music_id is not None
    music = s.query(Music).filter_by(id=clip.music_id).one()
    assert music.artist == "Artist"
    assert music.track == "Song"


def test_classify_clean_no_match_sets_false(seeded_db, monkeypatch):
    s, tmp_path = seeded_db

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[]}}'
    )
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is False
    assert clip.music_id is None


def test_classify_transient_exhaustion_leaves_row_retryable(seeded_db, monkeypatch):
    """TransientError must NOT terminal-mark — leave is_music_recognized=None
    so the predicate picks the row up on the next run."""
    s, tmp_path = seeded_db

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.side_effect = RuntimeError("network")
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(acr_max_attempts=2),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is None, (
        "transient errors must not terminal-mark; expected None, "
        f"got {clip.is_music_recognized!r}"
    )
    assert fake_acr.recognize_by_file.call_count == 2


def test_acr_transient_leaves_row_retryable(seeded_db, monkeypatch):
    """ACR TransientError must NOT terminal-mark the clip. The row must
    stay at is_music_recognized=None so the next run reprocesses it."""
    from modules.music.clients import TransientError

    s, tmp_path = seeded_db

    def raise_transient(*a, **kw):
        raise TransientError("acr down")

    monkeypatch.setattr("modules.music.classify._fingerprint", raise_transient)
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: MagicMock(),
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.is_music_recognized is None, (
        "transient errors must not terminal-mark; expected None, "
        f"got {clip.is_music_recognized!r}"
    )


def test_classify_existing_music_row_reused(seeded_db, monkeypatch):
    s, tmp_path = seeded_db
    s.add(Music(id=42, artist="Artist", track="Song"))
    s.commit()

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.return_value = (
        '{"status":{"code":0},"metadata":{"music":[{'
        '"title":"Song","artists":[{"name":"Artist"}],"score":90}]}}'
    )
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )

    clip = s.query(Clip).filter_by(id=10).one()
    assert clip.music_id == 42


def test_classify_skips_non_null_clips(seeded_db, monkeypatch):
    s, tmp_path = seeded_db
    clip = s.query(Clip).filter_by(id=10).one()
    clip.is_music_recognized = True
    s.commit()

    fake_acr = MagicMock()
    fake_acr.recognize_by_file.side_effect = AssertionError("should not be called")
    monkeypatch.setattr(
        "modules.music.classify.ACRCloudRecognizer",
        lambda cfg: fake_acr,
    )

    classify_music(
        music=_music_settings(),
        paths=_paths(str(tmp_path)),
        secrets=AcrSecrets(host="h", access_key="k", access_secret="s"),
    )


def test_reset_music_classify_nulls_clip_links_and_deletes_orphan_music(db_session):
    from core.database import Clip, Music, User
    from modules.music.state import reset_music_classify

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(Music(id=1, artist="a", track="t"))
    db_session.merge(Music(id=2, artist="x", track="y"))  # orphan after reset
    db_session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            music_id=1,
            music_confidence=0.9,
            is_music_recognized=True,
        )
    )
    db_session.commit()

    reset_music_classify(db_session)

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.music_id is None
    assert clip.music_confidence is None
    assert clip.is_music_recognized is None
    assert db_session.query(Music).all() == []


def test_reset_music_classify_also_clears_ineligible_clips(db_session):
    """Stale classify outputs on a currently-ineligible clip must be cleared
    too, otherwise re-selection in a later run skips re-fingerprinting."""
    from core.database import Clip, Music, User
    from modules.music.state import reset_music_classify

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(Music(id=1, artist="a", track="t"))
    db_session.merge(
        Clip(
            id=20,
            user_id=1,
            is_selected=False,
            is_downloaded=True,
            music_id=1,
            music_confidence=0.8,
            is_music_recognized=True,
        )
    )
    db_session.commit()

    reset_music_classify(db_session)

    clip = db_session.query(Clip).filter_by(id=20).one()
    assert clip.music_id is None
    assert clip.music_confidence is None
    assert clip.is_music_recognized is None
    assert db_session.query(Music).all() == []


def test_classify_config_payload_ignores_features_only_knobs():
    """Changing a features-only knob must not invalidate classify
    fingerprints — and vice versa."""
    from modules.music.state import classify_config_payload

    base = _music_settings()
    bumped_features = base.model_copy(update={"reccobeats_batch_size": 99})
    assert classify_config_payload(base) == classify_config_payload(bumped_features)

    bumped_classify = base.model_copy(update={"audio_fingerprint_confidence": 0.95})
    assert classify_config_payload(base) != classify_config_payload(bumped_classify)


def test_classify_music_features_only_knob_change_does_not_reset(
    monkeypatch, db_session
):
    """Bumping a features-only setting must NOT trigger a music-classify reset."""
    import modules.music.classify as classify_mod
    from core.database import StageState
    from modules.music.classify import classify_music
    from modules.music.state import SCOPE_MUSIC, STAGE_MUSIC_CLASSIFY

    class _NoOpAcr:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(classify_mod, "ACRCloudRecognizer", _NoOpAcr)

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(Music(id=1, artist="a", track="t"))
    db_session.merge(
        Clip(
            id=30,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            music_id=1,
            music_confidence=0.9,
            is_music_recognized=True,
        )
    )
    db_session.commit()

    paths = _paths("/tmp")
    secrets = AcrSecrets(host="h", access_key="k", access_secret="s")

    classify_music(_music_settings(), paths, secrets)
    sealed = db_session.get(StageState, (STAGE_MUSIC_CLASSIFY, SCOPE_MUSIC))
    assert sealed is not None
    config_hash_before = sealed.config_hash

    classify_music(_music_settings(reccobeats_batch_size=99), paths, secrets)

    # No reset: row preserved.
    clip = db_session.query(Clip).filter_by(id=30).one()
    assert clip.music_id == 1
    assert clip.is_music_recognized is True
    # Fingerprint unchanged: scope is stage-specific.
    db_session.expire_all()
    sealed = db_session.get(StageState, (STAGE_MUSIC_CLASSIFY, SCOPE_MUSIC))
    assert sealed.config_hash == config_hash_before


def test_classify_music_config_change_triggers_reset(monkeypatch, db_session):
    """A config change in MusicSettings must reset clip → music links so the
    next classify run re-fingerprints."""
    import modules.music.classify as classify_mod
    from core.config import MusicSettings, PathsSettings
    from core.database import Clip, Music, StageState, User
    from modules.music.classify import AcrSecrets, classify_music
    from modules.music.state import SCOPE_MUSIC, STAGE_MUSIC_CLASSIFY

    class _NoOpAcr:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(classify_mod, "ACRCloudRecognizer", _NoOpAcr)

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(Music(id=1, artist="a", track="t"))
    db_session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            music_id=1,
            music_confidence=0.9,
            is_music_recognized=True,
        )
    )
    db_session.commit()

    base_music = MusicSettings(
        audio_fingerprint_confidence=0.7,
        commit_every=1,
        http_timeout=10.0,
        spotify_search_limit=5,
        spotify_token_skew_seconds=60,
        spotify_request_timeout=10.0,
        reccobeats_batch_size=10,
        reccobeats_delay_min=0.1,
        reccobeats_delay_max=0.2,
        manual_features_max_seconds=20,
        manual_features_sample_rate=22050,
        manual_features_max_mb=4.0,
        manual_features_mp3_bitrate="64k",
        api_max_attempts=2,
        api_retry_delay=0.0,
        api_retry_jitter=0.0,
        acr_max_attempts=1,
        ffmpeg_timeout_seconds=5,
        reccobeats_upstream_fail_threshold=3,
    )
    paths = PathsSettings(
        video_dir="/tmp",
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        speech_audio_dir="/tmp",
        audio_dir="/tmp",
        data_csv_path="/tmp/x.csv",
    )
    secrets = AcrSecrets(host="h", access_key="k", access_secret="s")

    classify_music(base_music, paths, secrets)
    assert db_session.get(StageState, (STAGE_MUSIC_CLASSIFY, SCOPE_MUSIC)) is not None

    bumped = base_music.model_copy(update={"audio_fingerprint_confidence": 0.95})
    classify_music(bumped, paths, secrets)

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.music_id is None, "config drift must NULL music_id"
    assert clip.is_music_recognized is None


def test_classify_music_unchanged_config_does_not_reset(monkeypatch, db_session):
    import modules.music.classify as classify_mod
    from core.config import MusicSettings, PathsSettings
    from core.database import Clip, Music, User
    from modules.music.classify import AcrSecrets, classify_music

    class _NoOpAcr:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(classify_mod, "ACRCloudRecognizer", _NoOpAcr)

    db_session.merge(User(id=1, is_selected=True, is_eligible=True))
    db_session.merge(Music(id=1, artist="a", track="t"))
    db_session.merge(
        Clip(
            id=10,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            music_id=1,
            music_confidence=0.9,
            is_music_recognized=True,
        )
    )
    db_session.commit()

    music = MusicSettings(
        audio_fingerprint_confidence=0.7,
        commit_every=1,
        http_timeout=10.0,
        spotify_search_limit=5,
        spotify_token_skew_seconds=60,
        spotify_request_timeout=10.0,
        reccobeats_batch_size=10,
        reccobeats_delay_min=0.1,
        reccobeats_delay_max=0.2,
        manual_features_max_seconds=20,
        manual_features_sample_rate=22050,
        manual_features_max_mb=4.0,
        manual_features_mp3_bitrate="64k",
        api_max_attempts=2,
        api_retry_delay=0.0,
        api_retry_jitter=0.0,
        acr_max_attempts=1,
        ffmpeg_timeout_seconds=5,
        reccobeats_upstream_fail_threshold=3,
    )
    paths = PathsSettings(
        video_dir="/tmp",
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        speech_audio_dir="/tmp",
        audio_dir="/tmp",
        data_csv_path="/tmp/x.csv",
    )
    secrets = AcrSecrets(host="h", access_key="k", access_secret="s")

    classify_music(
        music, paths, secrets
    )  # first run — no prior state, must not reset existing seeded data
    classify_music(music, paths, secrets)  # second run — fingerprint match, no reset

    clip = db_session.query(Clip).filter_by(id=10).one()
    assert clip.music_id == 1, "unchanged config must not reset clip links"


def test_classify_config_payload_ignores_retry_knobs():
    """Retry attempts/delay/jitter influence reliability, not the
    classification outcome — must not invalidate the seal."""
    from core.config import MusicSettings
    from modules.music.state import classify_config_payload

    base = MusicSettings(
        audio_fingerprint_confidence=0.5,
        commit_every=50,
        http_timeout=20.0,
        spotify_search_limit=5,
        spotify_token_skew_seconds=30,
        spotify_request_timeout=8.0,
        reccobeats_batch_size=20,
        reccobeats_delay_min=0.0,
        reccobeats_delay_max=0.0,
        manual_features_max_seconds=20,
        manual_features_sample_rate=44100,
        manual_features_max_mb=5.0,
        manual_features_mp3_bitrate="128k",
        api_max_attempts=3,
        api_retry_delay=0.0,
        api_retry_jitter=0.0,
        acr_max_attempts=2,
        ffmpeg_timeout_seconds=60,
        reccobeats_upstream_fail_threshold=3,
    )
    for upd in (
        {"acr_max_attempts": base.acr_max_attempts + 3},
        {"api_retry_delay": base.api_retry_delay + 1.0},
        {"api_retry_jitter": base.api_retry_jitter + 1.0},
    ):
        bumped = base.model_copy(update=upd)
        assert classify_config_payload(base) == classify_config_payload(bumped), upd
