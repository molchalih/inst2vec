"""Behavior tests for the four extract_music_features sub-stages."""

from unittest.mock import MagicMock

import pytest

from core.config import MusicSettings
from core.database import (
    Base,
    Clip,
    Music,
    StageState,
    User,
    get_engine,
    get_session,
)
from modules.music.clients import TransientError
from modules.music.features import (
    _enrich_catalog_features,
    _enrich_upload_fallback,
    _resolve_reccobeats_ids,
    _resolve_spotify_ids,
)
from modules.music.state import _NO_MATCH, music_has_features


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
    )
    base.update(overrides)
    return MusicSettings(**base)


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


def test_spotify_transient_leaves_row_retryable(db_session):
    """SpotifyClient.search_id raising TransientError must NOT write the
    _NO_MATCH sentinel. The row must stay at spotify_id=None so the next
    run retries."""
    s = db_session
    s.add(Music(id=1, artist="a", track="t"))
    s.commit()
    spotify = MagicMock()
    spotify.search_id.side_effect = TransientError("boom")

    _resolve_spotify_ids(s, spotify, _music_settings())
    s.commit()

    row = s.query(Music).filter_by(id=1).one()
    assert row.spotify_id is None, (
        f"transient must not write sentinel; got {row.spotify_id!r}"
    )


def test_resolve_spotify_ids_writes_no_match_for_empty_response(db_session):
    s = db_session
    s.add(Music(id=1, artist="a", track="t"))
    s.commit()
    spotify = MagicMock()
    spotify.search_id.return_value = None

    _resolve_spotify_ids(s, spotify, _music_settings())
    s.commit()

    row = s.query(Music).filter_by(id=1).one()
    assert row.spotify_id == _NO_MATCH


def test_resolve_spotify_ids_writes_real_id(db_session):
    s = db_session
    s.add(Music(id=1, artist="a", track="t"))
    s.commit()
    spotify = MagicMock()
    spotify.search_id.return_value = "spotify-xyz"

    _resolve_spotify_ids(s, spotify, _music_settings())
    s.commit()

    assert s.query(Music).filter_by(id=1).one().spotify_id == "spotify-xyz"


def test_resolve_reccobeats_ids_collapses_missing_to_no_match(db_session):
    s = db_session
    s.add(Music(id=1, artist="a", track="t", spotify_id="sp-1"))
    s.add(Music(id=2, artist="b", track="u", spotify_id="sp-2"))
    s.commit()
    rb = MagicMock()
    rb.get_ids.return_value = {"sp-1": "rb-1"}

    _resolve_reccobeats_ids(s, rb)
    s.commit()

    assert s.query(Music).filter_by(id=1).one().reccobeats_id == "rb-1"
    assert s.query(Music).filter_by(id=2).one().reccobeats_id == _NO_MATCH


def test_enrich_catalog_features_writes_true_on_complete_features(db_session):
    s = db_session
    s.add(
        Music(
            id=1,
            artist="a",
            track="t",
            spotify_id="sp-1",
            reccobeats_id="rb-1",
        )
    )
    s.commit()
    rb = MagicMock()
    rb.get_features.return_value = {
        "rb-1": {
            "acousticness": 0.1,
            "danceability": 0.2,
            "energy": 0.3,
            "instrumentalness": 0.4,
            "key": 5,
            "liveness": 0.6,
            "loudness": -7.0,
            "mode": 1,
            "speechiness": 0.05,
            "tempo": 120.0,
            "valence": 0.5,
        }
    }

    _enrich_catalog_features(s, rb)
    s.commit()

    row = s.query(Music).filter_by(id=1).one()
    assert row.is_audio_features_extracted is True
    assert music_has_features(row)


def test_enrich_catalog_features_leaves_null_when_incomplete(db_session):
    s = db_session
    s.add(
        Music(
            id=1,
            artist="a",
            track="t",
            spotify_id="sp-1",
            reccobeats_id="rb-1",
        )
    )
    s.commit()
    rb = MagicMock()
    rb.get_features.return_value = {"rb-1": {"tempo": 120.0}}

    _enrich_catalog_features(s, rb)
    s.commit()

    row = s.query(Music).filter_by(id=1).one()
    assert row.is_audio_features_extracted is None


def test_upload_fallback_writes_true_on_success(db_session, tmp_path, monkeypatch):
    s = db_session
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Music(
            id=1,
            artist="a",
            track="t",
            spotify_id="sp-1",
            reccobeats_id=_NO_MATCH,
        )
    )
    s.add(
        Clip(
            id=10,
            user_id=1,
            music_id=1,
            is_selected=True,
            is_downloaded=True,
        )
    )
    s.commit()
    (tmp_path / "10.mp4").write_bytes(b"fake")

    fake_audio = tmp_path / "10.wav"
    fake_audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "modules.music.features.extract_audio_sample",
        lambda video, out, music: fake_audio,
    )
    rb = MagicMock()
    rb.upload_features.return_value = {
        "acousticness": 0.1,
        "danceability": 0.2,
        "energy": 0.3,
        "instrumentalness": 0.4,
        "liveness": 0.6,
        "loudness": -7.0,
        "speechiness": 0.05,
        "tempo": 120.0,
        "valence": 0.5,
    }

    _enrich_upload_fallback(s, rb, str(tmp_path), _music_settings())
    s.commit()

    row = s.query(Music).filter_by(id=1).one()
    assert row.is_audio_features_extracted is True
    assert row.tempo == 120.0


def test_upload_fallback_writes_false_when_no_video_on_disk(
    db_session, tmp_path, monkeypatch
):
    s = db_session
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Music(id=1, artist="a", track="t", spotify_id="spot1", reccobeats_id=_NO_MATCH)
    )
    s.add(
        Clip(
            id=10,
            user_id=1,
            music_id=1,
            is_selected=True,
            is_downloaded=True,
        )
    )
    s.commit()
    rb = MagicMock()

    _enrich_upload_fallback(s, rb, str(tmp_path), _music_settings())
    s.commit()

    assert s.query(Music).filter_by(id=1).one().is_audio_features_extracted is False


def test_upload_fallback_writes_false_on_transient_error(
    db_session, tmp_path, monkeypatch
):
    s = db_session
    s.add(User(id=1, parse_status="success", is_selected=True))
    s.add(
        Music(id=1, artist="a", track="t", spotify_id="spot1", reccobeats_id=_NO_MATCH)
    )
    s.add(
        Clip(
            id=10,
            user_id=1,
            music_id=1,
            is_selected=True,
            is_downloaded=True,
        )
    )
    s.commit()
    (tmp_path / "10.mp4").write_bytes(b"fake")
    fake_audio = tmp_path / "10.wav"
    fake_audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "modules.music.features.extract_audio_sample",
        lambda video, out, music: fake_audio,
    )
    rb = MagicMock()
    rb.upload_features.side_effect = TransientError("boom")

    _enrich_upload_fallback(s, rb, str(tmp_path), _music_settings())
    s.commit()

    assert s.query(Music).filter_by(id=1).one().is_audio_features_extracted is False


def test_upload_fallback_sweeps_remaining_null_to_false(db_session, tmp_path):
    """4b sweep: rows with resolved spotify_id and no downloadable clip after
    4a → False. Rows with NULL spotify_id (Stage-1 transient) stay NULL so
    the next run retries them."""
    s = db_session
    s.add(
        Music(id=1, artist="a", track="t", spotify_id="spot1", reccobeats_id=_NO_MATCH)
    )
    s.add(Music(id=2, artist="b", track="t", spotify_id=None))
    s.commit()
    rb = MagicMock()

    _enrich_upload_fallback(s, rb, str(tmp_path), _music_settings())
    s.commit()

    assert s.query(Music).filter_by(id=1).one().is_audio_features_extracted is False
    assert s.query(Music).filter_by(id=2).one().is_audio_features_extracted is None


def test_music_has_features_helper():
    m = Music(artist="a", track="t")
    assert music_has_features(m) is False
    for f in [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ]:
        setattr(m, f, 0.5)
    m.key = 5
    m.mode = 1
    assert music_has_features(m) is True


def test_features_config_payload_ignores_classify_only_knobs():
    """Changing a classify-only knob must not invalidate features
    fingerprints — and vice versa."""
    from core.config import MusicSettings
    from modules.music.state import features_config_payload

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
    )
    bumped_classify = base.model_copy(update={"audio_fingerprint_confidence": 0.95})
    assert features_config_payload(base) == features_config_payload(bumped_classify)

    bumped_features = base.model_copy(update={"reccobeats_batch_size": 99})
    assert features_config_payload(base) != features_config_payload(bumped_features)


def test_reset_music_features_nulls_feature_columns(db_session):
    from core.database import Music
    from modules.music.state import reset_music_features

    db_session.merge(
        Music(
            id=1,
            artist="a",
            track="t",
            spotify_id="sp1",
            reccobeats_id="rb1",
            is_audio_features_extracted=True,
            acousticness=0.5,
            danceability=0.4,
            energy=0.3,
            instrumentalness=0.2,
            key=1,
            liveness=0.1,
            loudness=-5.0,
            mode=1,
            speechiness=0.0,
            tempo=120.0,
            valence=0.8,
        )
    )
    db_session.commit()

    reset_music_features(db_session)
    m = db_session.query(Music).filter_by(id=1).one()
    assert m.spotify_id is None
    assert m.reccobeats_id is None
    assert m.is_audio_features_extracted is None
    assert m.acousticness is None
    assert m.tempo is None
    assert m.valence is None


def test_extract_music_features_config_change_triggers_reset(monkeypatch, db_session):
    """Bumping a feature-relevant MusicSettings field flips the
    features fingerprint and NULLs every feature column."""
    import modules.music.features as features_mod
    from core.config import MusicSettings, PathsSettings
    from core.database import Music
    from modules.music.features import MusicSecrets, extract_music_features

    # No-op every sub-stage so we exercise only the gate.
    monkeypatch.setattr(features_mod, "_resolve_spotify_ids", lambda *a, **kw: None)
    monkeypatch.setattr(features_mod, "_resolve_reccobeats_ids", lambda *a, **kw: None)
    monkeypatch.setattr(features_mod, "_enrich_catalog_features", lambda *a, **kw: None)
    monkeypatch.setattr(features_mod, "_enrich_upload_fallback", lambda *a, **kw: None)

    db_session.merge(
        Music(
            id=1,
            artist="a",
            track="t",
            spotify_id="sp1",
            reccobeats_id="rb1",
            is_audio_features_extracted=True,
            acousticness=0.5,
            danceability=0.4,
            energy=0.3,
            instrumentalness=0.2,
            key=1,
            liveness=0.1,
            loudness=-5.0,
            mode=1,
            speechiness=0.0,
            tempo=120.0,
            valence=0.8,
        )
    )
    db_session.commit()

    base = MusicSettings(
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
    )
    paths = PathsSettings(
        video_dir="/tmp",
        plots_dir="/tmp",
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        speech_audio_dir="/tmp",
        audio_dir="/tmp",
        data_csv_path="/tmp/x.csv",
    )
    secrets = MusicSecrets(spotify_client_id="x", spotify_client_secret="y")

    extract_music_features(base, paths, secrets)
    extract_music_features(
        base.model_copy(update={"reccobeats_batch_size": 99}), paths, secrets
    )

    m = db_session.query(Music).filter_by(id=1).one()
    assert m.spotify_id is None, "config drift must NULL feature columns"
    assert m.tempo is None


def test_extract_music_features_unchanged_config_does_not_reset(
    monkeypatch, db_session
):
    """Two consecutive calls with the same config must preserve seeded
    feature data (no prior state on the first call ⇒ no reset; second
    call hits the fingerprint match)."""
    import modules.music.features as features_mod
    from core.config import MusicSettings, PathsSettings
    from core.database import Music
    from modules.music.features import MusicSecrets, extract_music_features

    monkeypatch.setattr(features_mod, "_resolve_spotify_ids", lambda *a, **kw: None)
    monkeypatch.setattr(features_mod, "_resolve_reccobeats_ids", lambda *a, **kw: None)
    monkeypatch.setattr(features_mod, "_enrich_catalog_features", lambda *a, **kw: None)
    monkeypatch.setattr(features_mod, "_enrich_upload_fallback", lambda *a, **kw: None)

    db_session.merge(
        Music(
            id=1,
            artist="a",
            track="t",
            spotify_id="sp1",
            reccobeats_id="rb1",
            is_audio_features_extracted=True,
            acousticness=0.5,
            danceability=0.4,
            energy=0.3,
            instrumentalness=0.2,
            key=1,
            liveness=0.1,
            loudness=-5.0,
            mode=1,
            speechiness=0.0,
            tempo=120.0,
            valence=0.8,
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
    )
    paths = PathsSettings(
        video_dir="/tmp",
        plots_dir="/tmp",
        model_path="/tmp",
        profile_pic_dir="/tmp",
        thumbnail_dir="/tmp",
        speech_audio_dir="/tmp",
        audio_dir="/tmp",
        data_csv_path="/tmp/x.csv",
    )
    secrets = MusicSecrets(spotify_client_id="x", spotify_client_secret="y")

    extract_music_features(music, paths, secrets)
    extract_music_features(music, paths, secrets)

    m = db_session.query(Music).filter_by(id=1).one()
    assert m.spotify_id == "sp1"
    assert m.tempo == 120.0
