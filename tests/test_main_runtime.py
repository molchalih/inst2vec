from __future__ import annotations

from types import SimpleNamespace

import main


def test_run_pipeline_loads_config_once_and_wires_stages(monkeypatch):
    calls: list[str] = []

    settings = SimpleNamespace(
        paths=SimpleNamespace(
            video_dir="data/source/videos",
            plots_dir="data/plots",
            model_path="./models/Qwen3-VL-Embedding-8B",
            profile_pic_dir="data/source/profile_pics",
            thumbnail_dir="data/source/thumbnails",
            speech_audio_dir="data/source/audio",
            data_csv_path="data/data.csv",
        ),
        parse=SimpleNamespace(fetch_retry_delays_sec=[0, 30, 60, 90]),
        download=SimpleNamespace(
            max_attempts=3, retry_delay=2, retry_jitter=5, concurrency=5
        ),
        filter=SimpleNamespace(
            min_video_duration=3,
            max_video_duration=80,
            min_taken_at=1640995200,
            creator_min_median_views=10000,
            min_eligible_clips_per_user=10,
            global_low_percentile=5,
            global_high_percentile=99,
            creator_low_z_threshold=-3.5,
            selection_pool_percent=0.20,
            selected_clips_per_user=10,
            selection_random_seed=42,
        ),
        music=SimpleNamespace(
            audio_fingerprint_confidence=0.8,
            commit_every=50,
            http_timeout=20.0,
            spotify_search_limit=5,
            spotify_token_skew_seconds=30,
            spotify_request_timeout=8.0,
            reccobeats_batch_size=20,
            reccobeats_delay_min=2.0,
            reccobeats_delay_max=3.0,
            manual_features_max_seconds=20,
            manual_features_sample_rate=44100,
            manual_features_max_mb=5.0,
            manual_features_mp3_bitrate="128k",
        ),
        speech=SimpleNamespace(
            whisper_model="large-v3-turbo",
            commit_every=50,
            translate_model="google/translategemma-4b-it",
            translate_target_lang="en",
            translation_max_chars=1000,
            translate_max_new_tokens=200,
            logprob_threshold=-0.8,
            compression_threshold=2.4,
            min_meaningful_chars=8,
            vad_enabled=True,
            vad_sampling_rate=16000,
            vad_threshold=0.5,
            vad_min_speech_ms=250,
            vad_min_silence_ms=100,
            vad_speech_pad_ms=150,
            vad_min_total_speech_s=0.5,
        ),
        captions=SimpleNamespace(
            commit_every=50,
            translate_model="google/translategemma-4b-it",
            translate_target_lang="en",
            translation_max_chars=1000,
            translate_max_new_tokens=200,
        ),
        embeddings=SimpleNamespace(
            exclude_disqualified_users=True,
            embed_max_length=32768,
            adaptive_max_frames=96,
            adaptive_default_fps=2.0,
        ),
        search=SimpleNamespace(),
        validation=SimpleNamespace(plateau_drop_threshold=0.05),
        overrides=SimpleNamespace(video="", sandwich="", audio=""),
    )

    secrets = SimpleNamespace(
        database_url="sqlite:///:memory:",
        identity_db_url="sqlite:///:memory:",
        hiker_api_key="hiker",
        arc_host="arc-host",
        arc_access_key="arc-key",
        arc_secret_key="arc-secret",
        spotify_client_id="spotify-id",
        spotify_client_secret="spotify-secret",
        huggingface_token="hf",
    )

    monkeypatch.setattr(main, "load_runtime_config", lambda: (settings, secrets))
    monkeypatch.setattr(main, "startup", lambda: calls.append("startup"))
    monkeypatch.setattr(
        main,
        "init_db",
        lambda database_url, identity_db_url: calls.append(
            f"init:{database_url}:{identity_db_url}"
        ),
    )
    monkeypatch.setattr(
        main,
        "load_usernames_from_csv",
        lambda **kwargs: calls.append("import:csv"),
    )
    monkeypatch.setattr(main, "fetch_profiles", lambda **kwargs: calls.append("parse"))
    monkeypatch.setattr(
        main, "process_dataset", lambda *args, **kwargs: calls.append("filter")
    )
    monkeypatch.setattr(
        main, "download_files", lambda *args, **kwargs: calls.append("download")
    )
    monkeypatch.setattr(
        main, "classify_music", lambda **kwargs: calls.append("music:classify")
    )
    monkeypatch.setattr(
        main, "extract_music_features", lambda **kwargs: calls.append("music:features")
    )
    monkeypatch.setattr(
        main, "process_speech", lambda *args, **kwargs: calls.append("speech:process")
    )
    monkeypatch.setattr(
        main,
        "process_captions",
        lambda *args, **kwargs: calls.append("captions:process"),
    )
    monkeypatch.setattr(
        main,
        "embed_clip_embeddings",
        lambda settings, cases=None: calls.append("embed:clip"),
    )
    monkeypatch.setattr(
        main,
        "embed_user_embeddings",
        lambda settings, cases=None: calls.append("embed:user"),
    )
    monkeypatch.setattr(
        main, "run_cluster_search", lambda **kwargs: calls.append("cluster:search")
    )
    monkeypatch.setattr(
        main,
        "validate_clustering",
        lambda **kwargs: {"video": None, "sandwich": None, "audio": None},
    )
    monkeypatch.setattr(main, "plot_clusters", lambda **kwargs: calls.append("viz"))

    main.run_pipeline()

    assert calls[0] == "startup"
    assert calls[1].startswith("init:sqlite:///:memory:")
    assert calls[2] == "import:csv"
    assert calls[3] == "parse"
    assert "download" in calls
    assert "cluster:search" in calls
    assert calls[-1] == "viz"
