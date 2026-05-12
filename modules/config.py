from __future__ import annotations

import os
from types import SimpleNamespace


def load_runtime_config() -> SimpleNamespace:
    """Load runtime configuration from environment variables and build a runtime object.
    
    Returns a SimpleNamespace with all required configuration sections:
    - database_url, identity_db_url
    - pipeline (batch_size, max_clips)
    - paths (video_dir, plots_dir, model_path, profile_pic_dir, thumbnail_dir)
    - parse (fetch_retry_delays_sec)
    - download (max_attempts, retry_delay)
    - finalize (target_clips_per_user, require_min_text_clips, pass_a_recompute_from_scratch, global_min_plays, global_min_plays_percentile, creator_robust_z_threshold, creator_min_clips)
    - music (audio_fingerprint_confidence, commit_every, http_timeout, spotify_search_limit, spotify_token_skew_seconds, spotify_request_timeout, reccobeats_batch_size, reccobeats_delay_min, reccobeats_delay_max, manual_features_max_seconds, manual_features_sample_rate, manual_features_max_mb, manual_features_mp3_bitrate)
    - speech (whisper_model, commit_every, translate_model, translate_target_lang, translation_max_chars, translate_max_new_tokens, logprob_threshold, compression_threshold, min_meaningful_chars)
    - captions (commit_every, translate_model, translate_target_lang, translation_max_chars, translate_max_new_tokens)
    - embeddings (exclude_disqualified_users, embed_max_length, adaptive_max_frames, adaptive_default_fps)
    - search
    - validation (plateau_drop_threshold)
    - overrides (video, sandwich, audio)
    - secrets (hiker_api_key, arc_host, arc_access_key, arc_secret_key, spotify_client_id, spotify_client_secret, huggingface_token)
    """
    runtime = SimpleNamespace(
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/inst2vec.db"),
        identity_db_url=os.getenv("IDENTITY_DB_URL", "sqlite:///data/identity_map.db"),
        pipeline=SimpleNamespace(
            batch_size=int(os.getenv("BATCH_SIZE", "9999")),
            max_clips=int(os.getenv("MAX_CLIPS", "5")),
        ),
        paths=SimpleNamespace(
            video_dir=os.getenv("VIDEO_DIR", "data/source/videos"),
            plots_dir=os.getenv("PLOTS_DIR", "data/plots"),
            model_path=os.getenv("MODEL_PATH", "./models/Qwen3-VL-Embedding-8B"),
            profile_pic_dir=os.getenv("PROFILE_PIC_DIR", "data/source/profile_pics"),
            thumbnail_dir=os.getenv("THUMBNAIL_DIR", "data/source/thumbnails"),
        ),
        parse=SimpleNamespace(
            fetch_retry_delays_sec=[int(x) for x in os.getenv("PARSE_FETCH_RETRY_DELAYS", "0 30 60 90").split()],
        ),
        download=SimpleNamespace(
            max_attempts=int(os.getenv("MAX_DOWNLOAD_ATTEMPTS", "3")),
            retry_delay=int(os.getenv("DOWNLOAD_RETRY_DELAY", "2")),
        ),
        finalize=SimpleNamespace(
            target_clips_per_user=int(os.getenv("FINALIZE_TARGET_CLIPS_PER_USER", "4")),
            require_min_text_clips=bool(int(os.getenv("FINALIZE_REQUIRE_MIN_TEXT_CLIPS", "0"))),
            pass_a_recompute_from_scratch=bool(int(os.getenv("FINALIZE_PASS_A_RECOMPUTE_FROM_SCRATCH", "1"))),
            global_min_plays=int(os.getenv("FINALIZE_GLOBAL_MIN_PLAYS", "0")),
            global_min_plays_percentile=float(os.getenv("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "5")),
            creator_robust_z_threshold=float(os.getenv("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-2.5")),
            creator_min_clips=int(os.getenv("FINALIZE_CREATOR_MIN_CLIPS", "4")),
        ),
        music=SimpleNamespace(
            audio_fingerprint_confidence=float(os.getenv("AUDIO_FINGERPRINT_CONFIDENCE", "0.8")),
            commit_every=int(os.getenv("SPOTIFY_COMMIT_EVERY", "50")),
            http_timeout=float(os.getenv("MUSIC_HTTP_TIMEOUT", "20")),
            spotify_search_limit=int(os.getenv("SPOTIFY_SEARCH_LIMIT", "5")),
            spotify_token_skew_seconds=int(os.getenv("SPOTIFY_TOKEN_SKEW_SECONDS", "30")),
            spotify_request_timeout=float(os.getenv("SPOTIFY_REQUEST_TIMEOUT", "8")),
            reccobeats_batch_size=int(os.getenv("RECCOBEATS_BATCH_SIZE", "20")),
            reccobeats_delay_min=float(os.getenv("RECCOBEATS_DELAY_MIN", "2")),
            reccobeats_delay_max=float(os.getenv("RECCOBEATS_DELAY_MAX", "3")),
            manual_features_max_seconds=int(os.getenv("MANUAL_FEATURES_MAX_SECONDS", "20")),
            manual_features_sample_rate=int(os.getenv("MANUAL_FEATURES_SAMPLE_RATE", "44100")),
            manual_features_max_mb=float(os.getenv("MANUAL_FEATURES_MAX_MB", "5")),
            manual_features_mp3_bitrate=os.getenv("MANUAL_FEATURES_MP3_BITRATE", "128k"),
        ),
        speech=SimpleNamespace(
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
            commit_every=int(os.getenv("SPEECH_COMMIT_EVERY", "50")),
            translate_model=os.getenv("SPEECH_TRANSLATE_MODEL", "google/translategemma-4b-it"),
            translate_target_lang=os.getenv("SPEECH_TRANSLATE_TARGET_LANG", "en"),
            translation_max_chars=int(os.getenv("SPEECH_TRANSLATION_MAX_CHARS", "1000")),
            translate_max_new_tokens=int(os.getenv("SPEECH_TRANSLATE_MAX_NEW_TOKENS", "200")),
            logprob_threshold=float(os.getenv("SPEECH_LOGPROB_THRESHOLD", "-0.8")),
            compression_threshold=float(os.getenv("SPEECH_COMPRESSION_THRESHOLD", "2.4")),
            min_meaningful_chars=int(os.getenv("SPEECH_MIN_MEANINGFUL_CHARS", "8")),
        ),
        captions=SimpleNamespace(
            commit_every=int(os.getenv("CAPTIONS_COMMIT_EVERY", "50")),
            translate_model=os.getenv("CAPTIONS_TRANSLATE_MODEL", "google/translategemma-4b-it"),
            translate_target_lang=os.getenv("CAPTIONS_TRANSLATE_TARGET_LANG", "en"),
            translation_max_chars=int(os.getenv("CAPTIONS_TRANSLATION_MAX_CHARS", "1000")),
            translate_max_new_tokens=int(os.getenv("CAPTIONS_TRANSLATE_MAX_NEW_TOKENS", "200")),
        ),
        embeddings=SimpleNamespace(
            exclude_disqualified_users=bool(int(os.getenv("EMBEDDINGS_EXCLUDE_DISQUALIFIED_USERS", "1"))),
            embed_max_length=int(os.getenv("EMBEDDINGS_EMBED_MAX_LENGTH", "32768")),
            adaptive_max_frames=int(os.getenv("EMBEDDINGS_ADAPTIVE_MAX_FRAMES", "96")),
            adaptive_default_fps=float(os.getenv("EMBEDDINGS_ADAPTIVE_DEFAULT_FPS", "2.0")),
        ),
        search=SimpleNamespace(),
        validation=SimpleNamespace(
            plateau_drop_threshold=float(os.getenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")),
        ),
        overrides=SimpleNamespace(
            video=os.getenv("CLUSTER_OVERRIDE_VIDEO", ""),
            sandwich=os.getenv("CLUSTER_OVERRIDE_SANDWICH", ""),
            audio=os.getenv("CLUSTER_OVERRIDE_AUDIO", ""),
        ),
        secrets=SimpleNamespace(
            hiker_api_key=os.getenv("HIKER_API_KEY", ""),
            arc_host=os.getenv("ARC_HOST", ""),
            arc_access_key=os.getenv("ARC_ACCESS_KEY", ""),
            arc_secret_key=os.getenv("ARC_SECRET_KEY", ""),
            spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID", ""),
            spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", ""),
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN", ""),
        ),
    )
    return runtime
