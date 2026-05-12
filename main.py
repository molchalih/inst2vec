from modules.captions import clean_captions, detect_caption_language, translate_captions
from modules.cluster_search import run_cluster_search
from modules.cluster_validation import validate_clustering
from modules.clustering import cluster_users
from modules.config import load_runtime_config
from modules.console import log, phase, startup
from modules.database import init_db
from modules.download import download_files
from modules.embeddings import (
    embed_audio_clips,
    embed_sandwich_clips,
    embed_user_clips,
    embed_video_clips,
)
from modules.finalize import finalize_user_dataset
from modules.music import classify_music, extract_music_features
from modules.parse import fetch_profiles
from modules.speech import classify_speech, clean_speech, translate_speech
from modules.visualization import plot_clusters


def run_pipeline() -> None:

    settings, secrets = load_runtime_config()

    startup()

    phase("Database")
    init_db(secrets.database_url, secrets.identity_db_url)

    phase("Profile Parsing")
    fetch_profiles(
        batch_size=settings.pipeline.batch_size,
        max_clips=settings.pipeline.max_clips,
        hiker_api_key=secrets.hiker_api_key,
    )

    phase("Dataset Filtering — Pass A")
    finalize_user_dataset(
        "A",
        target_clips_per_user=settings.finalize.target_clips_per_user,
        require_min_text_clips=settings.finalize.require_min_text_clips,
        pass_a_recompute_from_scratch=settings.finalize.pass_a_recompute_from_scratch,
        global_min_plays=settings.finalize.global_min_plays,
        global_min_plays_percentile=settings.finalize.global_min_plays_percentile,
        creator_robust_z_threshold=settings.finalize.creator_robust_z_threshold,
        creator_min_clips=settings.finalize.creator_min_clips,
    )

    phase("Download")
    download_files(
        batch_size=settings.pipeline.batch_size,
        max_clips=settings.pipeline.max_clips,
        max_attempts=settings.download.max_attempts,
        retry_delay=settings.download.retry_delay,
        profile_pic_dir=settings.paths.profile_pic_dir,
        thumbnail_dir=settings.paths.thumbnail_dir,
        video_dir=settings.paths.video_dir,
    )

    phase("Music Classification")
    classify_music(
        video_dir=settings.paths.video_dir,
        min_confidence=settings.music.audio_fingerprint_confidence,
        commit_every=settings.music.commit_every,
        arc_host=secrets.arc_host,
        arc_access_key=secrets.arc_access_key,
        arc_secret_key=secrets.arc_secret_key,
    )

    phase("Music Feature Extraction")
    extract_music_features(
        video_dir=settings.paths.video_dir,
        http_timeout=settings.music.http_timeout,
        commit_every=settings.music.commit_every,
        spotify_client_id=secrets.spotify_client_id,
        spotify_client_secret=secrets.spotify_client_secret,
        spotify_token_skew_seconds=settings.music.spotify_token_skew_seconds,
        spotify_search_limit=settings.music.spotify_search_limit,
        spotify_request_timeout=settings.music.spotify_request_timeout,
        reccobeats_batch_size=settings.music.reccobeats_batch_size,
        reccobeats_delay_min=settings.music.reccobeats_delay_min,
        reccobeats_delay_max=settings.music.reccobeats_delay_max,
        manual_features_max_seconds=settings.music.manual_features_max_seconds,
        manual_features_sample_rate=settings.music.manual_features_sample_rate,
        manual_features_max_mb=settings.music.manual_features_max_mb,
        manual_features_mp3_bitrate=settings.music.manual_features_mp3_bitrate,
    )

    phase("Speech")
    classify_speech(
        video_dir=settings.paths.video_dir,
        whisper_model=settings.speech.whisper_model,
        commit_every=settings.speech.commit_every,
        logprob_threshold=settings.speech.logprob_threshold,
        compression_threshold=settings.speech.compression_threshold,
        min_meaningful_chars=settings.speech.min_meaningful_chars,
    )
    translate_speech(
        commit_every=settings.speech.commit_every,
        translate_model=settings.speech.translate_model,
        translate_target_lang=settings.speech.translate_target_lang,
        translation_max_chars=settings.speech.translation_max_chars,
        translate_max_new_tokens=settings.speech.translate_max_new_tokens,
    )
    clean_speech()

    phase("Captions")
    clean_captions(commit_every=settings.captions.commit_every)
    detect_caption_language()
    translate_captions(
        commit_every=settings.captions.commit_every,
        translate_model=settings.captions.translate_model,
        translate_target_lang=settings.captions.translate_target_lang,
        translation_max_chars=settings.captions.translation_max_chars,
        translate_max_new_tokens=settings.captions.translate_max_new_tokens,
    )

    phase("Dataset Filtering — Pass B")
    finalize_user_dataset(
        "B",
        target_clips_per_user=settings.finalize.target_clips_per_user,
        require_min_text_clips=settings.finalize.require_min_text_clips,
        pass_a_recompute_from_scratch=settings.finalize.pass_a_recompute_from_scratch,
        global_min_plays=settings.finalize.global_min_plays,
        global_min_plays_percentile=settings.finalize.global_min_plays_percentile,
        creator_robust_z_threshold=settings.finalize.creator_robust_z_threshold,
        creator_min_clips=settings.finalize.creator_min_clips,
    )

    phase("Video Embeddings")
    embed_video_clips(
        model_path=settings.paths.model_path,
        video_dir=settings.paths.video_dir,
        embed_max_length=settings.embeddings.embed_max_length,
        adaptive_max_frames=settings.embeddings.adaptive_max_frames,
        adaptive_default_fps=settings.embeddings.adaptive_default_fps,
        exclude_disqualified_users=settings.embeddings.exclude_disqualified_users,
    )
    embed_sandwich_clips(
        model_path=settings.paths.model_path,
        video_dir=settings.paths.video_dir,
        embed_max_length=settings.embeddings.embed_max_length,
        adaptive_max_frames=settings.embeddings.adaptive_max_frames,
        adaptive_default_fps=settings.embeddings.adaptive_default_fps,
        exclude_disqualified_users=settings.embeddings.exclude_disqualified_users,
    )
    embed_audio_clips(
        model_path=settings.paths.model_path,
        video_dir=settings.paths.video_dir,
        embed_max_length=settings.embeddings.embed_max_length,
        adaptive_max_frames=settings.embeddings.adaptive_max_frames,
        adaptive_default_fps=settings.embeddings.adaptive_default_fps,
        exclude_disqualified_users=settings.embeddings.exclude_disqualified_users,
    )

    phase("User Embeddings")
    embed_user_clips()

    phase("Cluster Search")
    run_cluster_search(
        settings=settings.search,
        clustering_grid_workers=getattr(settings.search, "clustering_grid_workers", 1),
    )

    phase("Cluster Validation")
    best_params = validate_clustering(
        settings=settings.validation,
        clustering_grid_workers=getattr(settings.search, "clustering_grid_workers", 1),
    )

    phase("Clustering")
    for case, params in best_params.items():
        if params is None:
            log("cluster", f"{case}: no valid run — skipping", level="warn")
            continue
        cluster_users(case, **params)

    phase("Visualization")
    plot_clusters(plots_dir=settings.paths.plots_dir)


if __name__ == "__main__":
    run_pipeline()
