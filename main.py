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
    runtime = load_runtime_config()

    startup()

    phase("Database")
    init_db(runtime.database_url, runtime.identity_db_url)

    phase("Profile Parsing")
    fetch_profiles(
        batch_size=runtime.pipeline.batch_size,
        max_clips=runtime.pipeline.max_clips,
        hiker_api_key=runtime.secrets.hiker_api_key,
    )

    phase("Dataset Filtering — Pass A")
    finalize_user_dataset(
        "A",
        target_clips_per_user=runtime.finalize.target_clips_per_user,
        require_min_text_clips=runtime.finalize.require_min_text_clips,
        pass_a_recompute_from_scratch=runtime.finalize.pass_a_recompute_from_scratch,
        global_min_plays=runtime.finalize.global_min_plays,
        global_min_plays_percentile=runtime.finalize.global_min_plays_percentile,
        creator_robust_z_threshold=runtime.finalize.creator_robust_z_threshold,
        creator_min_clips=runtime.finalize.creator_min_clips,
    )

    phase("Download")
    download_files(
        batch_size=runtime.pipeline.batch_size,
        max_clips=runtime.pipeline.max_clips,
        max_attempts=runtime.download.max_attempts,
        retry_delay=runtime.download.retry_delay,
        profile_pic_dir=runtime.paths.profile_pic_dir,
        thumbnail_dir=runtime.paths.thumbnail_dir,
        video_dir=runtime.paths.video_dir,
    )

    phase("Music Classification")
    classify_music(
        video_dir=runtime.paths.video_dir,
        min_confidence=runtime.music.audio_fingerprint_confidence,
        commit_every=runtime.music.commit_every,
        arc_host=runtime.secrets.arc_host,
        arc_access_key=runtime.secrets.arc_access_key,
        arc_secret_key=runtime.secrets.arc_secret_key,
    )

    phase("Music Feature Extraction")
    extract_music_features(
        video_dir=runtime.paths.video_dir,
        http_timeout=runtime.music.http_timeout,
        commit_every=runtime.music.commit_every,
        spotify_client_id=runtime.secrets.spotify_client_id,
        spotify_client_secret=runtime.secrets.spotify_client_secret,
        spotify_token_skew_seconds=runtime.music.spotify_token_skew_seconds,
        spotify_search_limit=runtime.music.spotify_search_limit,
        spotify_request_timeout=runtime.music.spotify_request_timeout,
        reccobeats_batch_size=runtime.music.reccobeats_batch_size,
        reccobeats_delay_min=runtime.music.reccobeats_delay_min,
        reccobeats_delay_max=runtime.music.reccobeats_delay_max,
        manual_features_max_seconds=runtime.music.manual_features_max_seconds,
        manual_features_sample_rate=runtime.music.manual_features_sample_rate,
        manual_features_max_mb=runtime.music.manual_features_max_mb,
        manual_features_mp3_bitrate=runtime.music.manual_features_mp3_bitrate,
    )

    phase("Speech")
    classify_speech(
        video_dir=runtime.paths.video_dir,
        whisper_model=runtime.speech.whisper_model,
        commit_every=runtime.speech.commit_every,
        logprob_threshold=runtime.speech.logprob_threshold,
        compression_threshold=runtime.speech.compression_threshold,
        min_meaningful_chars=runtime.speech.min_meaningful_chars,
    )
    translate_speech(
        commit_every=runtime.speech.commit_every,
        translate_model=runtime.speech.translate_model,
        translate_target_lang=runtime.speech.translate_target_lang,
        translation_max_chars=runtime.speech.translation_max_chars,
        translate_max_new_tokens=runtime.speech.translate_max_new_tokens,
    )
    clean_speech()

    phase("Captions")
    clean_captions(commit_every=runtime.captions.commit_every)
    detect_caption_language()
    translate_captions(
        commit_every=runtime.captions.commit_every,
        translate_model=runtime.captions.translate_model,
        translate_target_lang=runtime.captions.translate_target_lang,
        translation_max_chars=runtime.captions.translation_max_chars,
        translate_max_new_tokens=runtime.captions.translate_max_new_tokens,
    )

    phase("Dataset Filtering — Pass B")
    finalize_user_dataset(
        "B",
        target_clips_per_user=runtime.finalize.target_clips_per_user,
        require_min_text_clips=runtime.finalize.require_min_text_clips,
        pass_a_recompute_from_scratch=runtime.finalize.pass_a_recompute_from_scratch,
        global_min_plays=runtime.finalize.global_min_plays,
        global_min_plays_percentile=runtime.finalize.global_min_plays_percentile,
        creator_robust_z_threshold=runtime.finalize.creator_robust_z_threshold,
        creator_min_clips=runtime.finalize.creator_min_clips,
    )

    phase("Video Embeddings")
    embed_video_clips(
        model_path=runtime.paths.model_path,
        video_dir=runtime.paths.video_dir,
        embed_max_length=runtime.embeddings.embed_max_length,
        adaptive_max_frames=runtime.embeddings.adaptive_max_frames,
        adaptive_default_fps=runtime.embeddings.adaptive_default_fps,
        exclude_disqualified_users=runtime.embeddings.exclude_disqualified_users,
    )
    embed_sandwich_clips(
        model_path=runtime.paths.model_path,
        video_dir=runtime.paths.video_dir,
        embed_max_length=runtime.embeddings.embed_max_length,
        adaptive_max_frames=runtime.embeddings.adaptive_max_frames,
        adaptive_default_fps=runtime.embeddings.adaptive_default_fps,
        exclude_disqualified_users=runtime.embeddings.exclude_disqualified_users,
    )
    embed_audio_clips(
        model_path=runtime.paths.model_path,
        video_dir=runtime.paths.video_dir,
        embed_max_length=runtime.embeddings.embed_max_length,
        adaptive_max_frames=runtime.embeddings.adaptive_max_frames,
        adaptive_default_fps=runtime.embeddings.adaptive_default_fps,
        exclude_disqualified_users=runtime.embeddings.exclude_disqualified_users,
    )

    phase("User Embeddings")
    embed_user_clips()

    phase("Cluster Search")
    run_cluster_search(settings=runtime.search)

    phase("Cluster Validation")
    best_params = validate_clustering(settings=runtime.validation)

    phase("Clustering")
    for case, params in best_params.items():
        if params is None:
            log("cluster", f"{case}: no valid run — skipping", level="warn")
            continue
        cluster_users(case, **params)

    phase("Visualization")
    plot_clusters(plots_dir=runtime.paths.plots_dir)


if __name__ == "__main__":
    run_pipeline()
