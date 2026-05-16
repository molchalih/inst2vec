from modules.captions import process_captions
from modules.cluster_search import run_cluster_search
from modules.cluster_validation import validate_clustering
from modules.clustering import cluster_users
from modules.config import load_runtime_config
from modules.console import log, phase, startup
from modules.database import init_db
from modules.download import download_files
from modules._embeddings_legacy import (
    embed_audio_clips,
    embed_sandwich_clips,
    embed_user_clips,
    embed_video_clips,
)
from modules.filter import process_dataset
from modules.music import classify_music, extract_music_features
from modules.music.classify import AcrSecrets
from modules.music.features import MusicSecrets
from modules.parse import fetch_profiles
from modules.speech import VadConfig, classify_speech, clean_speech, translate_speech
from modules.utils import load_usernames_from_csv
from modules.visualization import plot_clusters


def run_pipeline() -> None:

    settings, secrets = load_runtime_config()

    startup()

    """
    0. DATABASE: initializes the databases (identity and main). Populate / seed from .csv if neccessary.
    """
    phase("Database")
    init_db(secrets.database_url, secrets.identity_db_url)

    phase("Importing")
    load_usernames_from_csv(csv_path=settings.paths.data_csv_path)

    """
    1. PARSING: fetches profiles and corresponding clips metadata via hiker api, populates the database.
    a) fetch_profiles: fetches profiles and corresponding clips metadata, populates the database.
    b) fetch_clips: fetches clips metadata, populates the database.
    """
    phase("Profile Parsing")
    fetch_profiles(
        hiker_api_key=secrets.hiker_api_key,
    )

    """
    2. PROCESSING: Filters low quality and unwanted clips, randomly selects appropriate ones. Generates statistics.
    a) hard: flags low quality clips and those that don't meet the basic policy.
    b) soft: flags clips that are outliers in the dataset.
    c) random: randomly selects clips from the remaining pool.
    """
    phase("Processing Dataset")
    process_dataset(settings.filter)

    """
    3. DOWNLOADING: downloads profile pics, videos and thumbnails of the filtered profiles.
    """
    phase("Download")
    download_files(settings.download, settings.paths)

    """
    4.1 MUSIC: fingerprints the music in videos.
    """
    phase("Music fingerprinting")
    classify_music(
        music=settings.music,
        paths=settings.paths,
        secrets=AcrSecrets(
            host=secrets.arc_host,
            access_key=secrets.arc_access_key,
            access_secret=secrets.arc_secret_key,
        ),
    )
    """
    4.2. MUSIC: extracts the music features (its textual representation).
    """
    phase("Music feature extraction")
    extract_music_features(
        music=settings.music,
        paths=settings.paths,
        secrets=MusicSecrets(
            spotify_client_id=secrets.spotify_client_id,
            spotify_client_secret=secrets.spotify_client_secret,
        ),
    )

    """
    5. SPEECH: transcribes speech with Whisper (writes is_speech_detected),
       translates detected non-English speech, then post-cleans
       hallucination-marker translations.
    """
    phase("Speech transcription")
    classify_speech(
        video_dir=settings.paths.video_dir,
        speech_audio_dir=settings.paths.speech_audio_dir,
        whisper_model=settings.speech.whisper_model,
        commit_every=settings.speech.commit_every,
        logprob_threshold=settings.speech.logprob_threshold,
        compression_threshold=settings.speech.compression_threshold,
        min_meaningful_chars=settings.speech.min_meaningful_chars,
        vad_config=VadConfig(
            enabled=settings.speech.vad_enabled,
            sampling_rate=settings.speech.vad_sampling_rate,
            threshold=settings.speech.vad_threshold,
            min_speech_ms=settings.speech.vad_min_speech_ms,
            min_silence_ms=settings.speech.vad_min_silence_ms,
            speech_pad_ms=settings.speech.vad_speech_pad_ms,
            min_total_speech_s=settings.speech.vad_min_total_speech_s,
        ),
    )
    translate_speech(
        commit_every=settings.speech.commit_every,
        translate_model=settings.speech.translate_model,
        translate_target_lang=settings.speech.translate_target_lang,
        translation_max_chars=settings.speech.translation_max_chars,
        translate_max_new_tokens=settings.speech.translate_max_new_tokens,
    )
    clean_speech()

    """
    6. CAPTIONS: translates applicable captions.
    """
    phase("Captions translation")
    process_captions(settings.captions)

    """
    8. EMBEDDINGS: embeds the features into a vector space (various modalities).
    - video: only video
    - sandwich: video + music features
    - audio: only audio
    """
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

    """
    9. USER EMBEDDINGS: calculates the average embedding of the clips belonging to a user, generating a user-level representation.
    """
    phase("User Embeddings")
    embed_user_clips()

    """
    10. CLUSTER SEARCH: ...
    """
    phase("Cluster Search")
    run_cluster_search(
        settings=settings.search,
        clustering_grid_workers=getattr(settings.search, "clustering_grid_workers", 1),
    )

    """
    11. CLUSTER VALIDATION: ...
    """
    phase("Cluster Validation")
    best_params = validate_clustering(
        settings=settings.validation,
        clustering_grid_workers=getattr(settings.search, "clustering_grid_workers", 1),
    )

    """
    12. CLUSTERING: ...
    """
    phase("Clustering")
    for case, params in best_params.items():
        if params is None:
            log("cluster", f"{case}: no valid run — skipping", level="warn")
            continue
        cluster_users(case, **params)

    """
    13. VISUALIZATION: ...
    """
    phase("Visualization")
    plot_clusters(plots_dir=settings.paths.plots_dir)


if __name__ == "__main__":
    run_pipeline()
