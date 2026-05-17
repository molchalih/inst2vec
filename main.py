from modules.captions import process_captions
from modules.clustering import assign_clusters, run_cluster_search, validate_clustering
from modules.config import load_runtime_config
from modules.console import phase, startup
from modules.database import init_db
from modules.download import download_files, extract_audio_stage
from modules.embeddings import (
    EmbeddingSecrets,
    embed_clip_embeddings,
    embed_user_embeddings,
)
from modules.filter import process_dataset
from modules.music import classify_music, extract_music_features
from modules.music.classify import AcrSecrets
from modules.music.features import MusicSecrets
from modules.parse import fetch_profiles
from modules.speech import process_speech
from modules.upload import upload_videos
from modules.ingest import load_usernames_from_csv
from modules.visualization.plots import plot_clusters


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
    3.5 UPLOAD: pushes selected+downloaded videos to the object store so the
    remote embedder GPU pod can fetch them. No-op when storage.bucket is unset.
    """
    phase("Upload")
    upload_videos(settings, secrets)

    """
    3.6 AUDIO EXTRACTION: extracts and fingerprints mp3 audio from downloaded
    videos for the Gemini multimodal embedding case. No-op when
    embeddings.gemini_enabled is false.
    """
    phase("Audio extraction")
    extract_audio_stage(settings)

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
    process_speech(
        settings.speech,
        video_dir=settings.paths.video_dir,
        speech_audio_dir=settings.paths.speech_audio_dir,
    )

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
    phase("Clip Embeddings")
    embed_clip_embeddings(
        settings,
        EmbeddingSecrets(
            gemini_api_key=secrets.gemini_api_key,
            embedder_remote_url=secrets.embedder_remote_url,
            embedder_token=secrets.embedder_token,
            object_store_endpoint=secrets.object_store_endpoint,
            object_store_access_key=secrets.object_store_access_key,
            object_store_secret_key=secrets.object_store_secret_key,
        ),
    )

    """
    9. USER EMBEDDINGS: calculates the average embedding of the clips belonging to a user, generating a user-level representation.
    """
    phase("User Embeddings")
    embed_user_embeddings(settings)

    workers = getattr(settings.search, "clustering_grid_workers", 1)

    """
    10. CLUSTER SEARCH: ...
    """
    phase("Cluster Search")
    run_cluster_search(settings=settings.search, clustering_grid_workers=workers)

    """
    11. CLUSTER VALIDATION: ...
    """
    phase("Cluster Validation")
    validate_clustering(settings=settings.validation, clustering_grid_workers=workers)

    """
    12. CLUSTERING: assign final cluster labels using the best run per case.
    """
    phase("Clustering")
    assign_clusters(settings=settings.validation)

    """
    13. VISUALIZATION: ...
    """
    phase("Visualization")
    plot_clusters(plots_dir=settings.paths.plots_dir)


if __name__ == "__main__":
    run_pipeline()
