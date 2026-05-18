from core.config import load_runtime_config
from core.console import phase, startup
from core.database import init_db
from modules import (
    captions,
    clustering,
    embeddings,
    filter,
    ingest,
    music,
    speech,
    upload,
)
from modules.visualization import plots


def run_pipeline() -> None:

    settings, secrets = load_runtime_config()

    startup()

    """
    0. DATABASE: initializes the databases (identity and main). Populate / seed from .csv if neccessary.
    """
    phase("Database")
    init_db(secrets.database_url, secrets.identity_db_url)

    phase("Importing")
    ingest.run_seed(settings, secrets)

    """
    1. PARSING: fetches profiles and corresponding clips metadata via hiker api, populates the database.
    a) fetch_profiles: fetches profiles and corresponding clips metadata, populates the database.
    b) fetch_clips: fetches clips metadata, populates the database.
    """
    phase("Profile Parsing")
    ingest.run_profiles(settings, secrets)

    """
    2. PROCESSING: Filters low quality and unwanted clips, randomly selects appropriate ones. Generates statistics.
    a) hard: flags low quality clips and those that don't meet the basic policy.
    b) soft: flags clips that are outliers in the dataset.
    c) random: randomly selects clips from the remaining pool.
    """
    phase("Processing Dataset")
    filter.run(settings, secrets)

    """
    3. DOWNLOADING: downloads profile pics, videos and thumbnails of the filtered profiles.
    """
    phase("Download")
    ingest.run_download(settings, secrets)

    """
    3.5 UPLOAD: pushes selected+downloaded videos to the object store so the
    remote embedder GPU pod can fetch them. No-op when storage.bucket is unset.
    """
    phase("Upload")
    upload.run(settings, secrets)

    """
    3.6 AUDIO EXTRACTION: extracts and fingerprints mp3 audio from downloaded
    videos. Always runs; idempotent via fingerprint seal + per-file mtime.
    """
    phase("Audio extraction")
    ingest.run_audio(settings, secrets)

    """
    4.1 MUSIC: fingerprints the music in videos.
    """
    phase("Music fingerprinting")
    music.run_classify(settings, secrets)

    """
    4.2. MUSIC: extracts the music features (its textual representation).
    """
    phase("Music feature extraction")
    music.run_features(settings, secrets)

    """
    5. SPEECH: transcribes speech with Whisper (writes is_speech_detected),
       translates detected non-English speech, then post-cleans
       hallucination-marker translations.
    """
    phase("Speech transcription")
    speech.run(settings, secrets)

    """
    6. CAPTIONS: translates applicable captions.
    """
    phase("Captions translation")
    captions.run(settings, secrets)

    """
    8. EMBEDDINGS: embeds the features into a vector space (various modalities).
    - video: only video
    - sandwich: video + music features
    - audio: only audio
    """
    phase("Clip Embeddings")
    embeddings.run_clip(settings, secrets)

    """
    9. USER EMBEDDINGS: calculates the average embedding of the clips belonging to a user, generating a user-level representation.
    """
    phase("User Embeddings")
    # Aggregate every case that downstream clustering will request, so
    # gemini (when embeddings.gemini_enabled=true) produces UserEmbedding
    # rows and is not silently sealed as an empty matrix.
    embeddings.run_users(settings, secrets)

    """
    10. CLUSTER SEARCH: ...
    """
    phase("Cluster Search")
    clustering.run_search(settings, secrets)

    """
    11. CLUSTER VALIDATION: ...
    """
    phase("Cluster Validation")
    clustering.run_validation(settings, secrets)

    """
    12. CLUSTERING: assign final cluster labels using the best run per case.
    """
    phase("Clustering")
    clustering.run_assign(settings, secrets)

    """
    13. VISUALIZATION: ...
    """
    phase("Visualization")
    plots.run(settings, secrets)


if __name__ == "__main__":
    run_pipeline()
