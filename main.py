from modules.captions import clean_captions, detect_caption_language, translate_captions
from modules.cluster_search import run_cluster_search
from modules.cluster_validation import validate_clustering
from modules.clustering import cluster_users
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

startup()

phase("Database")
init_db()

# phase("Importing")
# load_usernames_from_csv()

phase("Profile Parsing")
fetch_profiles()

phase("Dataset Filtering — Pass A")
finalize_user_dataset(pass_name="A")

phase("Download")
download_files()

phase("Music Classification")
classify_music()

phase("Music Feature Extraction")
extract_music_features()

phase("Speech")
classify_speech()
translate_speech()
clean_speech()

phase("Captions")
clean_captions()
detect_caption_language()
translate_captions()

phase("Dataset Filtering — Pass B")
finalize_user_dataset(pass_name="B")

phase("Video Embeddings")
embed_video_clips()
embed_sandwich_clips()
embed_audio_clips()

phase("User Embeddings")
embed_user_clips()

phase("Cluster Search")
run_cluster_search()

phase("Cluster Validation")
best_params = validate_clustering()

phase("Clustering")
for case, params in best_params.items():
    if params is None:
        log("cluster", f"{case}: no valid run — skipping", level="warn")
        continue
    cluster_users(case, **params)

phase("Visualization")
plot_clusters()
