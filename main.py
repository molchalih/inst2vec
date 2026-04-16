from modules.database import init_db, load_usernames_from_csv
from modules.parse import fetch_profiles
from modules.download import download_files
from modules.music import classify_music, extract_music_features
from modules.speech import classify_speech, translate_speech, clean_speech
from modules.captions import detect_caption_language, translate_captions, clean_captions
from modules.finalize import finalize_user_dataset
from modules.embeddings import embed_video_clips, embed_sandwich_clips, embed_audio_clips, embed_user_clips
from modules.clustering import cluster_users

# initialize database
init_db()

# load usernames from csv
load_usernames_from_csv()

# data parsing
fetch_profiles()

# dataset filtering pass A (post-parse)
finalize_user_dataset(pass_name="A")

# data download
download_files()

# Phase 1 – music
classify_music()

# music features extraction via Spotify and ReccoBeats
extract_music_features()

# Phase 2 – speech
classify_speech()
translate_speech()
clean_speech()

# Phase 3 – captions
clean_captions()
detect_caption_language()
translate_captions()

# dataset filtering pass B (pre-embedding)
finalize_user_dataset(pass_name="B")

# Phase 4 – embeddings (run sequentially by modality)
embed_video_clips()
embed_sandwich_clips()
embed_audio_clips()

# Phase 5 – user embeddings (mean-pool clip embeddings per user per case)
embed_user_clips()

# Phase 6 – clustering (UMAP + HDBSCAN per embedding case)
for case in ["video", "sandwich", "audio"]:
    cluster_users(case)
