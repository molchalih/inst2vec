from modules.database import init_db, load_usernames_from_csv
from modules.parse import fetch_profiles
from modules.download import download_files
from modules.music import classify_music, extract_music_features
from modules.speech import classify_speech, translate_speech
# from modules.embeddings import embed_clips

init_db()
load_usernames_from_csv()
fetch_profiles()
download_files()

# Phase 1 – music
classify_music()
extract_music_features()

# Phase 2 – speech
classify_speech()
translate_speech()

# embed_clips()
