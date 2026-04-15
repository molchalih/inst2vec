from modules.database import init_db, load_usernames_from_csv
from modules.parser import fetch_profiles
from modules.downloader import download_files
from modules.embedder import embed_clips
from modules.audio_processor import process_clip_audio, process_music_metadata

init_db()
load_usernames_from_csv()
fetch_profiles()
download_files()
process_clip_audio()
process_music_metadata()
# embed_clips()