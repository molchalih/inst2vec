from modules.database import init_db, load_usernames_from_csv
from modules.parser import fetch_profiles
from modules.downloader import download_files
from modules.gatherer import gather_info
from modules.embedder import embed_clips

init_db()
load_usernames_from_csv()
fetch_profiles()
download_files()
gather_info()
# embed_clips()