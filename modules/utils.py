import csv
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from modules.console import log
from modules.database import User, get_session

load_dotenv()

def load_usernames_from_csv(csv_path: str = os.environ(["DATA_CSV_PATH"])):
    with open(csv_path) as f:
        reader = csv.reader(f)
        urls = [row[0].strip() for row in reader if row]

    usernames = set()
    for url in urls:
        path = urlparse(url).path.strip("/")
        if path:
            # extract username from URL
            username = path.split("/")[0]
            if username:
                usernames.add(username)

    total = len(urls)
    unique = len(usernames)
    duplicates_in_csv = total - unique

    session = get_session()
    loaded = 0
    for username in sorted(usernames):
        if not session.query(User).filter_by(username=username).first():
            session.add(
                User(
                    pk=hash(username) & 0x7FFFFFFFFFFFFFFF,
                    username=username,
                    parse_status="pending",
                )
            )
            loaded += 1
    session.commit()
    already_in_db = unique - loaded
    log("database", f"loaded {loaded} usernames ({duplicates_in_csv} duplicates in csv, {already_in_db} already in db)")
    session.close()