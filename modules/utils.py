import csv
import os
from urllib.parse import urlparse

from modules.console import log
from modules.database import User, get_or_create_user_identity, get_session


def load_usernames_from_csv(csv_path: str = "data/data.csv"):
    if not os.path.exists(csv_path):
        log("database", f"data CSV not found at {csv_path!r} — skipping import")
        return

    with open(csv_path) as f:
        reader = csv.reader(f)
        urls = [row[0].strip() for row in reader if row]

    usernames: set[str] = set()
    for url in urls:
        path = urlparse(url).path.strip("/")
        if path:
            username = path.split("/")[0]
            if username:
                usernames.add(username)

    total = len(urls)
    unique = len(usernames)
    duplicates_in_csv = total - unique

    session = get_session()
    loaded = 0
    for username in usernames:
        user_id = get_or_create_user_identity(username)
        if not session.query(User).filter_by(id=user_id).first():
            session.add(User(id=user_id))
            loaded += 1
    session.commit()
    already_in_db = unique - loaded
    log(
        "database",
        f"loaded {loaded} usernames ({duplicates_in_csv} duplicates in csv, {already_in_db} already in db)",
    )
    session.close()
