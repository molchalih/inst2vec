import csv
import os
import time
from urllib.parse import urlparse

from core.console import log
from core.database import User, allocate_user_identity, get_session


def load_usernames_from_csv(csv_path: str = "data/data.csv"):
    t0 = time.perf_counter()
    if not os.path.exists(csv_path):
        log("seed", "LOAD", csv_path, "none")
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
    log("seed", "LOAD", csv_path, "ok", stats={"rows": total, "unique": unique})

    session = get_session()
    loaded = 0
    for username in usernames:
        with allocate_user_identity(username) as user_id:
            if session.query(User).filter_by(id=user_id).first():
                continue
            session.add(User(id=user_id))
            session.commit()
            loaded += 1
    already_in_db = unique - loaded
    log(
        "seed",
        "WRITE",
        "users",
        "ok",
        stats={
            "new": loaded,
            "skipped_db": already_in_db,
            "duplicates_csv": total - unique,
            "time": time.perf_counter() - t0,
        },
    )
    session.close()
