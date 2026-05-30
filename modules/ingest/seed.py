import csv
import os
from urllib.parse import urlparse

from core.config import Secrets, Settings
from core.database import User, allocate_user_identity, get_session
from core.log import StageResult, event, stage


def load_usernames_from_csv(csv_path: str = "data/data.csv"):
    if not os.path.exists(csv_path):
        return None

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

    session = get_session()
    loaded = 0
    for username in usernames:
        with allocate_user_identity(username) as user_id:
            if session.query(User).filter_by(id=user_id).first():
                continue
            session.add(User(id=user_id))
            session.commit()
            loaded += 1
    session.close()
    return total, unique, loaded


@stage("seed")
def run_seed(settings: Settings, secrets: Secrets) -> StageResult | None:
    csv_path = settings.paths.data_csv_path
    if not os.path.exists(csv_path):
        event("LOAD", str(csv_path), stats={"rows": 0})
        return None
    result = load_usernames_from_csv(csv_path=str(csv_path))
    if result is None:
        event("LOAD", str(csv_path), stats={"rows": 0})
        return None
    total, unique, loaded = result
    event("LOAD", str(csv_path), stats={"rows": total, "unique": unique})
    already_in_db = unique - loaded
    event(
        "WRITE",
        "users",
        stats={
            "new": loaded,
            "skipped_db": already_in_db,
            "duplicates_csv": total - unique,
        },
    )
    return StageResult(loaded=total, unique=unique)
