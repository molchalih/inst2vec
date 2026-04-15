import csv
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, BigInteger, Integer, String, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship, Session

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"])
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    pk = Column(BigInteger, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String)
    profile_pic_url = Column(String)
    profile_pic_url_hd = Column(String)
    following_count = Column(Integer)
    city_name = Column(String)

    clips = relationship("Clip", back_populates="user")


class Clip(Base):
    __tablename__ = "clips"

    pk = Column(BigInteger, primary_key=True)
    user_pk = Column(BigInteger, ForeignKey("users.pk"), nullable=False)
    thumbnail_url = Column(String)
    video_url = Column(String)
    caption_text = Column(Text)
    caption_translation = Column(Text)
    has_audio = Column(Boolean)
    comment_count = Column(Integer)
    reshare_count = Column(Integer)
    like_count = Column(Integer)
    play_count = Column(Integer)

    user = relationship("User", back_populates="clips")


def init_db():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)


def load_usernames_from_csv(csv_path: str = "data/data.csv"):
    with open(csv_path) as f:
        reader = csv.reader(f)
        urls = [row[0].strip() for row in reader if row]

    usernames = set()
    for url in urls:
        path = urlparse(url).path.strip("/")
        if path:
            # handle trailing /reels/ etc — take first segment
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
            session.add(User(pk=hash(username) & 0x7FFFFFFFFFFFFFFF, username=username))
            loaded += 1
    session.commit()
    already_in_db = unique - loaded
    print(f"Loaded {loaded} usernames ({duplicates_in_csv} duplicates in csv, {already_in_db} already in db)")
    session.close()
