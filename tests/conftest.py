import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("IDENTITY_DB_URL", "sqlite:///:memory:")
