from modules.database import Clip


def test_clip_has_is_uploaded_column():
    """is_uploaded should be a nullable bool, mirroring is_downloaded."""
    col = Clip.__table__.c["is_uploaded"]
    assert col.nullable is True
    assert col.type.python_type is bool


def test_clip_is_uploaded_defaults_to_none():
    """A fresh Clip row instantiated without is_uploaded should have None."""
    c = Clip(id=999_001, user_id=1)
    assert c.is_uploaded is None
