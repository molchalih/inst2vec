"""Engine helpers — robustness of the documented local run path."""

from sqlalchemy.orm import Session

from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import Comparison


def test_create_app_engine_creates_missing_sqlite_parent_dir(tmp_path) -> None:
    # The documented default url is sqlite:///data/swipe_anchor.db, and data/ is
    # gitignored/absent on a fresh checkout. Opening it must not fail.
    db_path = tmp_path / "data" / "nested" / "swipe_anchor.db"
    assert not db_path.parent.exists()

    engine = create_app_engine(f"sqlite:///{db_path}")
    with Session(engine) as s:
        s.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3))
        s.commit()

    assert db_path.exists()


def test_create_app_engine_handles_bare_path(tmp_path) -> None:
    db_path = tmp_path / "data2" / "app.db"
    engine = create_app_engine(str(db_path))  # bare path, no scheme
    with Session(engine) as s:
        s.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3))
        s.commit()
    assert db_path.exists()
