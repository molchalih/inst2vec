import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
from modules.console import log, phase, progress, startup


def test_log_info_does_not_raise():
    log("test", "hello world")


def test_log_all_levels_do_not_raise():
    for level in ("info", "ok", "warn", "err"):
        log("test", f"level={level}", level=level)


def test_phase_does_not_raise():
    phase("Test Phase")


def test_startup_does_not_raise():
    startup("data/inst2vec.db")


def test_progress_advances_to_completion():
    with progress(3, "Testing") as advance:
        advance(detail="item 1")
        advance(detail="item 2")
        advance()
