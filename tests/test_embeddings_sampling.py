from modules.embeddings.sampling import (
    frame_retry_schedule,
    is_token_mismatch_error,
    probe_duration_seconds,
)


def test_frame_retry_schedule_descends_and_dedupes():
    assert frame_retry_schedule(64) == [64, 48, 32, 24, 16]


def test_frame_retry_schedule_caps_below_initial():
    # Anything above the initial cap is filtered out.
    result = frame_retry_schedule(32)
    assert result[0] == 32
    assert all(c <= 32 for c in result)
    assert len(set(result)) == len(result)


def test_frame_retry_schedule_small_initial():
    assert frame_retry_schedule(16) == [16]


def test_is_token_mismatch_error_matches_video_token_count():
    exc = RuntimeError("Mismatch in `video` token count for batch item")
    assert is_token_mismatch_error(exc) is True


def test_is_token_mismatch_error_matches_truncation_hint():
    exc = RuntimeError("Likely due to `truncation='max_length'`")
    assert is_token_mismatch_error(exc) is True


def test_is_token_mismatch_error_ignores_other_errors():
    assert is_token_mismatch_error(RuntimeError("OOM")) is False


def test_probe_duration_seconds_missing_file_returns_none():
    # ffprobe will exit non-zero on a nonexistent file.
    assert probe_duration_seconds("/nonexistent/path/file.mp4") is None
