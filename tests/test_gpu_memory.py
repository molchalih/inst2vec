import logging

import pytest

from core.gpu_memory import (
    VramStats,
    _format_oom,
    log_vram,
    oom_guard,
    reset_peak,
    vram_fields,
    vram_stats,
)


def test_vram_stats_is_none_without_cuda():
    # No CUDA on this host → stats unavailable.
    assert vram_stats() is None


def test_vram_fields_is_none_without_cuda():
    # No CUDA → no fields to emit; callers skip the structured log line.
    assert vram_fields(at="point") is None


def test_log_vram_and_reset_peak_are_safe_noops_without_cuda():
    # Must not raise when CUDA is unavailable.
    log_vram("test point")
    reset_peak()


def test_oom_guard_passes_through_when_no_error():
    ran = []
    with oom_guard("unit"):
        ran.append(1)
    assert ran == [1]


def test_oom_guard_propagates_non_oom_errors_unchanged():
    with pytest.raises(ValueError), oom_guard("unit"):
        raise ValueError("not an oom")


def test_oom_guard_reraises_cuda_oom_and_logs(caplog):
    import torch

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(torch.cuda.OutOfMemoryError),
        oom_guard("cluster pass"),
    ):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert any("cluster pass" in r.getMessage() for r in caplog.records)


def test_format_oom_is_pure_and_handles_none():
    assert "cluster pass" in _format_oom("cluster pass", None)
    stats = VramStats(
        device_used_gb=18.0,
        free_gb=6.0,
        total_gb=24.0,
        proc_alloc_gb=0.0,
        proc_peak_gb=0.0,
    )
    msg = _format_oom("cluster pass", stats)
    assert "cluster pass" in msg and "24.0" in msg
