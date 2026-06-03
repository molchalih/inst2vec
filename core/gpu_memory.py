"""CUDA VRAM monitoring utilities for the labels pipeline.

Provides lightweight helpers to log VRAM at critical model-lifecycle transitions
(load / unload / inference) and to convert opaque CUDA OOM errors into
diagnostics carrying live allocation/peak/free numbers.

CPU-safe contract: every public function is a no-op (or pass-through) when
CUDA is unavailable so the module is safe to import and call on dev machines
and in CI without a GPU.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)


def cuda_available() -> bool:
    """Return True when a CUDA-capable GPU is present and torch reports it."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass(frozen=True)
class VramStats:
    """Snapshot of CUDA VRAM figures, all in GB.

    ``device_used_gb`` / ``free_gb`` / ``total_gb`` are DEVICE-WIDE (from
    ``cudaMemGetInfo``), so they include memory held by other processes — in
    particular vLLM's worker subprocesses, which the calling process's own
    torch counters (``proc_*``) cannot see. ``proc_*`` reflect only this
    process's torch allocator (meaningful for the in-process VL clip pass).
    """

    device_used_gb: float
    free_gb: float
    total_gb: float
    proc_alloc_gb: float
    proc_peak_gb: float

    def __str__(self) -> str:
        return (
            f"device_used={self.device_used_gb:.1f}/{self.total_gb:.1f} "
            f"free={self.free_gb:.1f} "
            f"proc_alloc={self.proc_alloc_gb:.1f} "
            f"proc_peak={self.proc_peak_gb:.1f} GB"
        )


def vram_stats() -> VramStats | None:
    """Return a VRAM snapshot, or None when CUDA is unavailable."""
    if not cuda_available():
        return None
    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    return VramStats(
        device_used_gb=(total_b - free_b) / 1e9,
        free_gb=free_b / 1e9,
        total_gb=total_b / 1e9,
        proc_alloc_gb=torch.cuda.memory_allocated() / 1e9,
        proc_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def reset_peak() -> None:
    """Reset the CUDA peak-memory counter; no-op when CUDA is unavailable."""
    if not cuda_available():
        return
    import torch

    torch.cuda.reset_peak_memory_stats()


def log_vram(label: str) -> None:
    """Log a single VRAM info line; silent no-op when CUDA is unavailable."""
    stats = vram_stats()
    if stats is None:
        return
    _log.info("[vram] %s: %s", label, stats)


def vram_fields(**extra: object) -> dict[str, object] | None:
    """Return rounded VRAM figures as a flat dict for structured logging.

    Suitable as ``event(..., stats=vram_fields(at="after-unload"))``. Returns
    ``None`` when CUDA is unavailable so callers can skip emitting a line.
    ``extra`` keys (e.g. ``at=``) are merged in first so they lead the stats.
    """
    stats = vram_stats()
    if stats is None:
        return None
    return {
        **extra,
        "used_gb": round(stats.device_used_gb, 1),
        "free_gb": round(stats.free_gb, 1),
        "total_gb": round(stats.total_gb, 1),
        "proc_peak_gb": round(stats.proc_peak_gb, 1),
    }


def _format_oom(label: str, stats: VramStats | None) -> str:
    """Return a diagnostic string for a CUDA OOM event (pure function)."""
    msg = f"CUDA OOM during {label}"
    if stats is not None:
        msg += f" — {stats}"
    return msg


@contextlib.contextmanager
def oom_guard(label: str):  # type: ignore[return]
    """Context manager that enriches CUDA OOM errors with VRAM diagnostics.

    On entry: logs VRAM (no-op without CUDA).
    On exit (success or error): logs VRAM (no-op without CUDA).
    On CUDA OOM or RuntimeError containing "out of memory": logs an ERROR with
    live VRAM figures via stdlib logging so test caplog captures it, then
    re-raises the original exception unchanged.
    """
    log_vram(f"{label} (enter)")
    try:
        yield
    except Exception as exc:
        import torch

        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
        if is_oom:
            _log.error(_format_oom(label, vram_stats()))
        raise
    finally:
        log_vram(f"{label} (exit)")
