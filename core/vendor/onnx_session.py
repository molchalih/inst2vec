"""onnxruntime session factory for the MIR GPU path.

Builds an InferenceSession that prefers CUDA and falls back to CPU, and
resolves output tensor names by their last dimension (e.g. 519 for MAEST
genre predictions, 1280 for the EffNet embedding) so node names never need
to be hardcoded across model revisions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_session(onnx_path: Path):
    """InferenceSession preferring CUDA, falling back to CPU.

    ``onnxruntime-gpu`` loads its CUDA/cuDNN provider from the pip-installed
    ``nvidia-*-cu12`` wheels; without ``preload_dlls()`` it cannot find
    ``libcudnn.so.9`` and silently runs on CPU. The call is best-effort —
    CPU-only builds (e.g. the macOS dev host) don't expose it.
    """
    import onnxruntime as ort

    try:
        ort.preload_dlls()
    except Exception:
        pass
    # EXHAUSTIVE cuDNN conv autotune picks the best kernel for the fixed
    # MAEST/EffNet patch shapes; the search cost is paid once at the first
    # forward. ``do_copy_in_default_stream=1`` keeps host↔device transfers on
    # the compute stream so they don't race with kernel launches. Both are
    # numerics-safe; the predictions are bit-stable across algos.
    provider_options: list[dict] = [
        {
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "1",
        },
        {},
    ]
    return ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=provider_options,
    )


def pick_output_by_lastdim(
    outputs: list[Any],
    last_dim: int,
    *,
    prefer: Callable[[Any], bool] | None = None,
) -> str:
    """Return the name of the output tensor whose last shape dim == last_dim.

    When several match (e.g. MAEST exposes both sigmoid predictions and linear
    logits at width 519), ``prefer`` breaks the tie; a matching ``prefer`` wins,
    else the first match is returned. Raises ValueError if none match.
    """
    matches = [o for o in outputs if list(o.shape) and o.shape[-1] == last_dim]
    if not matches:
        raise ValueError(f"no ONNX output with last dim {last_dim}")
    if prefer is not None:
        for o in matches:
            if prefer(o):
                return o.name
    return matches[0].name
