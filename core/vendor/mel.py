"""MusiCNN mel-spectrogram + numpy patch framing for the ONNX MIR path.

Essentia (CPU) produces the 96-band MusiCNN mel used by BOTH discogs-effnet
and discogs-maest; only the patch framing differs. These helpers keep the
mel extraction in Essentia and do the cheap framing in numpy so the framed
patches can be fed to onnxruntime on the GPU.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def musicnn_mel(audio: np.ndarray) -> np.ndarray:
    """Per-frame 96-band MusiCNN log-mel for 16 kHz mono audio -> (T, 96)."""
    import essentia

    essentia.log.warningActive = False  # ty: ignore[unresolved-attribute]
    essentia.log.infoActive = False  # ty: ignore[unresolved-attribute]
    from essentia.standard import TensorflowInputMusiCNN  # ty: ignore[unresolved-import]

    extractor = TensorflowInputMusiCNN()
    frames = [extractor(frame) for frame in _audio_frames(audio)]
    if not frames:
        return np.zeros((0, 96), dtype=np.float32)
    return np.asarray(frames, dtype=np.float32)


def _audio_frames(audio: np.ndarray, frame_size: int = 512, hop_size: int = 256):
    """Yield 16 kHz / hop-256 frames the way Essentia's MIR predictors do.

    ``startFromZero=False`` matches the centered FrameCutter that Essentia's
    TensorflowPredictMAEST / TensorflowPredictEffnetDiscogs use internally (it
    half-frame zero-pads the start). Using the default ``True`` shifts every
    frame and yields one fewer patch, which the MAEST transformer amplifies
    into completely different predictions — so this flag is load-bearing.
    """
    import essentia

    essentia.log.warningActive = False  # ty: ignore[unresolved-attribute]
    essentia.log.infoActive = False  # ty: ignore[unresolved-attribute]
    from essentia.standard import FrameGenerator  # ty: ignore[unresolved-import]

    yield from FrameGenerator(
        audio.astype(np.float32), frameSize=frame_size, hopSize=hop_size,
        startFromZero=False,
    )


def tile_to_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Loop 1-D ``audio`` up to ``n`` samples (no-op when already >= ``n``).

    Mirrors the old Essentia MAEST wrapper's short-clip rule: clips shorter
    than the model's window are repeated to fill it before mel extraction.
    """
    if audio.size == 0:
        return np.zeros(n, dtype=audio.dtype)
    if audio.size >= n:
        return audio
    reps = -(-n // audio.size)
    return np.tile(audio, reps)[:n].astype(audio.dtype, copy=False)


def pad_or_tile_frames(mel: np.ndarray, n: int) -> np.ndarray:
    """Loop mel frames to fill exactly ``n`` frames (zeros for empty input)."""
    if mel.shape[0] == 0:
        return np.zeros((n, mel.shape[1] if mel.ndim == 2 else 96), dtype=np.float32)
    reps = -(-n // mel.shape[0])
    return np.tile(mel, (reps, 1))[:n].astype(np.float32, copy=False)


def frame_patches(mel: np.ndarray, *, patch_size: int, hop_size: int) -> np.ndarray:
    """Slice (T, 96) mel into (n_patches, patch_size, 96).

    Short input (T < patch_size) is tiled up to exactly one patch so callers
    always get at least one window. Trailing frames that do not fill a whole
    patch are dropped (matches Essentia's discard-remainder patching).
    """
    if mel.shape[0] < patch_size:
        return pad_or_tile_frames(mel, patch_size)[None, :, :]
    starts = range(0, mel.shape[0] - patch_size + 1, hop_size)
    return np.stack([mel[s : s + patch_size] for s in starts]).astype(np.float32)
