"""Shape- and tiling-handling tests for the MAEST vendor wrapper."""

from __future__ import annotations

import numpy as np

from core.vendor.maest import MAEST, _tile_to_length


def _stub(predict_out: np.ndarray, *, min_samples: int = 1) -> MAEST:
    m = MAEST.__new__(MAEST)
    m._predict = lambda audio: predict_out
    m._min_samples = min_samples
    return m


def test_predict_reduces_savedmodel_shape_to_1d():
    n, k = 5, 519
    out = np.arange(n * k, dtype=np.float32).reshape(n, 1, 1, k)
    result = _stub(out).predict(np.zeros(16, dtype=np.float32))
    assert result.shape == (k,)
    np.testing.assert_allclose(result, out.reshape(n, k).mean(axis=0))


def test_predict_reduces_frozen_graph_shape_to_1d():
    n, k = 3, 519
    out = np.arange(n * k, dtype=np.float32).reshape(n, k)
    result = _stub(out).predict(np.zeros(16, dtype=np.float32))
    assert result.shape == (k,)
    np.testing.assert_allclose(result, out.mean(axis=0))


def test_predict_tiles_short_audio_to_min_samples():
    captured: list[np.ndarray] = []
    m = MAEST.__new__(MAEST)
    m._min_samples = 480_000

    def _capture(audio):
        captured.append(audio)
        return np.zeros((1, 519), dtype=np.float32)

    m._predict = _capture
    short = np.ones(160_000, dtype=np.float32)
    m.predict(short)
    assert captured[0].size == 480_000
    np.testing.assert_array_equal(captured[0][:160_000], short)
    np.testing.assert_array_equal(captured[0][160_000:320_000], short)


def test_predict_passes_long_audio_unchanged():
    captured: list[np.ndarray] = []
    m = MAEST.__new__(MAEST)
    m._min_samples = 480_000

    def _capture(audio):
        captured.append(audio)
        return np.zeros((2, 519), dtype=np.float32)

    m._predict = _capture
    long = np.ones(960_000, dtype=np.float32)
    m.predict(long)
    assert captured[0] is long


def test_tile_to_length_handles_empty_audio():
    out = _tile_to_length(np.empty(0, dtype=np.float32), 100)
    assert out.shape == (100,)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.zeros(100, dtype=np.float32))


def test_tile_to_length_loops_exactly():
    src = np.array([1, 2, 3], dtype=np.float32)
    out = _tile_to_length(src, 7)
    np.testing.assert_array_equal(
        out, np.array([1, 2, 3, 1, 2, 3, 1], dtype=np.float32)
    )
