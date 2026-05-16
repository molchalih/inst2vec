import numpy as np
import pytest

from modules.embeddings.vectors import bytes_to_array, to_bytes


def test_bytes_to_array_roundtrip():
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    blob = arr.tobytes()
    result = bytes_to_array(blob)
    np.testing.assert_array_almost_equal(result, arr)


def test_bytes_to_array_dtype_is_float32():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = bytes_to_array(arr.tobytes())
    assert result.dtype == np.float32


def test_bytes_to_array_returns_copy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    blob = arr.tobytes()
    result = bytes_to_array(blob)
    result[0] = 99.0
    result2 = bytes_to_array(blob)
    assert result2[0] == pytest.approx(1.0)


def test_to_bytes_accepts_torch_tensor_like():
    """to_bytes should accept any object with .cpu().float().numpy() chain."""

    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self._arr

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    blob = to_bytes(FakeTensor(arr))
    assert bytes_to_array(blob).tolist() == [1.0, 2.0, 3.0]
