import numpy as np

from modules.embeddings.users import aggregate_user_embeddings_from_rows
from modules.embeddings.vectors import bytes_to_array


def _make_blob(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def test_aggregate_single_clip_per_user():
    rows = [
        (_make_blob([1.0, 2.0, 3.0]), 101),
        (_make_blob([4.0, 5.0, 6.0]), 102),
    ]
    result = aggregate_user_embeddings_from_rows(rows)
    assert set(result.keys()) == {101, 102}
    np.testing.assert_array_almost_equal(bytes_to_array(result[101]), [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(bytes_to_array(result[102]), [4.0, 5.0, 6.0])


def test_aggregate_mean_of_multiple_clips():
    rows = [
        (_make_blob([1.0, 3.0]), 101),
        (_make_blob([3.0, 1.0]), 101),
        (_make_blob([0.0, 0.0]), 101),
    ]
    result = aggregate_user_embeddings_from_rows(rows)
    np.testing.assert_array_almost_equal(
        bytes_to_array(result[101]), [4.0 / 3.0, 4.0 / 3.0]
    )


def test_aggregate_output_dtype_is_float32():
    rows = [(_make_blob([1.0, 2.0]), 101)]
    result = aggregate_user_embeddings_from_rows(rows)
    assert bytes_to_array(result[101]).dtype == np.float32


def test_aggregate_empty_rows_returns_empty_dict():
    assert aggregate_user_embeddings_from_rows([]) == {}


def test_aggregate_mixed_users():
    rows = [
        (_make_blob([2.0, 4.0]), 1),
        (_make_blob([0.0, 0.0]), 1),
        (_make_blob([10.0, 10.0]), 2),
    ]
    result = aggregate_user_embeddings_from_rows(rows)
    assert set(result.keys()) == {1, 2}
    np.testing.assert_array_almost_equal(bytes_to_array(result[1]), [1.0, 2.0])
    np.testing.assert_array_almost_equal(bytes_to_array(result[2]), [10.0, 10.0])
