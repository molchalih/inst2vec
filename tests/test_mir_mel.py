import numpy as np

from core.vendor.mel import frame_patches, pad_or_tile_frames, tile_to_length


def test_frame_patches_splits_nonoverlapping():
    mel = np.arange(5 * 96, dtype=np.float32).reshape(5, 96)
    patches = frame_patches(mel, patch_size=2, hop_size=2)
    assert patches.shape == (
        2,
        2,
        96,
    )  # 5 frames -> two full patches, remainder dropped
    np.testing.assert_array_equal(patches[0], mel[0:2])
    np.testing.assert_array_equal(patches[1], mel[2:4])


def test_frame_patches_pads_short_input_to_one_patch():
    mel = np.ones((1, 96), dtype=np.float32)
    patches = frame_patches(mel, patch_size=4, hop_size=4)
    assert patches.shape == (1, 4, 96)
    np.testing.assert_array_equal(patches[0, 0], np.ones(96))  # first frame preserved


def test_pad_or_tile_frames_tiles_to_exact_length():
    mel = np.stack([np.full(96, i, dtype=np.float32) for i in range(3)])
    out = pad_or_tile_frames(mel, 7)
    assert out.shape == (7, 96)
    # tiled: frames 0,1,2,0,1,2,0
    np.testing.assert_array_equal(
        out[:, 0], np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.float32)
    )


def test_tile_to_length_loops_short_and_passes_long_through():
    short = np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(
        tile_to_length(short, 7), np.array([1, 2, 3, 1, 2, 3, 1], dtype=np.float32)
    )
    long = np.arange(10, dtype=np.float32)
    assert tile_to_length(long, 5) is long  # already long enough -> returned as-is
