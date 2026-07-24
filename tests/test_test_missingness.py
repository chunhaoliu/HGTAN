import numpy as np

from data.test_missingness import apply_frame_missingness, build_frame_missing_mask


def test_random_mask_is_exact_deterministic_and_preserves_first_frame():
    mask_a = build_frame_missing_mask(4, 64, ratio=0.15, mode="random", seed=123)
    mask_b = build_frame_missing_mask(4, 64, ratio=0.15, mode="random", seed=123)

    assert np.array_equal(mask_a, mask_b)
    assert not mask_a[:, 0].any()
    assert np.all(mask_a.sum(axis=1) == round(0.15 * 63))


def test_burst_mask_contains_one_contiguous_interval_per_track():
    mask = build_frame_missing_mask(5, 64, ratio=0.20, mode="burst", seed=456)

    assert not mask[:, 0].any()
    assert np.all(mask.sum(axis=1) == round(0.20 * 63))
    for row in mask:
        selected = np.flatnonzero(row)
        assert np.all(np.diff(selected) == 1)


def test_missing_frames_carry_forward_and_decay_confidence():
    sequences = np.zeros((1, 5, 16), dtype=np.float32)
    sequences[0, :, 8] = np.arange(5, dtype=np.float32)
    sequences[0, :, 15] = 1.0
    mask = np.asarray([[False, False, True, True, False]])

    adjusted = apply_frame_missingness(sequences, mask, confidence_decay=0.65)

    assert adjusted[0, 2, 8] == adjusted[0, 1, 8]
    assert adjusted[0, 3, 8] == adjusted[0, 2, 8]
    assert np.isclose(adjusted[0, 2, 15], 0.65)
    assert np.isclose(adjusted[0, 3, 15], 0.65**2)
    assert adjusted[0, 4, 8] == sequences[0, 4, 8]
