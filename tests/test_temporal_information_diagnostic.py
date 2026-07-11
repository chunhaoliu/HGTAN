from __future__ import annotations

import numpy as np

from scripts.diagnose_temporal_information import _shuffle_history


def test_shuffle_history_preserves_final_frame_and_frame_multiset() -> None:
    values = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    shuffled = _shuffle_history(values, seed=17)

    np.testing.assert_array_equal(shuffled[:, -1, :], values[:, -1, :])
    for row in range(len(values)):
        original = sorted(map(tuple, values[row, :-1, :]))
        changed = sorted(map(tuple, shuffled[row, :-1, :]))
        assert changed == original
    assert not np.array_equal(shuffled[:, :-1, :], values[:, :-1, :])
