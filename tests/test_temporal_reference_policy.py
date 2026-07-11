from __future__ import annotations

import numpy as np

from data.reference_policy import reference_assessment_components


def _metadata(n_tracks: int) -> dict[str, np.ndarray]:
    return {
        "mission_type": np.full(n_tracks, 2),
        "target_type": np.full(n_tracks, 2),
        "formation_type": np.full(n_tracks, 2),
        "defense_state": np.full(n_tracks, 1),
        "environment_type": np.full(n_tracks, 1),
        "asset_type": np.full(n_tracks, 2),
    }


def _track(distance: np.ndarray) -> np.ndarray:
    steps = len(distance)
    sequence = np.full((1, steps, 16), 0.5, dtype=np.float64)
    sequence[:, :, 6] = 0.25
    sequence[:, :, 7] = 0.65
    sequence[:, :, 8] = distance
    sequence[:, :, 9] = 0.65
    sequence[:, :, 10] = 0.35
    sequence[:, :, 11] = distance
    return sequence


def test_temporal_policy_distinguishes_histories_with_same_final_geometry() -> None:
    approaching = _track(np.linspace(0.9, 0.3, 16))
    static = _track(np.full(16, 0.3))
    metadata = _metadata(1)

    approaching_components = reference_assessment_components(
        approaching, metadata, variant="temporal_balanced"
    )
    static_components = reference_assessment_components(static, metadata, variant="temporal_balanced")

    assert approaching_components["temporal_escalation"][0, -1] > 0.5
    assert static_components["temporal_escalation"][0, -1] < 0.1
    assert (
        approaching_components["temporal_escalation"][0, -1]
        > static_components["temporal_escalation"][0, -1] + 0.4
    )
    assert approaching_components["threat_score"][0, -1] > static_components["threat_score"][0, -1]


def test_original_balanced_policy_remains_framewise_compatible() -> None:
    approaching = _track(np.linspace(0.9, 0.3, 16))
    static = _track(np.full(16, 0.3))
    metadata = _metadata(1)

    approaching_score = reference_assessment_components(approaching, metadata, variant="balanced")[
        "threat_score"
    ][0, -1]
    static_score = reference_assessment_components(static, metadata, variant="balanced")["threat_score"][0, -1]
    np.testing.assert_allclose(approaching_score, static_score)
