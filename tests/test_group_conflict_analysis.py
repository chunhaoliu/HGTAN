import numpy as np
import pytest

from scripts.analyze_group_conflict import build_strata, compute_group_conflict_scores, weighted_composite_f1


def test_group_conflict_ignores_masked_oracle_features():
    sequences = np.full((2, 8, 16), 0.5, dtype=np.float32)
    sequences[0, :, 0] = 0.0
    sequences[0, :, 4] = 0.0
    sequences[1, :, 0] = 1.0
    sequences[1, :, 4] = 1.0

    group_scores, conflict = compute_group_conflict_scores(sequences)

    np.testing.assert_allclose(group_scores[0], group_scores[1])
    np.testing.assert_allclose(conflict[0], conflict[1])


def test_group_conflict_orients_low_risk_features_before_aggregation():
    sequences = np.full((1, 8, 16), 0.5, dtype=np.float32)
    sequences[:, :, 8] = 0.0  # low distance means high risk
    sequences[:, :, 9] = 1.0
    sequences[:, :, 10] = 0.0  # low altitude means high risk
    sequences[:, :, 11] = 0.0  # low time-to-arrival means high risk

    group_scores, conflict = compute_group_conflict_scores(sequences)

    assert group_scores[0, 2] == 1.0
    assert conflict[0] == 0.5


def test_high_conflict_stratum_uses_requested_quantile():
    threshold, strata = build_strata(np.arange(8, dtype=float), 0.75)

    assert threshold == 5.25
    assert strata["high_conflict"].sum() == 2
    assert np.array_equal(strata["lower_conflict"], ~strata["high_conflict"])


def test_composite_f1_matches_paper_weighting():
    assert weighted_composite_f1(0.8, 0.4) == pytest.approx(0.7)
