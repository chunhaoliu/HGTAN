from __future__ import annotations

import numpy as np

from data.sequence_pipeline import select_observation_window
from exp.registry import apply_assessment_setting, get_suite_settings
from utils.config import HGTANConfig


def test_tail_window_keeps_common_terminal_labels() -> None:
    sequences = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    threat_seq = np.tile(np.arange(1, 7, dtype=np.int64), (2, 1))
    urgency_seq = np.tile(np.array([1, 1, 2, 2, 3, 3]), (2, 1))

    observed, threat, urgency, start, end = select_observation_window(
        sequences,
        threat_seq,
        urgency_seq,
        observed_len=3,
        observation_window="tail",
    )

    np.testing.assert_array_equal(observed, sequences[:, 3:6, :])
    np.testing.assert_array_equal(threat, threat_seq[:, 3:6])
    np.testing.assert_array_equal(urgency, urgency_seq[:, 3:6])
    np.testing.assert_array_equal(threat[:, -1], threat_seq[:, -1])
    np.testing.assert_array_equal(urgency[:, -1], urgency_seq[:, -1])
    assert (start, end) == (3, 6)


def test_prefix_window_retains_legacy_behavior() -> None:
    sequences = np.arange(1 * 5 * 2, dtype=np.float32).reshape(1, 5, 2)
    threat_seq = np.arange(1, 6, dtype=np.int64).reshape(1, 5)
    urgency_seq = np.array([[1, 1, 2, 2, 3]], dtype=np.int64)

    observed, threat, urgency, start, end = select_observation_window(
        sequences,
        threat_seq,
        urgency_seq,
        observed_len=2,
        observation_window="prefix",
    )

    np.testing.assert_array_equal(observed, sequences[:, :2, :])
    np.testing.assert_array_equal(threat, threat_seq[:, :2])
    np.testing.assert_array_equal(urgency, urgency_seq[:, :2])
    assert (start, end) == (0, 2)


def test_fixed_endpoint_suite_propagates_tail_window_to_config() -> None:
    settings = get_suite_settings("fixed_endpoint_observed_time")
    assert [setting["observed_len"] for setting in settings] == [32, 64, 96, 128]
    assert all(setting["seq_len"] == 128 for setting in settings)
    assert all(setting["observation_window"] == "tail" for setting in settings)

    config = HGTANConfig.get_config()
    updated = apply_assessment_setting(config, settings[0])
    assert updated["sequence"]["observation_window"] == "tail"
    assert updated["sequence"]["seq_len"] == 128
    assert updated["sequence"]["observed_len"] == 32
