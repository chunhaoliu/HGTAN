import pytest

from scripts.tune_temporal_hgtan import (
    CANDIDATE_SETS,
    aggregate_candidate,
    selection_score,
    validation_view,
)


def test_selection_score_rewards_final_and_temporal_quality():
    weaker = {
        "final_composite_f1": 0.85,
        "threat_temporal_macro_f1": 0.75,
        "urgency_temporal_macro_f1": 0.80,
        "threat_temporal_accuracy": 0.80,
        "urgency_temporal_accuracy": 0.85,
        "threat_mean_abs_ordinal_error": 0.20,
        "urgency_mean_abs_ordinal_error": 0.15,
    }
    stronger = {key: value + 0.02 for key, value in weaker.items()}
    stronger["threat_mean_abs_ordinal_error"] = 0.15
    stronger["urgency_mean_abs_ordinal_error"] = 0.10
    assert selection_score(stronger) > selection_score(weaker)


def test_validation_view_excludes_test_artifacts():
    bundle = {
        "train_loader": 1,
        "val_loader": 2,
        "test_loader": 3,
        "X_val": 4,
        "X_test": 5,
        "t_train_0": 6,
        "u_train_0": 7,
        "threat_seq_val": 8,
        "urgency_seq_val": 9,
        "threat_seq_test": 10,
    }
    view = validation_view(bundle)
    assert set(view) == {
        "train_loader",
        "val_loader",
        "X_val",
        "t_train_0",
        "u_train_0",
        "threat_seq_val",
        "urgency_seq_val",
    }
    assert all("test" not in key.lower() for key in view)


def test_aggregate_candidate_reports_mean_std_and_worst_seed():
    records = []
    for value in (0.80, 0.84, 0.88):
        metrics = {
            "selection_score": value,
            "final_composite_f1": value,
            "threat_final_f1": value,
            "urgency_final_f1": value,
            "threat_temporal_macro_f1": value,
            "urgency_temporal_macro_f1": value,
            "threat_temporal_accuracy": value,
            "urgency_temporal_accuracy": value,
            "threat_mean_abs_ordinal_error": 1.0 - value,
            "urgency_mean_abs_ordinal_error": 1.0 - value,
        }
        records.append({"metrics": metrics})
    summary = aggregate_candidate(records)
    assert summary["selection_score_mean"] == pytest.approx(0.84)
    assert summary["selection_score_min"] == pytest.approx(0.80)
    assert summary["selection_score_std"] > 0.0


def test_candidate_sets_have_unique_names():
    for candidates in CANDIDATE_SETS.values():
        names = [candidate["name"] for candidate in candidates]
        assert len(names) == len(set(names))

