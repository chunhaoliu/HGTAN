import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path

from models.model_factory import build_model
from scripts.paper_assets import DEFAULT_PAPER_TAG, selected_figure_stems
from scripts.export_dataset_statistics import count_values, stream_statistics
from scripts import make_taes_figures
from scripts.make_taes_tables import fmt, make_comparison_table, metric
from utils.config import ALL_FEATURES, HGTANConfig
from utils.metrics import count_parameters


def test_table_metric_uses_sample_standard_deviation() -> None:
    summary = pd.DataFrame(
        [
            {
                "source_suite": "r3_comparison_formal_c5_s3",
                "setting": "ATUAV-Core__latent_state_masked",
                "model": "TemporalHGTAN",
                "task": "joint",
                "metric": "composite_f1",
                "mean": 0.8883,
                "std": 0.0041,
                "ci95": 0.0047,
                "n": 3,
            }
        ]
    )
    stat = metric(
        summary,
        "r3_comparison_formal_c5_s3",
        "ATUAV-Core__latent_state_masked",
        "TemporalHGTAN",
        "joint",
        "composite_f1",
    )
    assert stat == {"mean": 0.8883, "std": 0.0041, "n": 3.0}
    assert fmt(stat) == "88.83$\\pm$0.41"


def test_comparison_table_reports_final_and_temporal_fidelity() -> None:
    rows = []
    for task, metric_name, mean, std in [
        ("threat", "f1", 0.8803, 0.0090),
        ("urgency", "f1", 0.9123, 0.0107),
        ("joint", "composite_f1", 0.8883, 0.0041),
        ("threat_track", "temporal_accuracy", 0.8607, 0.0155),
        ("threat_track", "temporal_macro_f1", 0.8414, 0.0182),
        ("threat_track", "mean_abs_ordinal_error", 0.139, 0.016),
    ]:
        rows.append(
            {
                "source_suite": "r3_comparison_formal_c5_s3",
                "setting": "ATUAV-Core__latent_state_masked",
                "model": "TemporalHGTAN",
                "task": task,
                "metric": metric_name,
                "mean": mean,
                "std": std,
                "n": 3,
            }
        )
    table = make_comparison_table(pd.DataFrame(rows))
    assert table is not None
    assert "T-Macro-F1" in table
    assert "Ord. MAE" in table
    assert "0.139$\\pm$0.016" in table
    assert "Crit. Miss" not in table


def test_paper_model_parameter_count_is_frozen() -> None:
    config = HGTANConfig.get_model_config("hgtan")
    config["dropout"] = 0.08
    config["prior_weight_alpha"] = 0.1
    assert count_parameters(build_model("TemporalHGTAN", config)) == 717173


def test_dataset_statistics_helpers_are_stable() -> None:
    values = np.arange(32, dtype=np.float32).reshape(1, 2, 16) / 31.0
    stats = stream_statistics(values)
    assert list(stats) == list(ALL_FEATURES)
    assert stats[ALL_FEATURES[0]]["min"] == 0.0
    assert count_values(np.array([2, 1, 2])) == {"1": 1, "2": 2}


def test_protocol_case_fallback_masks_privileged_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    class ProtocolCaptured(Exception):
        pass

    def capture_protocol(*args, sequence_cfg, **kwargs):
        assert sequence_cfg["type_as_input"] is False
        assert sequence_cfg["mission_as_input"] is False
        assert sequence_cfg["reference_policy_variant"] == "balanced"
        raise ProtocolCaptured

    monkeypatch.setattr(make_taes_figures, "generate_uav_track_sequences", capture_protocol)
    with pytest.raises(ProtocolCaptured):
        make_taes_figures.generate_protocol_case()


def test_fixed_endpoint_evidence_is_the_default_paper_bundle() -> None:
    config_path = Path("configs/paper/taes_r1_c5.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    suites = config["formal_suites"]
    figures = selected_figure_stems("main")

    assert DEFAULT_PAPER_TAG == "r3_c5_fixed_endpoint"
    assert suites["observed_time"] == "fixed_endpoint_window_formal_s3"
    assert suites["fixed_endpoint_ablation"] == "fixed_endpoint_ablation_obs32_formal_s3"
    assert "fig_ablation_fixed_summary_composite_f1" in figures
    assert "fig_ablation_fixed_summary_temporal_f1" in figures
    assert "fig_ablation_default_summary_composite_f1" in figures
    assert "fig_ablation_default_summary_temporal_f1" in figures
    assert "fig_policy_holdout_margins" not in figures
    assert "fig_ablation_short_composite_f1" not in figures


def test_classwise_f1_keeps_ordinal_levels_separate() -> None:
    rows = make_taes_figures.per_class_f1(
        np.array([1, 1, 2, 2, 3]),
        np.array([1, 2, 2, 2, 1]),
        n_classes=3,
    )
    assert rows[0] == (1, 2, 0.5)
    assert rows[1] == (2, 2, 0.8)
    assert rows[2] == (3, 1, 0.0)


def test_first_critical_frames_marks_missing_events() -> None:
    frames = make_taes_figures.first_critical_frames(
        np.array(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 3, 2],
                [4, 4, 5, 5, 5],
            ]
        )
    )
    np.testing.assert_array_equal(frames, np.array([3, -1, 0]))


def test_representative_transition_case_uses_full_pool_median_with_timing_tie_break() -> None:
    threat_true = np.tile(np.array([1, 2, 4, 4]), (5, 1))
    urgency_true = np.tile(np.array([1, 1, 2, 2]), (5, 1))
    hgtan_pred = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 4, 4],
            [1, 2, 3, 4],
            [1, 1, 3, 4],
            [1, 2, 4, 4],
        ]
    )
    gru_pred = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 1, 3, 4],
            [1, 1, 2, 4],
            [1, 1, 2, 3],
        ]
    )
    predictions = {}
    for model, threat_pred in [
        ("TOPSIS", gru_pred),
        ("TemporalHMM", gru_pred),
        ("TemporalGRU", gru_pred),
        ("TemporalHGTAN", hgtan_pred),
    ]:
        predictions[model] = {
            "threat_seq_true": threat_true.tolist(),
            "threat_seq_pred": threat_pred.tolist(),
            "urgency_seq_true": urgency_true.tolist(),
            "urgency_seq_pred": urgency_true.tolist(),
        }
    record = {
        "seed": 42,
        "predictions": predictions,
        "data_profile": [{"frame_interval": 0.2}],
    }

    selected = make_taes_figures.select_representative_transition_case([record])

    assert selected is not None
    assert selected["selection"]["candidate_tracks"] == 5
    assert selected["selection"]["track_index"] == 1
    assert selected["selection"]["median_gru_minus_hgtan_mae"] == pytest.approx(0.25)
    assert selected["selection"]["hgtan_timing_error_frames"] == 0
