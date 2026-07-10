import numpy as np
from data.reference_policy import REFERENCE_POLICY_NAME, build_reference_assessment_sequences
from data.sequence_generator import generate_uav_track_payload
from data.pipeline_common import build_split_indices
from exp.exp_main import load_seed_checkpoint
from exp.result_writer import build_training_rows, setting_context, write_json_gzip


def _metadata() -> dict[str, np.ndarray]:
    return {
        "mission_type": np.array([0, 3]),
        "target_type": np.array([0, 3]),
        "formation_type": np.array([0, 3]),
        "defense_state": np.array([0, 3]),
        "environment_type": np.array([0, 3]),
        "asset_type": np.array([0, 3]),
    }


def test_reference_policy_uses_latent_state_and_clean_geometry():
    clean = np.full((2, 4, 16), 0.5, dtype=np.float64)
    clean[:, :, 6] = 0.1
    clean[:, :, 8] = 0.1
    clean[:, :, 9] = 0.9
    clean[:, :, 10] = 0.1
    clean[:, :, 11] = 0.1

    threat, urgency, components = build_reference_assessment_sequences(clean, _metadata())

    assert threat.shape == (2, 4)
    assert urgency.shape == (2, 4)
    assert np.all(components["threat_score"][1] > components["threat_score"][0])
    assert np.all(components["urgency_score"][1] > components["urgency_score"][0])


def test_default_sequence_protocol_masks_latent_target_and_mission_codes():
    payload = generate_uav_track_payload(n_tracks=48, seq_len=16, seed=17)

    assert np.all(payload["sequences"][:, :, 0] == 0.0)
    assert np.all(payload["sequences"][:, :, 4] == 0.0)
    assert payload["metadata"]["reference_policy"] == REFERENCE_POLICY_NAME
    assert "threat_risk" not in payload["metadata"]
    assert payload["threat_seq"].shape == (48, 16)
    assert payload["urgency_seq"].shape == (48, 16)


def test_sequence_protocol_can_explicitly_restore_mission_code_for_ablation_only():
    payload = generate_uav_track_payload(n_tracks=48, seq_len=16, seed=17, mission_as_input=True)

    assert np.any(payload["sequences"][:, :, 4] > 0.0)
    assert np.all(payload["sequences"][:, :, 0] == 0.0)


def test_reference_policy_variants_change_reference_scores_without_changing_shapes():
    clean = np.full((2, 4, 16), 0.5, dtype=np.float64)
    _, _, balanced = build_reference_assessment_sequences(clean, _metadata(), variant="balanced")
    _, _, access_first = build_reference_assessment_sequences(clean, _metadata(), variant="access_first")

    assert balanced["threat_score"].shape == access_first["threat_score"].shape == (2, 4)
    assert not np.allclose(balanced["threat_score"], access_first["threat_score"])


def test_fixed_scenario_holdout_excludes_the_named_family_from_training():
    families = np.repeat(np.array(["Probe_Surveillance", "EW_Contested", "Strike_Penetration", "Saturation_Overload"]), 12)
    labels = np.tile(np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3]), 4)
    urgency = np.tile(np.array([1, 2, 3]), 16)
    train_idx, val_idx, test_idx = build_split_indices(
        threat_labels=labels,
        urgency_labels=urgency,
        metadata={"scenario_family": families},
        seed=7,
        data_cfg={"train_ratio": 0.7, "val_ratio": 0.15, "scenario_holdout_key": "scenario_family", "scenario_holdout_value": "Saturation_Overload"},
        split_strategy="fixed_holdout",
        validator=lambda *_: True,
    )

    assert np.all(families[test_idx] == "Saturation_Overload")
    assert not np.any(families[train_idx] == "Saturation_Overload")
    assert not np.any(families[val_idx] == "Saturation_Overload")


def test_result_context_records_the_named_scenario_holdout():
    setting = {
        "dataset": "ATUAV-Core",
        "protocol": "latent_state_masked",
        "scenario_profile": "ATUAV-Core",
        "split_strategy": "fixed_holdout",
        "detection_window": "standard",
        "noise_level": 0.0,
        "missing_ratio": 0.0,
        "scenario_holdout_key": "scenario_family",
        "scenario_holdout_value": "EW_Contested",
    }

    context = setting_context("holdout_ew", setting)

    assert context["scenario_holdout_key"] == "scenario_family"
    assert context["scenario_holdout_value"] == "EW_Contested"


def test_training_export_keeps_validation_selection_and_holdout_identity():
    records = [
        {
            "assessment_setting": {
                "dataset": "ATUAV-Core",
                "protocol": "latent_state_masked",
                "scenario_profile": "ATUAV-Core",
                "task_form": "sequential",
                "split_strategy": "fixed_holdout",
                "reference_policy_variant": "balanced",
                "scenario_holdout_key": "scenario_family",
                "scenario_holdout_value": "Strike_Penetration",
            },
            "run_index": 0,
            "seed": 42,
            "training": {
                "TemporalHGTAN": {
                    "info": {"best_epoch": 7, "best_val_score": 0.81, "overfitting": {"gap": 0.04}},
                    "curves": [{"epoch": 1, "train_loss": 1.2, "val_score": 0.55}],
                }
            },
        }
    ]

    summaries, curves = build_training_rows(records)

    assert summaries[0]["best_val_score"] == 0.81
    assert summaries[0]["overfitting_gap"] == 0.04
    assert summaries[0]["scenario_holdout_value"] == "Strike_Penetration"
    assert curves[0]["val_score"] == 0.55


def test_seed_checkpoint_round_trip_requires_matching_identity(tmp_path):
    path = tmp_path / "run_00_seed_42.json.gz"
    record = {"run_index": 0, "seed": 42, "results": {"TemporalHGTAN": {"threat": {"f1": 0.8}}}}
    write_json_gzip(path, record)

    loaded = load_seed_checkpoint(
        path,
        expected_models=["TemporalHGTAN"],
        expected_seed=42,
        expected_run_index=0,
    )

    assert loaded == record
    assert load_seed_checkpoint(path, expected_models=["TemporalLSTM"], expected_seed=42, expected_run_index=0) is None


def test_seed_checkpoint_ignores_truncated_json(tmp_path):
    path = tmp_path / "run_00_seed_42.json.gz"
    path.write_bytes(b"not-a-gzip-stream")

    assert load_seed_checkpoint(
        path,
        expected_models=["TemporalHGTAN"],
        expected_seed=42,
        expected_run_index=0,
    ) is None
