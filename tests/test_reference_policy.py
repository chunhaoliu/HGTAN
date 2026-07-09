import numpy as np

from data.reference_policy import REFERENCE_POLICY_NAME, build_reference_assessment_sequences
from data.sequence_generator import generate_uav_track_payload


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
