"""Track-sequence generator for sequential ATUAV assessment protocols."""

from __future__ import annotations

from typing import Any

import numpy as np

from data.generator import _clip01, generate_uav_swarm_payload
from data.reference_policy import REFERENCE_POLICY_NAME, build_reference_assessment_sequences
from utils.config import HGTANConfig


def generate_uav_track_payload(
    n_tracks: int | None = None,
    seq_len: int | None = None,
    seed: int | None = None,
    scenario_profile: str | None = None,
    detection_window: str | None = None,
    benchmark_dataset: str | None = None,
    type_as_input: bool | None = None,
    mission_as_input: bool | None = None,
    sequence_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one assessment payload for sequential ATUAV track data."""
    data_cfg = HGTANConfig.DATA
    sequence_cfg = sequence_cfg or HGTANConfig.SEQUENCE
    n_tracks = n_tracks or data_cfg["n_samples"]
    seq_len = seq_len or sequence_cfg["seq_len"]
    type_as_input = sequence_cfg["type_as_input"] if type_as_input is None else type_as_input
    mission_as_input = sequence_cfg["mission_as_input"] if mission_as_input is None else mission_as_input

    rng = np.random.default_rng(seed)
    sample_payload = generate_uav_swarm_payload(
        n_samples=n_tracks,
        seed=seed,
        scenario_profile=scenario_profile or data_cfg.get("benchmark_dataset", "ATUAV-Core"),
        detection_window=detection_window or data_cfg.get("detection_window", "standard"),
        benchmark_dataset=benchmark_dataset or data_cfg.get("benchmark_dataset"),
        apply_label_conditioned_perturbations=False,
        apply_static_observation_noise=False,
    )
    final_features = np.asarray(sample_payload["features"])
    metadata = dict(sample_payload["metadata"])
    # The instantaneous generator retains this legacy diagnostic only for its
    # own task.  It must not enter the sequential reference-policy path.
    metadata.pop("threat_risk", None)

    clean_sequences = _build_temporal_features(final_features, metadata, seq_len, rng)
    observed_sequences = _apply_sequence_noise(clean_sequences, metadata, rng, sequence_cfg)
    threat_seq, urgency_seq, reference_components = build_reference_assessment_sequences(clean_sequences, metadata)

    model_sequences = observed_sequences.copy()
    if not type_as_input:
        model_sequences[:, :, 0] = 0.0
    if not mission_as_input:
        model_sequences[:, :, 4] = 0.0

    track_metadata = _build_track_metadata(
        metadata,
        n_tracks,
        seq_len,
        type_as_input,
        mission_as_input,
        sequence_cfg,
    )
    track_metadata["clean_sequence"] = clean_sequences.astype(np.float32)
    track_metadata["noisy_sequence"] = observed_sequences.astype(np.float32)
    track_metadata["model_input_sequence"] = model_sequences.astype(np.float32)
    track_metadata["reference_policy"] = REFERENCE_POLICY_NAME
    track_metadata["reference_threat_score"] = reference_components["threat_score"].astype(np.float32)
    track_metadata["reference_urgency_score"] = reference_components["urgency_score"].astype(np.float32)

    return {
        "sequences": model_sequences.astype(np.float32),
        "threat_seq": threat_seq.astype(np.int64),
        "urgency_seq": urgency_seq.astype(np.int64),
        "metadata": track_metadata,
        "task_form": "sequential",
    }


def generate_uav_track_sequences(
    n_tracks: int | None = None,
    seq_len: int | None = None,
    seed: int | None = None,
    scenario_profile: str | None = None,
    detection_window: str | None = None,
    benchmark_dataset: str | None = None,
    type_as_input: bool | None = None,
    mission_as_input: bool | None = None,
    sequence_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate synthetic UAV track sequences with dynamic threat labels.

    The generator reuses the static scenario semantics, then creates a plausible
    temporal evolution from a safer early observation to a later risk state.
    """
    payload = generate_uav_track_payload(
        n_tracks=n_tracks,
        seq_len=seq_len,
        seed=seed,
        scenario_profile=scenario_profile,
        detection_window=detection_window,
        benchmark_dataset=benchmark_dataset,
        type_as_input=type_as_input,
        mission_as_input=mission_as_input,
        sequence_cfg=sequence_cfg,
    )
    return payload["sequences"], payload["threat_seq"], payload["urgency_seq"], payload["metadata"]


def _build_temporal_features(
    final_features: np.ndarray,
    metadata: dict[str, Any],
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_tracks, n_features = final_features.shape
    tau = np.linspace(0.0, 1.0, seq_len, dtype=np.float64)
    smooth_tau = 3.0 * tau**2 - 2.0 * tau**3

    initial = final_features.copy()
    mission_type = np.asarray(metadata["mission_type"])
    formation_type = np.asarray(metadata["formation_type"])
    high_intent = (mission_type >= 2).astype(np.float64)

    initial[:, 5] = _clip01(final_features[:, 5] - 0.12 - 0.05 * high_intent)   # coordination
    initial[:, 6] = _clip01(final_features[:, 6] + 0.22 + 0.05 * high_intent)   # heading angle
    initial[:, 7] = _clip01(final_features[:, 7] - 0.08)                        # route deviation
    initial[:, 8] = _clip01(final_features[:, 8] + 0.30 - 0.04 * high_intent)   # distance
    initial[:, 9] = _clip01(final_features[:, 9] - 0.10)                        # velocity
    initial[:, 10] = _clip01(final_features[:, 10] + 0.08)                      # altitude
    initial[:, 11] = _clip01(final_features[:, 11] + 0.28)                      # time to arrival
    initial[:, 12] = _clip01(final_features[:, 12] - 0.10 * (formation_type >= 1))
    initial[:, 15] = _clip01(final_features[:, 15] + 0.06)                      # track confidence

    sequences = np.empty((n_tracks, seq_len, n_features), dtype=np.float64)
    for step, alpha in enumerate(smooth_tau):
        sequences[:, step, :] = initial * (1.0 - alpha) + final_features * alpha

    turn_point = rng.uniform(0.35, 0.70, size=n_tracks)
    for track_idx in range(n_tracks):
        maneuver = np.maximum(tau - turn_point[track_idx], 0.0)
        maneuver = maneuver / max(1.0 - turn_point[track_idx], 1e-6)
        maneuver = maneuver**1.5
        sequences[track_idx, :, 6] = _clip01(sequences[track_idx, :, 6] - 0.10 * maneuver)
        sequences[track_idx, :, 8] = _clip01(sequences[track_idx, :, 8] - 0.08 * maneuver)
        sequences[track_idx, :, 11] = _clip01(sequences[track_idx, :, 11] - 0.08 * maneuver)

    drift = rng.normal(0.0, 0.01, size=sequences.shape)
    return _clip01(sequences + drift)


def _apply_sequence_noise(
    sequences: np.ndarray,
    metadata: dict[str, Any],
    rng: np.random.Generator,
    sequence_cfg: dict[str, Any],
) -> np.ndarray:
    base_noise = sequence_cfg.get("track_noise_std", 0.015)
    range_m = float(sequence_cfg.get("range_m", 1000.0))
    range_factor = 1.0 + 2.0 * max(range_m - 1000.0, 0.0) / 4000.0
    environment_type = np.asarray(metadata.get("environment_type", np.zeros(len(sequences))))
    range_signal = sequences[:, :, 8]
    noise_scale = base_noise * range_factor * (1.0 + 0.6 * range_signal + 0.4 * (environment_type[:, None] >= 2))
    noisy = sequences + rng.normal(0.0, noise_scale[..., None], size=sequences.shape)
    noisy = _apply_track_jitter(noisy, rng, sequence_cfg)
    noisy = _apply_track_missingness(noisy, rng, sequence_cfg)
    return _clip01(noisy)


def _apply_track_jitter(
    sequences: np.ndarray,
    rng: np.random.Generator,
    sequence_cfg: dict[str, Any],
) -> np.ndarray:
    jitter_std = float(sequence_cfg.get("track_jitter_std", 0.0) or 0.0)
    if jitter_std <= 0:
        return sequences

    adjusted = sequences.copy()
    jitter_features = [6, 8, 11, 15]  # heading, distance, time-to-arrival, track confidence
    state = np.zeros((sequences.shape[0], len(jitter_features)), dtype=np.float64)
    for step in range(sequences.shape[1]):
        state = 0.78 * state + rng.normal(0.0, jitter_std, size=state.shape)
        adjusted[:, step, jitter_features] += state
    return adjusted


def _apply_track_missingness(
    sequences: np.ndarray,
    rng: np.random.Generator,
    sequence_cfg: dict[str, Any],
) -> np.ndarray:
    missing_ratio = float(sequence_cfg.get("track_missing_ratio", sequence_cfg.get("missing_ratio", 0.0)) or 0.0)
    if missing_ratio <= 0:
        return sequences

    adjusted = sequences.copy()
    n_tracks, n_steps, _ = adjusted.shape
    frame_missing = rng.random((n_tracks, n_steps)) < missing_ratio
    frame_missing[:, 0] = False
    hold_features = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    for step in range(1, n_steps):
        mask = frame_missing[:, step]
        if not np.any(mask):
            continue
        indices = np.flatnonzero(mask)
        adjusted[indices[:, None], step, hold_features] = adjusted[indices[:, None], step - 1, hold_features]
        adjusted[indices, step, 15] = np.minimum(adjusted[indices, step, 15], adjusted[indices, step - 1, 15] * 0.65)
    return adjusted


def _build_track_metadata(
    metadata: dict[str, Any],
    n_tracks: int,
    seq_len: int,
    type_as_input: bool,
    mission_as_input: bool,
    sequence_cfg: dict[str, Any],
) -> dict[str, Any]:
    track_metadata = dict(metadata)
    track_metadata["track_id"] = np.arange(n_tracks)
    track_metadata["source"] = np.full(n_tracks, "simulated_sequence", dtype=object)
    track_metadata["seq_len"] = np.full(n_tracks, seq_len)
    track_metadata["type_as_input"] = np.full(n_tracks, type_as_input)
    track_metadata["mission_as_input"] = np.full(n_tracks, mission_as_input)
    track_metadata["range_m"] = np.full(n_tracks, sequence_cfg.get("range_m", 1000))
    track_metadata["track_noise_std"] = np.full(n_tracks, sequence_cfg.get("track_noise_std", 0.015))
    track_metadata["track_missing_ratio"] = np.full(n_tracks, sequence_cfg.get("track_missing_ratio", 0.0))
    track_metadata["track_jitter_std"] = np.full(n_tracks, sequence_cfg.get("track_jitter_std", 0.0))
    track_metadata["sensor_profile"] = np.full(n_tracks, sequence_cfg.get("sensor_profile", "nominal"), dtype=object)
    return track_metadata
