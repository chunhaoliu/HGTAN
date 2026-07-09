"""Reference assessment policy for the sequential UAV benchmark.

The policy deliberately operates on latent scenario state and clean engagement
geometry.  It is kept separate from the observed indicator stream supplied to
the models, so the sequential task is policy recovery under degraded sensing
rather than direct regression of an observed-feature score.
"""

from __future__ import annotations

from typing import Any

import numpy as np


REFERENCE_POLICY_NAME = "latent_consequence_v1"
REFERENCE_THREAT_THRESHOLDS = np.array([0.24, 0.41, 0.58, 0.74], dtype=np.float64)
REFERENCE_URGENCY_THRESHOLDS = np.array([0.32, 0.62], dtype=np.float64)


def build_reference_assessment_sequences(
    clean_sequences: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return frozen threat and urgency references for clean scenario states."""
    components = reference_assessment_components(clean_sequences, metadata)
    threat = np.clip(np.digitize(components["threat_score"], REFERENCE_THREAT_THRESHOLDS) + 1, 1, 5)
    urgency = np.clip(np.digitize(components["urgency_score"], REFERENCE_URGENCY_THRESHOLDS) + 1, 1, 3)
    return threat.astype(np.int64), urgency.astype(np.int64), components


def reference_assessment_components(
    clean_sequences: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Compute interpretable clean-state consequence and response components.

    Latent mission, platform, formation, defense, environment, and asset
    states define the reference policy.  The clean trajectory contributes only
    the physical approach and access geometry.  The model never receives the
    latent target- or mission-type codes under the default protocol.
    """
    if clean_sequences.ndim != 3 or clean_sequences.shape[-1] != 16:
        raise ValueError("clean_sequences must have shape (n_tracks, n_steps, 16)")

    n_tracks, n_steps, _ = clean_sequences.shape
    for key in ("mission_type", "target_type", "formation_type", "defense_state", "environment_type", "asset_type"):
        if key not in metadata:
            raise KeyError(f"Missing latent reference-policy state: {key}")
        if len(np.asarray(metadata[key])) != n_tracks:
            raise ValueError(f"Metadata field {key} does not match the number of tracks")

    def expand(name: str) -> np.ndarray:
        return np.asarray(metadata[name], dtype=np.float64)[:, None]

    mission = expand("mission_type") / 3.0
    platform = expand("target_type") / 3.0
    formation = expand("formation_type") / 3.0
    defense_state = expand("defense_state") / 3.0
    environment = expand("environment_type") / 3.0
    asset = expand("asset_type") / 3.0

    payload = clean_sequences[:, :, 1]
    adversarial = clean_sequences[:, :, 2]
    endurance = clean_sequences[:, :, 3]
    coordination = clean_sequences[:, :, 5]
    heading = clean_sequences[:, :, 6]
    route_deviation = clean_sequences[:, :, 7]
    distance = clean_sequences[:, :, 8]
    velocity = clean_sequences[:, :, 9]
    altitude = clean_sequences[:, :, 10]
    time_to_arrival = clean_sequences[:, :, 11]
    swarm = clean_sequences[:, :, 12]

    # True geometry is evaluated before the sensor process; it is not the
    # observed feature vector that the models receive after degradation.
    approach = np.clip(
        0.38 * (1.0 - time_to_arrival)
        + 0.30 * (1.0 - distance)
        + 0.18 * velocity
        + 0.08 * (1.0 - altitude)
        + 0.06 * (1.0 - heading),
        0.0,
        1.0,
    )
    platform_consequence = np.clip(
        0.48 * platform + 0.22 * payload + 0.20 * adversarial + 0.10 * endurance,
        0.0,
        1.0,
    )
    mission_commitment = np.clip(
        0.58 * mission + 0.20 * formation + 0.12 * coordination + 0.10 * route_deviation,
        0.0,
        1.0,
    )
    defense_margin = np.clip(
        0.52 * (1.0 - defense_state)
        + 0.28 * asset
        + 0.12 * environment
        + 0.08 * swarm,
        0.0,
        1.0,
    )

    consequence = np.clip(
        0.48 * platform_consequence * (0.45 + 0.55 * approach)
        + 0.32 * mission_commitment
        + 0.20 * defense_margin,
        0.0,
        1.0,
    )
    threat_score = np.clip(
        0.42 * consequence
        + 0.24 * platform_consequence
        + 0.22 * approach
        + 0.12 * defense_margin
        + 0.10 * mission_commitment * approach,
        0.0,
        1.0,
    )
    urgency_score = np.clip(
        0.56 * approach
        + 0.24 * defense_margin
        + 0.12 * mission_commitment * approach
        + 0.08 * formation,
        0.0,
        1.0,
    )

    expected_shape = (n_tracks, n_steps)
    return {
        "approach": np.broadcast_to(approach, expected_shape).copy(),
        "platform_consequence": np.broadcast_to(platform_consequence, expected_shape).copy(),
        "mission_commitment": np.broadcast_to(mission_commitment, expected_shape).copy(),
        "defense_margin": np.broadcast_to(defense_margin, expected_shape).copy(),
        "consequence": np.broadcast_to(consequence, expected_shape).copy(),
        "threat_score": threat_score,
        "urgency_score": urgency_score,
    }
