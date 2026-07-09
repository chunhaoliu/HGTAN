"""
Scenario-oriented UAV swarm threat assessment data generator.

This generator is designed for a sequential threat-assessment workflow:
1. Features encode capability / intent / opportunity / context.
2. Threat labels follow fixed operational risk thresholds rather than percentiles.
3. Urgency labels model decision window pressure instead of mirroring threat labels.
4. Metadata preserves scenario family information for scenario-shift evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data.dataset_protocol import difficulty_tier_for_dataset
from utils.config import (
    ALL_FEATURES,
    HGTANConfig,
    MISSION_TYPE_LABELS,
    N_FEATURES,
    TARGET_TYPE_LABELS,
)


DEFENSE_STATE_LABELS = {
    0: "Layered_Strong",
    1: "Balanced_Defense",
    2: "Degraded_Defense",
    3: "Resource_Saturated",
}

ENVIRONMENT_LABELS = {
    0: "Open_Clear",
    1: "Urban_Clutter",
    2: "LowAltitude_Masking",
    3: "Adverse_Weather",
}

FORMATION_LABELS = {
    0: "Loose_Screen",
    1: "Clustered_Assault",
    2: "MultiAxis_Encirclement",
    3: "Decoy_Split",
}

ASSET_LABELS = {
    0: "Local_Asset",
    1: "Air_Defense_Node",
    2: "Command_Post",
    3: "Strategic_Infrastructure",
}

THREAT_THRESHOLDS = np.array([0.26, 0.44, 0.64, 0.82], dtype=np.float64)
URGENCY_THRESHOLDS = np.array([0.36, 0.67], dtype=np.float64)

SCENARIO_PROFILES = {
    "ATUAV-Core": {
        "description": "Balanced air-target UAV swarm threat-assessment setting.",
        "mission_p": [0.28, 0.24, 0.28, 0.20],
        "defense_p": [0.24, 0.36, 0.25, 0.15],
        "environment_p": [0.30, 0.24, 0.26, 0.20],
        "asset_p": [0.20, 0.32, 0.26, 0.22],
    },
    "ATUAV-MissionShift": {
        "description": "Intent-shift setting with more strike and saturation missions.",
        "mission_p": [0.12, 0.18, 0.38, 0.32],
        "defense_p": [0.22, 0.34, 0.27, 0.17],
        "environment_p": [0.26, 0.24, 0.28, 0.22],
        "asset_p": [0.16, 0.28, 0.30, 0.26],
    },
    "ATUAV-DefenseShift": {
        "description": "Defense-state shift with degraded and saturated defense conditions.",
        "mission_p": [0.24, 0.23, 0.30, 0.23],
        "defense_p": [0.08, 0.22, 0.36, 0.34],
        "environment_p": [0.24, 0.27, 0.27, 0.22],
        "asset_p": [0.16, 0.30, 0.28, 0.26],
    },
    "ATUAV-EnvironmentShift": {
        "description": "Environmental shift toward clutter, masking, and adverse weather.",
        "mission_p": [0.22, 0.24, 0.30, 0.24],
        "defense_p": [0.20, 0.32, 0.29, 0.19],
        "environment_p": [0.08, 0.34, 0.34, 0.24],
        "asset_p": [0.18, 0.30, 0.28, 0.24],
    },
    "ATUAV-Saturation": {
        "description": "High-load swarm saturation setting.",
        "mission_p": [0.08, 0.18, 0.30, 0.44],
        "defense_p": [0.08, 0.18, 0.30, 0.44],
        "environment_p": [0.18, 0.24, 0.32, 0.26],
        "asset_p": [0.10, 0.26, 0.30, 0.34],
    },
    "ATUAV-SensorDegraded": {
        "description": "Contested sensing setting with clutter and EW-heavy targets.",
        "mission_p": [0.18, 0.34, 0.26, 0.22],
        "defense_p": [0.12, 0.27, 0.34, 0.27],
        "environment_p": [0.06, 0.38, 0.24, 0.32],
        "asset_p": [0.16, 0.30, 0.28, 0.26],
    },
}

DETECTION_WINDOW_PROFILES = {
    "standard": "Nominal detection and engagement window.",
    "early_detection": "Longer warning time and stronger defensive opportunity.",
    "terminal_detection": "Short warning time close to the defended asset.",
    "resource_saturation": "Defensive resource pressure under larger coordinated swarms.",
    "contested_sensing": "Tracking and sensing degradation under adversarial conditions.",
}


def generate_uav_swarm_payload(
    n_samples: int | None = None,
    seed: int | None = None,
    scenario_profile: str | None = None,
    detection_window: str | None = None,
    benchmark_dataset: str | None = None,
    apply_label_conditioned_perturbations: bool = True,
    apply_static_observation_noise: bool = True,
) -> dict[str, Any]:
    """Generate one assessment payload for instantaneous ATUAV samples."""
    features, threat_labels, metadata = generate_uav_swarm_data(
        n_samples=n_samples,
        seed=seed,
        return_metadata=True,
        scenario_profile=scenario_profile,
        detection_window=detection_window,
        benchmark_dataset=benchmark_dataset,
        apply_label_conditioned_perturbations=apply_label_conditioned_perturbations,
        apply_static_observation_noise=apply_static_observation_noise,
    )
    urgency_labels = generate_urgency_labels(features, threat_labels, metadata=metadata)
    return {
        "features": features.astype(np.float32, copy=False),
        "threat_labels": threat_labels.astype(np.int64, copy=False),
        "urgency_labels": urgency_labels.astype(np.int64, copy=False),
        "metadata": metadata,
        "task_form": "instantaneous",
    }


def generate_uav_swarm_data(
    n_samples: int | None = None,
    seed: int | None = None,
    return_metadata: bool = False,
    scenario_profile: str | None = None,
    detection_window: str | None = None,
    benchmark_dataset: str | None = None,
    apply_label_conditioned_perturbations: bool = True,
    apply_static_observation_noise: bool = True,
):
    """
    Generate a scenario-oriented UAV swarm threat assessment dataset.

    Returns:
        features: (n_samples, 16)
        threat_labels: (n_samples,) with 1-5 classes
        metadata: optional dict of latent scenario descriptors
    """
    if n_samples is None:
        n_samples = HGTANConfig.DATA["n_samples"]

    rng = np.random.default_rng(seed)
    data_cfg = HGTANConfig.DATA
    scenario_profile = scenario_profile or data_cfg.get("benchmark_dataset", "ATUAV-Core")
    benchmark_dataset = benchmark_dataset or data_cfg.get("benchmark_dataset", scenario_profile)
    detection_window = detection_window or data_cfg.get("detection_window", "standard")
    profile_cfg = _get_scenario_profile(scenario_profile)
    _validate_detection_window(detection_window)

    mission_type = rng.choice([0, 1, 2, 3], size=n_samples, p=profile_cfg["mission_p"])
    defense_state = rng.choice([0, 1, 2, 3], size=n_samples, p=profile_cfg["defense_p"])
    environment_type = rng.choice([0, 1, 2, 3], size=n_samples, p=profile_cfg["environment_p"])
    asset_type = rng.choice([0, 1, 2, 3], size=n_samples, p=profile_cfg["asset_p"])
    formation_type = _sample_formation_types(rng, mission_type, scenario_profile)
    target_type = _sample_target_types(rng, mission_type, scenario_profile)

    target_asset_value = _clip01(
        np.array([0.28, 0.55, 0.78, 0.92])[asset_type] + rng.normal(0, 0.05, n_samples)
    )

    payload_capability = _clip01(
        0.18
        + 0.18 * (target_type >= 2)
        + 0.12 * (mission_type >= 2)
        + 0.16 * (mission_type == 3)
        + rng.normal(0, 0.07, n_samples)
    )
    adversarial_capability = _clip01(
        0.22
        + 0.10 * (target_type == 1)
        + 0.12 * (target_type == 3)
        + 0.10 * (environment_type >= 2)
        + 0.12 * (mission_type >= 1)
        + rng.normal(0, 0.06, n_samples)
    )
    endurance_margin = _clip01(
        0.30
        + 0.15 * (mission_type == 0)
        + 0.10 * (mission_type == 1)
        + 0.08 * (target_type == 0)
        + 0.06 * (target_type == 2)
        - 0.05 * (environment_type == 3)
        + rng.normal(0, 0.07, n_samples)
    )

    coordination_level = _clip01(
        0.18
        + 0.12 * formation_type
        + 0.10 * (mission_type >= 2)
        + 0.14 * (mission_type == 3)
        + 0.06 * (target_type == 3)
        + rng.normal(0, 0.07, n_samples)
    )
    heading_angle = _clip01(
        0.65
        - 0.12 * (mission_type >= 2)
        - 0.14 * (mission_type == 3)
        - 0.06 * (formation_type == 2)
        + 0.06 * (mission_type == 0)
        + rng.normal(0, 0.08, n_samples)
    )
    route_deviation = _clip01(
        0.18
        + 0.10 * (mission_type == 1)
        + 0.16 * (mission_type >= 2)
        + 0.10 * (formation_type == 3)
        + 0.08 * (environment_type == 1)
        + rng.normal(0, 0.08, n_samples)
    )

    distance = _clip01(
        0.72
        - 0.12 * (mission_type >= 1)
        - 0.16 * (mission_type >= 2)
        - 0.08 * (mission_type == 3)
        + 0.10 * (defense_state == 0)
        + rng.normal(0, 0.08, n_samples)
    )
    velocity = _clip01(
        0.26
        + 0.12 * (target_type >= 2)
        + 0.12 * (mission_type >= 2)
        + 0.08 * (mission_type == 1)
        + 0.06 * coordination_level
        + rng.normal(0, 0.07, n_samples)
    )
    altitude = _clip01(
        0.45
        - 0.18 * (environment_type == 2)
        + 0.10 * (environment_type == 0)
        - 0.06 * (mission_type >= 2)
        + 0.05 * (defense_state == 0)
        + rng.normal(0, 0.07, n_samples)
    )

    swarm_size = _clip01(
        0.18
        + 0.10 * formation_type
        + 0.14 * (mission_type >= 1)
        + 0.14 * (mission_type == 3)
        + rng.normal(0, 0.08, n_samples)
    )
    defense_capability = _clip01(
        np.array([0.86, 0.66, 0.42, 0.24])[defense_state]
        + 0.05 * (asset_type >= 2)
        - 0.06 * (environment_type == 3)
        + rng.normal(0, 0.05, n_samples)
    )

    time_to_arrival = _compute_time_to_arrival(distance, velocity, coordination_level)
    track_confidence = _clip01(
        0.84
        - 0.18 * (environment_type == 1)
        - 0.24 * (environment_type == 3)
        - 0.16 * adversarial_capability
        - 0.10 * (altitude < 0.30)
        + 0.10 * defense_capability
        - 0.08 * route_deviation
        + rng.normal(0, 0.05, n_samples)
    )

    features = np.column_stack(
        [
            target_type / 3.0,
            payload_capability,
            adversarial_capability,
            endurance_margin,
            mission_type / 3.0,
            coordination_level,
            heading_angle,
            route_deviation,
            distance,
            velocity,
            altitude,
            time_to_arrival,
            swarm_size,
            defense_capability,
            target_asset_value,
            track_confidence,
        ]
    )
    features = _apply_detection_window(features, detection_window)

    threat_risk = compute_threat_scores(
        features,
        metadata={
            "mission_type": mission_type,
            "target_type": target_type,
            "formation_type": formation_type,
        },
    )
    threat_labels = np.digitize(threat_risk, bins=THREAT_THRESHOLDS) + 1
    threat_labels = np.clip(threat_labels, 1, 5)

    if apply_label_conditioned_perturbations:
        features, threat_labels = add_boundary_samples(
            features,
            threat_labels,
            threat_risk,
            ratio=data_cfg.get("boundary_ratio", 0.08),
            rng=rng,
        )
        features, threat_labels = add_confusing_samples(
            features,
            threat_labels,
            ratio=data_cfg.get("confusing_ratio", 0.03),
            rng=rng,
        )

    if apply_static_observation_noise:
        noise = rng.normal(0, data_cfg.get("noise_std", 0.02), features.shape)
        features = _clip01(features + noise)

    metadata = {
        "mission_type": mission_type,
        "target_type": target_type,
        "defense_state": defense_state,
        "environment_type": environment_type,
        "formation_type": formation_type,
        "asset_type": asset_type,
        "scenario_group": np.array(
            [
                f"{mission_type[i]}-{defense_state[i]}-{environment_type[i]}-{formation_type[i]}"
                for i in range(n_samples)
            ],
            dtype=object,
        ),
        "scenario_family": _scenario_family_names(mission_type, defense_state, environment_type, formation_type),
        "mission_name": np.array([MISSION_TYPE_LABELS[int(idx)] for idx in mission_type], dtype=object),
        "target_name": np.array([TARGET_TYPE_LABELS[int(idx)] for idx in target_type], dtype=object),
        "defense_name": np.array([DEFENSE_STATE_LABELS[int(idx)] for idx in defense_state], dtype=object),
        "environment_name": np.array([ENVIRONMENT_LABELS[int(idx)] for idx in environment_type], dtype=object),
        "formation_name": np.array([FORMATION_LABELS[int(idx)] for idx in formation_type], dtype=object),
        "asset_name": np.array([ASSET_LABELS[int(idx)] for idx in asset_type], dtype=object),
        "benchmark_dataset": np.full(n_samples, benchmark_dataset, dtype=object),
        "scenario_profile": np.full(n_samples, scenario_profile, dtype=object),
        "difficulty_tier": np.full(n_samples, difficulty_tier_for_dataset(benchmark_dataset, scenario_profile), dtype=object),
        "sensor_profile": np.full(n_samples, data_cfg.get("sensor_profile", benchmark_dataset), dtype=object),
        "detection_window": np.full(n_samples, detection_window, dtype=object),
        "threat_risk": threat_risk,
    }

    shuffle_idx = rng.permutation(n_samples)
    features = features[shuffle_idx]
    threat_labels = threat_labels[shuffle_idx]
    metadata = {
        key: value[shuffle_idx] if isinstance(value, np.ndarray) else value
        for key, value in metadata.items()
    }

    if return_metadata:
        return features, threat_labels, metadata
    return features, threat_labels


def compute_threat_scores(features: np.ndarray, metadata: dict[str, Any] | None = None) -> np.ndarray:
    """
    Compute a fixed operational threat risk score in [0, 1].

    The score is not percentile-based. Instead it combines four interpretable
    dimensions plus swarm synergy and defense pressure.
    """
    target_type_signal = features[:, 0]
    payload = features[:, 1]
    adversarial = features[:, 2]
    endurance = features[:, 3]

    mission_signal = features[:, 4]
    coordination = features[:, 5]
    heading = features[:, 6]
    route_deviation = features[:, 7]

    distance = features[:, 8]
    velocity = features[:, 9]
    altitude = features[:, 10]
    time_to_arrival = features[:, 11]

    swarm_size = features[:, 12]
    defense = features[:, 13]
    asset_value = features[:, 14]
    track_confidence = features[:, 15]

    capability_risk = (
        0.32 * payload
        + 0.28 * adversarial
        + 0.18 * target_type_signal
        + 0.22 * endurance
    )
    intent_risk = (
        0.28 * mission_signal
        + 0.28 * coordination
        + 0.22 * (1.0 - heading)
        + 0.22 * route_deviation
    )
    opportunity_risk = (
        0.32 * (1.0 - time_to_arrival)
        + 0.24 * (1.0 - distance)
        + 0.22 * velocity
        + 0.22 * (1.0 - altitude)
    )
    context_risk = (
        0.30 * swarm_size
        + 0.28 * asset_value
        + 0.24 * (1.0 - defense)
        + 0.18 * (1.0 - track_confidence)
    )

    formation_type = metadata.get("formation_type") if metadata is not None else None
    formation_bonus = np.zeros(len(features), dtype=np.float64)
    if formation_type is not None:
        formation_bonus += 0.05 * (formation_type == 2)
        formation_bonus += 0.04 * (formation_type == 3)

    mission_type = metadata.get("mission_type") if metadata is not None else None
    mission_bonus = 0.0 if mission_type is None else 0.08 * (mission_type == 3)

    synergy_factor = 1.0 + 0.32 * np.power(np.clip(swarm_size * coordination, 0.0, 1.0), 0.7)
    synergy_factor += mission_bonus + formation_bonus

    base_risk = (
        0.28 * capability_risk
        + 0.23 * intent_risk
        + 0.26 * opportunity_risk
        + 0.23 * context_risk
    )
    threat_risk = np.clip(base_risk * synergy_factor, 0.0, 1.0)
    return threat_risk


def add_boundary_samples(
    features: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    ratio: float = 0.08,
    rng: np.random.Generator | None = None,
):
    """Inject mild feature perturbations around fixed decision boundaries."""
    if rng is None:
        rng = np.random.default_rng()

    n_boundary = int(len(labels) * ratio)
    if n_boundary == 0:
        return features, labels

    near_boundary_mask = np.zeros(len(labels), dtype=bool)
    for boundary in THREAT_THRESHOLDS:
        near_boundary_mask |= np.abs(scores - boundary) < 0.05

    candidate_indices = np.where(near_boundary_mask)[0]
    if len(candidate_indices) == 0:
        return features, labels

    selected = rng.choice(candidate_indices, size=min(n_boundary, len(candidate_indices)), replace=False)
    perturbation = rng.normal(0, 0.06, size=(len(selected), features.shape[1]))
    features[selected] = _clip01(features[selected] + perturbation)
    return features, labels


def add_confusing_samples(
    features: np.ndarray,
    labels: np.ndarray,
    ratio: float = 0.03,
    rng: np.random.Generator | None = None,
):
    """Inject a small number of ambiguous samples without changing labels."""
    if rng is None:
        rng = np.random.default_rng()

    n_confusing = int(len(labels) * ratio)
    if n_confusing == 0:
        return features, labels

    confuse_indices = rng.choice(len(labels), size=n_confusing, replace=False)
    for idx in confuse_indices:
        label = labels[idx]
        if label <= 2:
            features[idx, 5] = _clip01(features[idx, 5] + 0.12)   # coordination_level
            features[idx, 7] = _clip01(features[idx, 7] + 0.10)   # route_deviation
            features[idx, 8] = _clip01(features[idx, 8] - 0.12)   # distance
            features[idx, 11] = _clip01(features[idx, 11] - 0.10) # time_to_arrival
        elif label >= 4:
            features[idx, 8] = _clip01(features[idx, 8] + 0.10)
            features[idx, 11] = _clip01(features[idx, 11] + 0.12)
            features[idx, 13] = _clip01(features[idx, 13] + 0.08) # defense_capability
            features[idx, 15] = _clip01(features[idx, 15] + 0.10) # track_confidence

    return features, labels


def generate_urgency_labels(
    features: np.ndarray,
    threat_labels: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Generate urgency labels from operational decision pressure.

    Urgency is linked to engagement window and defense pressure rather than
    being a direct remapping of threat labels.
    """
    mission_signal = features[:, 4]
    coordination = features[:, 5]
    distance = features[:, 8]
    velocity = features[:, 9]
    time_to_arrival = features[:, 11]
    swarm_size = features[:, 12]
    defense = features[:, 13]
    asset_value = features[:, 14]
    track_confidence = features[:, 15]

    threat_signal = metadata.get("threat_risk") if metadata is not None else (threat_labels - 1) / 4.0
    swarm_pressure = np.power(np.clip(swarm_size * coordination, 0.0, 1.0), 0.6)

    urgency_scores = (
        0.34 * (1.0 - time_to_arrival)
        + 0.14 * (1.0 - distance)
        + 0.10 * velocity
        + 0.15 * (1.0 - defense)
        + 0.12 * asset_value
        + 0.07 * swarm_pressure
        + 0.04 * mission_signal
        + 0.04 * (1.0 - track_confidence)
    )
    urgency_scores = np.clip(urgency_scores + 0.04 * threat_signal, 0.0, 1.0)

    urgency = np.digitize(urgency_scores, bins=URGENCY_THRESHOLDS) + 1
    return np.clip(urgency, 1, 3)


def get_feature_statistics(features: np.ndarray) -> dict[str, dict[str, float]]:
    """Return per-feature descriptive statistics."""
    stats = {}
    for i, name in enumerate(ALL_FEATURES):
        stats[name] = {
            "mean": float(np.mean(features[:, i])),
            "std": float(np.std(features[:, i])),
            "min": float(np.min(features[:, i])),
            "max": float(np.max(features[:, i])),
        }
    return stats


def compute_derived_features(features: np.ndarray) -> dict[str, np.ndarray]:
    """Return derived interpretable signals for analysis."""
    return {
        "time_to_arrival": features[:, 11],
        "inbound_directness": 1.0 - features[:, 6],
        "swarm_pressure": np.power(np.clip(features[:, 12] * features[:, 5], 0.0, 1.0), 0.6),
        "decision_pressure": (
            0.5 * (1.0 - features[:, 11]) +
            0.2 * (1.0 - features[:, 13]) +
            0.2 * features[:, 14] +
            0.1 * (1.0 - features[:, 15])
        ),
    }


def _scenario_family_names(
    mission_type: np.ndarray,
    defense_state: np.ndarray,
    environment_type: np.ndarray,
    formation_type: np.ndarray,
) -> np.ndarray:
    """Map latent scenario axes into paper-readable operational families."""
    names = []
    for mission, defense, environment, formation in zip(
        mission_type.tolist(),
        defense_state.tolist(),
        environment_type.tolist(),
        formation_type.tolist(),
    ):
        if mission == 3 or (defense == 3 and formation in {2, 3}):
            names.append("Saturation_Overload")
        elif mission == 2:
            names.append("Strike_Penetration")
        elif mission == 1 or environment in {1, 3}:
            names.append("EW_Contested")
        else:
            names.append("Probe_Surveillance")
    return np.asarray(names, dtype=object)


def _sample_target_types(
    rng: np.random.Generator,
    mission_type: np.ndarray,
    scenario_profile: str | None = None,
) -> np.ndarray:
    target_type = np.zeros_like(mission_type)
    for idx, mission in enumerate(mission_type):
        if scenario_profile == "ATUAV-SensorDegraded":
            if mission == 0:
                target_type[idx] = rng.choice([0, 1], p=[0.58, 0.42])
            elif mission == 1:
                target_type[idx] = rng.choice([1, 2], p=[0.80, 0.20])
            elif mission == 2:
                target_type[idx] = rng.choice([1, 2, 3], p=[0.24, 0.52, 0.24])
            else:
                target_type[idx] = rng.choice([1, 2, 3], p=[0.20, 0.28, 0.52])
        elif scenario_profile == "ATUAV-Saturation" and mission >= 2:
            target_type[idx] = rng.choice([2, 3], p=[0.24, 0.76])
        elif mission == 0:
            target_type[idx] = rng.choice([0, 1], p=[0.82, 0.18])
        elif mission == 1:
            target_type[idx] = rng.choice([1, 2], p=[0.70, 0.30])
        elif mission == 2:
            target_type[idx] = rng.choice([2, 3], p=[0.68, 0.32])
        else:
            target_type[idx] = rng.choice([2, 3], p=[0.34, 0.66])
    return target_type


def _sample_formation_types(
    rng: np.random.Generator,
    mission_type: np.ndarray,
    scenario_profile: str | None = None,
) -> np.ndarray:
    formation_type = np.zeros_like(mission_type)
    for idx, mission in enumerate(mission_type):
        if scenario_profile == "ATUAV-Saturation":
            if mission <= 1:
                formation_type[idx] = rng.choice([0, 1, 3], p=[0.25, 0.35, 0.40])
            else:
                formation_type[idx] = rng.choice([1, 2, 3], p=[0.20, 0.56, 0.24])
        elif scenario_profile == "ATUAV-SensorDegraded":
            if mission <= 1:
                formation_type[idx] = rng.choice([0, 3], p=[0.36, 0.64])
            else:
                formation_type[idx] = rng.choice([1, 2, 3], p=[0.30, 0.28, 0.42])
        elif mission == 0:
            formation_type[idx] = rng.choice([0, 1], p=[0.75, 0.25])
        elif mission == 1:
            formation_type[idx] = rng.choice([0, 3], p=[0.45, 0.55])
        elif mission == 2:
            formation_type[idx] = rng.choice([1, 2], p=[0.55, 0.45])
        else:
            formation_type[idx] = rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25])
    return formation_type


def _compute_time_to_arrival(
    distance: np.ndarray,
    velocity: np.ndarray,
    coordination_level: np.ndarray,
) -> np.ndarray:
    effective_velocity = 0.18 + velocity * (0.78 + 0.22 * coordination_level)
    raw_tta = distance / np.clip(effective_velocity, 0.18, None)
    return _clip01((raw_tta - 0.10) / 1.70)


def _get_scenario_profile(scenario_profile: str) -> dict[str, Any]:
    if scenario_profile not in SCENARIO_PROFILES:
        valid = ", ".join(sorted(SCENARIO_PROFILES))
        raise ValueError(f"Unknown scenario_profile={scenario_profile!r}. Valid options: {valid}")
    return SCENARIO_PROFILES[scenario_profile]


def _validate_detection_window(detection_window: str) -> None:
    if detection_window not in DETECTION_WINDOW_PROFILES:
        valid = ", ".join(sorted(DETECTION_WINDOW_PROFILES))
        raise ValueError(f"Unknown detection_window={detection_window!r}. Valid options: {valid}")


def _apply_detection_window(features: np.ndarray, detection_window: str) -> np.ndarray:
    if detection_window == "standard":
        return features

    adjusted = features.copy()
    tta_shift = 0.0

    if detection_window == "early_detection":
        adjusted[:, 8] = _clip01(adjusted[:, 8] + 0.16)   # distance
        adjusted[:, 13] = _clip01(adjusted[:, 13] + 0.06) # defense_capability
        adjusted[:, 15] = _clip01(adjusted[:, 15] + 0.04) # track_confidence
        tta_shift = 0.08
    elif detection_window == "terminal_detection":
        adjusted[:, 8] = _clip01(adjusted[:, 8] - 0.18)
        adjusted[:, 9] = _clip01(adjusted[:, 9] + 0.08)   # velocity
        adjusted[:, 12] = _clip01(adjusted[:, 12] + 0.05) # swarm_size
        tta_shift = -0.10
    elif detection_window == "resource_saturation":
        adjusted[:, 5] = _clip01(adjusted[:, 5] + 0.05)   # coordination_level
        adjusted[:, 12] = _clip01(adjusted[:, 12] + 0.18)
        adjusted[:, 13] = _clip01(adjusted[:, 13] - 0.16)
        tta_shift = -0.06
    elif detection_window == "contested_sensing":
        adjusted[:, 2] = _clip01(adjusted[:, 2] + 0.10)   # adversarial_capability
        adjusted[:, 7] = _clip01(adjusted[:, 7] + 0.06)   # route_deviation
        adjusted[:, 13] = _clip01(adjusted[:, 13] - 0.06)
        adjusted[:, 15] = _clip01(adjusted[:, 15] - 0.20)
    else:
        _validate_detection_window(detection_window)

    adjusted[:, 11] = _compute_time_to_arrival(adjusted[:, 8], adjusted[:, 9], adjusted[:, 5])
    adjusted[:, 11] = _clip01(adjusted[:, 11] + tta_shift)
    return _clip01(adjusted)


def _clip01(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 0.02, 0.98)
