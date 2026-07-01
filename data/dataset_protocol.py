"""Dataset protocol definitions for ATUAV sequential threat assessment.

This module makes the assessment assumptions explicit: scene taxonomy,
difficulty tiers, sensor degradation rules, and label-generation semantics are
exported as tables instead of being buried inside the generator code.
"""

from __future__ import annotations

from typing import Any

from utils.config import THREAT_LEVELS, URGENCY_LEVELS


SCENE_TAXONOMY: dict[str, list[dict[str, Any]]] = {
    "mission": [
        {
            "id": 0,
            "name": "Recon_Probe",
            "definition": "Low-lethality sensing or probing flight used to expose defenses.",
            "risk_role": "intent_low_opportunity_probe",
        },
        {
            "id": 1,
            "name": "EW_Harass",
            "definition": "Electronic-warfare harassment with degraded tracking and association.",
            "risk_role": "capability_sensor_contest",
        },
        {
            "id": 2,
            "name": "Precision_Strike",
            "definition": "Directed attack against a defended asset or air-defense node.",
            "risk_role": "intent_high_asset_focused",
        },
        {
            "id": 3,
            "name": "Saturation_Breakthrough",
            "definition": "Coordinated swarm pressure intended to overload available defense resources.",
            "risk_role": "swarm_synergy_and_defense_pressure",
        },
    ],
    "defense": [
        {
            "id": 0,
            "name": "Layered_Strong",
            "definition": "Layered sensing and interception resources remain available.",
            "risk_role": "mitigates_opportunity",
        },
        {
            "id": 1,
            "name": "Balanced_Defense",
            "definition": "Nominal defense posture with no dominant resource bottleneck.",
            "risk_role": "reference_defense_state",
        },
        {
            "id": 2,
            "name": "Degraded_Defense",
            "definition": "Sensing or intercept resources are partially degraded.",
            "risk_role": "raises_context_risk",
        },
        {
            "id": 3,
            "name": "Resource_Saturated",
            "definition": "Defense resources are saturated by simultaneous tracks or decoys.",
            "risk_role": "raises_critical_miss_risk",
        },
    ],
    "environment": [
        {
            "id": 0,
            "name": "Open_Clear",
            "definition": "Clear airspace with stable tracking and low clutter.",
            "risk_role": "low_sensor_degradation",
        },
        {
            "id": 1,
            "name": "Urban_Clutter",
            "definition": "Cluttered background that increases false associations and route ambiguity.",
            "risk_role": "moderate_tracking_degradation",
        },
        {
            "id": 2,
            "name": "LowAltitude_Masking",
            "definition": "Low-altitude masking reduces effective detection and engagement quality.",
            "risk_role": "opportunity_and_sensing_degradation",
        },
        {
            "id": 3,
            "name": "Adverse_Weather",
            "definition": "Weather-induced sensing degradation and lower track confidence.",
            "risk_role": "strong_sensor_degradation",
        },
    ],
    "scenario_family": [
        {
            "id": 0,
            "name": "Probe_Surveillance",
            "definition": "Reconnaissance or weak-intent tracks under mostly recoverable defense conditions.",
            "risk_role": "low_to_moderate_reference_case",
        },
        {
            "id": 1,
            "name": "EW_Contested",
            "definition": "Electronic warfare or cluttered tracks with degraded sensing confidence.",
            "risk_role": "sensor_degradation_case",
        },
        {
            "id": 2,
            "name": "Strike_Penetration",
            "definition": "Precision-strike tracks approaching valuable assets.",
            "risk_role": "high_intent_opportunity_case",
        },
        {
            "id": 3,
            "name": "Saturation_Overload",
            "definition": "High-coordination swarm pressure under degraded or saturated defenses.",
            "risk_role": "hard_critical_case",
        },
    ],
}


DIFFICULTY_CONFIGS: dict[str, dict[str, Any]] = {
    "ATUAV-Core": {
        "difficulty_tier": "standard",
        "scenario_profile": "ATUAV-Core",
        "protocol": "atuav_core",
        "split_strategy": "stratified",
        "detection_window": "standard",
        "observed_len": 64,
        "range_m": 1000,
        "track_noise_std": 0.015,
        "track_missing_ratio": 0.0,
        "track_jitter_std": 0.0,
        "description": "Balanced in-distribution sequential assessment setting.",
    },
    "ATUAV-Noise": {
        "difficulty_tier": "noise",
        "scenario_profile": "ATUAV-SensorDegraded",
        "protocol": "atuav_noise",
        "split_strategy": "stratified",
        "detection_window": "contested_sensing",
        "observed_len": 64,
        "range_m": 5000,
        "track_noise_std": 0.025,
        "track_missing_ratio": 0.0,
        "track_jitter_std": 0.010,
        "description": "Far-range and cluttered sensing degradation without heavy dropout.",
    },
    "ATUAV-Missing": {
        "difficulty_tier": "missing",
        "scenario_profile": "ATUAV-SensorDegraded",
        "protocol": "atuav_missing",
        "split_strategy": "stratified",
        "detection_window": "contested_sensing",
        "observed_len": 64,
        "range_m": 3000,
        "track_noise_std": 0.018,
        "track_missing_ratio": 0.15,
        "track_jitter_std": 0.006,
        "description": "Contested tracks with intermittent sensor dropout and confidence loss.",
    },
    "ATUAV-OOD": {
        "difficulty_tier": "ood",
        "scenario_profile": "ATUAV-MissionShift",
        "protocol": "atuav_ood",
        "split_strategy": "scenario_holdout",
        "scenario_holdout_key": "mission_type",
        "detection_window": "standard",
        "observed_len": 64,
        "range_m": 3000,
        "track_noise_std": 0.018,
        "track_missing_ratio": 0.03,
        "track_jitter_std": 0.006,
        "description": "Mission-family holdout for out-of-distribution generalization.",
    },
    "ATUAV-Hard": {
        "difficulty_tier": "hard",
        "scenario_profile": "ATUAV-Saturation",
        "protocol": "atuav_hard",
        "split_strategy": "scenario_holdout",
        "scenario_holdout_key": "scenario_family",
        "detection_window": "contested_sensing",
        "observed_len": 32,
        "range_m": 5000,
        "track_noise_std": 0.030,
        "track_missing_ratio": 0.10,
        "track_jitter_std": 0.015,
        "description": "Short-observation saturation setting with range noise, jitter, and dropout.",
    },
}


SENSOR_DEGRADATION_MODEL: list[dict[str, Any]] = [
    {
        "component": "range_noise",
        "formula": "sigma = base_std * (1 + 2 * max(range_m - 1000, 0) / 4000)",
        "rationale": "Longer observation range increases angular and range-bin uncertainty.",
    },
    {
        "component": "environment_noise",
        "formula": "sigma = sigma * (1 + 0.6 * normalized_distance + 0.4 * I(environment>=2))",
        "rationale": "Distant, low-altitude, and adverse-weather tracks receive lower measurement quality.",
    },
    {
        "component": "track_jitter",
        "formula": "AR(1) jitter is added to heading, distance, time-to-arrival, and track confidence.",
        "rationale": "Track association jitter creates temporally correlated perturbations instead of iid noise.",
    },
    {
        "component": "track_missingness",
        "formula": "missing frames are forward-filled and track_confidence is reduced at missing frames.",
        "rationale": "Operational sensors often hold the previous estimate when detections are intermittent.",
    },
]


LABEL_RULES: list[dict[str, Any]] = [
    {
        "task": "threat",
        "score": "0.28 capability + 0.23 intent + 0.26 opportunity + 0.23 context, then swarm synergy",
        "thresholds": "0.26, 0.44, 0.64, 0.82",
        "labels": "; ".join(f"{label}:{name}" for label, name in THREAT_LEVELS.items()),
        "rationale": "Threat is an ordinal operational risk level, not a percentile label.",
    },
    {
        "task": "urgency",
        "score": "arrival pressure, distance, velocity, defense pressure, asset value, swarm pressure, and confidence",
        "thresholds": "0.36, 0.67",
        "labels": "; ".join(f"{label}:{name}" for label, name in URGENCY_LEVELS.items()),
        "rationale": "Urgency captures decision-window pressure and can differ from threat severity.",
    },
]


def dataset_registry_entries() -> dict[str, dict[str, Any]]:
    """Return dataset entries used by the assessment registry."""
    entries = {}
    for dataset_id, cfg in DIFFICULTY_CONFIGS.items():
        entries[dataset_id] = {
            "scenario_profile": cfg["scenario_profile"],
            "n_samples": 4000,
            "difficulty_tier": cfg["difficulty_tier"],
            "description": cfg["description"],
        }
    return entries


def difficulty_protocol_entries() -> dict[str, dict[str, Any]]:
    """Return protocol entries for the five canonical difficulty settings."""
    protocols = {}
    for dataset_id, cfg in DIFFICULTY_CONFIGS.items():
        protocol = {
            "split_strategy": cfg["split_strategy"],
            "detection_window": cfg["detection_window"],
            "noise_level": 0.0,
            "missing_ratio": cfg["track_missing_ratio"],
            "seq_len": 64,
            "observed_len": cfg["observed_len"],
            "type_as_input": False,
            "range_m": cfg["range_m"],
            "track_noise_std": cfg["track_noise_std"],
            "track_missing_ratio": cfg["track_missing_ratio"],
            "track_jitter_std": cfg["track_jitter_std"],
            "difficulty_tier": cfg["difficulty_tier"],
            "sensor_profile": dataset_id,
            "description": cfg["description"],
        }
        if "scenario_holdout_key" in cfg:
            protocol["scenario_holdout_key"] = cfg["scenario_holdout_key"]
        protocols[cfg["protocol"]] = protocol
    return protocols


def difficulty_suite_specs() -> list[dict[str, str]]:
    """Return explicit dataset/protocol pairs for the canonical assessment suite."""
    return [
        {"dataset": dataset_id, "protocol": cfg["protocol"]}
        for dataset_id, cfg in DIFFICULTY_CONFIGS.items()
    ]


def taxonomy_rows() -> list[dict[str, Any]]:
    rows = []
    for axis, entries in SCENE_TAXONOMY.items():
        for entry in entries:
            rows.append({"axis": axis, **entry})
    return rows


def difficulty_rows() -> list[dict[str, Any]]:
    return [{"dataset": dataset_id, **cfg} for dataset_id, cfg in DIFFICULTY_CONFIGS.items()]


def sensor_degradation_rows() -> list[dict[str, Any]]:
    return list(SENSOR_DEGRADATION_MODEL)


def label_rule_rows() -> list[dict[str, Any]]:
    return list(LABEL_RULES)


def difficulty_tier_for_dataset(dataset_id: str, scenario_profile: str | None = None) -> str:
    """Map a dataset or scenario profile to a compact difficulty tier."""
    if dataset_id in DIFFICULTY_CONFIGS:
        return str(DIFFICULTY_CONFIGS[dataset_id]["difficulty_tier"])
    profile = scenario_profile or dataset_id
    if profile == "ATUAV-Saturation":
        return "hard"
    if profile in {"ATUAV-MissionShift", "ATUAV-DefenseShift", "ATUAV-EnvironmentShift"}:
        return "ood"
    if profile == "ATUAV-SensorDegraded":
        return "degraded"
    return "standard"
