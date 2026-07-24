"""Experiment registry for the air-target UAV threat-assessment study.

The paper evidence chain contains comparison, ablation, two focused
sensitivity axes, reference-policy robustness, and scenario-family holdout.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any


DEFAULT_TASK_FORM = "instantaneous"
SEQUENTIAL_TASK_FORM = "sequential"

SUITE_SEQUENCE_KEYS = [
    "seq_len",
    "observed_len",
    "observation_window",
    "frame_interval",
    "range_m",
    "track_noise_std",
    "track_missing_ratio",
    "track_jitter_std",
    "type_as_input",
    "mission_as_input",
    "reference_policy_variant",
]

DATA_CONFIG_KEYS = ["scenario_holdout_key", "scenario_holdout_value"]
SEQUENCE_CONFIG_KEYS = [
    "seq_len",
    "observed_len",
    "observation_window",
    "frame_interval",
    "range_m",
    "track_noise_std",
    "track_missing_ratio",
    "track_jitter_std",
    "type_as_input",
    "mission_as_input",
    "reference_policy_variant",
]


def _dataset_entry(
    scenario_profile: str,
    description: str,
    *,
    n_samples: int,
) -> dict[str, Any]:
    return {
        "scenario_profile": scenario_profile,
        "description": description,
        "n_samples": n_samples,
    }


def _protocol_entry(
    description: str,
    *,
    split_strategy: str = "stratified",
    detection_window: str = "standard",
    noise_level: float = 0.0,
    missing_ratio: float = 0.0,
    **overrides: Any,
) -> dict[str, Any]:
    return {
        "description": description,
        "split_strategy": split_strategy,
        "detection_window": detection_window,
        "noise_level": noise_level,
        "missing_ratio": missing_ratio,
        **overrides,
    }


def _suite_entry(
    description: str,
    *,
    datasets: list[str],
    protocols: list[str],
    default_n_samples: int,
    task_form: str = DEFAULT_TASK_FORM,
    **overrides: Any,
) -> dict[str, Any]:
    return {
        "description": description,
        "datasets": datasets,
        "protocols": protocols,
        "default_n_samples": default_n_samples,
        "task_form": task_form,
        **overrides,
    }


def _sequential_setting(
    dataset_id: str,
    protocol_id: str,
    *,
    setting_name: str | None = None,
    description: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    payload = {
        "dataset": dataset_id,
        "protocol": protocol_id,
        **overrides,
    }
    if setting_name is not None:
        payload["setting_name"] = setting_name
    if description is not None:
        payload["setting_description"] = description
    return payload


def _comparison_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    return [
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name="ATUAV-Core__latent_state_masked",
            description="Default full-observation comparison setting.",
        ),
    ]


def _ablation_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    return [
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name="ATUAV-Core__latent_state_masked",
            description="Default ablation setting on the full observed window.",
        ),
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name="ATUAV-Core__latent_state_masked__ablation_obs32",
            description="Short-history ablation stress at 32 observed frames (6.4 s).",
            observed_len=32,
        ),
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name="ATUAV-Core__latent_state_masked__ablation_range5000",
            description="Far-range ablation stress at 5000 m nominal sensing range.",
            range_m=5000.0,
            track_noise_std=0.015,
            track_missing_ratio=0.08,
            track_jitter_std=0.012,
        ),
    ]


def _observed_time_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    return [
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name=f"ATUAV-Core__latent_state_masked__long_obs{observed_len}",
            description=(
                f"Long observed-time sensitivity at {observed_len} frames "
                f"({observed_len * 0.2:.1f} s)."
            ),
            seq_len=128,
            observed_len=observed_len,
        )
        for observed_len in [32, 64, 96, 128]
    ]


def _fixed_endpoint_observed_time_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    return [
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name=f"ATUAV-Core__latent_state_masked__fixed_endpoint_obs{observed_len}",
            description=(
                f"Fixed-endpoint history sensitivity using the final {observed_len} frames "
                f"({observed_len * 0.2:.1f} s) of a common 128-frame track."
            ),
            seq_len=128,
            observed_len=observed_len,
            observation_window="tail",
        )
        for observed_len in [32, 64, 96, 128]
    ]


def _fixed_endpoint_ablation_settings() -> list[dict[str, Any]]:
    return [
        _sequential_setting(
            "ATUAV-Core",
            "latent_state_masked",
            setting_name="ATUAV-Core__latent_state_masked__ablation_fixed_endpoint_obs32",
            description=(
                "Short-history ablation using the final 32 frames of a common "
                "64-frame track and the shared terminal labels."
            ),
            seq_len=64,
            observed_len=32,
            observation_window="tail",
        )
    ]


def _distance_degradation_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    settings: list[dict[str, Any]] = []
    for range_m in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        range_factor = 1.0 + 2.0 * max(range_m - 1000, 0) / 4000
        settings.append(
            _sequential_setting(
                dataset_id,
                protocol_id,
                setting_name=f"ATUAV-Core__latent_state_masked__range{range_m}",
                description=(
                    f"Range-driven sensing degradation at {range_m} m "
                    f"(noise multiplier {range_factor:.2f})."
                ),
                range_m=float(range_m),
                track_noise_std=0.015,
                track_missing_ratio=0.02 if range_m < 3000 else 0.05 if range_m < 4500 else 0.08,
                track_jitter_std=0.004 * range_factor,
            )
        )
    return settings


def _policy_robustness_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    return [
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name=f"ATUAV-Core__latent_state_masked__policy_{variant}",
            description=f"Reference-policy robustness under the {variant} policy variant.",
            reference_policy_variant=variant,
        )
        for variant in ["balanced", "consequence_first", "access_first"]
    ]


def _scenario_holdout_settings() -> list[dict[str, Any]]:
    dataset_id = "ATUAV-Core"
    protocol_id = "latent_state_masked"
    families = ["Probe_Surveillance", "EW_Contested", "Strike_Penetration", "Saturation_Overload"]
    return [
        _sequential_setting(
            dataset_id,
            protocol_id,
            setting_name=f"ATUAV-Core__latent_state_masked__holdout_{family}",
            description=f"Leave-{family}-out scenario-family generalization setting.",
            split_strategy="fixed_holdout",
            scenario_holdout_key="scenario_family",
            scenario_holdout_value=family,
        )
        for family in families
    ]


def _missing_robustness_settings() -> list[dict[str, Any]]:
    return [
        _sequential_setting(
            "ATUAV-Core",
            "latent_state_masked",
            setting_name="ATUAV-Core__latent_state_masked__test_missing",
            description=(
                "Frozen-model test-time robustness under random and contiguous "
                "frame missingness."
            ),
            evaluation_mode="frozen_test_missing",
            test_missing_modes=["random", "burst"],
            test_missing_rates=[0.0, 0.05, 0.10, 0.15, 0.20],
            test_confidence_decay=0.65,
        )
    ]


ASSESSMENT_DATASETS: dict[str, dict[str, Any]] = {
    "ATUAV-Core": _dataset_entry(
        "ATUAV-Core",
        "Core sequential air-target UAV threat-assessment dataset used by the manuscript.",
        n_samples=4000,
    ),
}


ASSESSMENT_PROTOCOLS: dict[str, dict[str, Any]] = {
    "standard": _protocol_entry("Compact IID split used only for static smoke validation."),
    "latent_state_masked": _protocol_entry(
        "Default realistic sequential protocol without oracle target-type or mission-code input.",
        seq_len=64,
        observed_len=64,
        frame_interval=0.2,
        type_as_input=False,
        mission_as_input=False,
        reference_policy_variant="balanced",
    ),
}


ASSESSMENT_SUITES: dict[str, dict[str, Any]] = {
    "smoke": _suite_entry(
        "Tiny static sanity suite for runner validation.",
        datasets=["ATUAV-Core"],
        protocols=["standard"],
        default_n_samples=256,
    ),
    "seq_smoke": _suite_entry(
        "Tiny sequential sanity suite for runner validation.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=128,
        task_form=SEQUENTIAL_TASK_FORM,
    ),
    "comparison": _suite_entry(
        "Official manuscript comparison experiment on the default sequential protocol.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_comparison_settings(),
    ),
    "ablation": _suite_entry(
        "Official manuscript ablation experiment on the default sequential protocol.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_ablation_settings(),
    ),
    "observed_time": _suite_entry(
        "Comparison sensitivity axis with long observed windows from 6.4 s to 25.6 s.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_observed_time_settings(),
    ),
    "fixed_endpoint_observed_time": _suite_entry(
        "Fixed-endpoint history sensitivity using tail windows with shared terminal labels.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_fixed_endpoint_observed_time_settings(),
    ),
    "fixed_endpoint_ablation": _suite_entry(
        "Short-history module ablation with a fixed terminal decision point.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_fixed_endpoint_ablation_settings(),
    ),
    "distance_degradation": _suite_entry(
        "Comparison sensitivity axis with range-driven observation degradation from 1000 m to 5000 m.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_distance_degradation_settings(),
    ),
    "policy_robustness": _suite_entry(
        "Reference-policy robustness across pre-specified operational policy variants.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_policy_robustness_settings(),
    ),
    "scenario_holdout": _suite_entry(
        "Leave-one-scenario-family-out generalization across the four operational families.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_scenario_holdout_settings(),
    ),
    "missing_robustness": _suite_entry(
        "Frozen-model robustness to random and contiguous test-time frame missingness.",
        datasets=["ATUAV-Core"],
        protocols=["latent_state_masked"],
        default_n_samples=4000,
        task_form=SEQUENTIAL_TASK_FORM,
        settings=_missing_robustness_settings(),
    ),
}


TRADITIONAL_MODELS = ["TOPSIS", "GRA", "Fuzzy", "Entropy-TOPSIS", "Combined-TOPSIS", "TemporalHMM"]
TEMPORAL_MODELS = [
    "LastFrameMLP",
    "MeanPoolMLP",
    "FlatSequenceMLP",
    "TemporalGRU",
    "TemporalLSTM",
    "TemporalTransformer",
    "TemporalTCN",
    "TemporalHGTAN",
]
SEQ_ABLATION_MODELS = [
    "TemporalHGTAN",
    "TemporalHGTAN_LastFrame",
    "TemporalHGTAN_MeanPool",
    "TemporalHGTAN_NoSynergy",
    "TemporalHGTAN_NoPrior",
]


MODEL_GROUPS: dict[str, list[str]] = {
    "traditional": TRADITIONAL_MODELS,
    "seq_curve": ["TOPSIS", "TemporalHMM", "TemporalGRU", "TemporalLSTM", "TemporalHGTAN"],
    "seq_lite": ["LastFrameMLP", "TemporalHGTAN"],
    "seq_main": TRADITIONAL_MODELS + TEMPORAL_MODELS,
    "seq_ablation": SEQ_ABLATION_MODELS,
    "seq_missing": [
        "MeanPoolMLP",
        "FlatSequenceMLP",
        "TemporalGRU",
        "TemporalLSTM",
        "TemporalTransformer",
        "TemporalTCN",
        "TemporalHGTAN",
    ],
    "seq_window": TEMPORAL_MODELS,
    "lite": ["TOPSIS", "MLP", "HGTAN"],
}


def get_dataset(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in ASSESSMENT_DATASETS:
        raise KeyError(_valid_message("dataset", dataset_id, ASSESSMENT_DATASETS))
    return deepcopy(ASSESSMENT_DATASETS[dataset_id])


def get_protocol(protocol_id: str) -> dict[str, Any]:
    if protocol_id not in ASSESSMENT_PROTOCOLS:
        raise KeyError(_valid_message("protocol", protocol_id, ASSESSMENT_PROTOCOLS))
    return deepcopy(ASSESSMENT_PROTOCOLS[protocol_id])


def get_suite_settings(suite_name: str) -> list[dict[str, Any]]:
    if suite_name not in ASSESSMENT_SUITES:
        raise KeyError(_valid_message("suite", suite_name, ASSESSMENT_SUITES))

    suite = ASSESSMENT_SUITES[suite_name]
    task_form = suite.get("task_form", DEFAULT_TASK_FORM)
    settings: list[dict[str, Any]] = []
    for dataset_id, protocol_id, explicit_overrides in _iter_setting_specs(suite):
        dataset = get_dataset(dataset_id)
        protocol = get_protocol(protocol_id)
        setting = {
            "dataset": dataset_id,
            "protocol": protocol_id,
            "scenario_profile": dataset["scenario_profile"],
            "n_samples": suite.get("default_n_samples", dataset["n_samples"]),
            "task_form": task_form,
            **protocol,
            "dataset_description": dataset["description"],
            "protocol_description": protocol["description"],
        }
        setting.update(explicit_overrides)
        for key in SUITE_SEQUENCE_KEYS:
            if key in suite and key not in setting:
                setting[key] = suite[key]
        settings.append(setting)
    return settings


def apply_assessment_setting(config: dict[str, Any], setting: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    data_cfg = updated["data"]
    data_cfg["benchmark_dataset"] = setting["dataset"]
    data_cfg["scenario_profile"] = setting["scenario_profile"]
    data_cfg["n_samples"] = setting["n_samples"]
    data_cfg["split_strategy"] = setting["split_strategy"]
    data_cfg["detection_window"] = setting["detection_window"]
    data_cfg["noise_level"] = setting["noise_level"]
    data_cfg["missing_ratio"] = setting["missing_ratio"]
    for key in DATA_CONFIG_KEYS:
        if key in setting:
            data_cfg[key] = setting[key]

    sequence_cfg = updated.setdefault("sequence", {})
    for key in SEQUENCE_CONFIG_KEYS:
        if key in setting:
            sequence_cfg[key] = setting[key]
    return updated


# Backward-compatible aliases for older code paths.
BENCHMARK_DATASETS = ASSESSMENT_DATASETS
BENCHMARK_PROTOCOLS = ASSESSMENT_PROTOCOLS
BENCHMARK_SUITES = ASSESSMENT_SUITES
apply_benchmark_setting = apply_assessment_setting


def make_setting_name(setting: dict[str, Any], seed: int | None = None) -> str:
    if "setting_name" in setting:
        base = str(setting["setting_name"])
        parts = [base]
    else:
        parts = [setting["dataset"], setting["protocol"]]
    if seed is not None:
        parts.append(f"seed{seed}")
    return "__".join(_slug(part) for part in parts)


def parse_model_list(model_arg: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(model_arg, str):
        tokens = [token.strip() for token in model_arg.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in model_arg if str(token).strip()]

    if not tokens:
        return MODEL_GROUPS["lite"].copy()

    expanded: list[str] = []
    for token in tokens:
        if token in MODEL_GROUPS:
            expanded.extend(MODEL_GROUPS[token])
        else:
            expanded.append(token)
    return list(dict.fromkeys(expanded))


def _iter_setting_specs(suite: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    if "settings" in suite:
        return [
            (
                spec["dataset"],
                spec["protocol"],
                {key: value for key, value in spec.items() if key not in {"dataset", "protocol"}},
            )
            for spec in suite["settings"]
        ]
    return [
        (dataset_id, protocol_id, {})
        for dataset_id, protocol_id in product(suite["datasets"], suite["protocols"])
    ]


def _slug(value: str) -> str:
    return value.replace(" ", "").replace("/", "-").replace("\\", "-")


def _valid_message(kind: str, selected: str, registry: dict[str, Any]) -> str:
    valid = ", ".join(sorted(registry))
    return f"Unknown {kind}={selected!r}. Valid options: {valid}"
