"""Execution engine for ATUAV assessment experiments."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data import (
    build_data_profile_rows,
    build_feature_profile_rows,
    build_joint_train_kwargs,
    data_provider,
    resolve_class_weights,
    sequence_data_provider,
)
from exp.registry import apply_assessment_setting, make_setting_name
from exp.result_writer import setting_context, write_json, write_setting_outputs
from models.model_factory import (
    SEQUENTIAL_MODELS,
    build_model,
    get_selected_sequential_models,
    get_selected_traditional_models,
    get_selected_trainable_models,
    validate_model_names,
)
from utils.config import HGTANConfig, device, get_run_seeds, set_random_seed, to_serializable
from utils.metrics import (
    build_classification_metrics,
    compute_composite_f1,
    count_parameters,
    evaluate_model,
    measure_inference_time,
    train_model,
)
from utils.sequence_metrics import compute_track_metrics, predict_prefix_labels
from utils.tools import apply_cli_overrides


SCENARIO_GROUP_KEYS = [
    "mission_name",
    "target_name",
    "defense_name",
    "environment_name",
    "formation_name",
    "asset_name",
    "scenario_group",
    "scenario_family",
    "benchmark_dataset",
    "scenario_profile",
    "difficulty_tier",
    "sensor_profile",
    "detection_window",
]
ERROR_METADATA_KEYS = [
    "track_id",
    "mission_name",
    "target_name",
    "defense_name",
    "environment_name",
    "formation_name",
    "asset_name",
    "scenario_group",
    "scenario_family",
    "benchmark_dataset",
    "scenario_profile",
    "difficulty_tier",
    "sensor_profile",
    "detection_window",
    "range_m",
    "track_noise_std",
    "track_missing_ratio",
    "track_jitter_std",
    "type_as_input",
]

TRACK_METRIC_NON_SUMMARY_FIELDS = {
    "setting",
    "dataset",
    "protocol",
    "scenario_profile",
    "task_form",
    "split_strategy",
    "detection_window",
    "noise_level",
    "missing_ratio",
    "seq_len",
    "observed_len",
    "frame_interval",
    "track_noise_std",
    "range_m",
    "type_as_input",
    "run_index",
    "seed",
    "model",
    "task",
    "critical_labels",
    "support_tracks",
    "support_timepoints",
    "critical_track_support",
}
DEFAULT_TASK_FORM = "instantaneous"
SEQUENTIAL_TASK_FORM = "sequential"


@dataclass(frozen=True)
class AssessmentOptions:
    """Runtime overrides for an assessment experiment suite."""

    suite: str
    mode: str
    models: list[str]
    out_root: Path
    num_runs: int | None = None
    epochs: int | None = None
    skip_existing: bool = False
    cli_args: Any | None = None


class Exp_Main:
    """Run an assessment experiment suite composed of dataset/protocol settings."""

    def __init__(self, settings: list[dict[str, Any]], options: AssessmentOptions):
        self.settings = settings
        self.options = options
        self.options.out_root.mkdir(parents=True, exist_ok=True)

    def dry_run(self) -> None:
        self.write_manifest()
        print_header(self.options.suite, self.options.out_root, self.options.models, self.settings)
        for setting in self.settings:
            print("  - " + describe_setting(setting))
        print(f"\nDry-run manifest written to: {self.options.out_root / 'run_manifest.json'}")

    def run(self) -> list[dict[str, Any]]:
        self.write_manifest()
        print_header(self.options.suite, self.options.out_root, self.options.models, self.settings)

        global_rows: list[dict[str, Any]] = []
        for setting_idx, setting in enumerate(self.settings, start=1):
            print(f"\n[{setting_idx}/{len(self.settings)}] {describe_setting(setting)}")
            global_rows.extend(self.run_setting(setting))

        if global_rows:
            global_csv = self.options.out_root / "summary.csv"
            pd.DataFrame(global_rows).to_csv(global_csv, index=False)
            print(f"\nGlobal summary: {global_csv}")

        print("\nAssessment experiments completed.")
        return global_rows

    def run_setting(self, setting: dict[str, Any]) -> list[dict[str, Any]]:
        config = self._build_config(setting)
        seeds = get_run_seeds(config["run"])
        print_runtime(config)
        setting_name = make_setting_name(setting)
        setting_dir = self.options.out_root / setting_name
        if self.options.skip_existing and setting_output_complete(
            setting_dir,
            expected_models=self.options.models,
            expected_task_form=setting.get("task_form", DEFAULT_TASK_FORM),
        ):
            print(f"  Skipped existing complete setting: {setting_dir}")
            return read_existing_summary(setting_dir)

        seed_runner = run_sequence_single_seed if is_sequential_setting(setting) else run_single_seed
        records = []
        for run_idx, seed in enumerate(seeds):
            print(f"  Run {run_idx + 1}/{len(seeds)} | seed={seed}")
            records.append(seed_runner(setting, config, self.options.models, seed, run_idx, setting_name))

        summary_rows = annotate_summary_rows(summarize_results(records), setting_name, setting)
        write_setting_outputs(
            setting_dir=setting_dir,
            setting_name=setting_name,
            setting=setting,
            model_names=self.options.models,
            records=records,
            summary_rows=summary_rows,
            config=config,
            cli_args=self.options.cli_args,
        )
        print(f"  Saved: {setting_dir}")
        return summary_rows

    def write_manifest(self) -> None:
        manifest = {
            "suite": self.options.suite,
            "mode": self.options.mode,
            "models": self.options.models,
            "n_settings": len(self.settings),
            "settings": [to_serializable(setting) for setting in self.settings],
            "overrides": {
                "num_runs": self.options.num_runs,
                "epochs": self.options.epochs,
                "skip_existing": self.options.skip_existing,
                "cli_args": vars(self.options.cli_args) if self.options.cli_args is not None else None,
            },
        }
        write_json(self.options.out_root / "run_manifest.json", manifest)

    def _build_config(self, setting: dict[str, Any]) -> dict[str, Any]:
        config = HGTANConfig.get_experiment_config("benchmark", mode=self.options.mode)
        config = apply_assessment_setting(config, setting)
        apply_runtime_overrides(config, self.options)
        validate_runtime_config(config, self.options.mode)
        return config


def run_single_seed(
    setting: dict[str, Any],
    config: dict[str, Any],
    selected_models: list[str],
    seed: int,
    run_idx: int,
    setting_name: str,
) -> dict[str, Any]:
    """Run one instantaneous seed through the shared execution skeleton."""
    return run_seed_record(setting, config, selected_models, seed, run_idx, setting_name)


def run_sequence_single_seed(
    setting: dict[str, Any],
    config: dict[str, Any],
    selected_models: list[str],
    seed: int,
    run_idx: int,
    setting_name: str,
) -> dict[str, Any]:
    """Run one sequential seed through the shared execution skeleton."""
    return run_seed_record(setting, config, selected_models, seed, run_idx, setting_name)


def run_seed_record(
    setting: dict[str, Any],
    config: dict[str, Any],
    selected_models: list[str],
    seed: int,
    run_idx: int,
    setting_name: str,
) -> dict[str, Any]:
    """Run one seed for either task form using one shared orchestration skeleton."""
    set_random_seed(seed, config)
    data_bundle = load_seed_data_bundle(setting, config, seed)
    print_loader_profile(data_bundle)
    data_profile_rows, feature_profile_rows = build_audit_rows(data_bundle, setting, setting_name, run_idx, seed)

    execution = execute_model_suite(
        setting=setting,
        config=config,
        data_bundle=data_bundle,
        selected_models=selected_models,
        setting_name=setting_name,
        run_idx=run_idx,
        seed=seed,
    )
    scenario_rows = build_scenario_metric_rows(
        predictions=execution["predictions"],
        metadata=data_bundle["metadata_test"],
        setting=setting,
        setting_name=setting_name,
        run_idx=run_idx,
        seed=seed,
    )
    error_rows = build_error_case_rows(
        predictions=execution["predictions"],
        metadata=data_bundle["metadata_test"],
        setting=setting,
        setting_name=setting_name,
        run_idx=run_idx,
        seed=seed,
    )

    record = {
        "assessment_setting": build_assessment_setting_record(setting),
        "run_index": run_idx,
        "seed": seed,
        "split_strategy": data_bundle["split_strategy"],
        "results": to_serializable(execution["results"]),
        "efficiency": to_serializable(execution["efficiency"]),
        "predictions": to_serializable(execution["predictions"]),
        "training": to_serializable(execution["training"]),
        "scenario_metrics": to_serializable(scenario_rows),
        "error_cases": to_serializable(error_rows),
        "data_profile": to_serializable(data_profile_rows),
        "feature_profile": to_serializable(feature_profile_rows),
    }
    if is_sequential_setting(setting):
        record.update(
            {
                "seq_len": data_bundle["seq_len"],
                "observed_len": data_bundle["observed_len"],
                "track_metrics": to_serializable(execution["track_metric_rows"]),
                "operational_case": to_serializable(
                    build_operational_case_record(
                        data_bundle=data_bundle,
                        predictions=execution["predictions"],
                        setting=setting,
                        setting_name=setting_name,
                        run_idx=run_idx,
                        seed=seed,
                    )
                ),
            }
        )
    return record


def load_seed_data_bundle(
    setting: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Load the task-appropriate data bundle for one seed."""
    if is_sequential_setting(setting):
        return sequence_data_provider(config, seed).copy()
    return data_provider(config, seed, enforce_min_class_samples=True).as_dict()


def execute_model_suite(
    *,
    setting: dict[str, Any],
    config: dict[str, Any],
    data_bundle: dict[str, Any],
    selected_models: list[str],
    setting_name: str,
    run_idx: int,
    seed: int,
) -> dict[str, Any]:
    """Execute traditional and trainable models under one shared task-aware skeleton."""
    sequential = is_sequential_setting(setting)
    traditional_models, trainable_models, traditional_label, trainable_label = resolve_model_groups_for_setting(
        setting,
        selected_models,
    )

    results: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    training: dict[str, Any] = {}
    track_metric_rows: list[dict[str, Any]] = []

    if traditional_models:
        print(f"    [{traditional_label}]")
        if sequential:
            traditional_results, traditional_efficiency, traditional_predictions, traditional_track_rows = (
                evaluate_sequence_traditional_methods(
                    data_bundle,
                    traditional_models,
                    setting=setting,
                    setting_name=setting_name,
                    run_idx=run_idx,
                    seed=seed,
                    frame_interval=config["sequence"].get("frame_interval", 1.0),
                )
            )
            track_metric_rows.extend(traditional_track_rows)
        else:
            traditional_results, traditional_efficiency, traditional_predictions = evaluate_traditional_methods(
                data_bundle,
                traditional_models,
            )
        results.update(traditional_results)
        efficiency.update(traditional_efficiency)
        predictions.update(traditional_predictions)

    if trainable_models:
        print(f"    [{trainable_label}]")
    for model_name in trainable_models:
        set_random_seed(seed, config)
        print(f"    Training {model_name} ...")
        if sequential:
            model_results, model_efficiency, model_predictions, model_track_rows, model_training = (
                run_sequence_trainable_model(
                    model_name,
                    data_bundle,
                    config,
                    setting=setting,
                    setting_name=setting_name,
                    run_idx=run_idx,
                    seed=seed,
                )
            )
            track_metric_rows.extend(model_track_rows)
        else:
            model_results, model_efficiency, model_predictions, model_training = run_trainable_model(
                model_name,
                data_bundle,
                config,
            )
        results[model_name] = model_results
        efficiency[model_name] = model_efficiency
        predictions[model_name] = model_predictions
        training[model_name] = model_training
        release_runtime_cache()

    return {
        "results": results,
        "efficiency": efficiency,
        "predictions": predictions,
        "training": training,
        "track_metric_rows": track_metric_rows,
    }


def resolve_model_groups_for_setting(
    setting: dict[str, Any],
    selected_models: list[str],
) -> tuple[dict[str, Any], list[str], str, str]:
    """Resolve traditional/trainable model groups for the current task form."""
    traditional_models = get_selected_traditional_models(selected_models)
    if not is_sequential_setting(setting):
        return traditional_models, get_selected_trainable_models(selected_models), "Traditional", "Trainable"

    sequence_models = get_selected_sequential_models(selected_models)
    traditional_names = set(traditional_models)
    unsupported = [name for name in selected_models if name not in SEQUENTIAL_MODELS and name not in traditional_names]
    if unsupported:
        valid_text = ", ".join(sorted(SEQUENTIAL_MODELS))
        raise ValueError(
            f"Sequential task currently supports: {valid_text}. "
            f"Unsupported selected model(s): {', '.join(unsupported)}"
        )
    return traditional_models, sequence_models, "Sequential Traditional", "Sequential"


def build_assessment_setting_record(setting: dict[str, Any]) -> dict[str, Any]:
    """Return compact setting identity fields stored inside each run record."""
    record = {
        "dataset": setting["dataset"],
        "protocol": setting["protocol"],
        "scenario_profile": setting["scenario_profile"],
    }
    if is_sequential_setting(setting):
        record["task_form"] = SEQUENTIAL_TASK_FORM
    return record


def is_sequential_setting(setting: dict[str, Any]) -> bool:
    """Return True when the setting uses sequential track inputs."""
    return setting.get("task_form", DEFAULT_TASK_FORM) == SEQUENTIAL_TASK_FORM


def evaluate_traditional_methods(
    data_bundle: dict[str, Any],
    traditional_models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit and evaluate traditional baselines on the current split."""
    results: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    predictions: dict[str, Any] = {}

    for name, model in traditional_models.items():
        try:
            fit_start = time.perf_counter()
            model.fit(data_bundle["X_train"], data_bundle["t_train"], data_bundle["u_train"])
            fit_time = time.perf_counter() - fit_start

            threat_pred, urgency_pred, predict_time = timed_traditional_predict(
                model,
                data_bundle["X_test"],
            )

            threat_metrics = build_classification_metrics(
                data_bundle["t_test"],
                threat_pred,
                critical_labels_1based=[4, 5],
            )
            urgency_metrics = build_classification_metrics(
                data_bundle["u_test"],
                urgency_pred,
                critical_labels_1based=[3],
            )
            results[name] = {"threat": threat_metrics, "urgency": urgency_metrics}
            efficiency[name] = build_efficiency_record(
                params=0,
                train_time=fit_time,
                inference_time=predict_time * 1000.0,
                n_inference_samples=len(data_bundle["X_test"]),
                inference_unit="full_test_set",
            )
            predictions[name] = build_prediction_payload(
                data_bundle["t_test"],
                data_bundle["u_test"],
                threat_pred,
                urgency_pred,
            )
            print(
                f"    {name}: Threat F1={threat_metrics['f1']:.4f}, "
                f"Urgency F1={urgency_metrics['f1']:.4f}, "
                f"Composite={compute_composite_f1(results[name]):.4f}"
            )
        except Exception as exc:
            print(f"    {name}: Error - {exc}")
            results[name] = empty_metric_record()

    return results, efficiency, predictions


def evaluate_sequence_traditional_methods(
    data_bundle: dict[str, Any],
    traditional_models: dict[str, Any],
    *,
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
    frame_interval: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Apply domain threat-assessment baselines to sequential tracks.

    Traditional methods are fitted on the final observed frame, then evaluated
    both at the final frame and over every prefix. This makes TOPSIS/GRA/Fuzzy
    comparable with recurrent models in observed-time, distance, and holdout
    suites without pretending that they are sequence learners.
    """
    results: dict[str, Any] = {}
    efficiency: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    track_metric_rows: list[dict[str, Any]] = []

    for name, model in traditional_models.items():
        try:
            fit_start = time.perf_counter()
            if getattr(model, "uses_sequence_input", False):
                model.fit_sequence(
                    data_bundle["X_train"],
                    data_bundle["threat_seq_train"],
                    data_bundle["urgency_seq_train"],
                )
                predict_input = data_bundle["X_test"]
            else:
                model.fit(data_bundle["X_train"][:, -1, :], data_bundle["t_train"], data_bundle["u_train"])
                predict_input = data_bundle["X_test"][:, -1, :]
            fit_time = time.perf_counter() - fit_start

            threat_pred, urgency_pred, predict_time = timed_traditional_predict(model, predict_input)
            threat_seq_pred, urgency_seq_pred = predict_sequence_traditional_prefix_labels(
                model,
                data_bundle["X_test"],
            )

            threat_metrics = build_classification_metrics(
                data_bundle["t_test"],
                threat_pred,
                critical_labels_1based=[4, 5],
            )
            urgency_metrics = build_classification_metrics(
                data_bundle["u_test"],
                urgency_pred,
                critical_labels_1based=[3],
            )
            results[name] = {"threat": threat_metrics, "urgency": urgency_metrics}
            efficiency[name] = build_efficiency_record(
                params=0,
                train_time=fit_time,
                inference_time=predict_time * 1000.0,
                n_inference_samples=len(data_bundle["X_test"]),
                inference_unit="full_test_set",
            )
            predictions[name] = build_prediction_payload(
                data_bundle["t_test"],
                data_bundle["u_test"],
                threat_pred,
                urgency_pred,
            )
            predictions[name].update(
                {
                    "threat_seq_true": data_bundle["threat_seq_test"],
                    "threat_seq_pred": threat_seq_pred,
                    "urgency_seq_true": data_bundle["urgency_seq_test"],
                    "urgency_seq_pred": urgency_seq_pred,
                }
            )
            track_metric_rows.extend(
                build_track_metric_rows(
                    model_name=name,
                    predictions=predictions[name],
                    setting=setting,
                    setting_name=setting_name,
                    run_idx=run_idx,
                    seed=seed,
                    frame_interval=frame_interval,
                )
            )
            print(
                f"    {name}: Threat F1={threat_metrics['f1']:.4f}, "
                f"Urgency F1={urgency_metrics['f1']:.4f}, "
                f"Composite={compute_composite_f1(results[name]):.4f}"
            )
        except Exception as exc:
            print(f"    {name}: Error - {exc}")
            results[name] = empty_metric_record()

    return results, efficiency, predictions, track_metric_rows


def predict_sequence_traditional_prefix_labels(
    model: Any,
    sequences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels for each prefix by applying a traditional model to its last frame."""
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError(f"Expected sequence tensor with shape (n, time, features), got {sequences.shape}")
    if getattr(model, "uses_sequence_input", False) and hasattr(model, "predict_sequence"):
        return model.predict_sequence(sequences)

    threat_steps = []
    urgency_steps = []
    for end_step in range(1, sequences.shape[1] + 1):
        threat_pred, urgency_pred = model.predict(sequences[:, end_step - 1, :])
        threat_steps.append(np.asarray(threat_pred, dtype=np.int64))
        urgency_steps.append(np.asarray(urgency_pred, dtype=np.int64))
    return np.stack(threat_steps, axis=1), np.stack(urgency_steps, axis=1)


def timed_traditional_predict(
    model: Any,
    x_test: np.ndarray,
    *,
    min_total_seconds: float = 0.005,
    max_repeats: int = 25,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict once for outputs and repeat briefly for stable latency on tiny tests."""
    start = time.perf_counter()
    threat_pred, urgency_pred = model.predict(x_test)
    elapsed = time.perf_counter() - start

    total = elapsed
    repeats = 1
    while total < min_total_seconds and repeats < max_repeats:
        start = time.perf_counter()
        model.predict(x_test)
        total += time.perf_counter() - start
        repeats += 1

    return threat_pred, urgency_pred, total / repeats


def run_trainable_model(
    model_name: str,
    data_bundle: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Train and evaluate one neural baseline."""
    results, efficiency, predictions, _, training = fit_trainable_model(model_name, data_bundle, config)
    return results, efficiency, predictions, training


def run_sequence_trainable_model(
    model_name: str,
    data_bundle: dict[str, Any],
    config: dict[str, Any],
    *,
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Train, evaluate, and compute temporal track metrics for one model."""
    results, efficiency, predictions, trained_model, training = fit_trainable_model(model_name, data_bundle, config)
    threat_seq_pred, urgency_seq_pred = predict_prefix_labels(
        trained_model,
        data_bundle["X_test"],
        batch_size=_loader_batch_size(data_bundle["test_loader"], fallback=config["train"]["batch_size"]),
        use_amp=config["train"].get("use_amp", False),
    )
    predictions.update(
        {
            "threat_seq_true": data_bundle["threat_seq_test"],
            "threat_seq_pred": threat_seq_pred,
            "urgency_seq_true": data_bundle["urgency_seq_test"],
            "urgency_seq_pred": urgency_seq_pred,
        }
    )
    track_rows = build_track_metric_rows(
        model_name=model_name,
        predictions=predictions,
        setting=setting,
        setting_name=setting_name,
        run_idx=run_idx,
        seed=seed,
        frame_interval=config["sequence"].get("frame_interval", 1.0),
    )
    return results, efficiency, predictions, track_rows, training


def fit_trainable_model(
    model_name: str,
    data_bundle: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    """Shared train/evaluate path used by static and sequential neural models."""
    train_cfg = config["train"]
    model_cfg = config["model"]

    class_weight_threat, class_weight_urgency = resolve_class_weights(
        train_cfg,
        data_bundle["t_train_0"],
        data_bundle["u_train_0"],
    )
    train_kwargs = build_joint_train_kwargs(
        train_cfg,
        class_weight_threat=class_weight_threat,
        class_weight_urgency=class_weight_urgency,
    )

    model = build_model(model_name, model_cfg)
    start_time = time.time()
    train_result = train_model(
        model,
        data_bundle["train_loader"],
        data_bundle["val_loader"],
        **train_kwargs,
    )
    train_time = time.time() - start_time

    trained_model = train_result["model"]
    threat_metrics, urgency_metrics, threat_pred_0, urgency_pred_0 = evaluate_model(
        trained_model,
        data_bundle["test_loader"],
        use_amp=train_cfg.get("use_amp", False),
    )
    inference_time = measure_inference_time(
        trained_model,
        data_bundle["test_loader"],
        n_runs=config["run"].get("latency_batches", 20),
        use_amp=train_cfg.get("use_amp", False),
    )
    measured_eval_batch = _loader_batch_size(data_bundle["test_loader"], fallback=len(data_bundle["X_test"]))
    results = {"threat": threat_metrics, "urgency": urgency_metrics}
    efficiency = build_efficiency_record(
        params=count_parameters(trained_model),
        train_time=train_time,
        inference_time=inference_time,
        n_inference_samples=measured_eval_batch,
        inference_unit="batch",
    )
    predictions = build_prediction_payload(
        data_bundle["t_test"],
        data_bundle["u_test"],
        np.asarray(threat_pred_0, dtype=np.int64) + 1,
        np.asarray(urgency_pred_0, dtype=np.int64) + 1,
    )
    training = build_training_record(train_result, train_time)
    training["info"].update(
        {
            "train_batch_size": _loader_batch_size(data_bundle["train_loader"], fallback=train_cfg["batch_size"]),
            "val_batch_size": _loader_batch_size(data_bundle["val_loader"], fallback=train_cfg["batch_size"]),
            "test_batch_size": measured_eval_batch,
        }
    )
    return results, efficiency, predictions, trained_model, training


def build_training_record(train_result: dict[str, Any], train_time: float) -> dict[str, Any]:
    """Normalize neural training curves and early-stopping metadata."""
    train_losses = list(train_result.get("train_losses", []))
    val_losses = list(train_result.get("val_losses", []))
    val_scores = list(train_result.get("val_scores", []))
    learning_rates = list(train_result.get("training_info", {}).get("learning_rates", []))
    n_epochs = max(len(train_losses), len(val_losses), len(val_scores), len(learning_rates))

    curves = []
    for epoch_idx in range(n_epochs):
        curves.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": _list_get(train_losses, epoch_idx),
                "val_loss": _list_get(val_losses, epoch_idx),
                "val_score": _list_get(val_scores, epoch_idx),
                "learning_rate": _list_get(learning_rates, epoch_idx),
            }
        )

    info = dict(train_result.get("training_info", {}))
    info.pop("learning_rates", None)
    info["train_time"] = float(train_time)
    info["n_curve_epochs"] = int(n_epochs)
    return {"info": info, "curves": curves}


def _list_get(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def build_efficiency_record(
    *,
    params: int,
    train_time: float,
    inference_time: float,
    n_inference_samples: int,
    inference_unit: str,
) -> dict[str, Any]:
    """Normalize model efficiency fields across neural and traditional baselines."""
    n_inference_samples = max(int(n_inference_samples), 1)
    return {
        "params": int(params),
        "train_time": float(train_time),
        "inference_time_ms": float(inference_time),
        "inference_time_ms_per_sample": float(inference_time / n_inference_samples),
        "throughput_samples_per_second": float(1000.0 * n_inference_samples / max(inference_time, 1e-12)),
        "measured_batch_size": int(n_inference_samples),
        "inference_unit": inference_unit,
    }


def _loader_batch_size(loader: Any, *, fallback: int) -> int:
    batch_size = getattr(loader, "batch_size", None)
    return max(int(batch_size or fallback), 1)


def build_prediction_payload(
    threat_true: np.ndarray,
    urgency_true: np.ndarray,
    threat_pred: np.ndarray,
    urgency_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """Store predictions in 1-based label space for readable artifacts."""
    return {
        "threat_true": np.asarray(threat_true, dtype=np.int64),
        "threat_pred": np.asarray(threat_pred, dtype=np.int64),
        "urgency_true": np.asarray(urgency_true, dtype=np.int64),
        "urgency_pred": np.asarray(urgency_pred, dtype=np.int64),
    }


def build_scenario_metric_rows(
    *,
    predictions: dict[str, Any],
    metadata: dict[str, Any],
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute metrics by mission/defense/environment/formation scenario groups."""
    rows: list[dict[str, Any]] = []
    context = setting_context(setting_name, setting)

    for model_name, payload in predictions.items():
        n_samples = len(payload["threat_true"])
        for group_key in SCENARIO_GROUP_KEYS:
            if group_key not in metadata:
                continue
            group_values = np.asarray(metadata[group_key])
            if group_values.ndim == 0 or len(group_values) != n_samples:
                continue

            for group_value in sorted(np.unique(group_values).tolist(), key=str):
                mask = group_values == group_value
                support = int(mask.sum())
                if support == 0:
                    continue

                for task, critical_labels in [("threat", [4, 5]), ("urgency", [3])]:
                    metrics = build_classification_metrics(
                        np.asarray(payload[f"{task}_true"])[mask],
                        np.asarray(payload[f"{task}_pred"])[mask],
                        critical_labels_1based=critical_labels,
                    )
                    row = {
                        **context,
                        "run_index": run_idx,
                        "seed": seed,
                        "model": model_name,
                        "task": task,
                        "group_key": group_key,
                        "group_value": str(group_value),
                        "support": support,
                    }
                    row.update(metrics)
                    rows.append(row)
    return rows


def build_error_case_rows(
    *,
    predictions: dict[str, Any],
    metadata: dict[str, Any],
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Record final-step misclassification cases for later qualitative analysis."""
    rows: list[dict[str, Any]] = []
    context = setting_context(setting_name, setting)
    metadata_arrays = {
        key: np.asarray(value)
        for key, value in metadata.items()
        if key in ERROR_METADATA_KEYS and isinstance(value, np.ndarray) and np.asarray(value).ndim > 0
    }

    for model_name, payload in predictions.items():
        for task, critical_labels in [("threat", [4, 5]), ("urgency", [3])]:
            true_labels = np.asarray(payload[f"{task}_true"], dtype=np.int64)
            pred_labels = np.asarray(payload[f"{task}_pred"], dtype=np.int64)
            error_indices = np.flatnonzero(true_labels != pred_labels)
            for sample_index in error_indices.tolist():
                true_label = int(true_labels[sample_index])
                pred_label = int(pred_labels[sample_index])
                row = {
                    **context,
                    "run_index": run_idx,
                    "seed": seed,
                    "model": model_name,
                    "task": task,
                    "sample_index": sample_index,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "ordinal_error": int(abs(true_label - pred_label)),
                    "under_estimation": bool(pred_label < true_label),
                    "over_estimation": bool(pred_label > true_label),
                    "critical_miss": bool(true_label in critical_labels and pred_label < true_label),
                }
                for key, values in metadata_arrays.items():
                    if len(values) == len(true_labels):
                        row[key] = str(values[sample_index])
                rows.append(row)
    return rows


def build_audit_rows(
    data_bundle: dict[str, Any],
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build data audit artifacts shared by static and sequential settings."""
    context = setting_context(setting_name, setting)
    return (
        build_data_profile_rows(data_bundle, context=context, run_idx=run_idx, seed=seed),
        build_feature_profile_rows(data_bundle, context=context, run_idx=run_idx, seed=seed),
    )


def build_track_metric_rows(
    *,
    model_name: str,
    predictions: dict[str, Any],
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
    frame_interval: float,
) -> list[dict[str, Any]]:
    """Compute temporal metrics over the full observed track for one model."""
    context = setting_context(setting_name, setting)
    task_specs = [
        ("threat", [4, 5], predictions["threat_seq_true"], predictions["threat_seq_pred"]),
        ("urgency", [3], predictions["urgency_seq_true"], predictions["urgency_seq_pred"]),
    ]
    rows = []
    for task, critical_labels, true_seq, pred_seq in task_specs:
        metrics = compute_track_metrics(
            true_seq,
            pred_seq,
            critical_labels=critical_labels,
            frame_interval=frame_interval,
        )
        row = {
            **context,
            "run_index": run_idx,
            "seed": seed,
            "model": model_name,
            "task": task,
            "critical_labels": ",".join(str(label) for label in critical_labels),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def build_operational_case_record(
    *,
    data_bundle: dict[str, Any],
    predictions: dict[str, Any],
    setting: dict[str, Any],
    setting_name: str,
    run_idx: int,
    seed: int,
) -> dict[str, Any]:
    """Select one high-risk sequential case for dynamic threat-curve figures."""
    if "threat_seq_test" not in data_bundle or not predictions:
        return {}

    threat_true = np.asarray(data_bundle["threat_seq_test"], dtype=np.int64)
    urgency_true = np.asarray(data_bundle["urgency_seq_test"], dtype=np.int64)
    if threat_true.ndim != 2 or len(threat_true) == 0:
        return {}

    critical = threat_true >= 4
    escalates = threat_true[:, -1] > threat_true[:, 0]
    candidates = np.flatnonzero(np.any(critical, axis=1) & escalates)
    if len(candidates) == 0:
        candidates = np.flatnonzero(np.any(critical, axis=1))
    if len(candidates) == 0:
        candidates = np.asarray([int(np.argmax(threat_true[:, -1]))])

    metadata = data_bundle.get("metadata_test", {})
    observed_len = int(data_bundle.get("observed_len", threat_true.shape[1]))
    frame_interval = float(setting.get("frame_interval", 0.2))
    scored_candidates: list[tuple[float, int, dict[str, Any]]] = []
    for candidate_index in candidates.tolist():
        model_curves = {}
        for model_name, payload in predictions.items():
            if "threat_seq_pred" not in payload or "urgency_seq_pred" not in payload:
                continue
            model_curves[model_name] = {
                "threat_pred": np.asarray(payload["threat_seq_pred"])[candidate_index],
                "urgency_pred": np.asarray(payload["urgency_seq_pred"])[candidate_index],
            }
        clean_features = _case_sequence(metadata, "clean_sequence", candidate_index, observed_len)
        noisy_features = _case_sequence(metadata, "noisy_sequence", candidate_index, observed_len)
        score = _score_operational_case_candidate(
            threat_true=np.asarray(threat_true[candidate_index], dtype=np.int64),
            model_curves=model_curves,
            frame_interval=frame_interval,
            clean_features=clean_features,
            noisy_features=noisy_features,
        )
        scored_candidates.append((score, int(candidate_index), model_curves))

    if scored_candidates:
        _, case_index, model_curves = max(scored_candidates, key=lambda item: item[0])
    else:
        case_index = int(candidates[0])
        model_curves = {}

    metadata_row = {}
    for key in ERROR_METADATA_KEYS:
        values = metadata.get(key)
        if isinstance(values, np.ndarray) and len(values) > case_index:
            metadata_row[key] = str(values[case_index])

    return {
        **setting_context(setting_name, setting),
        "run_index": run_idx,
        "seed": seed,
        "case_index": case_index,
        "case_key": f"run{run_idx}_seed{seed}_case{case_index}",
        "frame_interval": frame_interval,
        "true_first_critical_frame": _first_critical_frame(threat_true[case_index], critical_labels=[4, 5]),
        "urgency_first_immediate_frame": _first_critical_frame(urgency_true[case_index], critical_labels=[3]),
        "features": np.asarray(data_bundle["X_test"])[case_index],
        "clean_features": _case_sequence(metadata, "clean_sequence", case_index, observed_len),
        "noisy_features": _case_sequence(metadata, "noisy_sequence", case_index, observed_len),
        "model_input_features": _case_sequence(metadata, "model_input_sequence", case_index, observed_len),
        "threat_true": threat_true[case_index],
        "urgency_true": urgency_true[case_index],
        "metadata": metadata_row,
        "models": model_curves,
    }


def _first_critical_frame(sequence: np.ndarray, *, critical_labels: list[int]) -> int:
    mask = np.isin(np.asarray(sequence, dtype=np.int64), critical_labels)
    if not np.any(mask):
        return -1
    return int(np.argmax(mask))


def _case_sequence(metadata: dict[str, Any], key: str, case_index: int, observed_len: int) -> np.ndarray:
    values = metadata.get(key)
    if isinstance(values, np.ndarray) and values.ndim == 3 and len(values) > case_index:
        return np.asarray(values[case_index, :observed_len, :], dtype=np.float32)
    return np.empty((0, 0), dtype=np.float32)


def _score_operational_case_candidate(
    *,
    threat_true: np.ndarray,
    model_curves: dict[str, Any],
    frame_interval: float,
    clean_features: np.ndarray,
    noisy_features: np.ndarray,
) -> float:
    true_first = _first_critical_frame(threat_true, critical_labels=[4, 5])
    if true_first < 0 or len(threat_true) <= 1:
        return -1e9

    n_steps = len(threat_true)
    transitions = int(np.sum(np.diff(threat_true) != 0))
    severity_gain = float(threat_true[-1] - threat_true[0])
    centrality = 1.0 - abs((true_first / max(n_steps - 1, 1)) - 0.5) * 2.0
    score = 6.0 * max(severity_gain, 0.0) + 3.0 * transitions + 12.0 * max(centrality, 0.0)
    score += 4.0 * _signal_gap_score(clean_features, noisy_features)

    hgtan_curves = model_curves.get("TemporalHGTAN")
    if not hgtan_curves:
        return score

    hgtan_stats = _case_alarm_stats(
        threat_true,
        np.asarray(hgtan_curves["threat_pred"], dtype=np.int64),
        frame_interval=frame_interval,
    )
    score -= 5.0 * hgtan_stats["abs_delay_seconds"]
    score -= 0.35 * hgtan_stats["false_frames"]
    score -= 1.5 * hgtan_stats["mae"]
    if hgtan_stats["lead_seconds"] < -0.8:
        score += 3.0 * (hgtan_stats["lead_seconds"] + 0.8)
    else:
        score += 2.0 * max(hgtan_stats["lead_seconds"], 0.0)

    for baseline_name in ["TemporalGRU", "TemporalLSTM", "TemporalHMM", "TOPSIS"]:
        baseline_curves = model_curves.get(baseline_name)
        if not baseline_curves:
            continue
        baseline_stats = _case_alarm_stats(
            threat_true,
            np.asarray(baseline_curves["threat_pred"], dtype=np.int64),
            frame_interval=frame_interval,
        )
        score += 0.8 * (baseline_stats["false_frames"] - hgtan_stats["false_frames"])
        score += 0.6 * (baseline_stats["abs_delay_seconds"] - hgtan_stats["abs_delay_seconds"])
        score += 0.5 * (baseline_stats["mae"] - hgtan_stats["mae"])
    return float(score)


def _signal_gap_score(clean_features: np.ndarray, noisy_features: np.ndarray) -> float:
    if clean_features.size == 0 or noisy_features.size == 0 or clean_features.shape != noisy_features.shape:
        return 0.0
    focus_indices = [6, 8, 11]
    diff = np.abs(np.asarray(noisy_features, dtype=np.float64)[:, focus_indices] - np.asarray(clean_features, dtype=np.float64)[:, focus_indices])
    return float(np.clip(diff.mean(), 0.0, 1.0))


def _case_alarm_stats(
    threat_true: np.ndarray,
    threat_pred: np.ndarray,
    *,
    frame_interval: float,
) -> dict[str, float]:
    true_first = _first_critical_frame(threat_true, critical_labels=[4, 5])
    pred_first = _first_critical_frame(threat_pred, critical_labels=[4, 5])
    if pred_first < 0:
        lead_seconds = -float((len(threat_true) - true_first) * frame_interval)
        abs_delay_seconds = float((len(threat_true) - true_first) * frame_interval)
    else:
        lead_seconds = float((true_first - pred_first) * frame_interval)
        abs_delay_seconds = abs(lead_seconds)
    false_frames = int(np.sum((threat_pred >= 4) & (np.arange(len(threat_pred)) < true_first)))
    mae = float(np.mean(np.abs(np.asarray(threat_pred, dtype=np.float64) - np.asarray(threat_true, dtype=np.float64))))
    return {
        "lead_seconds": lead_seconds,
        "abs_delay_seconds": abs_delay_seconds,
        "false_frames": float(false_frames),
        "mae": mae,
    }


def summarize_results(all_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate run-level metrics into mean/std/CI rows."""
    rows: list[dict[str, Any]] = []
    if not all_records:
        return rows

    model_names = list(
        dict.fromkeys(
            model_name
            for record in all_records
            for model_name in record.get("results", {}).keys()
        )
    )
    for model_name in model_names:
        rows.extend(summarize_model_metrics(all_records, model_name))
        rows.extend(summarize_per_level_metrics(all_records, model_name))
        rows.extend(summarize_track_metrics(all_records, model_name))
        rows.extend(summarize_model_efficiency(all_records, model_name))
    return rows


def summarize_model_metrics(all_records: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in ["threat", "urgency"]:
        metric_names = sorted(
            {
                metric_name
                for record in all_records
                for metric_name in record["results"].get(model_name, {}).get(task, {}).keys()
            }
        )
        for metric in metric_names:
            rows.extend(make_summary_row(model_name, task, metric, collect_metric_values(all_records, model_name, task, metric)))

    composite_values = []
    for record in all_records:
        result = record["results"].get(model_name)
        if result and result.get("threat", {}).get("f1") is not None and result.get("urgency", {}).get("f1") is not None:
            composite_values.append(compute_composite_f1(result))
    rows.extend(make_summary_row(model_name, "joint", "composite_f1", composite_values))
    return rows


def summarize_per_level_metrics(all_records: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    """Summarize ordinal level behavior so high-risk classes are not hidden by macro scores."""
    rows: list[dict[str, Any]] = []
    task_specs = [("threat", 5), ("urgency", 3)]
    for task, n_classes in task_specs:
        for level in range(1, n_classes + 1):
            recalls: list[float] = []
            f1_scores: list[float] = []
            supports: list[float] = []
            for record in all_records:
                payload = record.get("predictions", {}).get(model_name)
                if not payload:
                    continue
                true = np.asarray(payload.get(f"{task}_true", []), dtype=np.int64)
                pred = np.asarray(payload.get(f"{task}_pred", []), dtype=np.int64)
                if true.size == 0 or pred.size == 0 or true.shape != pred.shape:
                    continue
                mask = true == level
                support = int(mask.sum())
                if support == 0:
                    continue
                tp = int(np.sum(mask & (pred == level)))
                fp = int(np.sum((true != level) & (pred == level)))
                fn = int(np.sum(mask & (pred != level)))
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1_value = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                recalls.append(float(recall))
                f1_scores.append(float(f1_value))
                supports.append(float(support))
            rows.extend(make_summary_row(model_name, f"{task}_level", f"L{level}_recall", recalls))
            rows.extend(make_summary_row(model_name, f"{task}_level", f"L{level}_f1", f1_scores))
            rows.extend(make_summary_row(model_name, f"{task}_level", f"L{level}_support", supports))
    return rows


def summarize_model_efficiency(all_records: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = sorted(
        {
            metric_name
            for record in all_records
            for metric_name in record.get("efficiency", {}).get(model_name, {}).keys()
        }
    )
    for metric in metric_names:
        values = []
        for record in all_records:
            if model_name not in record.get("efficiency", {}) or metric not in record["efficiency"][model_name]:
                continue
            numeric = _to_finite_float(record["efficiency"][model_name][metric])
            if numeric is not None:
                values.append(numeric)
        rows.extend(make_summary_row(model_name, "efficiency", metric, values))
    return rows


def summarize_track_metrics(all_records: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    track_rows = [
        row
        for record in all_records
        for row in record.get("track_metrics", [])
        if row.get("model") == model_name
    ]
    if not track_rows:
        return rows

    tasks = sorted({str(row["task"]) for row in track_rows if "task" in row})
    for task in tasks:
        task_rows = [row for row in track_rows if row.get("task") == task]
        metric_names = sorted(
            {
                metric_name
                for row in task_rows
                for metric_name, value in row.items()
                if metric_name not in TRACK_METRIC_NON_SUMMARY_FIELDS and _to_finite_float(value) is not None
            }
        )
        for metric in metric_names:
            values = [
                numeric
                for row in task_rows
                if (numeric := _to_finite_float(row.get(metric))) is not None
            ]
            rows.extend(make_summary_row(model_name, f"{task}_track", metric, values))
    return rows


def collect_metric_values(
    all_records: list[dict[str, Any]],
    model_name: str,
    task: str,
    metric: str,
) -> list[float]:
    values = []
    for record in all_records:
        value = record["results"].get(model_name, {}).get(task, {}).get(metric)
        if value is not None:
            values.append(float(value))
    return values


def make_summary_row(model_name: str, task: str, metric: str, values: list[float]) -> list[dict[str, Any]]:
    if not values:
        return []
    n = len(values)
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return [
        {
            "model": model_name,
            "task": task,
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": std,
            "ci95": float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0,
            "n": n,
        }
    ]


def _to_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def validate_models(model_names: list[str]) -> list[str]:
    """Backward-compatible wrapper used by run.py."""
    return validate_model_names(model_names)


def release_runtime_cache() -> None:
    """Release per-model CUDA/Python cache before the next long sequence run."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def apply_runtime_overrides(config: dict[str, Any], options: AssessmentOptions) -> None:
    if options.cli_args is not None:
        apply_cli_overrides(config, options.cli_args)
    if options.num_runs is not None:
        config["run"]["num_runs"] = options.num_runs
    if options.epochs is not None:
        config["train"]["num_epochs"] = options.epochs
        config["train"]["patience"] = min(config["train"]["patience"], max(1, options.epochs))
        config["train"]["min_epochs"] = min(config["train"].get("min_epochs", options.epochs), options.epochs)


def annotate_summary_rows(
    summary_rows: list[dict[str, Any]],
    setting_name: str,
    setting: dict[str, Any],
) -> list[dict[str, Any]]:
    annotated = []
    for row in summary_rows:
        enriched = setting_context(setting_name, setting)
        enriched.update(row)
        annotated.append(enriched)
    return annotated


def setting_output_complete(
    setting_dir: Path,
    *,
    expected_models: list[str],
    expected_task_form: str,
) -> bool:
    """Return True when a setting directory satisfies the compact paper-stage contract."""
    del expected_task_form
    required = ["summary.csv", "operational_cases.npz"]
    if not setting_dir.exists():
        return False
    if any(not (setting_dir / name).exists() for name in required):
        return False

    try:
        summary = pd.read_csv(setting_dir / "summary.csv")
    except (pd.errors.EmptyDataError, OSError):
        return False
    if summary.empty or "model" not in summary.columns:
        return False

    present_models = set(summary["model"].dropna().astype(str))
    return set(expected_models).issubset(present_models)


def read_existing_summary(setting_dir: Path) -> list[dict[str, Any]]:
    """Read a previously completed setting summary for global aggregation."""
    try:
        summary = pd.read_csv(setting_dir / "summary.csv")
    except pd.errors.EmptyDataError:
        return []
    return summary.to_dict(orient="records")


def describe_setting(setting: dict[str, Any]) -> str:
    holdout = setting.get("scenario_holdout_key")
    holdout_text = f", holdout={holdout}" if holdout else ""
    sequence_text = ""
    if is_sequential_setting(setting):
        sequence_text = f", observed={setting.get('observed_len', 'default')}/{setting.get('seq_len', 'default')}"
        if "range_m" in setting:
            sequence_text += f", range={setting['range_m']}m"
        if "difficulty_tier" in setting:
            sequence_text += f", difficulty={setting['difficulty_tier']}"
    return (
        f"{make_setting_name(setting)} "
        f"(task={setting.get('task_form', DEFAULT_TASK_FORM)}, samples={setting['n_samples']}, "
        f"split={setting['split_strategy']}, "
        f"window={setting['detection_window']}, noise={setting['noise_level']}, "
        f"missing={setting['missing_ratio']}{sequence_text}{holdout_text})"
    )


def print_header(suite: str, out_root: Path, models: list[str], settings: list[dict[str, Any]]) -> None:
    print(f"\nAssessment suite: {suite}")
    print(f"Output: {out_root}")
    print(f"Models: {', '.join(models)}")
    print(f"Settings: {len(settings)}")


def print_runtime(config: dict[str, Any]) -> None:
    train_cfg = config["train"]
    repro_cfg = config["reproducibility"]
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    print(
        "  Runtime: "
        f"device={device}, "
        f"cuda={torch.cuda.is_available()} ({cuda_name}), "
        f"batch={train_cfg['batch_size']}, workers={train_cfg.get('num_workers', 0)}, "
        f"pin_memory={train_cfg.get('pin_memory', False)}, amp={train_cfg.get('use_amp', False)}, "
        f"compile={train_cfg.get('compile_model', False)}, tf32={repro_cfg.get('allow_tf32', False)}"
    )


def print_loader_profile(data_bundle: dict[str, Any]) -> None:
    """Show resolved DataLoader batch sizes after split-aware adjustment."""
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    print(
        "    Loaders: "
        f"train_batch={train_loader.batch_size} ({len(train_loader)} steps/epoch, workers={getattr(train_loader, 'num_workers', 0)}), "
        f"val_batch={val_loader.batch_size} (workers={getattr(val_loader, 'num_workers', 0)}), "
        f"test_batch={test_loader.batch_size} (workers={getattr(test_loader, 'num_workers', 0)})"
    )


def validate_runtime_config(config: dict[str, Any], mode: str) -> None:
    """Fail fast when a GPU-only run would silently fall back to CPU."""
    if config.get("run", {}).get("require_cuda", False) and not torch.cuda.is_available():
        raise RuntimeError(
            f"--mode {mode} requires a CUDA-enabled PyTorch build, but this Python environment has "
            f"torch={torch.__version__}, cuda_build={torch.version.cuda}, cuda_available=False. "
            "Install a CUDA PyTorch wheel first, or use --mode speed/--mode repro for CPU/debug runs."
        )


def empty_metric_record() -> dict[str, Any]:
    return {
        "threat": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "critical_recall": 0.0,
            "critical_miss_rate": None,
            "decision_cost": 0.0,
            "ece": None,
            "brier": None,
        },
        "urgency": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "critical_recall": 0.0,
            "critical_miss_rate": None,
            "decision_cost": 0.0,
            "ece": None,
            "brier": None,
        },
    }


AssessmentExperiment = Exp_Main

# Backward-compatible aliases for older scripts/imports.
BenchmarkExperiment = Exp_Main
BenchmarkOptions = AssessmentOptions
build_benchmark_setting_record = build_assessment_setting_record
