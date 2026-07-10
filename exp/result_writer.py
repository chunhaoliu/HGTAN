"""Compact result serialization for the paper-stage assessment pipeline."""

from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import to_serializable


def write_setting_outputs(
    *,
    setting_dir: Path,
    setting_name: str,
    setting: dict[str, Any],
    model_names: list[str],
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    config: dict[str, Any],
    cli_args: Namespace | None,
) -> None:
    """Write the minimal per-setting contract used by the current paper workflow."""
    del setting_name, setting, model_names, config, cli_args
    setting_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary_rows).to_csv(setting_dir / "summary.csv", index=False)
    run_metric_rows = build_run_metric_rows(records)
    if run_metric_rows:
        pd.DataFrame(run_metric_rows).to_csv(setting_dir / "run_metrics.csv", index=False)
    write_operational_cases_npz(setting_dir / "operational_cases.npz", records)


def build_run_metric_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-seed metrics so formal runs can be audited beyond mean rows."""
    rows: list[dict[str, Any]] = []
    for record in records:
        setting = record.get("assessment_setting", {})
        common = {
            "setting": _record_setting_name(record),
            "dataset": setting.get("dataset"),
            "protocol": setting.get("protocol"),
            "scenario_profile": setting.get("scenario_profile"),
            "task_form": setting.get("task_form", "instantaneous"),
            "run_index": record.get("run_index"),
            "seed": record.get("seed"),
        }
        for model_name, result in record.get("results", {}).items():
            rows.extend(_flatten_result_metrics(common, model_name, result))
        for model_name, efficiency in record.get("efficiency", {}).items():
            rows.extend(_flatten_numeric_mapping(common, model_name, "efficiency", efficiency))
        for track_row in record.get("track_metrics", []):
            rows.extend(_flatten_track_metric_row(common, track_row))
    return rows


def _record_setting_name(record: dict[str, Any]) -> str | None:
    for row in record.get("track_metrics", []):
        if row.get("setting"):
            return str(row["setting"])
    case = record.get("operational_case") or {}
    if case.get("setting"):
        return str(case["setting"])
    setting = record.get("assessment_setting", {})
    dataset = setting.get("dataset")
    protocol = setting.get("protocol")
    if dataset and protocol:
        return f"{dataset}__{protocol}"
    return None


def _flatten_result_metrics(common: dict[str, Any], model_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in ["threat", "urgency"]:
        rows.extend(_flatten_numeric_mapping(common, model_name, task, result.get(task, {})))
    threat_f1 = _to_float(result.get("threat", {}).get("f1"))
    urgency_f1 = _to_float(result.get("urgency", {}).get("f1"))
    if threat_f1 is not None and urgency_f1 is not None:
        rows.append(
            {
                **common,
                "model": model_name,
                "task": "joint",
                "metric": "composite_f1",
                "value": 0.75 * threat_f1 + 0.25 * urgency_f1,
            }
        )
    return rows


def _flatten_track_metric_row(common: dict[str, Any], track_row: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = {
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
        "track_missing_ratio",
        "track_jitter_std",
        "type_as_input",
        "mission_as_input",
        "reference_policy_variant",
        "run_index",
        "seed",
        "model",
        "task",
        "critical_labels",
    }
    merged_common = {
        **common,
        "setting": track_row.get("setting", common.get("setting")),
        "dataset": track_row.get("dataset", common.get("dataset")),
        "protocol": track_row.get("protocol", common.get("protocol")),
        "scenario_profile": track_row.get("scenario_profile", common.get("scenario_profile")),
        "task_form": track_row.get("task_form", common.get("task_form")),
        "run_index": track_row.get("run_index", common.get("run_index")),
        "seed": track_row.get("seed", common.get("seed")),
    }
    rows = []
    for metric, value in track_row.items():
        if metric in excluded:
            continue
        numeric = _to_float(value)
        if numeric is None:
            continue
        rows.append(
            {
                **merged_common,
                "model": track_row.get("model"),
                "task": f"{track_row.get('task', 'track')}_track",
                "metric": metric,
                "value": numeric,
            }
        )
    return rows


def _flatten_numeric_mapping(
    common: dict[str, Any],
    model_name: str,
    task: str,
    values: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for metric, value in values.items():
        numeric = _to_float(value)
        if numeric is None:
            continue
        rows.append({**common, "model": model_name, "task": task, "metric": metric, "value": numeric})
    return rows


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def write_operational_cases_npz(path: Path, records: list[dict[str, Any]]) -> None:
    """Store the selected operational case curves used by the main manuscript figures."""
    arrays: dict[str, np.ndarray] = {}
    for record in records:
        case = record.get("operational_case") or {}
        if not case:
            continue
        prefix = _safe_key(str(case.get("case_key", f"run{record['run_index']}_case0")))
        arrays[f"{prefix}__features"] = np.asarray(case["features"])
        for key in ["clean_features", "noisy_features", "model_input_features"]:
            if key in case and np.asarray(case[key]).size:
                arrays[f"{prefix}__{key}"] = np.asarray(case[key])
        arrays[f"{prefix}__threat_true"] = np.asarray(case["threat_true"], dtype=np.int64)
        arrays[f"{prefix}__urgency_true"] = np.asarray(case["urgency_true"], dtype=np.int64)
        arrays[f"{prefix}__frame_interval"] = np.asarray([case.get("frame_interval", 0.2)], dtype=np.float32)
        for model_name, curves in case.get("models", {}).items():
            model_prefix = _safe_key(model_name)
            arrays[f"{prefix}__{model_prefix}__threat_pred"] = np.asarray(curves["threat_pred"], dtype=np.int64)
            arrays[f"{prefix}__{model_prefix}__urgency_pred"] = np.asarray(curves["urgency_pred"], dtype=np.int64)

    if not arrays:
        arrays["empty"] = np.asarray([], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def setting_context(setting_name: str, setting: dict[str, Any]) -> dict[str, Any]:
    """Return common setting columns used by aggregated summary outputs."""
    context = {
        "setting": setting_name,
        "dataset": setting["dataset"],
        "protocol": setting["protocol"],
        "scenario_profile": setting["scenario_profile"],
        "task_form": setting.get("task_form", "instantaneous"),
        "split_strategy": setting["split_strategy"],
        "detection_window": setting["detection_window"],
        "noise_level": setting["noise_level"],
        "missing_ratio": setting["missing_ratio"],
    }
    for key in [
        "seq_len",
        "observed_len",
        "frame_interval",
        "track_noise_std",
        "range_m",
        "track_missing_ratio",
        "track_jitter_std",
        "difficulty_tier",
        "sensor_profile",
        "type_as_input",
        "mission_as_input",
        "reference_policy_variant",
    ]:
        if key in setting:
            context[key] = setting[key]
    return context


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with numpy-safe conversion."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, ensure_ascii=False, indent=2)


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
