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
    write_operational_cases_npz(setting_dir / "operational_cases.npz", records)


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
