"""Dataset audit tables for assessment reproducibility."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.config import ALL_FEATURES, THREAT_LEVELS, URGENCY_LEVELS


AUDIT_GROUP_KEYS = [
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
    "track_missing_ratio",
    "track_jitter_std",
]


def build_data_profile_rows(
    data_bundle: dict[str, Any],
    *,
    context: dict[str, Any],
    run_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return label-distribution and scenario-coverage rows for one split."""
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        rows.extend(
            _label_distribution_rows(
                labels=data_bundle[f"t_{split}"],
                label_names=THREAT_LEVELS,
                context=context,
                run_idx=run_idx,
                seed=seed,
                split=split,
                task="threat",
                profile_type="label_distribution",
            )
        )
        rows.extend(
            _label_distribution_rows(
                labels=data_bundle[f"u_{split}"],
                label_names=URGENCY_LEVELS,
                context=context,
                run_idx=run_idx,
                seed=seed,
                split=split,
                task="urgency",
                profile_type="label_distribution",
            )
        )
        rows.extend(
            _scenario_distribution_rows(
                metadata=data_bundle.get(f"metadata_{split}", {}),
                context=context,
                run_idx=run_idx,
                seed=seed,
                split=split,
            )
        )

    rows.extend(
        _sequence_label_rows(
            data_bundle=data_bundle,
            context=context,
            run_idx=run_idx,
            seed=seed,
        )
    )
    return rows


def build_feature_profile_rows(
    data_bundle: dict[str, Any],
    *,
    context: dict[str, Any],
    run_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return per-feature descriptive statistics for train/val/test splits."""
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        values = np.asarray(data_bundle[f"X_{split}"], dtype=np.float64)
        n_samples = int(values.shape[0])
        n_timepoints = int(values.shape[1]) if values.ndim == 3 else 1
        if values.ndim == 3:
            values = values.reshape(-1, values.shape[-1])
        for feature_index, feature_name in enumerate(ALL_FEATURES):
            feature_values = values[:, feature_index]
            rows.append(
                {
                    **context,
                    "run_index": run_idx,
                    "seed": seed,
                    "split": split,
                    "feature_index": feature_index,
                    "feature": feature_name,
                    "n_samples": n_samples,
                    "n_timepoints": n_timepoints,
                    "mean": float(np.mean(feature_values)),
                    "std": float(np.std(feature_values)),
                    "min": float(np.min(feature_values)),
                    "q25": float(np.quantile(feature_values, 0.25)),
                    "median": float(np.quantile(feature_values, 0.50)),
                    "q75": float(np.quantile(feature_values, 0.75)),
                    "max": float(np.max(feature_values)),
                }
            )
    return rows


def _label_distribution_rows(
    *,
    labels: np.ndarray,
    label_names: dict[int, str],
    context: dict[str, Any],
    run_idx: int,
    seed: int,
    split: str,
    task: str,
    profile_type: str,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    total = int(labels.size)
    counts = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    rows = []
    for label, label_name in label_names.items():
        count = counts.get(label, 0)
        rows.append(
            {
                **context,
                "run_index": run_idx,
                "seed": seed,
                "profile_type": profile_type,
                "split": split,
                "task": task,
                "label": label,
                "label_name": label_name,
                "count": count,
                "ratio": float(count / total) if total else 0.0,
                "support": total,
                "group_key": "",
                "group_value": "",
            }
        )
    return rows


def _scenario_distribution_rows(
    *,
    metadata: dict[str, Any],
    context: dict[str, Any],
    run_idx: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for group_key in AUDIT_GROUP_KEYS:
        if group_key not in metadata:
            continue
        values = np.asarray(metadata[group_key])
        if values.ndim == 0:
            continue
        total = int(values.size)
        unique_values, counts = np.unique(values, return_counts=True)
        for value, count in sorted(zip(unique_values.tolist(), counts.tolist()), key=lambda item: str(item[0])):
            rows.append(
                {
                    **context,
                    "run_index": run_idx,
                    "seed": seed,
                    "profile_type": "scenario_distribution",
                    "split": split,
                    "task": "",
                    "label": "",
                    "label_name": "",
                    "count": int(count),
                    "ratio": float(count / total) if total else 0.0,
                    "support": total,
                    "group_key": group_key,
                    "group_value": str(value),
                }
            )
    return rows


def _sequence_label_rows(
    *,
    data_bundle: dict[str, Any],
    context: dict[str, Any],
    run_idx: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("threat", "threat_seq", THREAT_LEVELS),
        ("urgency", "urgency_seq", URGENCY_LEVELS),
    ]
    for split in ["train", "val", "test"]:
        for task, prefix, label_names in specs:
            key = f"{prefix}_{split}"
            if key not in data_bundle:
                continue
            rows.extend(
                _label_distribution_rows(
                    labels=np.asarray(data_bundle[key]).reshape(-1),
                    label_names=label_names,
                    context=context,
                    run_idx=run_idx,
                    seed=seed,
                    split=split,
                    task=task,
                    profile_type="sequence_label_distribution",
                )
            )
    return rows
