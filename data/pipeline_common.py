"""Common split and metadata helpers shared by ATUAV data pipelines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SplitValidator = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool]


def build_split_indices(
    *,
    threat_labels: np.ndarray,
    urgency_labels: np.ndarray,
    metadata: dict[str, object],
    seed: int,
    data_cfg: dict[str, object],
    split_strategy: str,
    validator: SplitValidator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build train/validation/test indices with a shared fallback policy."""
    if split_strategy == "scenario_holdout":
        group_key = str(data_cfg.get("scenario_holdout_key", "scenario_group"))
        if group_key not in metadata:
            valid = ", ".join(sorted(metadata.keys()))
            raise KeyError(f"Unknown scenario_holdout_key={group_key!r}. Valid metadata keys: {valid}")

        indices = _scenario_holdout_split(
            threat_labels=threat_labels,
            urgency_labels=urgency_labels,
            groups=np.asarray(metadata[group_key]),
            seed=seed,
            data_cfg=data_cfg,
            validator=validator,
        )
        if indices is not None:
            return indices

    return stratified_split_indices(threat_labels, seed, data_cfg)


def stratified_split_indices(
    labels: np.ndarray,
    seed: int,
    data_cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices with safe stratification fallback for tiny class counts."""
    indices = np.arange(len(labels))
    test_ratio = float(data_cfg["test_ratio"])
    val_ratio = float(data_cfg["val_ratio"]) / (1.0 - test_ratio)

    temp_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=safe_stratify_labels(labels),
    )

    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=safe_stratify_labels(labels[temp_idx]),
    )
    return train_idx, val_idx, test_idx


def safe_stratify_labels(labels: np.ndarray) -> np.ndarray | None:
    """Return labels when stratification is safe; otherwise let sklearn split randomly."""
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or np.min(counts) < 2:
        return None
    return labels


def split_has_full_label_support(
    threat_labels: np.ndarray,
    urgency_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> bool:
    """Require every split to contain the full threat and urgency label space."""
    expected_threat = set(np.unique(threat_labels))
    expected_urgency = set(np.unique(urgency_labels))

    for indices in (train_idx, val_idx, test_idx):
        if set(np.unique(threat_labels[indices])) != expected_threat:
            return False
        if set(np.unique(urgency_labels[indices])) != expected_urgency:
            return False
    return True


def split_has_minimum_threat_support(
    threat_labels: np.ndarray,
    urgency_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> bool:
    """Require each split to keep at least two threat classes for sequential studies."""
    del urgency_labels
    return all(len(np.unique(threat_labels[idx])) >= 2 for idx in (train_idx, val_idx, test_idx))


def slice_metadata(metadata: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    """Slice per-sample metadata while leaving scalar config fields untouched."""
    reference_length = None
    for value in metadata.values():
        if isinstance(value, np.ndarray):
            reference_length = len(value)
            break

    sliced: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray) and reference_length is not None and len(value) == reference_length:
            sliced[key] = value[indices]
        else:
            sliced[key] = value
    return sliced


def _scenario_holdout_split(
    *,
    threat_labels: np.ndarray,
    urgency_labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    data_cfg: dict[str, object],
    validator: SplitValidator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    indices = np.arange(len(threat_labels))
    test_ratio = float(data_cfg.get("scenario_holdout_ratio", data_cfg["test_ratio"]))
    val_ratio = float(data_cfg["val_ratio"]) / (1.0 - float(data_cfg["test_ratio"]))

    for offset in range(20):
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed + offset,
        )
        temp_idx, test_idx = next(outer.split(indices, threat_labels, groups=groups))

        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed + 100 + offset,
        )
        train_inner, val_inner = next(inner.split(temp_idx, threat_labels[temp_idx], groups=groups[temp_idx]))
        train_idx = temp_idx[train_inner]
        val_idx = temp_idx[val_inner]

        if validator(threat_labels, urgency_labels, train_idx, val_idx, test_idx):
            return train_idx, val_idx, test_idx

    return None
