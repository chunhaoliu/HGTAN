"""Dataset wrappers used by ATUAV data providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ATUAVDataset(Dataset):
    """Simple dual-task tensor dataset."""

    def __init__(self, features: np.ndarray, threat_labels_0: np.ndarray, urgency_labels_0: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.threat_labels = torch.as_tensor(threat_labels_0, dtype=torch.long)
        self.urgency_labels = torch.as_tensor(urgency_labels_0, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.threat_labels[index], self.urgency_labels[index]


class TrackSequenceDataset(Dataset):
    """Dual-task dataset for track sequences with shape (T, F) per sample."""

    def __init__(self, sequences: np.ndarray, threat_labels_0: np.ndarray, urgency_labels_0: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.threat_labels = torch.as_tensor(threat_labels_0, dtype=torch.long)
        self.urgency_labels = torch.as_tensor(urgency_labels_0, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.threat_labels[index], self.urgency_labels[index]


@dataclass
class DataBundle:
    """Container returned by the data factory."""

    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.payload

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]


def build_loader_kwargs(
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    dataset_size: int | None = None,
    shuffle: bool = False,
) -> dict[str, Any]:
    """Build GPU-aware DataLoader kwargs from the shared training config."""
    train_cfg = config.get("train", {})
    pin_memory = bool(train_cfg.get("pin_memory", False) and torch.cuda.is_available())
    requested_batch_size = int(batch_size or train_cfg.get("batch_size", 64))
    effective_batch_size = resolve_effective_batch_size(
        train_cfg,
        requested_batch_size,
        dataset_size,
        is_train=shuffle,
    )
    num_workers = resolve_num_workers(
        train_cfg,
        dataset_size,
        is_train=shuffle,
    )

    kwargs: dict[str, Any] = {
        "batch_size": effective_batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2) or 2)
    return kwargs


def resolve_effective_batch_size(
    train_cfg: dict[str, Any],
    requested_batch_size: int,
    dataset_size: int | None,
    *,
    is_train: bool,
) -> int:
    """Prevent GPU presets from collapsing tiny train splits into one-step epochs."""
    requested = max(1, int(requested_batch_size))
    if not dataset_size or int(dataset_size) <= 0:
        return requested

    size = int(dataset_size)
    effective = min(requested, size)
    if not is_train or not bool(train_cfg.get("adaptive_batching", True)):
        return effective

    min_steps = max(1, int(train_cfg.get("min_train_steps_per_epoch", 4) or 4))
    min_batch_size = max(1, int(train_cfg.get("min_batch_size", 32) or 32))
    step_capped = max(1, size // min_steps) if min_steps > 1 else size
    effective = min(effective, step_capped)
    effective = min(size, max(1, _round_down_power_of_two(effective)))
    if size >= min_batch_size * min_steps:
        effective = max(min_batch_size, effective)
    return min(size, max(1, effective))


def _round_down_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value.bit_length() - 1)


def resolve_num_workers(
    train_cfg: dict[str, Any],
    dataset_size: int | None,
    *,
    is_train: bool,
) -> int:
    """Scale worker count down for small in-memory splits, especially on Windows."""
    requested = max(0, int(train_cfg.get("num_workers", 0) or 0))
    if requested == 0:
        return 0
    if not bool(train_cfg.get("adaptive_num_workers", True)):
        return requested

    size = int(dataset_size or 0)
    cpu_cap = max(1, (os.cpu_count() or requested) - 2)
    max_allowed = min(requested, cpu_cap)

    if not is_train:
        if size <= 4096:
            return 0
        return min(max_allowed, 2)

    if size <= 512:
        return min(max_allowed, 1)
    if size <= 2048:
        return min(max_allowed, 2)
    if size <= 8192:
        return min(max_allowed, 4)
    return max_allowed
