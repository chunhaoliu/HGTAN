"""Data preparation pipeline for sequential ATUAV assessment protocols."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from data.data_loader import TrackSequenceDataset, build_loader_kwargs
from data.pipeline_common import (
    build_split_indices,
    slice_metadata,
    split_has_minimum_threat_support,
)
from data.sequence_generator import generate_uav_track_payload


def sequence_data_provider(
    config: dict[str, Any],
    seed: int,
    *,
    split_strategy: str | None = None,
    enforce_min_class_samples: bool = False,
) -> dict[str, Any]:
    """Build train/validation/test loaders for one sequential run."""
    del enforce_min_class_samples
    return prepare_sequence_data(
        seed=seed,
        config=config,
        split_strategy=split_strategy or config["data"].get("split_strategy", "stratified"),
    )


def prepare_sequence_data(
    *,
    seed: int,
    config: dict[str, Any],
    split_strategy: str | None = None,
) -> dict[str, Any]:
    """Generate, split, scale, crop, and package track sequences."""
    data_cfg = config["data"]
    sequence_cfg = config["sequence"]
    seq_len = sequence_cfg["seq_len"]
    observed_len = min(sequence_cfg.get("observed_len", seq_len), seq_len)

    payload = generate_uav_track_payload(
        n_tracks=data_cfg["n_samples"],
        seq_len=seq_len,
        seed=seed,
        scenario_profile=data_cfg.get("scenario_profile", data_cfg.get("benchmark_dataset", "ATUAV-Core")),
        detection_window=data_cfg.get("detection_window", "standard"),
        benchmark_dataset=data_cfg.get("benchmark_dataset"),
        type_as_input=sequence_cfg.get("type_as_input", False),
        sequence_cfg=sequence_cfg,
    )
    sequences = np.asarray(payload["sequences"])
    threat_seq = np.asarray(payload["threat_seq"])
    urgency_seq = np.asarray(payload["urgency_seq"])
    metadata = dict(payload["metadata"])

    final_step = observed_len - 1
    observed_sequences = sequences[:, :observed_len, :]
    threat_labels = threat_seq[:, final_step]
    urgency_labels = urgency_seq[:, final_step]
    selected_split = split_strategy or data_cfg.get("split_strategy", "stratified")

    train_idx, val_idx, test_idx = _build_sequence_split_indices(
        threat_labels=threat_labels,
        urgency_labels=urgency_labels,
        metadata=metadata,
        seed=seed,
        data_cfg=data_cfg,
        split_strategy=selected_split,
    )

    x_train, x_val, x_test = observed_sequences[train_idx], observed_sequences[val_idx], observed_sequences[test_idx]
    x_train, x_val, x_test, scaler = _scale_sequences(x_train, x_val, x_test)

    t_train, t_val, t_test = threat_labels[train_idx], threat_labels[val_idx], threat_labels[test_idx]
    u_train, u_val, u_test = urgency_labels[train_idx], urgency_labels[val_idx], urgency_labels[test_idx]
    t_train_0, t_val_0, t_test_0 = t_train - 1, t_val - 1, t_test - 1
    u_train_0, u_val_0, u_test_0 = u_train - 1, u_val - 1, u_test - 1

    return {
        "train_loader": _make_loader(x_train, t_train_0, u_train_0, config, shuffle=True),
        "val_loader": _make_loader(x_val, t_val_0, u_val_0, config, shuffle=False),
        "test_loader": _make_loader(x_test, t_test_0, u_test_0, config, shuffle=False),
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "t_train": t_train,
        "t_val": t_val,
        "t_test": t_test,
        "u_train": u_train,
        "u_val": u_val,
        "u_test": u_test,
        "t_train_0": t_train_0,
        "t_val_0": t_val_0,
        "t_test_0": t_test_0,
        "u_train_0": u_train_0,
        "u_val_0": u_val_0,
        "u_test_0": u_test_0,
        "threat_seq_train": threat_seq[train_idx, :observed_len],
        "threat_seq_val": threat_seq[val_idx, :observed_len],
        "threat_seq_test": threat_seq[test_idx, :observed_len],
        "urgency_seq_train": urgency_seq[train_idx, :observed_len],
        "urgency_seq_val": urgency_seq[val_idx, :observed_len],
        "urgency_seq_test": urgency_seq[test_idx, :observed_len],
        "metadata_train": slice_metadata(metadata, train_idx),
        "metadata_val": slice_metadata(metadata, val_idx),
        "metadata_test": slice_metadata(metadata, test_idx),
        "split_strategy": selected_split,
        "scaler": scaler,
        "seq_len": seq_len,
        "observed_len": observed_len,
        "task_form": "sequential",
    }


def _make_loader(
    sequences: np.ndarray,
    threat_labels_0: np.ndarray,
    urgency_labels_0: np.ndarray,
    config: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TrackSequenceDataset(sequences, threat_labels_0, urgency_labels_0),
        **build_loader_kwargs(config, dataset_size=len(sequences), shuffle=shuffle),
    )


def _scale_sequences(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    n_train, t_steps, n_features = x_train.shape
    scaler.fit(x_train.reshape(-1, n_features))

    def transform(values: np.ndarray) -> np.ndarray:
        shape = values.shape
        scaled = scaler.transform(values.reshape(-1, n_features)).reshape(shape)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    return transform(x_train), transform(x_val), transform(x_test), scaler


def _build_sequence_split_indices(
    *,
    threat_labels: np.ndarray,
    urgency_labels: np.ndarray,
    metadata: dict[str, Any],
    seed: int,
    data_cfg: dict[str, Any],
    split_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_split_indices(
        threat_labels=threat_labels,
        urgency_labels=urgency_labels,
        metadata=metadata,
        seed=seed,
        data_cfg=data_cfg,
        split_strategy=split_strategy,
        validator=split_has_minimum_threat_support,
    )
