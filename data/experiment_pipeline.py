"""
Shared data preparation utilities for experiment scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from data.data_loader import build_loader_kwargs
from data.generator import generate_uav_swarm_payload
from data.pipeline_common import (
    build_split_indices,
    slice_metadata,
    split_has_full_label_support,
)
from utils.config import N_CLASSES, N_URGENCY, validate_class_distribution


def compute_balanced_class_weights(
    labels_0_based: np.ndarray,
    max_weight: float = 5.0,
    n_classes: int | None = None,
) -> list[float]:
    """
    Compute clipped sqrt-balanced class weights from 0-based labels.
    """
    labels = np.asarray(labels_0_based, dtype=np.int64)
    n_classes = n_classes or int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    total = counts.sum()

    weights = np.sqrt(total / (n_classes * counts))
    weights = np.clip(weights, 0.5, max_weight)
    return weights.tolist()


def resolve_class_weights(
    train_cfg: dict[str, Any],
    threat_labels_0: np.ndarray,
    urgency_labels_0: np.ndarray,
) -> tuple[list[float] | None, list[float] | None]:
    """
    Resolve class weights from config and training labels.
    """
    class_weight_threat = train_cfg.get("class_weight_threat")
    class_weight_urgency = train_cfg.get("class_weight_urgency")

    if not train_cfg.get("auto_class_weight", False):
        return class_weight_threat, class_weight_urgency

    max_weight = train_cfg.get("class_weight_max", 5.0)
    class_weight_threat = compute_balanced_class_weights(threat_labels_0, max_weight, n_classes=N_CLASSES)
    class_weight_urgency = compute_balanced_class_weights(urgency_labels_0, max_weight, n_classes=N_URGENCY)
    return class_weight_threat, class_weight_urgency


def build_joint_train_kwargs(
    train_cfg: dict[str, Any],
    class_weight_threat: list[float] | None = None,
    class_weight_urgency: list[float] | None = None,
) -> dict[str, Any]:
    """
    Build a unified train_model kwargs dictionary from config.
    """
    return {
        "num_epochs": train_cfg["num_epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "urgency_weight": train_cfg["urgency_weight"],
        "patience": train_cfg["patience"],
        "weight_decay": train_cfg["weight_decay"],
        "loss_type_threat": train_cfg.get("loss_type_threat", "ce"),
        "loss_type_urgency": train_cfg.get("loss_type_urgency", "ce"),
        "class_weight_threat": class_weight_threat,
        "class_weight_urgency": class_weight_urgency,
        "focal_gamma": train_cfg.get("focal_gamma", 2.0),
        "label_smoothing": train_cfg.get("label_smoothing", 0.05),
        "warmup_epochs": train_cfg.get("warmup_epochs", 8),
        "min_delta": train_cfg.get("min_delta", 1e-4),
        "gradient_clip_norm": train_cfg.get("gradient_clip_norm", 1.0),
        "threat_weight": train_cfg.get("threat_weight", 0.75),
        "use_mixup": train_cfg.get("use_mixup", False),
        "mixup_alpha": train_cfg.get("mixup_alpha", 0.2),
        "min_epochs": train_cfg.get("min_epochs", 60),
        "ema_alpha": train_cfg.get("ema_alpha", 0.3),
        "use_amp": train_cfg.get("use_amp", False),
        "compile_model": train_cfg.get("compile_model", False),
    }


def prepare_experiment_data(
    seed: int,
    config: dict[str, Any],
    *,
    n_samples: int | None = None,
    noise_level: float = 0.0,
    missing_ratio: float = 0.0,
    split_strategy: str | None = None,
    enforce_min_class_samples: bool = False,
    max_retries: int = 5,
) -> dict[str, Any]:
    """
    Generate, split, scale, and package a dataset for experiments.
    """
    features, threat_labels, urgency_labels, metadata = _generate_dataset(
        seed=seed,
        config=config,
        n_samples=n_samples,
        enforce_min_class_samples=enforce_min_class_samples,
        max_retries=max_retries,
    )

    return _split_scale_and_pack(
        features=features,
        threat_labels=threat_labels,
        urgency_labels=urgency_labels,
        metadata=metadata,
        seed=seed,
        config=config,
        noise_level=noise_level,
        missing_ratio=missing_ratio,
        split_strategy=split_strategy,
    )


def _generate_dataset(
    *,
    seed: int,
    config: dict[str, Any],
    n_samples: int | None,
    enforce_min_class_samples: bool,
    max_retries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_samples = n_samples or config["data"]["n_samples"]
    min_samples = config["data"].get("min_class_samples", 30)
    retries = max_retries if enforce_min_class_samples else 1

    features = threat_labels = urgency_labels = metadata = None
    for retry in range(retries):
        current_seed = seed + retry * 1000 if retry else seed
        payload = generate_uav_swarm_payload(
            n_samples=n_samples,
            seed=current_seed,
            scenario_profile=config["data"].get(
                "scenario_profile",
                config["data"].get("benchmark_dataset", "ATUAV-Core"),
            ),
            detection_window=config["data"].get("detection_window", "standard"),
            benchmark_dataset=config["data"].get("benchmark_dataset"),
        )
        features = np.asarray(payload["features"])
        threat_labels = np.asarray(payload["threat_labels"])
        urgency_labels = np.asarray(payload["urgency_labels"])
        metadata = dict(payload["metadata"])

        threat_ok = validate_class_distribution(threat_labels, min_samples)
        urgency_ok = validate_class_distribution(urgency_labels, max(10, min_samples // 2))
        if not enforce_min_class_samples or (threat_ok and urgency_ok):
            break

    assert features is not None and threat_labels is not None and urgency_labels is not None and metadata is not None
    return features, threat_labels, urgency_labels, metadata


def _split_scale_and_pack(
    *,
    features: np.ndarray,
    threat_labels: np.ndarray,
    urgency_labels: np.ndarray,
    metadata: dict[str, Any],
    seed: int,
    config: dict[str, Any],
    noise_level: float,
    missing_ratio: float,
    split_strategy: str | None,
) -> dict[str, Any]:
    data_cfg = config["data"]
    selected_split_strategy = split_strategy or data_cfg.get("split_strategy", "stratified")

    train_idx, val_idx, test_idx = _build_split_indices(
        threat_labels=threat_labels,
        urgency_labels=urgency_labels,
        metadata=metadata,
        seed=seed,
        data_cfg=data_cfg,
        split_strategy=selected_split_strategy,
    )

    x_train, x_val, x_test = features[train_idx], features[val_idx], features[test_idx]
    t_train, t_val, t_test = threat_labels[train_idx], threat_labels[val_idx], threat_labels[test_idx]
    u_train, u_val, u_test = urgency_labels[train_idx], urgency_labels[val_idx], urgency_labels[test_idx]

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, x_test_scaled.shape)
        x_test_scaled = np.clip(x_test_scaled + noise, 0, 1)

    if missing_ratio > 0:
        mask = np.random.random(x_test_scaled.shape) < missing_ratio
        x_test_scaled = np.where(mask, 0, x_test_scaled)

    t_train_0 = t_train - 1
    t_val_0 = t_val - 1
    t_test_0 = t_test - 1
    u_train_0 = u_train - 1
    u_val_0 = u_val - 1
    u_test_0 = u_test - 1

    def make_loader(x: np.ndarray, t: np.ndarray, u: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.FloatTensor(x),
                torch.LongTensor(t),
                torch.LongTensor(u),
            ),
            **build_loader_kwargs(config, dataset_size=len(x), shuffle=shuffle),
        )

    return {
        "train_loader": make_loader(x_train_scaled, t_train_0, u_train_0, True),
        "val_loader": make_loader(x_val_scaled, t_val_0, u_val_0, False),
        "test_loader": make_loader(x_test_scaled, t_test_0, u_test_0, False),
        "X_train": x_train_scaled,
        "X_val": x_val_scaled,
        "X_test": x_test_scaled,
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
        "metadata_train": slice_metadata(metadata, train_idx),
        "metadata_val": slice_metadata(metadata, val_idx),
        "metadata_test": slice_metadata(metadata, test_idx),
        "split_strategy": selected_split_strategy,
        "scaler": scaler,
    }


def _build_split_indices(
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
        validator=split_has_full_label_support,
    )
