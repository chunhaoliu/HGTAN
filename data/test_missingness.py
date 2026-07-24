"""Deterministic test-time frame missingness for frozen-model robustness runs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from torch.utils.data import DataLoader

from data.data_loader import TrackSequenceDataset, build_loader_kwargs


MissingMode = Literal["random", "burst"]
HELD_FEATURES = np.asarray([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int64)
CONFIDENCE_INDEX = 15


def build_frame_missing_mask(
    n_tracks: int,
    n_steps: int,
    *,
    ratio: float,
    mode: MissingMode,
    seed: int,
) -> np.ndarray:
    """Create an exact-rate frame mask while preserving the first frame."""
    if n_tracks < 1 or n_steps < 1:
        raise ValueError("n_tracks and n_steps must both be positive.")
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must lie in [0, 1), got {ratio}.")
    if mode not in {"random", "burst"}:
        raise ValueError(f"Unsupported missingness mode: {mode}.")

    mask = np.zeros((n_tracks, n_steps), dtype=bool)
    available_steps = max(n_steps - 1, 0)
    n_missing = min(int(round(ratio * available_steps)), available_steps)
    if n_missing == 0:
        return mask

    rng = np.random.default_rng(seed)
    if mode == "random":
        candidates = np.arange(1, n_steps, dtype=np.int64)
        for track_idx in range(n_tracks):
            selected = rng.choice(candidates, size=n_missing, replace=False)
            mask[track_idx, selected] = True
    else:
        latest_start = n_steps - n_missing
        starts = rng.integers(1, latest_start + 1, size=n_tracks)
        for track_idx, start in enumerate(starts):
            mask[track_idx, start : start + n_missing] = True
    return mask


def apply_frame_missingness(
    sequences: np.ndarray,
    frame_missing: np.ndarray,
    *,
    confidence_decay: float = 0.65,
) -> np.ndarray:
    """Carry observations forward and decay confidence on missing frames."""
    values = np.asarray(sequences, dtype=np.float32)
    mask = np.asarray(frame_missing, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"Expected (tracks, time, features), got {values.shape}.")
    if mask.shape != values.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match {values.shape[:2]}.")
    if np.any(mask[:, 0]):
        raise ValueError("The first frame must remain observed.")
    if values.shape[2] <= CONFIDENCE_INDEX:
        raise ValueError("Expected the 16-indicator sequential input.")
    if not 0.0 < confidence_decay <= 1.0:
        raise ValueError("confidence_decay must lie in (0, 1].")

    adjusted = values.copy()
    for step in range(1, adjusted.shape[1]):
        indices = np.flatnonzero(mask[:, step])
        if indices.size == 0:
            continue
        adjusted[indices[:, None], step, HELD_FEATURES] = adjusted[
            indices[:, None], step - 1, HELD_FEATURES
        ]
        adjusted[indices, step, CONFIDENCE_INDEX] = np.minimum(
            adjusted[indices, step, CONFIDENCE_INDEX],
            adjusted[indices, step - 1, CONFIDENCE_INDEX] * confidence_decay,
        )
    return adjusted


def build_missing_test_bundle(
    data_bundle: dict[str, Any],
    config: dict[str, Any],
    *,
    ratio: float,
    mode: MissingMode,
    mask_seed: int,
    confidence_decay: float = 0.65,
) -> dict[str, Any]:
    """Return a test-only corruption view with train/validation data untouched."""
    raw_test = np.asarray(data_bundle["X_test_raw"], dtype=np.float32)
    mask = build_frame_missing_mask(
        raw_test.shape[0],
        raw_test.shape[1],
        ratio=ratio,
        mode=mode,
        seed=mask_seed,
    )
    corrupted_raw = apply_frame_missingness(
        raw_test,
        mask,
        confidence_decay=confidence_decay,
    )
    scaler = data_bundle["scaler"]
    shape = corrupted_raw.shape
    corrupted_scaled = scaler.transform(corrupted_raw.reshape(-1, shape[-1])).reshape(shape)
    corrupted_scaled = np.clip(corrupted_scaled, 0.0, 1.0).astype(np.float32)

    view = data_bundle.copy()
    view["X_test_raw"] = corrupted_raw
    view["X_test"] = corrupted_scaled
    view["test_missing_mask"] = mask
    view["test_missing_mode"] = mode
    view["test_missing_ratio"] = float(ratio)
    view["test_missing_realized_ratio"] = float(mask[:, 1:].mean()) if mask.shape[1] > 1 else 0.0
    view["test_missing_mask_seed"] = int(mask_seed)
    view["test_loader"] = DataLoader(
        TrackSequenceDataset(corrupted_scaled, view["t_test_0"], view["u_test_0"]),
        **build_loader_kwargs(config, dataset_size=len(corrupted_scaled), shuffle=False),
    )
    return view
