"""Temporal evaluation utilities for sequential threat assessment."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from utils.config import device
from utils.tools import cuda_amp_enabled, cuda_autocast


def predict_prefix_labels(
    model: torch.nn.Module,
    sequences: np.ndarray,
    *,
    batch_size: int,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict threat and urgency labels for every observed prefix length.

    Returned labels are 1-based, matching the CSV/NPZ artifact convention.
    """
    sequences = np.asarray(sequences, dtype=np.float32)
    if sequences.ndim != 3:
        raise ValueError(f"Expected sequence tensor with shape (n, time, features), got {sequences.shape}")
    if sequences.shape[1] < 1:
        raise ValueError("At least one observed time step is required for prefix evaluation.")

    model.eval()
    model = model.to(device)
    amp_enabled = cuda_amp_enabled(use_amp)
    threat_steps: list[np.ndarray] = []
    urgency_steps: list[np.ndarray] = []

    with torch.no_grad():
        for end_step in range(1, sequences.shape[1] + 1):
            threat_pred = []
            urgency_pred = []
            for start in range(0, len(sequences), batch_size):
                batch = torch.as_tensor(
                    sequences[start : start + batch_size, :end_step, :],
                    dtype=torch.float32,
                    device=device,
                )
                with cuda_autocast(amp_enabled):
                    threat_logits, urgency_logits = model(batch)
                threat_pred.append(threat_logits.argmax(dim=1).cpu().numpy() + 1)
                urgency_pred.append(urgency_logits.argmax(dim=1).cpu().numpy() + 1)
            threat_steps.append(np.concatenate(threat_pred, axis=0))
            urgency_steps.append(np.concatenate(urgency_pred, axis=0))

    return np.stack(threat_steps, axis=1), np.stack(urgency_steps, axis=1)


def compute_track_metrics(
    true_seq: np.ndarray,
    pred_seq: np.ndarray,
    *,
    critical_labels: list[int],
    frame_interval: float,
) -> dict[str, float | int | None]:
    """Compute temporal consistency and first-critical-event metrics."""
    true_seq = np.asarray(true_seq, dtype=np.int64)
    pred_seq = np.asarray(pred_seq, dtype=np.int64)
    if true_seq.shape != pred_seq.shape:
        raise ValueError(f"true_seq and pred_seq must have the same shape, got {true_seq.shape} and {pred_seq.shape}")
    if true_seq.ndim != 2:
        raise ValueError(f"Expected sequence labels with shape (n, time), got {true_seq.shape}")

    n_tracks, n_steps = true_seq.shape
    abs_error = np.abs(true_seq - pred_seq)
    metrics: dict[str, float | int | None] = {
        "support_tracks": int(n_tracks),
        "support_timepoints": int(n_tracks * n_steps),
        "temporal_accuracy": float(np.mean(true_seq == pred_seq)),
        "temporal_macro_f1": float(f1_score(true_seq.ravel(), pred_seq.ravel(), average="macro", zero_division=0)),
        "mean_abs_ordinal_error": float(np.mean(abs_error)),
        "final_step_accuracy": float(np.mean(true_seq[:, -1] == pred_seq[:, -1])),
        "final_abs_ordinal_error": float(np.mean(abs_error[:, -1])),
    }

    pred_flip_rate = _flip_rate(pred_seq)
    true_flip_rate = _flip_rate(true_seq)
    metrics.update(
        {
            "pred_flip_rate": pred_flip_rate,
            "true_flip_rate": true_flip_rate,
            "stability_gap": abs(pred_flip_rate - true_flip_rate),
            "terminal_trend_agreement": _terminal_trend_agreement(true_seq, pred_seq),
            "risk_escalation_recall": _risk_escalation_recall(true_seq, pred_seq),
        }
    )
    metrics.update(_critical_event_metrics(true_seq, pred_seq, critical_labels, frame_interval))
    return metrics


def _flip_rate(sequence: np.ndarray) -> float:
    if sequence.shape[1] <= 1:
        return 0.0
    return float(np.mean(sequence[:, 1:] != sequence[:, :-1]))


def _terminal_trend_agreement(true_seq: np.ndarray, pred_seq: np.ndarray) -> float:
    true_trend = np.sign(true_seq[:, -1] - true_seq[:, 0])
    pred_trend = np.sign(pred_seq[:, -1] - pred_seq[:, 0])
    return float(np.mean(true_trend == pred_trend))


def _risk_escalation_recall(true_seq: np.ndarray, pred_seq: np.ndarray) -> float | None:
    true_escalates = true_seq[:, -1] > true_seq[:, 0]
    if not np.any(true_escalates):
        return None
    pred_escalates = pred_seq[:, -1] > pred_seq[:, 0]
    return float(np.mean(pred_escalates[true_escalates]))


def _critical_event_metrics(
    true_seq: np.ndarray,
    pred_seq: np.ndarray,
    critical_labels: list[int],
    frame_interval: float,
) -> dict[str, float | int | None]:
    true_critical = np.isin(true_seq, critical_labels)
    pred_critical = np.isin(pred_seq, critical_labels)
    true_track_critical = np.any(true_critical, axis=1)
    pred_track_critical = np.any(pred_critical, axis=1)
    critical_track_support = int(true_track_critical.sum())

    metrics: dict[str, float | int | None] = {
        "critical_track_support": critical_track_support,
        "critical_recall_over_time": None,
        "critical_track_miss_rate": None,
        "critical_false_alarm_rate": None,
        "mean_detection_delay_steps": None,
        "mean_abs_detection_delay_steps": None,
        "mean_detection_delay_seconds": None,
        "mean_early_warning_seconds": None,
        "early_alarm_rate": None,
    }
    if np.any(true_critical):
        metrics["critical_recall_over_time"] = float(np.mean(pred_critical[true_critical]))

    no_true_critical = ~true_track_critical
    if np.any(no_true_critical):
        metrics["critical_false_alarm_rate"] = float(np.mean(pred_track_critical[no_true_critical]))

    if critical_track_support == 0:
        return metrics

    true_first = _first_true_index(true_critical[true_track_critical])
    pred_first = _first_true_index(pred_critical[true_track_critical])
    detected = pred_first >= 0
    metrics["critical_track_miss_rate"] = float(np.mean(~detected))

    if np.any(detected):
        delays = pred_first[detected] - true_first[detected]
        metrics["mean_detection_delay_steps"] = float(np.mean(delays))
        metrics["mean_abs_detection_delay_steps"] = float(np.mean(np.abs(delays)))
        metrics["mean_detection_delay_seconds"] = float(np.mean(delays) * frame_interval)
        metrics["mean_early_warning_seconds"] = float(np.mean(np.maximum(-delays, 0)) * frame_interval)
        metrics["early_alarm_rate"] = float(np.mean(delays < 0))
    return metrics


def _first_true_index(mask: np.ndarray) -> np.ndarray:
    has_event = np.any(mask, axis=1)
    first = np.argmax(mask, axis=1).astype(np.int64)
    first[~has_event] = -1
    return first
