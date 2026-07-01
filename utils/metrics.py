"""Training, evaluation, calibration, and run-summary utilities."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.config import (
    CRITICAL_THREAT_CLASSES,
    CRITICAL_URGENCY_CLASSES,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    URGENCY_WEIGHT,
    WEIGHT_DECAY,
    device,
)
from utils.tools import cuda_amp_enabled, cuda_autocast, move_to_device


DEFAULT_THREAT_WEIGHT = 0.75


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _maybe_compile_model(model, compile_model: bool):
    if not compile_model or device.type != "cuda" or not hasattr(torch, "compile"):
        return model, False
    try:
        return torch.compile(model), True
    except Exception as exc:  # pragma: no cover - depends on local PyTorch backend.
        print(f"  torch.compile disabled for this run: {exc}")
        return model, False


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            target,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class WarmupCosineScheduler:
    """Simple epoch-level warmup plus cosine learning-rate schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lr_ratio = min_lr_ratio
        self.current_epoch = 0

    def step(self) -> None:
        self.current_epoch += 1
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_epoch <= self.warmup_epochs:
                lr = base_lr * self.current_epoch / max(self.warmup_epochs, 1)
            else:
                progress = (self.current_epoch - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs,
                    1,
                )
                cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    urgency_weight: float = URGENCY_WEIGHT,
    patience: int = PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    loss_type_threat: str = "ce",
    loss_type_urgency: str = "ce",
    class_weight_threat: list[float] | None = None,
    class_weight_urgency: list[float] | None = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    warmup_epochs: int = 5,
    min_delta: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    threat_weight: float = DEFAULT_THREAT_WEIGHT,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    min_epochs: int = 60,
    ema_alpha: float = 0.3,
    use_amp: bool = False,
    compile_model: bool = False,
) -> dict[str, Any]:
    """Train a dual-task model and return the best checkpoint plus curves."""
    model = model.to(device)
    model, compiled_model = _maybe_compile_model(model, compile_model)
    amp_enabled = cuda_amp_enabled(use_amp)
    scaler = _make_grad_scaler(amp_enabled)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, min_lr_ratio=0.01)

    criterion_threat = _build_criterion(
        loss_type=loss_type_threat,
        class_weight=class_weight_threat,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    criterion_urgency = _build_criterion(
        loss_type=loss_type_urgency,
        class_weight=class_weight_urgency,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )

    best_state = None
    best_epoch = 0
    best_val_score = -float("inf")
    best_ema_score = -float("inf")
    ema_score = None
    no_improve_count = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_scores: list[float] = []
    learning_rates: list[float] = []

    for epoch in range(int(num_epochs)):
        train_loss = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_threat=criterion_threat,
            criterion_urgency=criterion_urgency,
            urgency_weight=urgency_weight,
            gradient_clip_norm=gradient_clip_norm,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        if train_loss is None:
            break

        val_loss, threat_f1, urgency_f1 = _validate_one_epoch(
            model=model,
            val_loader=val_loader,
            criterion_threat=criterion_threat,
            criterion_urgency=criterion_urgency,
            urgency_weight=urgency_weight,
            amp_enabled=amp_enabled,
        )
        val_score = threat_weight * threat_f1 + (1.0 - threat_weight) * urgency_f1

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_scores.append(val_score)
        learning_rates.append(scheduler.get_last_lr()[0])
        scheduler.step()

        ema_score = val_score if ema_score is None else ema_alpha * val_score + (1.0 - ema_alpha) * ema_score
        if val_score > best_val_score + min_delta:
            best_val_score = val_score
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if ema_score > best_ema_score + min_delta:
            best_ema_score = ema_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 25 == 0 or epoch + 1 == int(num_epochs):
            print(
                f"  Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Threat F1={threat_f1:.4f}, Urgency F1={urgency_f1:.4f}"
            )

        if epoch + 1 >= min_epochs and no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_info = {
        "best_epoch": best_epoch,
        "final_epoch": len(train_losses),
        "best_val_score": float(best_val_score if best_val_score > -float("inf") else 0.0),
        "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
        "final_val_loss": float(val_losses[-1]) if val_losses else 0.0,
        "overfitting": _compute_overfitting_metrics(train_losses, val_losses),
        "learning_rates": learning_rates,
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "compiled_model": bool(compiled_model),
    }
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_scores": val_scores,
        "training_info": training_info,
    }


def train_model_two_stage(
    model,
    train_loader,
    val_loader,
    num_epochs_stage1: int = 80,
    num_epochs_stage2: int = 70,
    learning_rate: float = LEARNING_RATE,
    patience: int = PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    loss_type_threat: str = "ce",
    loss_type_urgency: str = "ce",
    class_weight_threat: list[float] | None = None,
    class_weight_urgency: list[float] | None = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    warmup_epochs: int = 5,
    min_delta: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    stage2_unfreeze_last_sam: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper for older two-stage experiments.

    The experiment runner now uses the unified joint recipe for fair comparison.
    This wrapper keeps legacy imports working without carrying duplicate loops.
    """
    del stage2_unfreeze_last_sam
    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs_stage1 + num_epochs_stage2,
        learning_rate=learning_rate,
        patience=patience,
        weight_decay=weight_decay,
        loss_type_threat=loss_type_threat,
        loss_type_urgency=loss_type_urgency,
        class_weight_threat=class_weight_threat,
        class_weight_urgency=class_weight_urgency,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        warmup_epochs=warmup_epochs,
        min_delta=min_delta,
        gradient_clip_norm=gradient_clip_norm,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        min_epochs=min(num_epochs_stage1, num_epochs_stage1 + num_epochs_stage2),
    )


def evaluate_model(model, test_loader, *, use_amp: bool = False):
    """Return threat metrics, urgency metrics, and 0-based predictions."""
    model.eval()
    threat_pred, threat_true, urgency_pred, urgency_true = [], [], [], []
    threat_probs, urgency_probs = [], []
    amp_enabled = cuda_amp_enabled(use_amp)

    with torch.no_grad():
        for batch_x, batch_threat, batch_urgency in test_loader:
            batch_x = move_to_device(batch_x)
            with cuda_autocast(amp_enabled):
                threat_logits, urgency_logits = model(batch_x)
            threat_pred.extend(threat_logits.argmax(dim=1).cpu().numpy())
            urgency_pred.extend(urgency_logits.argmax(dim=1).cpu().numpy())
            threat_true.extend(batch_threat.numpy())
            urgency_true.extend(batch_urgency.numpy())
            threat_probs.append(torch.softmax(threat_logits, dim=1).cpu().numpy())
            urgency_probs.append(torch.softmax(urgency_logits, dim=1).cpu().numpy())

    threat_probs_arr = np.concatenate(threat_probs, axis=0) if threat_probs else None
    urgency_probs_arr = np.concatenate(urgency_probs, axis=0) if urgency_probs else None
    threat_metrics = build_classification_metrics(
        np.asarray(threat_true),
        np.asarray(threat_pred),
        probs=threat_probs_arr,
        critical_labels_1based=CRITICAL_THREAT_CLASSES,
    )
    urgency_metrics = build_classification_metrics(
        np.asarray(urgency_true),
        np.asarray(urgency_pred),
        probs=urgency_probs_arr,
        critical_labels_1based=CRITICAL_URGENCY_CLASSES,
    )
    return threat_metrics, urgency_metrics, list(threat_pred), list(urgency_pred)


def build_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probs: np.ndarray | None = None,
    critical_labels_1based: list[int] | None = None,
) -> dict[str, float | None]:
    """Compute classification, calibration, and ordinal decision metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    critical_labels_1based = critical_labels_1based or []

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "critical_recall": _critical_recall(y_true, y_pred, critical_labels_1based),
        "critical_miss_rate": _critical_miss_rate(y_true, y_pred, critical_labels_1based),
        "decision_cost": _decision_cost(y_true, y_pred, critical_labels_1based),
        "ece": None,
        "brier": None,
    }
    if probs is not None:
        metrics["ece"] = _expected_calibration_error(probs, y_true)
        metrics["brier"] = _multiclass_brier_score(probs, y_true)
    return metrics


def evaluate_per_class(model, test_loader, n_classes: int = 5, n_urgency: int = 3) -> dict[str, Any]:
    """Return per-class support, precision, recall, and F1 diagnostics."""
    threat_metrics, urgency_metrics, threat_pred, urgency_pred = evaluate_model(model, test_loader)
    del threat_metrics, urgency_metrics

    threat_true, urgency_true = [], []
    for _, batch_threat, batch_urgency in test_loader:
        threat_true.extend(batch_threat.numpy())
        urgency_true.extend(batch_urgency.numpy())

    return {
        "threat_per_class": _per_class_table(np.asarray(threat_true), np.asarray(threat_pred), n_classes),
        "urgency_per_class": _per_class_table(np.asarray(urgency_true), np.asarray(urgency_pred), n_urgency),
    }


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def measure_inference_time(model, data_loader, n_runs: int = 100, *, use_amp: bool = False) -> float:
    """Return average inference latency in milliseconds per batch."""
    model.eval()
    model = model.to(device)
    amp_enabled = cuda_amp_enabled(use_amp)
    times = []
    with torch.no_grad():
        for batch_idx, (batch_x, _, _) in enumerate(data_loader):
            if batch_idx >= n_runs:
                break
            batch_x = move_to_device(batch_x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with cuda_autocast(amp_enabled):
                _ = model(batch_x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(times)) if times else 0.0


def analyze_group_attention(model, test_loader):
    """Return mean group attention matrix when the model exposes attention weights."""
    model.eval()
    attention_batches = []
    with torch.no_grad():
        for batch_x, _, _ in test_loader:
            _ = model(move_to_device(batch_x))
            if hasattr(model, "get_attention_weights"):
                attention = model.get_attention_weights()
                if attention is not None:
                    attention_batches.append(attention.mean(dim=1).cpu().numpy())
    if not attention_batches:
        return None
    return np.concatenate(attention_batches, axis=0).mean(axis=0)


def compute_composite_f1(results_dict: dict[str, Any], threat_weight: float | None = None) -> float:
    """Compute weighted F1 over threat and urgency tasks."""
    if threat_weight is None:
        threat_weight = DEFAULT_THREAT_WEIGHT
    return float(
        threat_weight * results_dict["threat"]["f1"]
        + (1.0 - threat_weight) * results_dict["urgency"]["f1"]
    )


def summarize_runs(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated runs as mean/std/95% CI by model and metric."""
    summary = {"num_runs": len(run_records), "models": {}, "efficiency": {}}
    if not run_records:
        return summary

    model_names = list(
        dict.fromkeys(
            model_name
            for record in run_records
            for model_name in record.get("results", {}).keys()
        )
    )
    for model_name in model_names:
        summary["models"][model_name] = {"threat": {}, "urgency": {}, "composite_f1": {}}
        for task in ["threat", "urgency"]:
            metric_names = sorted(
                {
                    metric_name
                    for record in run_records
                    for metric_name in record["results"].get(model_name, {}).get(task, {}).keys()
                }
            )
            for metric in metric_names:
                values = [
                    float(record["results"][model_name][task][metric])
                    for record in run_records
                    if model_name in record.get("results", {})
                    and metric in record["results"][model_name].get(task, {})
                    and record["results"][model_name][task][metric] is not None
                ]
                if values:
                    summary["models"][model_name][task][metric] = _summarize_values(values)

        composite_values = [
            compute_composite_f1(record["results"][model_name])
            for record in run_records
            if model_name in record.get("results", {})
        ]
        if composite_values:
            summary["models"][model_name]["composite_f1"] = _summarize_values(composite_values)

        efficiency_metrics = sorted(
            {
                metric
                for record in run_records
                for metric in record.get("efficiency", {}).get(model_name, {}).keys()
            }
        )
        if efficiency_metrics:
            summary["efficiency"][model_name] = {}
        for metric in efficiency_metrics:
            values = [
                float(record["efficiency"][model_name][metric])
                for record in run_records
                if model_name in record.get("efficiency", {}) and metric in record["efficiency"][model_name]
            ]
            if values:
                summary["efficiency"][model_name][metric] = _summarize_values(values)
    return summary


def mixup_data(x: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, alpha: float = 0.2):
    """Apply mixup to one batch with two label targets."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y1, y1[index], y2, y2[index], lam


def mixup_data_single(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Apply mixup to one batch with one label target."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def _train_one_epoch(
    *,
    model,
    train_loader,
    optimizer,
    criterion_threat,
    criterion_urgency,
    urgency_weight: float,
    gradient_clip_norm: float,
    use_mixup: bool,
    mixup_alpha: float,
    amp_enabled: bool,
    scaler,
) -> float | None:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_threat, batch_urgency in train_loader:
        batch_x = move_to_device(batch_x)
        batch_threat = move_to_device(batch_threat)
        batch_urgency = move_to_device(batch_urgency)
        optimizer.zero_grad(set_to_none=True)

        with cuda_autocast(amp_enabled):
            if use_mixup and mixup_alpha > 0:
                mixed_x, t_a, t_b, u_a, u_b, lam = mixup_data(batch_x, batch_threat, batch_urgency, mixup_alpha)
                threat_logits, urgency_logits = model(mixed_x)
                loss_threat = mixup_criterion(criterion_threat, threat_logits, t_a, t_b, lam)
                loss_urgency = mixup_criterion(criterion_urgency, urgency_logits, u_a, u_b, lam)
            else:
                threat_logits, urgency_logits = model(batch_x)
                loss_threat = criterion_threat(threat_logits, batch_threat)
                loss_urgency = criterion_urgency(urgency_logits, batch_urgency)

            loss = loss_threat + urgency_weight * loss_urgency
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        scaler.scale(loss).backward()
        if gradient_clip_norm and gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        num_batches += 1

    if num_batches == 0:
        return None
    return total_loss / num_batches


def _validate_one_epoch(
    *,
    model,
    val_loader,
    criterion_threat,
    criterion_urgency,
    urgency_weight: float,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    threat_pred, threat_true, urgency_pred, urgency_true = [], [], [], []

    with torch.no_grad():
        for batch_x, batch_threat, batch_urgency in val_loader:
            batch_x = move_to_device(batch_x)
            batch_threat_dev = move_to_device(batch_threat)
            batch_urgency_dev = move_to_device(batch_urgency)
            with cuda_autocast(amp_enabled):
                threat_logits, urgency_logits = model(batch_x)
                loss_threat = criterion_threat(threat_logits, batch_threat_dev)
                loss_urgency = criterion_urgency(urgency_logits, batch_urgency_dev)
            total_loss += float((loss_threat + urgency_weight * loss_urgency).item())
            num_batches += 1

            threat_pred.extend(threat_logits.argmax(dim=1).cpu().numpy())
            urgency_pred.extend(urgency_logits.argmax(dim=1).cpu().numpy())
            threat_true.extend(batch_threat.numpy())
            urgency_true.extend(batch_urgency.numpy())

    avg_loss = total_loss / max(num_batches, 1)
    threat_f1 = f1_score(threat_true, threat_pred, average="macro", zero_division=0)
    urgency_f1 = f1_score(urgency_true, urgency_pred, average="macro", zero_division=0)
    return float(avg_loss), float(threat_f1), float(urgency_f1)


def _build_criterion(
    loss_type: str = "ce",
    class_weight: list[float] | None = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
):
    weight = torch.tensor(class_weight, dtype=torch.float32, device=device) if class_weight is not None else None
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=weight, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss_type={loss_type!r}")


def _compute_overfitting_metrics(train_losses: list[float], val_losses: list[float]) -> dict[str, Any]:
    if len(train_losses) < 10 or len(val_losses) < 10:
        return {"status": "insufficient_data"}

    train_arr = np.asarray(train_losses, dtype=np.float64)
    val_arr = np.asarray(val_losses, dtype=np.float64)
    final_gap = val_arr[-1] - train_arr[-1]
    mid = len(train_arr) // 2
    early_gap = np.mean(val_arr[:mid] - train_arr[:mid])
    late_gap = np.mean(val_arr[mid:] - train_arr[mid:])
    third = max(len(val_arr) // 3, 1)
    val_trend = np.mean(val_arr[-third:]) - np.mean(val_arr[:third])

    if final_gap < -0.05 and val_trend <= 0:
        status = "regularized_gap"
    elif late_gap - early_gap > 0.1 and val_trend > 0.05:
        status = "severe_overfitting"
    elif late_gap - early_gap > 0.05 or val_trend > 0.02:
        status = "mild_overfitting"
    elif final_gap < 0.05:
        status = "healthy"
    else:
        status = "normal"

    return {
        "status": status,
        "final_gap": float(final_gap),
        "gap_increase": float(late_gap - early_gap),
        "val_trend": float(val_trend),
    }


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    labels_0 = _to_zero_based_for_probs(labels, probs.shape[1])
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels_0).astype(np.float64)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= start) & (confidences <= end if end == 1.0 else confidences < end)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(accuracies[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def _multiclass_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    labels_0 = _to_zero_based_for_probs(labels, probs.shape[1])
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels_0]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _critical_recall(y_true: np.ndarray, y_pred: np.ndarray, critical_labels_1based: list[int]) -> float:
    if not critical_labels_1based:
        return 0.0
    true_ord, pred_ord = _to_ordinal_labels(y_true, y_pred)
    mask = np.isin(true_ord, critical_labels_1based)
    if not np.any(mask):
        return 0.0
    return float(np.mean(true_ord[mask] == pred_ord[mask]))


def _critical_miss_rate(y_true: np.ndarray, y_pred: np.ndarray, critical_labels_1based: list[int]) -> float | None:
    if not critical_labels_1based:
        return None
    true_ord, pred_ord = _to_ordinal_labels(y_true, y_pred)
    mask = np.isin(true_ord, critical_labels_1based)
    if not np.any(mask):
        return None
    return float(np.mean(pred_ord[mask] < true_ord[mask]))


def _decision_cost(y_true: np.ndarray, y_pred: np.ndarray, critical_labels_1based: list[int]) -> float:
    true_ord, pred_ord = _to_ordinal_labels(y_true, y_pred)
    distance = np.abs(true_ord - pred_ord).astype(np.float64)
    underestimation = np.maximum(true_ord - pred_ord, 0).astype(np.float64)
    overestimation = np.maximum(pred_ord - true_ord, 0).astype(np.float64)
    critical_miss = (np.isin(true_ord, critical_labels_1based) & (pred_ord < true_ord)).astype(np.float64)
    return float(np.mean(distance + 1.5 * underestimation + 0.5 * overestimation + 2.0 * critical_miss))


def _to_ordinal_labels(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zero_based = np.min(y_true) == 0 or np.min(y_pred) == 0
    if zero_based:
        return y_true + 1, y_pred + 1
    return y_true, y_pred


def _to_zero_based_for_probs(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and labels.min() == 1 and labels.max() <= n_classes:
        return labels - 1
    return labels


def _per_class_table(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[int, dict[str, float | int]]:
    table = {}
    for class_id in range(n_classes):
        mask = y_true == class_id
        if not np.any(mask):
            continue
        table[class_id] = {
            "support": int(mask.sum()),
            "precision": float(precision_score(y_true == class_id, y_pred == class_id, zero_division=0)),
            "recall": float(recall_score(y_true == class_id, y_pred == class_id, zero_division=0)),
            "f1": float(f1_score(y_true == class_id, y_pred == class_id, zero_division=0)),
        }
    return table


def _summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }
