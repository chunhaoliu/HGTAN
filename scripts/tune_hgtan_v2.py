"""Validation-only search for prefix-supervised Temporal HGTAN-v2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import resolve_class_weights, sequence_data_provider  # noqa: E402
from data.data_loader import build_loader_kwargs  # noqa: E402
from models import build_model  # noqa: E402
from scripts.tune_temporal_hgtan import (  # noqa: E402
    aggregate_candidate,
    base_config,
    selection_score,
    write_json,
    write_summary_csv,
)
from utils.config import device, set_random_seed  # noqa: E402
from utils.project_paths import RESULTS_DIR  # noqa: E402


PREFIX_GRID = [8, 16, 24, 32, 48, 64]
CANDIDATES = [
    {"name": "legacy_final", "model": "TemporalHGTAN", "prefix_weight": 0.0, "ordinal_weight": 0.0},
    {"name": "legacy_prefix05", "model": "TemporalHGTAN", "prefix_weight": 0.5, "ordinal_weight": 0.0},
    {"name": "gru_prefix05", "model": "TemporalGRU", "prefix_weight": 0.5, "ordinal_weight": 0.0},
    {"name": "v2_final", "model": "TemporalHGTANV2", "prefix_weight": 0.0, "ordinal_weight": 0.0},
    {"name": "v2_prefix025", "model": "TemporalHGTANV2", "prefix_weight": 0.25, "ordinal_weight": 0.0},
    {"name": "v2_prefix05", "model": "TemporalHGTANV2", "prefix_weight": 0.5, "ordinal_weight": 0.0},
    {"name": "v2_prefix10", "model": "TemporalHGTANV2", "prefix_weight": 1.0, "ordinal_weight": 0.0},
    {"name": "v2_prefix05_ord005", "model": "TemporalHGTANV2", "prefix_weight": 0.5, "ordinal_weight": 0.05},
    {
        "name": "v2_noreliability_prefix10",
        "model": "TemporalHGTANV2_NoReliability",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
    },
    {
        "name": "v2_notemporal_prefix10",
        "model": "TemporalHGTANV2_NoTemporal",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
    },
    {
        "name": "v2_nosynergy_prefix10",
        "model": "TemporalHGTANV2_NoSynergy",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
    },
]


class PrefixDataset(Dataset):
    def __init__(self, x: np.ndarray, threat_seq: np.ndarray, urgency_seq: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.threat_seq = torch.as_tensor(threat_seq - 1, dtype=torch.long)
        self.urgency_seq = torch.as_tensor(urgency_seq - 1, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.threat_seq[index], self.urgency_seq[index]


def _class_tensor(values: list[float] | None) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.tensor(values, dtype=torch.float32, device=device)


def _ordinal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = torch.arange(logits.size(-1), device=logits.device, dtype=logits.dtype)
    expected = (torch.softmax(logits, dim=-1) * classes).sum(dim=-1)
    scale = max(logits.size(-1) - 1, 1)
    return F.smooth_l1_loss(expected / scale, labels.to(logits.dtype) / scale)


def _dual_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    threat: torch.Tensor,
    urgency: torch.Tensor,
    threat_criterion: nn.Module,
    urgency_criterion: nn.Module,
    urgency_weight: float,
    ordinal_weight: float,
) -> torch.Tensor:
    threat_logits, urgency_logits = outputs
    loss = threat_criterion(threat_logits, threat) + urgency_weight * urgency_criterion(urgency_logits, urgency)
    if ordinal_weight > 0:
        loss = loss + ordinal_weight * (
            _ordinal_loss(threat_logits, threat) + urgency_weight * _ordinal_loss(urgency_logits, urgency)
        )
    return loss


@torch.no_grad()
def evaluate_prefix_grid(model: nn.Module, bundle: dict[str, Any], batch_size: int) -> dict[str, float]:
    model.eval()
    x = torch.as_tensor(bundle["X_val"], dtype=torch.float32)
    threat_true_seq = np.asarray(bundle["threat_seq_val"], dtype=np.int64)
    urgency_true_seq = np.asarray(bundle["urgency_seq_val"], dtype=np.int64)
    threat_predictions = []
    urgency_predictions = []
    threat_targets = []
    urgency_targets = []
    final_threat = final_urgency = None
    grid = [prefix for prefix in PREFIX_GRID if prefix <= x.shape[1]]
    if x.shape[1] not in grid:
        grid.append(x.shape[1])
    for prefix in grid:
        prefix_threat = []
        prefix_urgency = []
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size, :prefix, :].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                threat_logits, urgency_logits = model(batch)
            prefix_threat.append(threat_logits.argmax(dim=-1).cpu().numpy() + 1)
            prefix_urgency.append(urgency_logits.argmax(dim=-1).cpu().numpy() + 1)
        threat_pred = np.concatenate(prefix_threat)
        urgency_pred = np.concatenate(prefix_urgency)
        threat_predictions.append(threat_pred)
        urgency_predictions.append(urgency_pred)
        threat_targets.append(threat_true_seq[:, prefix - 1])
        urgency_targets.append(urgency_true_seq[:, prefix - 1])
        if prefix == x.shape[1]:
            final_threat, final_urgency = threat_pred, urgency_pred

    threat_true = np.concatenate(threat_targets)
    urgency_true = np.concatenate(urgency_targets)
    threat_pred = np.concatenate(threat_predictions)
    urgency_pred = np.concatenate(urgency_predictions)
    final_threat_true = threat_true_seq[:, x.shape[1] - 1]
    final_urgency_true = urgency_true_seq[:, x.shape[1] - 1]
    assert final_threat is not None and final_urgency is not None
    threat_final_f1 = f1_score(final_threat_true, final_threat, average="macro", zero_division=0)
    urgency_final_f1 = f1_score(final_urgency_true, final_urgency, average="macro", zero_division=0)
    metrics = {
        "final_composite_f1": 0.75 * threat_final_f1 + 0.25 * urgency_final_f1,
        "threat_final_f1": threat_final_f1,
        "urgency_final_f1": urgency_final_f1,
        "threat_temporal_macro_f1": f1_score(threat_true, threat_pred, average="macro", zero_division=0),
        "urgency_temporal_macro_f1": f1_score(urgency_true, urgency_pred, average="macro", zero_division=0),
        "threat_temporal_accuracy": accuracy_score(threat_true, threat_pred),
        "urgency_temporal_accuracy": accuracy_score(urgency_true, urgency_pred),
        "threat_mean_abs_ordinal_error": float(np.mean(np.abs(threat_true - threat_pred))),
        "urgency_mean_abs_ordinal_error": float(np.mean(np.abs(urgency_true - urgency_pred))),
    }
    metrics["selection_score"] = selection_score(metrics)
    return {key: float(value) for key, value in metrics.items()}


def train_candidate(
    candidate: dict[str, Any],
    bundle: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    set_random_seed(seed, config)
    train_cfg = config["train"]
    class_weight_threat, class_weight_urgency = resolve_class_weights(
        train_cfg, bundle["t_train_0"], bundle["u_train_0"]
    )
    threat_criterion = nn.CrossEntropyLoss(
        weight=_class_tensor(class_weight_threat), label_smoothing=train_cfg.get("label_smoothing", 0.0)
    )
    urgency_criterion = nn.CrossEntropyLoss(
        weight=_class_tensor(class_weight_urgency), label_smoothing=train_cfg.get("label_smoothing", 0.0)
    )
    dataset = PrefixDataset(bundle["X_train"], bundle["threat_seq_train"], bundle["urgency_seq_train"])
    loader = DataLoader(
        dataset,
        **build_loader_kwargs(config, dataset_size=len(dataset), shuffle=True),
    )
    model = build_model(candidate["model"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(train_cfg["num_epochs"]), 1), eta_min=train_cfg["learning_rate"] * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed + 7919)
    prefix_choices = [p for p in PREFIX_GRID[:-1] if p < bundle["observed_len"]]
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        model.train()
        for x, threat_seq, urgency_seq in loader:
            x = x.to(device)
            threat_seq = threat_seq.to(device)
            urgency_seq = urgency_seq.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                final_loss = _dual_loss(
                    model(x),
                    threat_seq[:, -1],
                    urgency_seq[:, -1],
                    threat_criterion,
                    urgency_criterion,
                    train_cfg["urgency_weight"],
                    candidate["ordinal_weight"],
                )
                loss = final_loss
                if candidate["prefix_weight"] > 0 and prefix_choices:
                    prefix = int(rng.choice(prefix_choices))
                    prefix_loss = _dual_loss(
                        model(x[:, :prefix, :]),
                        threat_seq[:, prefix - 1],
                        urgency_seq[:, prefix - 1],
                        threat_criterion,
                        urgency_criterion,
                        train_cfg["urgency_weight"],
                        candidate["ordinal_weight"],
                    )
                    loss = loss + candidate["prefix_weight"] * prefix_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("gradient_clip_norm", 1.0))
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if epoch % 5 == 0 or epoch == int(train_cfg["num_epochs"]):
            metrics = evaluate_prefix_grid(model, bundle, train_cfg["batch_size"])
            if best_metrics is None or metrics["selection_score"] > best_metrics["selection_score"] + 1e-4:
                best_state = deepcopy(model.state_dict())
                best_metrics = metrics
                best_epoch = epoch
                stale = 0
            else:
                stale += 5
            print(
                f"  {candidate['name']} seed={seed} epoch={epoch}: "
                f"score={metrics['selection_score']:.4f} final={metrics['final_composite_f1']:.4f} "
                f"tF1={metrics['threat_temporal_macro_f1']:.4f}",
                flush=True,
            )
            if epoch >= train_cfg.get("min_epochs", 60) and stale >= train_cfg.get("patience", 25):
                break
    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return {
        "candidate": candidate["name"],
        "model": candidate["model"],
        "seed": seed,
        "metrics": best_metrics,
        "training": {
            "best_epoch": best_epoch,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=[c["name"] for c in CANDIDATES])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--policy-variant", default="balanced")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [candidate for candidate in CANDIDATES if candidate["name"] in args.candidates]
    unknown = sorted(set(args.candidates) - {candidate["name"] for candidate in selected})
    if unknown:
        raise ValueError(f"Unknown candidates: {', '.join(unknown)}")
    config = base_config()
    config["data"]["n_samples"] = args.n_samples
    config["train"]["num_epochs"] = args.epochs
    config["train"]["min_epochs"] = min(config["train"]["min_epochs"], args.epochs)
    config["train"]["use_amp"] = True
    config["sequence"]["reference_policy_variant"] = args.policy_variant
    if args.out_dir is None:
        args.out_dir = RESULTS_DIR / "optimization" / "hgtan_v2_prefix_search_v1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "manifest.json",
        {
            "scope": "validation_only",
            "prefix_grid": PREFIX_GRID,
            "seeds": args.seeds,
            "candidates": selected,
            "config": config,
        },
    )
    records = []
    bundles: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        bundles[seed] = sequence_data_provider(config, seed)
        for candidate in selected:
            path = args.out_dir / candidate["name"] / f"seed_{seed}.json"
            if path.exists() and not args.rerun:
                records.append(json.loads(path.read_text(encoding="utf-8")))
                print(f"Reused {candidate['name']} seed={seed}", flush=True)
                continue
            print(f"Running {candidate['name']} seed={seed}", flush=True)
            record = train_candidate(candidate, bundles[seed], config, seed)
            write_json(path, record)
            records.append(record)

    summaries = []
    for candidate in selected:
        seed_records = [record for record in records if record["candidate"] == candidate["name"]]
        summaries.append({"candidate": candidate["name"], **aggregate_candidate(seed_records)})
    ranking = sorted(summaries, key=lambda row: row["selection_score_mean"], reverse=True)
    write_json(args.out_dir / "ranking.json", ranking)
    write_summary_csv(args.out_dir / "summary.csv", ranking)
    print("\nHGTAN-v2 prefix-search ranking:")
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank:2d}. {row['candidate']:<24} score={row['selection_score_mean']:.4f} "
            f"final={row['final_composite_f1_mean']:.4f} "
            f"tF1={row['threat_temporal_macro_f1_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
