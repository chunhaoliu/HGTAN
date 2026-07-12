"""Multi-axis validation search for TemporalHGTAN_Cursor.

Trains with multi-horizon prefix supervision and random length truncation, then
scores each candidate on default / short-history (obs32) / far-range (5000 m).
Validation only: never reads the test split.
"""

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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import resolve_class_weights, sequence_data_provider  # noqa: E402
from data.data_loader import build_loader_kwargs  # noqa: E402
from models import build_model  # noqa: E402
from scripts.tune_hgtan_v2 import (  # noqa: E402
    PREFIX_GRID,
    PrefixDataset,
    _augment_prefix,
    _class_tensor,
    _dual_loss,
    evaluate_prefix_grid,
)
from scripts.tune_temporal_hgtan import (  # noqa: E402
    aggregate_candidate,
    base_config,
    write_json,
    write_summary_csv,
)
from utils.config import device, set_random_seed  # noqa: E402
from utils.project_paths import RESULTS_DIR  # noqa: E402

LENGTH_CHOICES = [24, 32, 40, 48, 64]

# Round-1 architecture / recipe screen (keep list tight; expand after survivors).
CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "cursor_mh_robust",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_mh_noreliability",
        "model": "TemporalHGTAN_CursorNoReliability",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_mh_mix",
        "model": "TemporalHGTAN_CursorWithMix",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_mh_deep",
        "model": "TemporalHGTAN_CursorDeep",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_mh_strong_prefix",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 1.5,
        "ordinal_weight": 0.05,
        "multi_prefix": 3,
        "length_mix_prob": 0.7,
        "noise_std": 0.01,
        "frame_drop_prob": 0.04,
        "group_drop_prob": 0.12,
    },
    {
        "name": "cursor_mh_far_aug",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.02,
        "frame_drop_prob": 0.08,
        "group_drop_prob": 0.15,
    },
    # Matched baselines under the same multi-horizon recipe.
    {
        "name": "legacy_mh_robust",
        "model": "TemporalHGTAN",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "gru_mh_robust",
        "model": "TemporalGRU",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "flat_mh_robust",
        "model": "FlatSequenceMLP",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "transformer_mh_robust",
        "model": "TemporalTransformer",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    # Recipe expansions used by later autonomous rounds.
    {
        "name": "cursor_lr2e4_mh",
        "model": "TemporalHGTAN_Cursor",
        "learning_rate": 2e-4,
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 3,
        "length_mix_prob": 0.6,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_lr4e4_mh",
        "model": "TemporalHGTAN_Cursor",
        "learning_rate": 4e-4,
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_d005_w1e3",
        "model": "TemporalHGTAN_Cursor",
        "dropout": 0.05,
        "weight_decay": 1e-3,
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.5,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
    {
        "name": "cursor_short_heavy",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 2.0,
        "ordinal_weight": 0.05,
        "multi_prefix": 4,
        "length_mix_prob": 0.85,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.08,
    },
    {
        "name": "cursor_noreliab_short_heavy",
        "model": "TemporalHGTAN_CursorNoReliability",
        "prefix_weight": 2.0,
        "ordinal_weight": 0.05,
        "multi_prefix": 4,
        "length_mix_prob": 0.85,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.08,
    },
    {
        "name": "cursor_deep_short_heavy",
        "model": "TemporalHGTAN_CursorDeep",
        "prefix_weight": 1.5,
        "ordinal_weight": 0.0,
        "multi_prefix": 3,
        "length_mix_prob": 0.7,
        "noise_std": 0.015,
        "frame_drop_prob": 0.05,
        "group_drop_prob": 0.10,
    },
    # Far-range focused recipes for v3+.
    {
        "name": "cursor_range_mix_light",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.4,
        "range_mix_prob": 0.35,
        "noise_std": 0.015,
        "frame_drop_prob": 0.06,
        "group_drop_prob": 0.12,
    },
    {
        "name": "cursor_range_mix_strong",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.4,
        "range_mix_prob": 0.55,
        "noise_std": 0.02,
        "frame_drop_prob": 0.08,
        "group_drop_prob": 0.15,
    },
    {
        "name": "cursor_far_aug_range_mix",
        "model": "TemporalHGTAN_Cursor",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.45,
        "range_mix_prob": 0.45,
        "noise_std": 0.02,
        "frame_drop_prob": 0.08,
        "group_drop_prob": 0.15,
    },
    {
        "name": "cursor_mix_range_mix",
        "model": "TemporalHGTAN_CursorWithMix",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.4,
        "range_mix_prob": 0.45,
        "noise_std": 0.015,
        "frame_drop_prob": 0.06,
        "group_drop_prob": 0.12,
    },
    {
        "name": "cursor_lr4e4_range_mix",
        "model": "TemporalHGTAN_Cursor",
        "learning_rate": 4e-4,
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.4,
        "range_mix_prob": 0.45,
        "noise_std": 0.015,
        "frame_drop_prob": 0.06,
        "group_drop_prob": 0.12,
    },
    {
        "name": "core_lr4e4_range_mix",
        "model": "TemporalHGTANV2_Core",
        "learning_rate": 4e-4,
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.4,
        "range_mix_prob": 0.45,
        "noise_std": 0.015,
        "frame_drop_prob": 0.06,
        "group_drop_prob": 0.12,
    },
    {
        "name": "core_mh_robust",
        "model": "TemporalHGTANV2_Core",
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.4,
        "range_mix_prob": 0.0,
        "noise_std": 0.015,
        "frame_drop_prob": 0.06,
        "group_drop_prob": 0.12,
    },
    {
        "name": "core_lr4e4_range_mix_soft",
        "model": "TemporalHGTANV2_Core",
        "learning_rate": 4e-4,
        "prefix_weight": 1.0,
        "ordinal_weight": 0.0,
        "multi_prefix": 2,
        "length_mix_prob": 0.3,
        "range_mix_prob": 0.35,
        "noise_std": 0.01,
        "frame_drop_prob": 0.03,
        "group_drop_prob": 0.10,
    },
]


def axis_score(default_f1: float, short_f1: float, far_f1: float) -> float:
    return float(0.4 * default_f1 + 0.3 * short_f1 + 0.3 * far_f1)


@torch.no_grad()
def evaluate_final_at_prefix(
    model: nn.Module, bundle: dict[str, Any], batch_size: int, prefix: int
) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    x = torch.as_tensor(bundle["X_val"], dtype=torch.float32)
    threat_true = np.asarray(bundle["threat_seq_val"][:, prefix - 1], dtype=np.int64)
    urgency_true = np.asarray(bundle["urgency_seq_val"][:, prefix - 1], dtype=np.int64)
    threat_pred = []
    urgency_pred = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size, :prefix, :].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            threat_logits, urgency_logits = model(batch)
        threat_pred.append(threat_logits.argmax(dim=-1).cpu().numpy() + 1)
        urgency_pred.append(urgency_logits.argmax(dim=-1).cpu().numpy() + 1)
    threat_pred = np.concatenate(threat_pred)
    urgency_pred = np.concatenate(urgency_pred)
    threat_f1 = f1_score(threat_true, threat_pred, average="macro", zero_division=0)
    urgency_f1 = f1_score(urgency_true, urgency_pred, average="macro", zero_division=0)
    return float(0.75 * threat_f1 + 0.25 * urgency_f1)


def evaluate_axes(
    model: nn.Module,
    default_bundle: dict[str, Any],
    far_bundle: dict[str, Any],
    batch_size: int,
) -> dict[str, float]:
    default_metrics = evaluate_prefix_grid(model, default_bundle, batch_size)
    short_f1 = evaluate_final_at_prefix(model, default_bundle, batch_size, prefix=32)
    far_metrics = evaluate_prefix_grid(model, far_bundle, batch_size)
    metrics = {
        **{f"default_{k}": v for k, v in default_metrics.items()},
        "short32_final_composite_f1": short_f1,
        "far5000_final_composite_f1": far_metrics["final_composite_f1"],
        "far5000_threat_temporal_macro_f1": far_metrics["threat_temporal_macro_f1"],
        "axis_score": axis_score(
            default_metrics["final_composite_f1"],
            short_f1,
            far_metrics["final_composite_f1"],
        ),
    }
    # Keep legacy keys for aggregate helpers where possible.
    metrics["final_composite_f1"] = default_metrics["final_composite_f1"]
    metrics["threat_temporal_macro_f1"] = default_metrics["threat_temporal_macro_f1"]
    metrics["urgency_temporal_macro_f1"] = default_metrics["urgency_temporal_macro_f1"]
    metrics["threat_temporal_accuracy"] = default_metrics["threat_temporal_accuracy"]
    metrics["urgency_temporal_accuracy"] = default_metrics["urgency_temporal_accuracy"]
    metrics["threat_final_f1"] = default_metrics["threat_final_f1"]
    metrics["urgency_final_f1"] = default_metrics["urgency_final_f1"]
    metrics["threat_mean_abs_ordinal_error"] = default_metrics["threat_mean_abs_ordinal_error"]
    metrics["urgency_mean_abs_ordinal_error"] = default_metrics["urgency_mean_abs_ordinal_error"]
    metrics["selection_score"] = metrics["axis_score"]
    return metrics


def train_candidate(
    candidate: dict[str, Any],
    default_bundle: dict[str, Any],
    far_bundle: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    set_random_seed(seed, config)
    train_cfg = config["train"]
    class_weight_threat, class_weight_urgency = resolve_class_weights(
        train_cfg, default_bundle["t_train_0"], default_bundle["u_train_0"]
    )
    threat_criterion = nn.CrossEntropyLoss(
        weight=_class_tensor(class_weight_threat),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )
    urgency_criterion = nn.CrossEntropyLoss(
        weight=_class_tensor(class_weight_urgency),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )
    dataset = PrefixDataset(
        default_bundle["X_train"],
        default_bundle["threat_seq_train"],
        default_bundle["urgency_seq_train"],
    )
    far_dataset = PrefixDataset(
        far_bundle["X_train"],
        far_bundle["threat_seq_train"],
        far_bundle["urgency_seq_train"],
    )
    loader = DataLoader(
        dataset,
        **build_loader_kwargs(config, dataset_size=len(dataset), shuffle=True),
    )
    far_loader = DataLoader(
        far_dataset,
        **build_loader_kwargs(config, dataset_size=len(far_dataset), shuffle=True),
    )
    far_iter = iter(far_loader)
    model_cfg = deepcopy(config["model"])
    if candidate.get("dropout") is not None:
        model_cfg["dropout"] = float(candidate["dropout"])
    model = build_model(candidate["model"], model_cfg).to(device)
    learning_rate = float(candidate.get("learning_rate", train_cfg["learning_rate"]))
    weight_decay = float(candidate.get("weight_decay", train_cfg["weight_decay"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(train_cfg["num_epochs"]), 1),
        eta_min=learning_rate * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(seed + 7919)
    observed_len = int(default_bundle["observed_len"])
    prefix_choices = [p for p in PREFIX_GRID if p < observed_len]
    length_choices = [p for p in LENGTH_CHOICES if p <= observed_len]
    multi_prefix = int(candidate.get("multi_prefix", 1))
    length_mix_prob = float(candidate.get("length_mix_prob", 0.0))
    range_mix_prob = float(candidate.get("range_mix_prob", 0.0))

    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        model.train()
        for x, threat_seq, urgency_seq in loader:
            if range_mix_prob > 0 and rng.random() < range_mix_prob:
                try:
                    x, threat_seq, urgency_seq = next(far_iter)
                except StopIteration:
                    far_iter = iter(far_loader)
                    x, threat_seq, urgency_seq = next(far_iter)
            x = x.to(device)
            threat_seq = threat_seq.to(device)
            urgency_seq = urgency_seq.to(device)

            # Random length truncation to force short-window competence.
            if length_mix_prob > 0 and rng.random() < length_mix_prob and length_choices:
                cut = int(rng.choice(length_choices))
                x = x[:, :cut, :]
                threat_seq = threat_seq[:, :cut]
                urgency_seq = urgency_seq[:, :cut]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                x_aug = _augment_prefix(x, candidate)
                loss = _dual_loss(
                    model(x_aug),
                    threat_seq[:, -1],
                    urgency_seq[:, -1],
                    threat_criterion,
                    urgency_criterion,
                    train_cfg["urgency_weight"],
                    float(candidate.get("ordinal_weight", 0.0)),
                )
                if candidate.get("prefix_weight", 0.0) > 0 and prefix_choices:
                    prefix_losses = []
                    usable = [p for p in prefix_choices if p < x.size(1)]
                    if usable:
                        n_draw = min(multi_prefix, len(usable))
                        for prefix in rng.choice(usable, size=n_draw, replace=False):
                            prefix = int(prefix)
                            prefix_x = _augment_prefix(x[:, :prefix, :], candidate)
                            prefix_losses.append(
                                _dual_loss(
                                    model(prefix_x),
                                    threat_seq[:, prefix - 1],
                                    urgency_seq[:, prefix - 1],
                                    threat_criterion,
                                    urgency_criterion,
                                    train_cfg["urgency_weight"],
                                    float(candidate.get("ordinal_weight", 0.0)),
                                )
                            )
                        if prefix_losses:
                            loss = loss + float(candidate["prefix_weight"]) * (
                                sum(prefix_losses) / len(prefix_losses)
                            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("gradient_clip_norm", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if epoch % 5 == 0 or epoch == int(train_cfg["num_epochs"]):
            metrics = evaluate_axes(model, default_bundle, far_bundle, train_cfg["batch_size"])
            improved = best_metrics is None or metrics["axis_score"] > best_metrics["axis_score"] + 1e-4
            if improved:
                best_state = deepcopy(model.state_dict())
                best_metrics = metrics
                best_epoch = epoch
                stale = 0
            else:
                stale += 5
            print(
                f"  {candidate['name']} seed={seed} epoch={epoch}: "
                f"axis={metrics['axis_score']:.4f} "
                f"def={metrics['final_composite_f1']:.4f} "
                f"s32={metrics['short32_final_composite_f1']:.4f} "
                f"far={metrics['far5000_final_composite_f1']:.4f}",
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


def far_range_config(config: dict[str, Any]) -> dict[str, Any]:
    far = deepcopy(config)
    far["sequence"]["range_m"] = 5000.0
    far["sequence"]["track_noise_std"] = 0.015
    far["sequence"]["track_missing_ratio"] = 0.08
    far["sequence"]["track_jitter_std"] = 0.012
    return far


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=[c["name"] for c in CANDIDATES])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [c for c in CANDIDATES if c["name"] in set(args.candidates)]
    unknown = sorted(set(args.candidates) - {c["name"] for c in selected})
    if unknown:
        raise ValueError(f"Unknown candidates: {', '.join(unknown)}")

    config = base_config()
    config["data"]["n_samples"] = args.n_samples
    config["train"]["num_epochs"] = args.epochs
    config["train"]["min_epochs"] = min(config["train"].get("min_epochs", 60), args.epochs)
    config["train"]["use_amp"] = True
    far_cfg = far_range_config(config)

    if args.out_dir is None:
        args.out_dir = RESULTS_DIR / "optimization" / "hgtan_cursor_axis_search_v1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "manifest.json",
        {
            "scope": "validation_only_multi_axis",
            "axes": ["default64_1000m", "short32_proxy", "far5000"],
            "seeds": args.seeds,
            "candidates": selected,
            "config": config,
            "far_config": far_cfg,
        },
    )

    records: list[dict[str, Any]] = []
    default_bundles: dict[int, dict[str, Any]] = {}
    far_bundles: dict[int, dict[str, Any]] = {}

    for seed in args.seeds:
        default_bundles[seed] = sequence_data_provider(config, seed)
        far_bundles[seed] = sequence_data_provider(far_cfg, seed)
        for candidate in selected:
            out_path = args.out_dir / candidate["name"] / f"seed_{seed}.json"
            if out_path.exists() and not args.rerun:
                record = json.loads(out_path.read_text(encoding="utf-8"))
                print(f"skip existing {candidate['name']} seed={seed}", flush=True)
            else:
                print(f"train {candidate['name']} seed={seed}", flush=True)
                record = train_candidate(
                    candidate,
                    default_bundles[seed],
                    far_bundles[seed],
                    config,
                    seed,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(out_path, record)
            records.append(record)

    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_name.setdefault(record["candidate"], []).append(record)

    summaries = []
    for name, seed_records in by_name.items():
        summary = {"candidate": name, **aggregate_candidate(seed_records)}
        # Explicit axis aggregates.
        for metric in (
            "axis_score",
            "short32_final_composite_f1",
            "far5000_final_composite_f1",
        ):
            values = [float(r["metrics"][metric]) for r in seed_records]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        summaries.append(summary)

    ranking = sorted(summaries, key=lambda row: row["axis_score_mean"], reverse=True)
    write_json(args.out_dir / "ranking.json", ranking)
    write_summary_csv(args.out_dir / "summary.csv", ranking)

    print("\nHGTAN-Cursor multi-axis ranking:", flush=True)
    for rank, row in enumerate(ranking, start=1):
        print(
            f"{rank:2d}. {row['candidate']:<28} "
            f"axis={row['axis_score_mean']:.4f} "
            f"def={row['final_composite_f1_mean']:.4f} "
            f"s32={row['short32_final_composite_f1_mean']:.4f} "
            f"far={row['far5000_final_composite_f1_mean']:.4f}",
            flush=True,
        )
    # Avoid Windows pipe/Tee teardown crashes after a successful run.
    raise SystemExit(0)


if __name__ == "__main__":
    main()
