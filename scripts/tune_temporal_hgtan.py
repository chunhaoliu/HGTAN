"""Validation-only hyperparameter search for Temporal HGTAN.

This script deliberately never evaluates a candidate on the test split. The
formal test suites are run only after a configuration has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_joint_train_kwargs, resolve_class_weights, sequence_data_provider  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import HGTANConfig, set_random_seed, to_serializable  # noqa: E402
from utils.metrics import evaluate_model, train_model  # noqa: E402
from utils.project_paths import RESULTS_DIR  # noqa: E402
from utils.sequence_metrics import compute_track_metrics, predict_prefix_labels  # noqa: E402


SEEDS = [42, 123, 456]
OBJECTIVE_WEIGHTS = {
    "final_composite_f1": 0.40,
    "threat_temporal_macro_f1": 0.20,
    "urgency_temporal_macro_f1": 0.10,
    "threat_temporal_accuracy": 0.15,
    "urgency_temporal_accuracy": 0.10,
    "mean_abs_ordinal_error": -0.05,
}


RECIPE_CANDIDATES: list[dict[str, Any]] = [
    {"name": "c5_reference"},
    {"name": "lr_2e4", "train": {"learning_rate": 2e-4}},
    {"name": "lr_4e4", "train": {"learning_rate": 4e-4}},
    {"name": "dropout_005", "model": {"dropout": 0.05}},
    {"name": "dropout_012", "model": {"dropout": 0.12}},
    {"name": "wd_2e4", "train": {"weight_decay": 2e-4}},
    {"name": "wd_1e3", "train": {"weight_decay": 1e-3}},
    {"name": "smooth_002", "train": {"label_smoothing": 0.02}},
    {"name": "prior_020", "model": {"prior_weight_alpha": 0.20}},
    {"name": "urgency_020", "train": {"urgency_weight": 0.20}},
    {"name": "urgency_030", "train": {"urgency_weight": 0.30}},
    {"name": "warmup_004", "train": {"warmup_epochs": 4}},
]

COMBINATION_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "d005_w1e3",
        "model": {"dropout": 0.05},
        "train": {"weight_decay": 1e-3},
    },
    {
        "name": "d004_w1e3",
        "model": {"dropout": 0.04},
        "train": {"weight_decay": 1e-3},
    },
    {
        "name": "d006_w1e3",
        "model": {"dropout": 0.06},
        "train": {"weight_decay": 1e-3},
    },
    {
        "name": "d005_w8e4",
        "model": {"dropout": 0.05},
        "train": {"weight_decay": 8e-4},
    },
    {
        "name": "d005_w12e4",
        "model": {"dropout": 0.05},
        "train": {"weight_decay": 1.2e-3},
    },
    {
        "name": "d005_w1e3_u030",
        "model": {"dropout": 0.05},
        "train": {"weight_decay": 1e-3, "urgency_weight": 0.30},
    },
    {
        "name": "d005_w1e3_s002",
        "model": {"dropout": 0.05},
        "train": {"weight_decay": 1e-3, "label_smoothing": 0.02},
    },
]

CANDIDATE_SETS = {
    "recipe": RECIPE_CANDIDATES,
    "combination": COMBINATION_CANDIDATES,
}


def selection_score(metrics: dict[str, float]) -> float:
    """Return the pre-specified validation objective."""
    average_ordinal_error = 0.5 * (
        metrics["threat_mean_abs_ordinal_error"]
        + metrics["urgency_mean_abs_ordinal_error"]
    )
    values = {
        **metrics,
        "mean_abs_ordinal_error": average_ordinal_error,
    }
    return float(sum(OBJECTIVE_WEIGHTS[key] * values[key] for key in OBJECTIVE_WEIGHTS))


def aggregate_candidate(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one candidate over seeds without hiding seed instability."""
    metric_names = [
        "selection_score",
        "final_composite_f1",
        "threat_final_f1",
        "urgency_final_f1",
        "threat_temporal_macro_f1",
        "urgency_temporal_macro_f1",
        "threat_temporal_accuracy",
        "urgency_temporal_accuracy",
        "threat_mean_abs_ordinal_error",
        "urgency_mean_abs_ordinal_error",
    ]
    summary: dict[str, Any] = {"num_seeds": len(seed_records)}
    for metric_name in metric_names:
        values = [float(record["metrics"][metric_name]) for record in seed_records]
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary[f"{metric_name}_min"] = min(values)
    return summary


def base_config() -> dict[str, Any]:
    """Reconstruct the current c5 manuscript recipe."""
    config = HGTANConfig.get_config("gpu")
    config["data"]["n_samples"] = 4000
    config["model"].update(
        {
            "embed_dim": 128,
            "num_heads": 4,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.08,
            "use_prior_weights": True,
            "prior_weight_alpha": 0.10,
        }
    )
    config["train"].update(
        {
            "batch_size": 256,
            "num_epochs": 100,
            "patience": 25,
            "min_epochs": 60,
            "learning_rate": 3e-4,
            "weight_decay": 5e-4,
            "urgency_weight": 0.25,
            "threat_weight": 0.75,
            "label_smoothing": 0.0,
            "use_mixup": False,
            "num_workers": 0,
            "persistent_workers": False,
            "use_amp": True,
        }
    )
    return config


def apply_candidate(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    for section in ("model", "train"):
        updated[section].update(candidate.get(section, {}))
    return updated


def validation_view(bundle: dict[str, Any]) -> dict[str, Any]:
    """Expose only train/validation artifacts to the tuning path."""
    allowed = (
        "train_loader",
        "val_loader",
        "X_val",
        "t_train_0",
        "u_train_0",
        "threat_seq_val",
        "urgency_seq_val",
    )
    return {key: bundle[key] for key in allowed}


def evaluate_validation(model, bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    threat, urgency, _, _ = evaluate_model(
        model,
        bundle["val_loader"],
        use_amp=config["train"].get("use_amp", False),
    )
    threat_seq_pred, urgency_seq_pred = predict_prefix_labels(
        model,
        bundle["X_val"],
        batch_size=config["train"]["batch_size"],
        use_amp=config["train"].get("use_amp", False),
    )
    threat_track = compute_track_metrics(
        bundle["threat_seq_val"],
        threat_seq_pred,
        critical_labels=[4, 5],
        frame_interval=config["sequence"]["frame_interval"],
    )
    urgency_track = compute_track_metrics(
        bundle["urgency_seq_val"],
        urgency_seq_pred,
        critical_labels=[3],
        frame_interval=config["sequence"]["frame_interval"],
    )
    metrics = {
        "final_composite_f1": 0.75 * float(threat["f1"]) + 0.25 * float(urgency["f1"]),
        "threat_final_f1": float(threat["f1"]),
        "urgency_final_f1": float(urgency["f1"]),
        "threat_temporal_macro_f1": float(threat_track["temporal_macro_f1"]),
        "urgency_temporal_macro_f1": float(urgency_track["temporal_macro_f1"]),
        "threat_temporal_accuracy": float(threat_track["temporal_accuracy"]),
        "urgency_temporal_accuracy": float(urgency_track["temporal_accuracy"]),
        "threat_mean_abs_ordinal_error": float(threat_track["mean_abs_ordinal_error"]),
        "urgency_mean_abs_ordinal_error": float(urgency_track["mean_abs_ordinal_error"]),
    }
    metrics["selection_score"] = selection_score(metrics)
    return metrics


def run_seed(candidate: dict[str, Any], seed: int, config: dict[str, Any]) -> dict[str, Any]:
    set_random_seed(seed, config)
    bundle = validation_view(sequence_data_provider(config, seed))
    class_weight_threat, class_weight_urgency = resolve_class_weights(
        config["train"], bundle["t_train_0"], bundle["u_train_0"]
    )
    train_kwargs = build_joint_train_kwargs(
        config["train"], class_weight_threat, class_weight_urgency
    )
    model = build_model("TemporalHGTAN", config["model"])
    started = time.perf_counter()
    trained = train_model(
        model,
        bundle["train_loader"],
        bundle["val_loader"],
        **train_kwargs,
    )
    elapsed = time.perf_counter() - started
    metrics = evaluate_validation(trained["model"], bundle, config)
    return {
        "candidate": candidate["name"],
        "seed": seed,
        "metrics": metrics,
        "training": {
            "elapsed_seconds": elapsed,
            **trained["training_info"],
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-set", choices=sorted(CANDIDATE_SETS), default="recipe")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = CANDIDATE_SETS[args.candidate_set][: args.max_candidates]
    if args.out_dir is None:
        args.out_dir = RESULTS_DIR / "optimization" / f"hgtan_{args.candidate_set}_search_v1"
    root_config = base_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "search_manifest.json",
        {
            "selection_scope": "validation_only",
            "objective_weights": OBJECTIVE_WEIGHTS,
            "seeds": args.seeds,
            "base_config": root_config,
            "candidates": candidates,
        },
    )

    summaries = []
    for candidate in candidates:
        config = apply_candidate(root_config, candidate)
        seed_records = []
        for seed in args.seeds:
            record_path = args.out_dir / candidate["name"] / f"seed_{seed}.json"
            if record_path.exists() and not args.rerun:
                seed_records.append(json.loads(record_path.read_text(encoding="utf-8")))
                print(f"Reused {candidate['name']} seed={seed}")
                continue
            print(f"Running {candidate['name']} seed={seed}")
            record = run_seed(candidate, seed, config)
            write_json(record_path, record)
            seed_records.append(record)
        aggregate = aggregate_candidate(seed_records)
        summaries.append(
            {
                "candidate": candidate["name"],
                **aggregate,
            }
        )
        write_json(args.out_dir / candidate["name"] / "summary.json", summaries[-1])
        write_summary_csv(args.out_dir / "candidate_summary.csv", summaries)

    ranked = sorted(
        summaries,
        key=lambda row: (
            row["selection_score_mean"],
            row["selection_score_min"],
            row["final_composite_f1_mean"],
        ),
        reverse=True,
    )
    write_json(args.out_dir / "ranking.json", ranked)
    print("\nValidation-only ranking:")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank:2d}. {row['candidate']:<16} "
            f"score={row['selection_score_mean']:.4f} "
            f"final={row['final_composite_f1_mean']:.4f} "
            f"threat-temporal={row['threat_temporal_macro_f1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
