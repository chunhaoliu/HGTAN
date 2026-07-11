"""Validation-only diagnostics for temporal information in the UAV protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_joint_train_kwargs, resolve_class_weights, sequence_data_provider  # noqa: E402
from data.data_loader import TrackSequenceDataset, build_loader_kwargs  # noqa: E402
from models import build_model  # noqa: E402
from scripts.tune_temporal_hgtan import (  # noqa: E402
    aggregate_candidate,
    base_config,
    evaluate_validation,
    write_json,
    write_summary_csv,
)
from utils.config import set_random_seed  # noqa: E402
from utils.metrics import train_model  # noqa: E402
from utils.project_paths import RESULTS_DIR  # noqa: E402


DEFAULT_MODELS = ["FlatSequenceMLP", "TemporalGRU", "TemporalHGTAN"]
DEFAULT_VARIANTS = ["observed", "final_repeat", "history_shuffle", "clean"]


def _loader(
    x: np.ndarray,
    threat: np.ndarray,
    urgency: np.ndarray,
    config: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TrackSequenceDataset(x, threat, urgency),
        **build_loader_kwargs(config, dataset_size=len(x), shuffle=shuffle),
    )


def _scale_clean(bundle: dict[str, Any], split: str) -> np.ndarray:
    clean = np.asarray(bundle[f"metadata_{split}"]["clean_sequence"], dtype=np.float32)
    observed_len = int(bundle["observed_len"])
    clean = clean[:, :observed_len, :]
    shape = clean.shape
    scaled = bundle["scaler"].transform(clean.reshape(-1, shape[-1])).reshape(shape)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _shuffle_history(x: np.ndarray, seed: int) -> np.ndarray:
    shuffled = np.asarray(x, dtype=np.float32).copy()
    if shuffled.shape[1] <= 2:
        return shuffled
    rng = np.random.default_rng(seed)
    for row in range(len(shuffled)):
        order = rng.permutation(shuffled.shape[1] - 1)
        shuffled[row, :-1, :] = shuffled[row, order, :]
    return shuffled


def _variant_arrays(
    bundle: dict[str, Any],
    variant: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [np.asarray(bundle[f"X_{split}"], dtype=np.float32) for split in ("train", "val", "test")]
    if variant == "observed":
        return tuple(arrays)  # type: ignore[return-value]
    if variant == "clean":
        return tuple(_scale_clean(bundle, split) for split in ("train", "val", "test"))  # type: ignore[return-value]
    if variant == "final_repeat":
        return tuple(np.repeat(x[:, -1:, :], x.shape[1], axis=1) for x in arrays)  # type: ignore[return-value]
    if variant == "history_shuffle":
        return tuple(_shuffle_history(x, seed + 1009 * idx) for idx, x in enumerate(arrays))  # type: ignore[return-value]
    raise ValueError(f"Unknown diagnostic variant: {variant}")


def diagnostic_bundle(
    source: dict[str, Any],
    config: dict[str, Any],
    variant: str,
    seed: int,
) -> dict[str, Any]:
    bundle = deepcopy(source)
    x_train, x_val, x_test = _variant_arrays(source, variant, seed)
    bundle.update({"X_train": x_train, "X_val": x_val, "X_test": x_test})
    bundle["train_loader"] = _loader(x_train, bundle["t_train_0"], bundle["u_train_0"], config, shuffle=True)
    bundle["val_loader"] = _loader(x_val, bundle["t_val_0"], bundle["u_val_0"], config, shuffle=False)
    bundle["test_loader"] = _loader(x_test, bundle["t_test_0"], bundle["u_test_0"], config, shuffle=False)
    return bundle


def run_one(
    model_name: str,
    variant: str,
    seed: int,
    source: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    set_random_seed(seed, config)
    bundle = diagnostic_bundle(source, config, variant, seed)
    class_weight_threat, class_weight_urgency = resolve_class_weights(
        config["train"], bundle["t_train_0"], bundle["u_train_0"]
    )
    train_kwargs = build_joint_train_kwargs(
        config["train"], class_weight_threat, class_weight_urgency
    )
    model = build_model(model_name, config["model"])
    started = time.perf_counter()
    trained = train_model(model, bundle["train_loader"], bundle["val_loader"], **train_kwargs)
    metrics = evaluate_validation(trained["model"], bundle, config)
    return {
        "candidate": f"{model_name}__{variant}",
        "model": model_name,
        "variant": variant,
        "seed": seed,
        "metrics": metrics,
        "training": {
            "elapsed_seconds": time.perf_counter() - started,
            **trained["training_info"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--policy-variant", default="balanced")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = base_config()
    config["data"]["n_samples"] = args.n_samples
    config["train"]["num_epochs"] = args.epochs
    config["train"]["min_epochs"] = min(config["train"]["min_epochs"], args.epochs)
    config["sequence"]["reference_policy_variant"] = args.policy_variant
    if args.out_dir is None:
        args.out_dir = RESULTS_DIR / "optimization" / "temporal_information_diagnostic_v1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "manifest.json",
        {
            "scope": "validation_only",
            "models": args.models,
            "variants": args.variants,
            "seeds": args.seeds,
            "config": config,
        },
    )

    records: list[dict[str, Any]] = []
    sources: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        sources[seed] = sequence_data_provider(config, seed)
        for model_name in args.models:
            for variant in args.variants:
                record_path = args.out_dir / f"{model_name}__{variant}" / f"seed_{seed}.json"
                if record_path.exists() and not args.rerun:
                    records.append(json.loads(record_path.read_text(encoding="utf-8")))
                    print(f"Reused {model_name} {variant} seed={seed}", flush=True)
                    continue
                print(f"Running {model_name} {variant} seed={seed}", flush=True)
                record = run_one(model_name, variant, seed, sources[seed], config)
                write_json(record_path, record)
                records.append(record)

    summaries = []
    for model_name in args.models:
        for variant in args.variants:
            selected = [r for r in records if r["model"] == model_name and r["variant"] == variant]
            if not selected:
                continue
            summaries.append(
                {
                    "candidate": f"{model_name}__{variant}",
                    "model": model_name,
                    "variant": variant,
                    **aggregate_candidate(selected),
                }
            )
    write_json(args.out_dir / "summary.json", summaries)
    write_summary_csv(args.out_dir / "summary.csv", summaries)
    print("\nTemporal-information diagnostic:")
    for row in summaries:
        print(
            f"{row['candidate']:<38} "
            f"final={row['final_composite_f1_mean']:.4f} "
            f"threat-temporal={row['threat_temporal_macro_f1_mean']:.4f} "
            f"score={row['selection_score_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
