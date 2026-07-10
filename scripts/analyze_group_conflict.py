"""Audit whether group synergy helps on conflicting observed evidence.

The diagnostic re-creates each test split from its recorded seed, verifies that
the regenerated labels match the stored predictions, and then evaluates the
full and no-synergy HGTAN variants on a pre-defined high-conflict stratum.
No model is retrained and no test label is used to define the stratum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import sequence_data_provider  # noqa: E402
from exp.registry import apply_assessment_setting, make_setting_name  # noqa: E402
from exp.result_writer import read_json_gzip  # noqa: E402
from utils.config import (  # noqa: E402
    ALL_FEATURES,
    CONTINUOUS_FEATURE_INDICES,
    FEATURE_GROUPS,
    FEATURE_RISK_DIRECTION,
    HGTANConfig,
    set_random_seed,
)
from utils.metrics import build_classification_metrics  # noqa: E402


FULL_MODEL = "TemporalHGTAN"
NO_SYNERGY_MODEL = "TemporalHGTAN_NoSynergy"
DEFAULT_SETTING = "ATUAV-Core__latent_state_masked"
THREAT_WEIGHT = 0.75
URGENCY_WEIGHT = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path, help="Completed ablation experiment directory.")
    parser.add_argument("--setting", default=DEFAULT_SETTING, help="Setting directory to audit.")
    parser.add_argument("--quantile", type=float, default=0.75, help="High-conflict quantile threshold.")
    parser.add_argument("--tail-frames", type=int, default=8, help="Trailing observed frames used for group scores.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <experiment_root>/mechanism_audit.")
    return parser.parse_args()


def continuous_group_indices() -> dict[str, list[int]]:
    """Return non-oracle feature indices for each semantic indicator group."""
    allowed = set(CONTINUOUS_FEATURE_INDICES)
    return {
        group_name: [ALL_FEATURES.index(name) for name in payload["features"] if ALL_FEATURES.index(name) in allowed]
        for group_name, payload in FEATURE_GROUPS.items()
    }


def compute_group_conflict_scores(sequences: np.ndarray, *, tail_frames: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return per-track group risk scores and their max-minus-min conflict."""
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != len(ALL_FEATURES):
        raise ValueError(f"Expected sequences with shape (n, t, {len(ALL_FEATURES)}), got {values.shape}")
    if tail_frames <= 0:
        raise ValueError("tail_frames must be positive")

    tail = values[:, -min(tail_frames, values.shape[1]) :, :].copy()
    for feature_index, feature_name in enumerate(ALL_FEATURES):
        if FEATURE_RISK_DIRECTION[feature_name] == "low":
            tail[:, :, feature_index] = 1.0 - tail[:, :, feature_index]

    group_scores = np.column_stack(
        [tail[:, :, indices].mean(axis=(1, 2)) for indices in continuous_group_indices().values()]
    )
    conflict = group_scores.max(axis=1) - group_scores.min(axis=1)
    return group_scores, conflict


def build_strata(conflict: np.ndarray, quantile: float) -> tuple[float, dict[str, np.ndarray]]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    threshold = float(np.quantile(conflict, quantile))
    high = np.asarray(conflict) >= threshold
    return threshold, {"all": np.ones(len(conflict), dtype=bool), "high_conflict": high, "lower_conflict": ~high}


def weighted_composite_f1(threat_f1: float, urgency_f1: float) -> float:
    return THREAT_WEIGHT * float(threat_f1) + URGENCY_WEIGHT * float(urgency_f1)


def evaluate_prediction_strata(
    payload: dict,
    strata: dict[str, np.ndarray],
    *,
    model: str,
    seed: int,
    conflict_threshold: float,
) -> list[dict]:
    rows: list[dict] = []
    for stratum, mask in strata.items():
        threat_true = np.asarray(payload["threat_true"], dtype=np.int64)[mask]
        threat_pred = np.asarray(payload["threat_pred"], dtype=np.int64)[mask]
        urgency_true = np.asarray(payload["urgency_true"], dtype=np.int64)[mask]
        urgency_pred = np.asarray(payload["urgency_pred"], dtype=np.int64)[mask]
        threat = build_classification_metrics(threat_true, threat_pred, critical_labels_1based=[4, 5])
        urgency = build_classification_metrics(urgency_true, urgency_pred, critical_labels_1based=[3])
        rows.append(
            {
                "seed": seed,
                "model": model,
                "stratum": stratum,
                "support": int(mask.sum()),
                "conflict_threshold": conflict_threshold,
                "threat_f1": threat["f1"],
                "urgency_f1": urgency["f1"],
                "composite_f1": weighted_composite_f1(threat["f1"], urgency["f1"]),
                "threat_accuracy": threat["accuracy"],
                "urgency_accuracy": urgency["accuracy"],
                "threat_mae": float(np.mean(np.abs(threat_true - threat_pred))),
                "urgency_mae": float(np.mean(np.abs(urgency_true - urgency_pred))),
            }
        )
    return rows


def find_setting(manifest: dict, setting_name: str) -> dict:
    for setting in manifest["settings"]:
        if make_setting_name(setting) == setting_name:
            return setting
    raise KeyError(f"Setting {setting_name!r} is absent from run_manifest.json")


def validate_prediction_alignment(record: dict, data_bundle: dict, model_names: tuple[str, ...]) -> None:
    expected = {
        "threat_true": np.asarray(data_bundle["t_test"], dtype=np.int64),
        "urgency_true": np.asarray(data_bundle["u_test"], dtype=np.int64),
    }
    for model_name in model_names:
        if model_name not in record["predictions"]:
            raise KeyError(f"Checkpoint does not contain {model_name}")
        payload = record["predictions"][model_name]
        for key, labels in expected.items():
            if not np.array_equal(np.asarray(payload[key], dtype=np.int64), labels):
                raise ValueError(f"Regenerated {key} does not align with stored {model_name} predictions")


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "support",
        "conflict_threshold",
        "threat_f1",
        "urgency_f1",
        "composite_f1",
        "threat_accuracy",
        "urgency_accuracy",
        "threat_mae",
        "urgency_mae",
    ]
    summary = runs.groupby(["stratum", "model"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        name if not statistic else f"{name}_{statistic}"
        for name, statistic in summary.columns.to_flat_index()
    ]
    return summary


def paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["threat_f1", "urgency_f1", "composite_f1", "threat_accuracy", "urgency_accuracy"]
    wide = runs.pivot(index=["seed", "stratum"], columns="model", values=metrics)
    rows = []
    for seed, stratum in wide.index:
        row = {"seed": seed, "stratum": stratum}
        for metric in metrics:
            row[f"delta_{metric}"] = wide.loc[(seed, stratum), (metric, FULL_MODEL)] - wide.loc[
                (seed, stratum), (metric, NO_SYNERGY_MODEL)
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_dir = (args.output_dir or experiment_root / "mechanism_audit").resolve()
    manifest = json.loads((experiment_root / "run_manifest.json").read_text(encoding="utf-8"))
    setting = find_setting(manifest, args.setting)
    setting_dir = experiment_root / args.setting
    checkpoints = sorted((setting_dir / "seed_checkpoints").glob("run_*_seed_*.json.gz"))
    if not checkpoints:
        raise FileNotFoundError(f"No completed seed checkpoints found under {setting_dir}")

    rows: list[dict] = []
    for checkpoint_path in checkpoints:
        record = read_json_gzip(checkpoint_path)
        seed = int(record["seed"])
        config = HGTANConfig.get_experiment_config("benchmark", mode=manifest["mode"])
        config = apply_assessment_setting(config, setting)
        set_random_seed(seed, config)
        data_bundle = sequence_data_provider(config, seed)
        validate_prediction_alignment(record, data_bundle, (FULL_MODEL, NO_SYNERGY_MODEL))

        _, conflict = compute_group_conflict_scores(data_bundle["X_test"], tail_frames=args.tail_frames)
        threshold, strata = build_strata(conflict, args.quantile)
        for model_name in (FULL_MODEL, NO_SYNERGY_MODEL):
            rows.extend(
                evaluate_prediction_strata(
                    record["predictions"][model_name],
                    strata,
                    model=model_name,
                    seed=seed,
                    conflict_threshold=threshold,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    summary = summarize_runs(runs)
    deltas = paired_deltas(runs)
    runs.to_csv(output_dir / "group_conflict_runs.csv", index=False)
    summary.to_csv(output_dir / "group_conflict_summary.csv", index=False)
    deltas.to_csv(output_dir / "group_conflict_paired_deltas.csv", index=False)

    print(summary.to_string(index=False))
    print("\nPaired full-minus-no-synergy deltas:")
    print(deltas.groupby("stratum").mean(numeric_only=True).to_string())
    print(f"\nSaved mechanism audit to: {output_dir}")


if __name__ == "__main__":
    main()
