"""Paired statistical audit for prefix-search candidate seed records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


DEFAULT_METRICS = [
    "final_composite_f1",
    "threat_temporal_macro_f1",
    "threat_temporal_accuracy",
    "threat_mean_abs_ordinal_error",
]


def load_candidate(root: Path, candidate: str) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    for path in sorted((root / candidate).glob("seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records[int(payload["seed"])] = {key: float(value) for key, value in payload["metrics"].items()}
    if not records:
        raise FileNotFoundError(f"No seed records found for {candidate!r} under {root}")
    return records


def bootstrap_interval(values: np.ndarray, *, seed: int = 20260712, repeats: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min((total - rank) * p_values[index], 1.0)
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def audit_candidates(
    root: Path,
    reference: str,
    comparators: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    reference_records = load_candidate(root, reference)
    rows = []
    for comparator in comparators:
        comparator_records = load_candidate(root, comparator)
        seeds = sorted(set(reference_records) & set(comparator_records))
        if len(seeds) < 2:
            raise ValueError(f"At least two paired seeds are required for {comparator}")
        for metric in metrics:
            reference_values = np.asarray([reference_records[seed][metric] for seed in seeds])
            comparator_values = np.asarray([comparator_records[seed][metric] for seed in seeds])
            direction = -1.0 if "error" in metric or "mae" in metric.lower() else 1.0
            paired_advantage = direction * (reference_values - comparator_values)
            ci_low, ci_high = bootstrap_interval(paired_advantage)
            t_result = ttest_rel(direction * reference_values, direction * comparator_values)
            try:
                wilcoxon_p = float(wilcoxon(paired_advantage).pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "reference": reference,
                    "comparator": comparator,
                    "metric": metric,
                    "num_seeds": len(seeds),
                    "seeds": seeds,
                    "reference_mean": float(reference_values.mean()),
                    "comparator_mean": float(comparator_values.mean()),
                    "advantage_mean": float(paired_advantage.mean()),
                    "advantage_ci95_low": ci_low,
                    "advantage_ci95_high": ci_high,
                    "wins": int(np.sum(paired_advantage > 0)),
                    "ties": int(np.sum(paired_advantage == 0)),
                    "paired_t_p": float(t_result.pvalue),
                    "wilcoxon_p": wilcoxon_p,
                }
            )

    for metric in metrics:
        metric_rows = [row for row in rows if row["metric"] == metric]
        adjusted_t = holm_adjust([row["paired_t_p"] for row in metric_rows])
        adjusted_w = holm_adjust([row["wilcoxon_p"] for row in metric_rows])
        for row, t_value, w_value in zip(metric_rows, adjusted_t, adjusted_w):
            row["paired_t_holm_p"] = t_value
            row["wilcoxon_holm_p"] = w_value
    return rows


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_rows = [{**row, "seeds": ";".join(map(str, row["seeds"]))} for row in rows]
    with (out_dir / "paired_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comparators", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit_candidates(args.root, args.reference, args.comparators, args.metrics)
    write_outputs(rows, args.out_dir)
    for row in rows:
        print(
            f"{row['metric']:<34} vs {row['comparator']:<28} "
            f"delta={100 * row['advantage_mean']:+.3f} pp "
            f"CI=[{100 * row['advantage_ci95_low']:+.3f}, {100 * row['advantage_ci95_high']:+.3f}] "
            f"t-Holm={row['paired_t_holm_p']:.4f} W-Holm={row['wilcoxon_holm_p']:.4f}"
        )


if __name__ == "__main__":
    main()
