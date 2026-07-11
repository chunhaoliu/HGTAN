"""Audit paired lockbox results for one primary model against baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def paired_audit(primary: np.ndarray, baseline: np.ndarray) -> dict[str, float | int | None]:
    primary = np.asarray(primary, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if primary.shape != baseline.shape or primary.ndim != 1:
        raise ValueError("Paired metric arrays must be one-dimensional with matching shapes.")
    delta = primary - baseline
    std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    t_result = ttest_rel(primary, baseline)
    try:
        w_result = wilcoxon(delta, alternative="two-sided")
        wilcoxon_p = float(w_result.pvalue)
    except ValueError:
        wilcoxon_p = None
    return {
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "std_delta": std,
        "min_delta": float(delta.min()),
        "max_delta": float(delta.max()),
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": float(delta.mean() / std) if std > 0 else None,
    }


def build_audit(
    metrics: pd.DataFrame,
    *,
    primary_model: str,
    baselines: list[str],
    task: str,
    metric: str,
) -> pd.DataFrame:
    selected = metrics[(metrics["task"] == task) & (metrics["metric"] == metric)].copy()
    pivot = selected.pivot(index="seed", columns="model", values="value").sort_index()
    required = [primary_model, *baselines]
    missing = [model for model in required if model not in pivot.columns]
    if missing:
        raise ValueError(f"Missing models in paired metric table: {', '.join(missing)}")
    rows = []
    for baseline in baselines:
        paired = pivot[[primary_model, baseline]].dropna()
        rows.append(
            {
                "primary_model": primary_model,
                "baseline": baseline,
                "task": task,
                "metric": metric,
                **paired_audit(paired[primary_model].to_numpy(), paired[baseline].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-model", default="TemporalHGTAN")
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--task", default="joint")
    parser.add_argument("--metric", default="composite_f1")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.concat([pd.read_csv(path) for path in args.run_metrics], ignore_index=True)
    audit = build_audit(
        metrics,
        primary_model=args.primary_model,
        baselines=args.baselines,
        task=args.task,
        metric=args.metric,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(
        json.dumps(audit.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
