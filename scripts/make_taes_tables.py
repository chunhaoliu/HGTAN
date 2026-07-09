"""Generate compact manuscript tables for the current two-experiment paper stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_assets import DEFAULT_PAPER_TAG, selected_table_keys
from utils.project_paths import COMPILED_ROOT, as_str


COMPARISON_MODELS = [
    "TOPSIS",
    "GRA",
    "Fuzzy",
    "Entropy-TOPSIS",
    "Combined-TOPSIS",
    "TemporalHMM",
    "LastFrameMLP",
    "MeanPoolMLP",
    "FlatSequenceMLP",
    "TemporalGRU",
    "TemporalLSTM",
    "TemporalTransformer",
    "TemporalTCN",
    "TemporalHGTAN",
]
COMPARISON_LABELS = {
    "LastFrameMLP": "Last-frame MLP",
    "MeanPoolMLP": "Mean-pooling MLP",
    "FlatSequenceMLP": "Flat-sequence MLP",
    "TemporalGRU": "Temporal GRU",
    "TemporalLSTM": "Temporal LSTM",
    "TemporalTransformer": "Temporal Transformer",
    "TemporalTCN": "Temporal TCN",
    "TemporalHGTAN": "Temporal HGTAN",
}
ABLATION_MODELS = [
    "TemporalHGTAN",
    "TemporalHGTAN_LastFrame",
    "TemporalHGTAN_MeanPool",
    "TemporalHGTAN_NoSynergy",
    "TemporalHGTAN_NoPrior",
]
ABLATION_LABELS = {
    "TemporalHGTAN": "Temporal HGTAN",
    "TemporalHGTAN_LastFrame": "Last-frame variant",
    "TemporalHGTAN_MeanPool": "Mean-pooling variant",
    "TemporalHGTAN_NoSynergy": "No-synergy variant",
    "TemporalHGTAN_NoPrior": "No-prior variant",
}
TABLE_BUILDERS = [
    ("comparison", "make_comparison_table"),
    ("ablation", "make_ablation_table"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact LaTeX tables for the TAES manuscript.")
    parser.add_argument("--compiled", default=as_str(COMPILED_ROOT), help="Compiled result directory.")
    parser.add_argument("--tag", default=DEFAULT_PAPER_TAG, help="Compiled CSV prefix.")
    parser.add_argument("--suite-prefix", default=None, help="Keep source_suite values that start with this prefix.")
    parser.add_argument("--out", default=None, help="Optional output path for the legacy all-table LaTeX file.")
    parser.add_argument("--paper-out-dir", default=None, help="Directory for curated main/appendix table snippets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compiled = Path(args.compiled)
    summary = filter_suites(read_compiled(compiled, args.tag, "summary"), args)

    tables = build_tables(summary)
    paper_out_dir = Path(args.paper_out_dir) if args.paper_out_dir else None
    if paper_out_dir is not None:
        paper_out_dir.mkdir(parents=True, exist_ok=True)
        write_table_layers(tables, paper_out_dir)
        print(f"Wrote {paper_out_dir / 'main_tables.tex'}")
        print(f"Wrote {paper_out_dir / 'appendix_tables.tex'}")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(join_tables(tables.values()), encoding="utf-8")
        print(f"Wrote {out_path}")


def read_compiled(compiled: Path, tag: str, artifact: str) -> pd.DataFrame:
    path = compiled / f"{tag}_{artifact}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def filter_suites(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if table.empty or "source_suite" not in table.columns:
        return table
    suite_prefix = args.suite_prefix or args.tag
    matches = table["source_suite"].astype(str).str.startswith(suite_prefix)
    return table[matches].copy() if matches.any() else table


def make_comparison_table(summary: pd.DataFrame) -> str | None:
    suite = resolve_suite(summary, "comparison")
    if suite is None:
        return None
    setting = first_existing_setting(summary, suite, ["ATUAV-Core__type_unknown", "ATUAV-Core__standard"])
    if setting is None:
        return None

    raw_rows = []
    for model in COMPARISON_MODELS:
        if not has_model(summary, suite, setting, model):
            continue
        raw_rows.append(
            {
                "model": model,
                "label": latex_escape(COMPARISON_LABELS.get(model, model)),
                "threat_f1": metric(summary, suite, setting, model, "threat", "f1"),
                "urgency_f1": metric(summary, suite, setting, model, "urgency", "f1"),
                "composite_f1": metric(summary, suite, setting, model, "joint", "composite_f1"),
                "temporal_accuracy": metric(summary, suite, setting, model, "threat_track", "temporal_accuracy"),
                "critical_miss": metric(summary, suite, setting, model, "threat_track", "critical_track_miss_rate"),
                "latency": metric(summary, suite, setting, model, "efficiency", "inference_time_ms_per_sample"),
            }
        )
    if not raw_rows:
        return None
    highlighted = {
        key: highlight_flags(raw_rows, key, larger_is_better=True)
        for key in ["threat_f1", "urgency_f1", "composite_f1", "temporal_accuracy"]
    }
    rows = []
    for row in raw_rows:
        model = row["model"]
        rows.append(
            [
                row["label"],
                fmt(row["threat_f1"], highlight=highlighted["threat_f1"].get(model)),
                fmt(row["urgency_f1"], highlight=highlighted["urgency_f1"].get(model)),
                fmt(row["composite_f1"], highlight=highlighted["composite_f1"].get(model)),
                fmt(row["temporal_accuracy"], highlight=highlighted["temporal_accuracy"].get(model)),
                fmt(row["critical_miss"]),
                fmt_number(row["latency"]),
            ]
        )
    return latex_table(
        caption=(
            "Comparison experiment on the default sequential protocol without oracle target-type input. "
            "Bold and underline mark the best and second-best values among reported models for the four main score columns."
        ),
        label="tab:comparison_experiment",
        columns=["Model", "Threat F1", "Urgency F1", "Comp. F1", "Threat T-Acc.", "Crit. Miss", "ms/sample"],
        rows=rows,
    )


def make_ablation_table(summary: pd.DataFrame) -> str | None:
    suite = resolve_suite(summary, "ablation")
    if suite is None:
        return None
    default_setting = first_existing_setting(summary, suite, ["ATUAV-Core__type_unknown", "ATUAV-Core__standard"])
    short_setting = first_existing_setting(summary, suite, ["ATUAV-Core__type_unknown__ablation_obs32"])
    far_setting = first_existing_setting(summary, suite, ["ATUAV-Core__type_unknown__ablation_range5000"])
    if default_setting is None:
        return None

    raw_rows = []
    for model in ABLATION_MODELS:
        if not has_model(summary, suite, default_setting, model):
            continue
        raw_rows.append(
            {
                "model": model,
                "label": latex_escape(ABLATION_LABELS.get(model, model)),
                "default_f1": metric(summary, suite, default_setting, model, "joint", "composite_f1"),
                "short_f1": metric(summary, suite, short_setting, model, "joint", "composite_f1") if short_setting else None,
                "far_f1": metric(summary, suite, far_setting, model, "joint", "composite_f1") if far_setting else None,
                "critical_recall": metric(summary, suite, default_setting, model, "threat", "critical_recall"),
                "false_alarm": metric(summary, suite, default_setting, model, "threat_track", "critical_false_alarm_rate"),
            }
        )
    if len(raw_rows) <= 1:
        return None
    highlighted = {
        "default_f1": highlight_flags(raw_rows, "default_f1", larger_is_better=True),
        "short_f1": highlight_flags(raw_rows, "short_f1", larger_is_better=True),
        "far_f1": highlight_flags(raw_rows, "far_f1", larger_is_better=True),
        "critical_recall": highlight_flags(raw_rows, "critical_recall", larger_is_better=True),
    }
    rows = []
    for row in raw_rows:
        model = row["model"]
        rows.append(
            [
                row["label"],
                fmt(row["default_f1"], highlight=highlighted["default_f1"].get(model)),
                fmt(row["short_f1"], highlight=highlighted["short_f1"].get(model)) if row["short_f1"] else "--",
                fmt(row["far_f1"], highlight=highlighted["far_f1"].get(model)) if row["far_f1"] else "--",
                fmt(row["critical_recall"], highlight=highlighted["critical_recall"].get(model)),
                fmt(row["false_alarm"]),
            ]
        )
    return latex_table(
        caption=(
            "Targeted ablation experiment of Temporal HGTAN. "
            "The short 6.4 s column stresses temporal evidence fusion, "
            "the 5000 m column stresses reliability-gated prior fusion, "
            "and critical recall plus critical false alarms expose high-risk behavior."
        ),
        label="tab:ablation_experiment",
        columns=["Variant", "Default F1", "Short 6.4 s F1", "Far 5000 m F1", "Critical Recall", "False Alarm"],
        rows=rows,
    )


def resolve_suite(summary: pd.DataFrame, suffix: str) -> str | None:
    if summary.empty or "source_suite" not in summary.columns:
        return None
    sources = sorted(str(value) for value in summary["source_suite"].dropna().unique())
    for source in sources:
        if source.endswith(f"_{suffix}"):
            return source
    for source in sources:
        if suffix in source:
            return source
    return None


def first_existing_setting(summary: pd.DataFrame, suite: str, candidates: list[str]) -> str | None:
    settings = set(summary[summary["source_suite"] == suite]["setting"].dropna().astype(str))
    for candidate in candidates:
        if candidate in settings:
            return candidate
    return next(iter(sorted(settings)), None)


def has_model(summary: pd.DataFrame, suite: str, setting: str, model: str) -> bool:
    match = summary[
        (summary["source_suite"] == suite)
        & (summary["setting"] == setting)
        & (summary["model"] == model)
    ]
    return not match.empty


def metric(
    summary: pd.DataFrame,
    suite: str,
    setting: str,
    model: str,
    task: str,
    metric_name: str,
) -> dict[str, float] | None:
    if summary.empty:
        return None
    match = summary[
        (summary["source_suite"] == suite)
        & (summary["setting"] == setting)
        & (summary["model"] == model)
        & (summary["task"] == task)
        & (summary["metric"] == metric_name)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "mean": float(row["mean"]),
        "ci95": float(row["ci95"]) if "ci95" in row and pd.notna(row["ci95"]) else 0.0,
        "n": float(row["n"]) if "n" in row and pd.notna(row["n"]) else 1.0,
    }


def fmt(stat: dict[str, float] | None, *, highlight: str | None = None) -> str:
    if stat is None:
        return "--"
    mean = 100.0 * stat["mean"]
    ci95 = 100.0 * stat.get("ci95", 0.0)
    if stat.get("n", 1.0) > 1 and ci95 > 0:
        value = f"{mean:.1f}$\\pm${ci95:.1f}"
    else:
        value = f"{mean:.1f}"
    return apply_highlight(value, highlight)


def fmt_number(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "--"
    mean = stat["mean"]
    ci95 = stat.get("ci95", 0.0)
    if stat.get("n", 1.0) > 1 and ci95 > 0:
        return f"{mean:.3f}$\\pm${ci95:.3f}"
    return f"{mean:.3f}"


def highlight_flags(rows: list[dict[str, object]], key: str, *, larger_is_better: bool) -> dict[str, str]:
    scored = [
        (str(row["model"]), float(stat["mean"]))
        for row in rows
        if (stat := row.get(key)) is not None
    ]
    scored.sort(key=lambda item: item[1], reverse=larger_is_better)
    flags: dict[str, str] = {}
    if scored:
        flags[scored[0][0]] = "best"
    if len(scored) > 1:
        flags[scored[1][0]] = "second"
    return flags


def apply_highlight(value: str, highlight: str | None) -> str:
    if highlight == "best":
        return rf"\textbf{{{value}}}"
    if highlight == "second":
        return rf"\underline{{{value}}}"
    return value


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def latex_table(caption: str, label: str, columns: list[str], rows: list[list[str]]) -> str:
    spec = "l" + "c" * (len(columns) - 1)
    header = " & ".join(columns) + r" \\"
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\n".join(
        [
            r"\begin{table*}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\tablefont",
            rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}",
            r"\hline",
            header,
            r"\hline",
            body,
            r"\hline",
            r"\end{tabular*}",
            r"\end{table*}",
        ]
    )


def build_tables(summary: pd.DataFrame) -> dict[str, str]:
    return {
        key: table
        for key, table in (
            (key, globals()[builder_name](summary))
            for key, builder_name in TABLE_BUILDERS
        )
        if table
    }


def write_table_layers(tables: dict[str, str], paper_out_dir: Path) -> None:
    for layer, snippet_name in [("main", "main_tables.tex"), ("appendix", "appendix_tables.tex")]:
        selected = [tables[key] for key in selected_table_keys(layer) if key in tables]
        (paper_out_dir / snippet_name).write_text(join_tables(selected), encoding="utf-8")


def join_tables(tables) -> str:
    blocks = [table for table in tables if table]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


if __name__ == "__main__":
    main()
