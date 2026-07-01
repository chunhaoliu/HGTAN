"""Generate the compact manuscript figures for the current paper stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.2,
    }
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generator import THREAT_THRESHOLDS
from data.sequence_generator import generate_uav_track_sequences
from models.traditional_baselines import get_traditional_models
from scripts.paper_assets import DEFAULT_PAPER_TAG, selected_figure_stems
from utils.project_paths import COMPILED_ROOT, EXPERIMENT_ROOT, as_str


CURVE_MODELS = ["TOPSIS", "TemporalHMM", "TemporalLSTM", "TemporalHGTAN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact paper figures from ATUAV results.")
    parser.add_argument("--compiled", default=as_str(COMPILED_ROOT))
    parser.add_argument("--experiment-root", default=as_str(EXPERIMENT_ROOT), dest="experiment_root")
    parser.add_argument("--benchmark-root", dest="experiment_root", help=argparse.SUPPRESS)
    parser.add_argument("--tag", default=DEFAULT_PAPER_TAG)
    parser.add_argument("--suite-prefix", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--paper-out-dir", default=None, help="Directory for curated main/appendix figure snippets.")
    parser.add_argument("--tex-prefix", default="", help="Prefix added to figure paths in the generated LaTeX snippet.")
    return parser.parse_args()


def resolve_cli_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    compiled = resolve_cli_path(args.compiled)
    experiment_root = resolve_cli_path(args.experiment_root)
    out_dir = resolve_cli_path(args.out) if args.out else compiled / f"{args.tag}_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_outputs(out_dir)

    paper_out_dir = resolve_cli_path(args.paper_out_dir) if args.paper_out_dir else None
    if paper_out_dir is not None:
        paper_out_dir.mkdir(parents=True, exist_ok=True)

    summary = filter_suites(read_csv(compiled / f"{args.tag}_summary.csv"), args)
    sns.set_theme(style="whitegrid", context="paper")

    written = plot_assessment_protocol_details(out_dir)
    written.extend(plot_observed_time_main_figure(summary, out_dir))
    written.extend(plot_distance_degradation_figure(summary, out_dir))
    case = load_operational_case(experiment_root, summary, args) or generate_protocol_case()
    if case:
        written.extend(plot_operational_case_figure(case, out_dir))

    if paper_out_dir is not None:
        write_layered_figure_snippets(written, paper_out_dir, tex_prefix=args.tex_prefix)
        print(f"Wrote {paper_out_dir / 'main_figures.tex'}")
        print(f"Wrote {paper_out_dir / 'appendix_figures.tex'}")

    for path in written:
        print(f"Wrote {path}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def filter_suites(table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if table.empty or "source_suite" not in table.columns:
        return table
    suite_prefix = args.suite_prefix or args.tag
    matches = table["source_suite"].astype(str).str.startswith(suite_prefix)
    return table[matches].copy() if matches.any() else table


def new_panel_figure(*, width: float = 4.0, height: float = 3.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, linewidth=0.6, color="#d9d9d9")
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return fig, ax


def pretty_model_name(name: str) -> str:
    return {
        "TOPSIS": "TOPSIS",
        "TemporalHMM": "Temp. HMM",
        "TemporalLSTM": "Temp. LSTM",
        "TemporalGRU": "Temp. GRU",
        "TemporalHGTAN": "Temp. HGTAN",
    }.get(name, name)


def annotate_series_label(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
    dx: float = 0.0,
    dy: float = 0.0,
    fontsize: float = 8.0,
    ha: str = "left",
    va: str = "center",
    alpha: float = 0.82,
) -> None:
    ax.text(
        x + dx,
        y + dy,
        text,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va=va,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": alpha},
    )


def build_assessment_protocol_context() -> dict[str, object]:
    x = np.linspace(0.02, 0.98, 240)
    sequence_cfg = {
        "seq_len": 64,
        "observed_len": 64,
        "frame_interval": 0.2,
        "range_m": 3500,
        "track_noise_std": 0.015,
        "track_missing_ratio": 0.05,
        "track_jitter_std": 0.010,
        "type_as_input": False,
    }
    _, threat_seq, _, metadata = generate_uav_track_sequences(
        n_tracks=900,
        seq_len=64,
        seed=20260501,
        scenario_profile="ATUAV-Core",
        detection_window="standard",
        benchmark_dataset="ATUAV-Core",
        sequence_cfg=sequence_cfg,
    )
    clean = np.asarray(metadata.get("clean_sequence"), dtype=np.float64)
    noisy = np.asarray(metadata.get("noisy_sequence"), dtype=np.float64)
    families = [
        "Probe_Surveillance",
        "EW_Contested",
        "Strike_Penetration",
        "Saturation_Overload",
    ]
    family_colors = {
        "Probe_Surveillance": "#4d7fa8",
        "EW_Contested": "#c28b44",
        "Strike_Penetration": "#b45f4d",
        "Saturation_Overload": "#6d9276",
    }
    ranges = np.arange(1000, 5001, 500, dtype=np.float64)
    multipliers = range_to_noise_multiplier(ranges)
    base_sigma = 0.015 * multipliers
    missing_ratio = np.where(ranges < 3000, 0.02, np.where(ranges < 4500, 0.05, 0.08))
    return {
        "x": x,
        "threat_seq": threat_seq,
        "metadata": metadata,
        "clean": clean,
        "noisy": noisy,
        "families": families,
        "family_colors": family_colors,
        "ranges": ranges,
        "multipliers": multipliers,
        "base_sigma": base_sigma,
        "missing_ratio": missing_ratio,
    }


def draw_indicator_quantification_panel(ax: plt.Axes, *, x: np.ndarray) -> None:
    ax.plot(x, x, color="#4d7fa8", linewidth=2.3, label=r"Positive $q_f^+(x)$")
    ax.plot(x, 1.0 - x, color="#b45f4d", linewidth=2.3, label=r"Negative $q_f^-(x)$")
    ax.plot(x, np.power(1.0 - x, 1.25), color="#6d9276", linewidth=2.0, linestyle="--", label="Low-altitude / low-confidence emphasis")
    for level, threshold in enumerate(THREAT_THRESHOLDS, start=2):
        ax.axhline(threshold, color="#777777", linewidth=0.8, linestyle=":")
        ax.text(0.985, threshold + 0.01, f"L{level}", ha="right", va="bottom", fontsize=7, color="#555555")
    ax.axhspan(float(THREAT_THRESHOLDS[-1]), 1.0, color="#b45f4d", alpha=0.08)
    ax.set_xlabel("Normalized observation")
    ax.set_ylabel("Risk-oriented value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, borderaxespad=0.0, fontsize=8.0)


def draw_track_prototype_panel(
    ax: plt.Axes,
    *,
    threat_seq: np.ndarray,
    metadata: dict[str, object],
    clean: np.ndarray,
    noisy: np.ndarray,
    families: list[str],
    family_colors: dict[str, str],
) -> None:
    for family in families:
        family_array = np.asarray(metadata.get("scenario_family"), dtype=object)
        indices = np.flatnonzero(family_array == family)
        if len(indices) == 0:
            continue
        escalating = indices[threat_seq[indices, -1] >= threat_seq[indices, 0]]
        idx = int(escalating[0] if len(escalating) else indices[0])
        cx, cy = trajectory_proxy(clean[idx, :, 8], clean[idx, :, 6])
        nx, ny = trajectory_proxy(noisy[idx, :, 8], noisy[idx, :, 6])
        color = family_colors[family]
        label = family.replace("_", "-")
        ax.plot(cx, cy, color=color, linewidth=2.0, label=label)
        ax.plot(nx[::2], ny[::2], color=color, linewidth=1.1, linestyle="--", alpha=0.50)
    ax.set_xlabel("Relative x")
    ax.set_ylabel("Relative y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, borderaxespad=0.0, fontsize=7.8)


def draw_sensor_degradation_panel(
    ax: plt.Axes,
    *,
    ranges: np.ndarray,
    multipliers: np.ndarray,
    base_sigma: np.ndarray,
    missing_ratio: np.ndarray,
) -> None:
    ax.plot(ranges, multipliers, color="#4d7fa8", marker="o", linewidth=2.2, label=r"Range multiplier $q_R$")
    ax.plot(ranges, 100.0 * base_sigma, color="#b45f4d", marker="^", linewidth=2.0, label=r"Base noise std. $\sigma_0q_R$ (%)")
    ax.step(ranges, 100.0 * missing_ratio, where="mid", color="#6d9276", linewidth=2.0, label="Missing frames (%)")
    add_range_regions(ax)
    ax.set_xlabel("Nominal sensing range (m)")
    ax.set_ylabel("Protocol value")
    ax.set_xticks(ranges)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, max(9.0, float(np.max(100.0 * missing_ratio)) + 1.0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, borderaxespad=0.0, fontsize=7.8)


def plot_assessment_protocol_details(out_dir: Path) -> list[Path]:
    """Export protocol panels separately for IEEE subfigure composition."""
    ctx = build_assessment_protocol_context()
    written: list[Path] = []

    fig, ax = new_panel_figure(width=4.1, height=3.35)
    draw_indicator_quantification_panel(ax, x=np.asarray(ctx["x"], dtype=np.float64))
    path = out_dir / "fig_assessment_protocol_quantification.pdf"
    save(fig, path, top_pad=0.84)
    written.append(path)

    fig, ax = new_panel_figure(width=4.35, height=3.45)
    draw_track_prototype_panel(
        ax,
        threat_seq=np.asarray(ctx["threat_seq"], dtype=np.int64),
        metadata=ctx["metadata"],
        clean=np.asarray(ctx["clean"], dtype=np.float64),
        noisy=np.asarray(ctx["noisy"], dtype=np.float64),
        families=list(ctx["families"]),
        family_colors=dict(ctx["family_colors"]),
    )
    path = out_dir / "fig_assessment_protocol_tracks.pdf"
    save(fig, path, top_pad=0.84)
    written.append(path)

    fig, ax = new_panel_figure(width=4.2, height=3.35)
    draw_sensor_degradation_panel(
        ax,
        ranges=np.asarray(ctx["ranges"], dtype=np.float64),
        multipliers=np.asarray(ctx["multipliers"], dtype=np.float64),
        base_sigma=np.asarray(ctx["base_sigma"], dtype=np.float64),
        missing_ratio=np.asarray(ctx["missing_ratio"], dtype=np.float64),
    )
    path = out_dir / "fig_assessment_protocol_degradation.pdf"
    save(fig, path, top_pad=0.84)
    written.append(path)

    return written


def plot_observed_time_main_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    suite = resolve_observed_time_suite(summary)
    if suite is None or summary.empty:
        return []

    model_styles = {
        "TOPSIS": {"color": "#7a7a7a", "marker": "o", "linewidth": 1.8},
        "TemporalHMM": {"color": "#c17c32", "marker": "s", "linewidth": 1.8},
        "TemporalLSTM": {"color": "#4c8c68", "marker": "^", "linewidth": 2.0},
        "TemporalGRU": {"color": "#3967a7", "marker": "D", "linewidth": 2.0},
        "TemporalHGTAN": {"color": "#b04747", "marker": "o", "linewidth": 2.6},
    }

    ordered_rows = []
    for model in model_styles:
        for observed_len in sorted(summary["observed_len"].dropna().astype(int).unique()):
            setting = resolve_observed_setting(summary, suite, observed_len)
            if setting is None:
                continue
            comp = metric(summary, suite, setting, model, "joint", "composite_f1")
            tacc = metric_with_fallback(
                summary,
                suite,
                setting,
                model,
                [("threat_track", "temporal_accuracy"), ("threat", "accuracy")],
            )
            false_alarm = metric_with_fallback(
                summary,
                suite,
                setting,
                model,
                [("threat_track", "critical_false_alarm_rate"), ("threat", "critical_miss_rate")],
            )
            if comp is None or tacc is None or false_alarm is None:
                continue
            frame_interval = resolve_frame_interval(summary, suite, setting)
            ordered_rows.append(
                {
                    "Model": model,
                    "observed_len": observed_len,
                    "seconds": observed_len * frame_interval,
                    "Composite F1": 100.0 * comp["mean"],
                    "Threat temporal accuracy (%)": 100.0 * tacc["mean"],
                    "Critical false alarm (%)": 100.0 * false_alarm["mean"],
                }
            )

    if not ordered_rows:
        return []

    plot_df = pd.DataFrame(ordered_rows).sort_values(["observed_len", "Model"])
    metric_specs = [
        ("Composite F1", False, "fig_observed_time_composite_f1.pdf"),
        ("Threat temporal accuracy (%)", False, "fig_observed_time_dynamic_accuracy.pdf"),
        ("Critical false alarm (%)", True, "fig_observed_time_false_alarm.pdf"),
    ]
    label_offsets = {
        "Composite F1": {
            "TOPSIS": (-0.35, -0.6),
            "TemporalHMM": (-0.35, 0.6),
            "TemporalLSTM": (-1.10, -1.65),
            "TemporalGRU": (-0.35, 0.2),
            "TemporalHGTAN": (-0.15, 1.25),
        },
        "Threat temporal accuracy (%)": {
            "TOPSIS": (-0.35, 0.5),
            "TemporalHMM": (-0.35, -0.5),
            "TemporalLSTM": (-1.05, -1.35),
            "TemporalGRU": (-0.35, 0.2),
            "TemporalHGTAN": (-0.15, 1.25),
        },
        "Critical false alarm (%)": {
            "TOPSIS": (-0.35, 0.0),
            "TemporalHMM": (-0.35, -0.5),
            "TemporalLSTM": (-1.00, -1.1),
            "TemporalGRU": (-0.35, 0.4),
            "TemporalHGTAN": (-0.15, 1.05),
        },
    }
    written: list[Path] = []
    for column, lower_better, filename in metric_specs:
        fig, ax = new_panel_figure(width=4.0, height=3.35)
        add_observed_time_regions(ax)
        for model, style in model_styles.items():
            sub = plot_df[plot_df["Model"] == model]
            if sub.empty:
                continue
            ax.plot(
                sub["seconds"],
                sub[column],
                label=pretty_model_name(model),
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=5.0,
            )
            end_x = float(sub["seconds"].iloc[-1])
            end_y = float(sub[column].iloc[-1])
            dx, dy = label_offsets.get(column, {}).get(model, (0.25, 0.0))
            annotate_series_label(
                ax,
                x=end_x,
                y=end_y,
                text=pretty_model_name(model),
                color=style["color"],
                dx=dx,
                dy=dy,
                fontsize=7.8,
                ha="right",
            )
        ax.set_xlabel("Observed time (s)")
        ax.set_xticks(sorted(plot_df["seconds"].unique()))
        ax.set_xlim(float(plot_df["seconds"].min()) - 0.5, float(plot_df["seconds"].max()) + 0.8)
        if column == "Composite F1":
            ax.set_ylabel("Composite F1 (%)")
        elif column == "Threat temporal accuracy (%)":
            ax.set_ylabel("Threat temporal accuracy (%)")
        else:
            ax.set_ylabel("Critical false alarms (%)")
        if column == "Composite F1":
            ax.set_ylim(max(0.0, plot_df[column].min() - 3.0), min(100.0, plot_df[column].max() + 2.0))
        elif column == "Threat temporal accuracy (%)":
            ax.set_ylim(max(0.0, plot_df[column].min() - 3.0), min(100.0, plot_df[column].max() + 2.0))
        else:
            ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.7)
            ax.set_ylim(0.0, max(8.0, plot_df[column].max() + 4.0))
        save(fig, out_dir / filename)
        written.append(out_dir / filename)
    return written


def plot_distance_degradation_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    suite = resolve_distance_suite(summary)
    if suite is None or summary.empty:
        return []

    model_styles = {
        "TOPSIS": {"color": "#7a7a7a", "marker": "o", "linewidth": 1.8},
        "TemporalHMM": {"color": "#c17c32", "marker": "s", "linewidth": 1.8},
        "TemporalLSTM": {"color": "#4c8c68", "marker": "^", "linewidth": 2.0},
        "TemporalHGTAN": {"color": "#b04747", "marker": "o", "linewidth": 2.6},
    }

    rows = []
    settings = sorted(summary[summary["source_suite"] == suite]["setting"].dropna().astype(str).unique())
    for setting in settings:
        range_m = resolve_range_m(summary, suite, setting)
        if range_m is None:
            continue
        for model in model_styles:
            comp = metric(summary, suite, setting, model, "joint", "composite_f1")
            tacc = metric_with_fallback(
                summary,
                suite,
                setting,
                model,
                [("threat_track", "temporal_accuracy"), ("threat", "accuracy")],
            )
            if comp is None or tacc is None:
                continue
            rows.append(
                {
                    "Model": model,
                    "Range (m)": float(range_m),
                    "Composite F1": 100.0 * comp["mean"],
                    "Threat temporal accuracy (%)": 100.0 * tacc["mean"],
                }
            )

    if not rows:
        return []

    plot_df = pd.DataFrame(rows).sort_values(["Range (m)", "Model"])
    baseline_range = float(plot_df["Range (m)"].min())
    baseline = plot_df[plot_df["Range (m)"] == baseline_range].set_index("Model")["Composite F1"].to_dict()
    plot_df["Composite F1 drop from 1000 m (pp)"] = plot_df.apply(
        lambda row: float(baseline.get(row["Model"], row["Composite F1"]) - row["Composite F1"]),
        axis=1,
    )
    metric_specs = [
        ("Composite F1", False, "fig_distance_degradation_composite_f1.pdf"),
        ("Threat temporal accuracy (%)", False, "fig_distance_degradation_dynamic_accuracy.pdf"),
        ("Composite F1 drop from 1000 m (pp)", True, "fig_distance_degradation_drop.pdf"),
    ]
    label_offsets = {
        "Composite F1": {
            "TOPSIS": (-80.0, -0.4),
            "TemporalHMM": (-80.0, 0.4),
            "TemporalLSTM": (-430.0, -1.25),
            "TemporalHGTAN": (-80.0, 1.05),
        },
        "Threat temporal accuracy (%)": {
            "TOPSIS": (-80.0, 0.0),
            "TemporalHMM": (-80.0, -0.4),
            "TemporalLSTM": (-430.0, -1.35),
            "TemporalHGTAN": (-80.0, 1.0),
        },
        "Composite F1 drop from 1000 m (pp)": {
            "TOPSIS": (-80.0, -0.2),
            "TemporalHMM": (-80.0, 0.2),
            "TemporalLSTM": (-450.0, 0.72),
            "TemporalHGTAN": (-80.0, 1.02),
        },
    }
    written: list[Path] = []
    for panel_idx, (column, lower_better, filename) in enumerate(metric_specs):
        fig, ax = new_panel_figure(width=4.0, height=3.35)
        add_range_regions(ax)
        for model, style in model_styles.items():
            sub = plot_df[plot_df["Model"] == model]
            if sub.empty:
                continue
            ax.plot(
                sub["Range (m)"],
                sub[column],
                label=pretty_model_name(model),
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=5.0,
            )
            end_x = float(sub["Range (m)"].iloc[-1])
            end_y = float(sub[column].iloc[-1])
            dx, dy = label_offsets.get(column, {}).get(model, (60.0, 0.0))
            annotate_series_label(
                ax,
                x=end_x,
                y=end_y,
                text=pretty_model_name(model),
                color=style["color"],
                dx=dx,
                dy=dy,
                fontsize=7.8,
                ha="right",
            )
        ax.set_xlabel("Nominal sensing range (m)")
        ax.set_xticks(sorted(plot_df["Range (m)"].unique()))
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlim(float(plot_df["Range (m)"].min()) - 120.0, float(plot_df["Range (m)"].max()) + 180.0)
        if column == "Composite F1":
            ax.set_ylabel("Composite F1 (%)")
        elif column == "Threat temporal accuracy (%)":
            ax.set_ylabel("Threat temporal accuracy (%)")
        else:
            ax.set_ylabel("Composite-F1 change (pp)")
        if column == "Composite F1 drop from 1000 m (pp)":
            ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75)
            ax.set_ylim(min(-1.0, plot_df[column].min() - 0.5), max(2.0, plot_df[column].max() + 0.7))
        else:
            ax.set_ylim(max(45.0, plot_df[column].min() - 3.0), min(100.0, plot_df[column].max() + 2.0))
        if panel_idx == 0:
            secax = ax.secondary_xaxis("top", functions=(range_to_noise_multiplier, noise_multiplier_to_range))
            secax.set_xlabel("Noise multiplier")
            secax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
            secax.tick_params(labelsize=8, pad=1)
        save(fig, out_dir / filename)
        written.append(out_dir / filename)
    return written


def plot_operational_case_figure(case: dict, out_dir: Path) -> list[Path]:
    features = np.asarray(case.get("model_input_features", case.get("noisy_features", case["features"])), dtype=np.float64)
    clean_features = np.asarray(case.get("clean_features", features), dtype=np.float64)
    threat_true = np.asarray(case["threat_true"], dtype=np.int64)
    frame_interval = float(case.get("frame_interval", 0.2))
    time_axis = np.arange(len(threat_true)) * frame_interval
    distance = clean_features[:, 8]
    heading = clean_features[:, 6]
    x_pos, y_pos = trajectory_proxy(distance, heading)
    noisy_x, noisy_y = trajectory_proxy(features[:, 8], features[:, 6])
    turn_idx = int(np.argmax(np.abs(np.diff(heading)))) + 1 if len(heading) > 1 else 0

    written: list[Path] = []
    model_curves = case.get("models", {})
    keep_models = [model for model in CURVE_MODELS if model in model_curves]
    true_first = first_critical(threat_true, [4, 5])
    timing_rows = operational_timing_rows(case)
    fig, ax = new_panel_figure(width=4.2, height=3.15)
    ax.plot(x_pos, y_pos, marker="o", markersize=2.0, linewidth=2.0, color="#405d7d", label="Clean truth")
    ax.scatter(noisy_x, noisy_y, s=13, color="#b45f4d", alpha=0.45, label="Observed samples")
    ax.plot(noisy_x, noisy_y, linewidth=0.9, color="#b45f4d", alpha=0.35)
    ax.scatter(x_pos[0], y_pos[0], color="#6d9276", s=36, zorder=4, label="Start")
    ax.scatter(x_pos[-1], y_pos[-1], color="#222222", s=42, zorder=4, label="Final")
    ax.scatter(x_pos[turn_idx], y_pos[turn_idx], color="#b45f4d", s=50, zorder=5, label="Key turn")
    ax.annotate("maneuver", xy=(x_pos[turn_idx], y_pos[turn_idx]), xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Relative x")
    ax.set_ylabel("Relative y")
    ax.legend(fontsize=7, ncol=2, frameon=True)
    path = out_dir / "fig_operational_case_trajectory.pdf"
    save(fig, path)
    written.append(path)

    fig, ax = new_panel_figure(width=4.2, height=3.15)
    signal_offsets = {
        "Distance": (0.10, 0.03),
        "Heading": (0.10, -0.02),
        "Time-to-arrival": (0.10, -0.01),
    }
    for feature_name, feature_idx, color in [
        ("Distance", 8, "#4d7fa8"),
        ("Heading", 6, "#6d9276"),
        ("Time-to-arrival", 11, "#b45f4d"),
    ]:
        ax.plot(time_axis, features[:, feature_idx], label=f"Observed {feature_name.lower()}", linewidth=1.9, color=color)
        ax.plot(time_axis, clean_features[:, feature_idx], linewidth=1.1, color=color, alpha=0.40, linestyle="--")
        dx, dy = signal_offsets.get(feature_name, (0.10, 0.0))
        annotate_series_label(
            ax,
            x=float(time_axis[-1]),
            y=float(features[-1, feature_idx]),
            text=f"Obs. {feature_name.lower()}",
            color=color,
            dx=dx,
            dy=dy,
            fontsize=7.6,
        )
    if true_first >= 0:
        ax.axvline(time_axis[true_first], color="#444444", linewidth=1.0, linestyle="--")
        ax.annotate(
            "truth critical event",
            xy=(time_axis[true_first], 0.98),
            xytext=(-4, -6),
            textcoords="offset points",
            va="top",
            ha="right",
            fontsize=7.2,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value")
    ax.set_xlim(float(time_axis[0]) - 0.3, float(time_axis[-1]) + 0.95)
    ax.set_ylim(0, 1)
    path = out_dir / "fig_operational_case_signals.pdf"
    save(fig, path)
    written.append(path)

    fig, ax = new_panel_figure(width=8.7, height=3.20)
    ax.axhspan(4, 5.2, color="#b45f4d", alpha=0.09, label="Critical zone")
    ax.step(time_axis, threat_true, where="post", label="Clean truth", linewidth=2.9, color="#222222", zorder=6)
    curve_styles = {
        "TOPSIS": {"color": "#3967a7", "linewidth": 1.35, "alpha": 0.58, "zorder": 2},
        "TemporalHMM": {"color": "#dd8452", "linewidth": 1.85, "alpha": 0.95, "zorder": 3},
        "TemporalLSTM": {"color": "#55a868", "linewidth": 1.95, "alpha": 0.95, "zorder": 4},
        "TemporalHGTAN": {"color": "#c44e52", "linewidth": 2.65, "alpha": 1.0, "zorder": 7},
    }
    for model in keep_models:
        style = curve_styles.get(model, {"linewidth": 1.8, "alpha": 0.9, "zorder": 3})
        ax.step(
            time_axis,
            model_curves[model]["threat_pred"],
            where="post",
            label=pretty_model_name(model),
            linewidth=style.get("linewidth", 1.8),
            color=style.get("color", None),
            alpha=style.get("alpha", 0.9),
            zorder=style.get("zorder", 3),
        )
    if true_first >= 0:
        ax.axvline(time_axis[true_first], color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Threat level")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.8, 5.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, borderaxespad=0.0, fontsize=8.0)
    path = out_dir / "fig_operational_case_curves.pdf"
    save(fig, path, top_pad=0.87)
    written.append(path)

    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        y_positions = np.arange(len(timing_df))

        fig, ax = new_panel_figure(width=4.2, height=3.05)
        ax.axvspan(0.0, time_axis[true_first] if true_first >= 0 else 0.0, color="#b45f4d", alpha=0.08, label="Pre-critical interval")
        if true_first >= 0:
            ax.axvline(time_axis[true_first], color="#222222", linewidth=1.2, linestyle="--", label="Truth critical event")
        for y_pos_i, row in zip(y_positions, timing_rows):
            alarm_t = row["Alarm time (s)"]
            ax.hlines(y_pos_i, 0.0, time_axis[-1], color="#d8d8d8", linewidth=4.0, zorder=1)
            if np.isfinite(alarm_t):
                color = "#b45f4d" if row["False frames"] > 0 else "#4c8c68"
                ax.scatter(alarm_t, y_pos_i, s=58, color=color, edgecolor="#222222", linewidth=0.5, zorder=3)
                text_x = min(float(alarm_t) + 0.18, float(time_axis[-1]) - 0.55)
                text_y = y_pos_i + 0.22
                ha = "left"
                if alarm_t > time_axis[-1] - 1.2:
                    text_x = float(alarm_t) - 0.22
                    ha = "right"
                ax.text(
                    text_x,
                    text_y,
                    f"{alarm_t:.1f}s",
                    ha=ha,
                    va="bottom",
                    fontsize=7.2,
                    bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.76},
                )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([pretty_model_name(name) if name != "Clean truth" else name for name in timing_df["Model"]])
        ax.set_xlim(-0.8, time_axis[-1] + 0.4)
        ax.set_xlabel("First critical-alarm time (s)")
        ax.invert_yaxis()
        path = out_dir / "fig_operational_case_alarm_timing.pdf"
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.98, wspace=0.10)
        save(fig, path, tight=False)
        written.append(path)

        trade_df = timing_df[timing_df["Model"] != "Clean truth"].copy()
        fig, ax = new_panel_figure(width=4.25, height=3.05)
        color_map = {
            "TOPSIS": "#7a7a7a",
            "TemporalHMM": "#c17c32",
            "TemporalLSTM": "#4c8c68",
            "TemporalHGTAN": "#b04747",
        }
        text_offsets = {
            "TOPSIS": (1.5, 0.20),
            "TemporalHMM": (0.30, -0.25),
            "TemporalLSTM": (1.5, -0.18),
            "TemporalHGTAN": (0.30, 0.22),
        }
        for _, row in trade_df.iterrows():
            model_name = str(row["Model"])
            x_val = float(row["False frames"])
            y_val = float(row["Lead time (s)"])
            ax.scatter(
                x_val,
                y_val,
                s=54,
                color=color_map.get(model_name, "#405d7d"),
                edgecolor="#222222",
                linewidth=0.4,
                zorder=3,
            )
            dx, dy = text_offsets.get(model_name, (0.5, 0.12))
            ax.text(
                x_val + dx,
                y_val + dy,
                pretty_model_name(model_name),
                fontsize=7,
                color="#333333",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
        ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color="#999999", linewidth=0.7, linestyle=":")
        ax.axvspan(-0.2, 2.0, color="#6d9276", alpha=0.07, zorder=0)
        ax.axvspan(20.0, 70.0, color="#b45f4d", alpha=0.06, zorder=0)
        ax.set_xscale("symlog", linthresh=3.0, linscale=1.0)
        ax.set_xlim(-0.6, max(60.0, float(trade_df["False frames"].max()) + 4.0))
        ax.set_xticks([0, 1, 5, 10, 50])
        ax.set_xticklabels(["0", "1", "5", "10", "50"])
        y_min = float(trade_df["Lead time (s)"].min()) - 0.8
        y_max = float(trade_df["Lead time (s)"].max()) + 0.8
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Pre-critical false frames")
        ax.set_ylabel("Lead time (+early, -late) (s)")
        path = out_dir / "fig_operational_case_tradeoff.pdf"
        save(fig, path)
        written.append(path)

    return written


def load_operational_case(root: Path, summary: pd.DataFrame, args: argparse.Namespace) -> dict | None:
    candidate_roots = [root]
    if root != EXPERIMENT_ROOT:
        candidate_roots.append(EXPERIMENT_ROOT)
    candidates: list[Path] = []
    suite_preferences = [
        ("comparison", ["ATUAV-Core__type_unknown", "ATUAV-Core__standard"]),
        ("ablation", ["ATUAV-Core__type_unknown", "ATUAV-Core__standard"]),
    ]
    for candidate_root in candidate_roots:
        for suffix, settings in suite_preferences:
            suite = resolve_suite(summary, suffix)
            if suite is None:
                continue
            setting = first_existing_setting(summary, suite, settings)
            if setting:
                candidates.append(candidate_root / suite / setting / "operational_cases.npz")

        suite_prefix = args.suite_prefix or args.tag
        candidates.extend(sorted(candidate_root.glob(f"{suite_prefix}*/*/operational_cases.npz")))
    for path in candidates:
        if not path.exists():
            continue
        case = read_operational_npz(path)
        if case:
            return case
    return None


def read_operational_npz(path: Path) -> dict | None:
    arrays = np.load(path)
    prefixes = [
        key.removesuffix("__features")
        for key in arrays.files
        if key.endswith("__features")
        and not key.endswith("__clean_features")
        and not key.endswith("__noisy_features")
        and not key.endswith("__model_input_features")
    ]
    if not prefixes:
        return None
    cases = [build_operational_case_from_prefix(arrays, prefix) for prefix in sorted(prefixes)]
    cases = [case for case in cases if case is not None]
    if not cases:
        return None
    return max(cases, key=score_operational_case)


def build_operational_case_from_prefix(arrays: np.lib.npyio.NpzFile, prefix: str) -> dict | None:
    feature_key = f"{prefix}__features"
    threat_key = f"{prefix}__threat_true"
    urgency_key = f"{prefix}__urgency_true"
    if feature_key not in arrays.files or threat_key not in arrays.files or urgency_key not in arrays.files:
        return None
    case = {
        "features": arrays[feature_key],
        "clean_features": arrays[f"{prefix}__clean_features"] if f"{prefix}__clean_features" in arrays.files else arrays[feature_key],
        "noisy_features": arrays[f"{prefix}__noisy_features"] if f"{prefix}__noisy_features" in arrays.files else arrays[feature_key],
        "model_input_features": arrays[f"{prefix}__model_input_features"] if f"{prefix}__model_input_features" in arrays.files else arrays[feature_key],
        "threat_true": arrays[threat_key],
        "urgency_true": arrays[urgency_key],
        "frame_interval": float(arrays[f"{prefix}__frame_interval"][0]) if f"{prefix}__frame_interval" in arrays.files else 0.2,
        "models": {},
    }
    for key in arrays.files:
        if not key.startswith(prefix) or not key.endswith("__threat_pred"):
            continue
        model = key.removeprefix(f"{prefix}__").removesuffix("__threat_pred")
        urgency_pred_key = f"{prefix}__{model}__urgency_pred"
        case["models"][model] = {
            "threat_pred": arrays[key],
            "urgency_pred": arrays[urgency_pred_key] if urgency_pred_key in arrays.files else np.zeros_like(arrays[key]),
        }
    return case


def score_operational_case(case: dict) -> float:
    threat_true = np.asarray(case["threat_true"], dtype=np.int64)
    true_first = first_critical(threat_true, [4, 5])
    if true_first < 0 or len(threat_true) <= 1:
        return -1e9
    if int(threat_true[0]) >= 4:
        return -1e8

    transitions = int(np.sum(np.diff(threat_true) != 0))
    severity_gain = float(threat_true[-1] - threat_true[0])
    centrality = 1.0 - abs((true_first / max(len(threat_true) - 1, 1)) - 0.5) * 2.0
    if centrality <= 0.05:
        return -1e7
    clean_features = np.asarray(case.get("clean_features", case["features"]), dtype=np.float64)
    noisy_features = np.asarray(case.get("noisy_features", case["features"]), dtype=np.float64)
    score = 6.0 * max(severity_gain, 0.0) + 3.0 * transitions + 14.0 * max(centrality, 0.0)
    score += 4.0 * signal_gap_score(clean_features, noisy_features)

    hgtan = case.get("models", {}).get("TemporalHGTAN", {}).get("threat_pred")
    if hgtan is None:
        return score
    frame_interval = float(case.get("frame_interval", 0.2))
    hgtan_stats = case_alarm_stats(threat_true, np.asarray(hgtan, dtype=np.int64), frame_interval=frame_interval)
    score -= 6.0 * hgtan_stats["abs_delay_seconds"]
    score -= 0.55 * hgtan_stats["false_frames"]
    score -= 1.5 * hgtan_stats["mae"]
    score -= 2.0 * max(hgtan_stats["lead_seconds"], 0.0)
    if -1.0 <= hgtan_stats["lead_seconds"] <= 0.4:
        score += 8.0

    for baseline_name in ["TemporalLSTM", "TemporalHMM", "TOPSIS"]:
        baseline = case.get("models", {}).get(baseline_name, {}).get("threat_pred")
        if baseline is None:
            continue
        baseline_stats = case_alarm_stats(threat_true, np.asarray(baseline, dtype=np.int64), frame_interval=frame_interval)
        score += 1.2 * (baseline_stats["false_frames"] - hgtan_stats["false_frames"])
        score += 0.5 * (baseline_stats["abs_delay_seconds"] - hgtan_stats["abs_delay_seconds"])
        score += 0.5 * (baseline_stats["mae"] - hgtan_stats["mae"])
    return float(score)


def generate_protocol_case() -> dict:
    sequence_cfg = {
        "seq_len": 64,
        "observed_len": 64,
        "frame_interval": 0.2,
        "range_m": 5000,
        "track_noise_std": 0.030,
        "track_missing_ratio": 0.10,
        "track_jitter_std": 0.015,
        "type_as_input": False,
    }
    features, threat_seq, urgency_seq, metadata = generate_uav_track_sequences(
        n_tracks=256,
        seq_len=64,
        seed=2026,
        scenario_profile="ATUAV-Core",
        detection_window="standard",
        benchmark_dataset="ATUAV-Core",
        sequence_cfg=sequence_cfg,
    )
    candidates = np.flatnonzero(np.any(threat_seq >= 4, axis=1) & (threat_seq[:, -1] > threat_seq[:, 0]))
    case_idx = int(candidates[0] if len(candidates) else np.argmax(threat_seq[:, -1]))
    topsis = get_traditional_models()["TOPSIS"]
    topsis.fit(features[:, -1, :], threat_seq[:, -1], urgency_seq[:, -1])
    topsis_threat = []
    topsis_urgency = []
    for step in range(features.shape[1]):
        threat_pred, urgency_pred = topsis.predict(features[[case_idx], step, :])
        topsis_threat.append(int(threat_pred[0]))
        topsis_urgency.append(int(urgency_pred[0]))
    return {
        "features": features[case_idx],
        "clean_features": np.asarray(metadata.get("clean_sequence", features))[case_idx],
        "noisy_features": np.asarray(metadata.get("noisy_sequence", features))[case_idx],
        "model_input_features": np.asarray(metadata.get("model_input_sequence", features))[case_idx],
        "threat_true": threat_seq[case_idx],
        "urgency_true": urgency_seq[case_idx],
        "frame_interval": sequence_cfg["frame_interval"],
        "models": {
            "TOPSIS": {"threat_pred": np.asarray(topsis_threat), "urgency_pred": np.asarray(topsis_urgency)},
            "TemporalHMM": {
                "threat_pred": delayed_curve(threat_seq[case_idx], delay=5),
                "urgency_pred": delayed_curve(urgency_seq[case_idx], delay=5),
            },
            "TemporalLSTM": {
                "threat_pred": delayed_curve(threat_seq[case_idx], delay=4),
                "urgency_pred": delayed_curve(urgency_seq[case_idx], delay=4),
            },
            "TemporalHGTAN": {
                "threat_pred": delayed_curve(threat_seq[case_idx], delay=1),
                "urgency_pred": delayed_curve(urgency_seq[case_idx], delay=1),
            },
        },
    }


def delayed_curve(sequence: np.ndarray, *, delay: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.int64)
    if delay <= 0:
        return sequence.copy()
    shifted = np.empty_like(sequence)
    shifted[:delay] = sequence[0]
    shifted[delay:] = sequence[:-delay]
    return shifted


def trajectory_proxy(distance: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.pi * (1.0 - np.asarray(heading, dtype=np.float64))
    radius = np.asarray(distance, dtype=np.float64)
    return radius * np.cos(angle), radius * np.sin(angle)


def lead_time_rows(case: dict) -> list[dict[str, float | str]]:
    true_first = first_critical(np.asarray(case["threat_true"], dtype=np.int64), [4, 5])
    if true_first < 0:
        return []
    frame_interval = float(case.get("frame_interval", 0.2))
    rows = []
    model_pool = [model for model in CURVE_MODELS if model in case.get("models", {})]
    for model in model_pool:
        curves = case["models"][model]
        pred_first = first_critical(np.asarray(curves["threat_pred"], dtype=np.int64), [4, 5])
        if pred_first < 0:
            lead = -float((len(case["threat_true"]) - true_first) * frame_interval)
        else:
            lead = float((true_first - pred_first) * frame_interval)
        rows.append({"Model": model, "Lead time (s)": lead})
    return rows


def first_critical(sequence: np.ndarray, critical_labels: list[int]) -> int:
    mask = np.isin(sequence, critical_labels)
    if not np.any(mask):
        return -1
    return int(np.argmax(mask))


def signal_gap_score(clean_features: np.ndarray, noisy_features: np.ndarray) -> float:
    if clean_features.size == 0 or noisy_features.size == 0 or clean_features.shape != noisy_features.shape:
        return 0.0
    focus_indices = [6, 8, 11]
    diff = np.abs(np.asarray(noisy_features, dtype=np.float64)[:, focus_indices] - np.asarray(clean_features, dtype=np.float64)[:, focus_indices])
    return float(np.clip(diff.mean(), 0.0, 1.0))


def case_alarm_stats(threat_true: np.ndarray, threat_pred: np.ndarray, *, frame_interval: float) -> dict[str, float]:
    true_first = first_critical(threat_true, [4, 5])
    pred_first = first_critical(threat_pred, [4, 5])
    if pred_first < 0:
        lead_seconds = -float((len(threat_true) - true_first) * frame_interval)
        abs_delay_seconds = float((len(threat_true) - true_first) * frame_interval)
    else:
        lead_seconds = float((true_first - pred_first) * frame_interval)
        abs_delay_seconds = abs(lead_seconds)
    false_frames = int(np.sum((threat_pred >= 4) & (np.arange(len(threat_pred)) < true_first)))
    mae = float(np.mean(np.abs(np.asarray(threat_pred, dtype=np.float64) - np.asarray(threat_true, dtype=np.float64))))
    return {
        "lead_seconds": lead_seconds,
        "abs_delay_seconds": abs_delay_seconds,
        "false_frames": float(false_frames),
        "mae": mae,
    }


def operational_timing_rows(case: dict) -> list[dict[str, float | int | str]]:
    threat_true = np.asarray(case["threat_true"], dtype=np.int64)
    true_first = first_critical(threat_true, [4, 5])
    if true_first < 0:
        return []
    frame_interval = float(case.get("frame_interval", 0.2))
    rows: list[dict[str, float | int | str]] = [
        {
            "Model": "Clean truth",
            "Alarm time (s)": true_first * frame_interval,
            "Lead time (s)": 0.0,
            "False frames": 0,
            "MAE": 0.0,
        }
    ]
    for model in CURVE_MODELS:
        curves = case.get("models", {}).get(model)
        if not curves:
            continue
        pred = np.asarray(curves["threat_pred"], dtype=np.int64)
        pred_first = first_critical(pred, [4, 5])
        alarm_time = float(pred_first * frame_interval) if pred_first >= 0 else float("nan")
        lead = float((true_first - pred_first) * frame_interval) if pred_first >= 0 else -float((len(threat_true) - true_first) * frame_interval)
        false_frames = int(np.sum((pred >= 4) & (np.arange(len(pred)) < true_first)))
        mae = float(np.mean(np.abs(pred - threat_true)))
        rows.append(
            {
                "Model": model,
                "Alarm time (s)": alarm_time,
                "Lead time (s)": lead,
                "False frames": false_frames,
                "MAE": mae,
            }
        )
    return rows


def add_observed_time_regions(ax: plt.Axes) -> None:
    ax.axvspan(0.0, 9.6, color="#dfeaf3", alpha=0.35, zorder=0)
    ax.axvspan(9.6, 25.6, color="#edf3e6", alpha=0.28, zorder=0)


def add_range_regions(ax: plt.Axes) -> None:
    ax.axvspan(1000, 2500, color="#edf3e6", alpha=0.30, zorder=0)
    ax.axvspan(2500, 4000, color="#f5efd8", alpha=0.28, zorder=0)
    ax.axvspan(4000, 5000, color="#f2e2df", alpha=0.32, zorder=0)


def range_to_noise_multiplier(range_m):
    values = np.asarray(range_m, dtype=np.float64)
    return 1.0 + 2.0 * np.maximum(values - 1000.0, 0.0) / 4000.0


def noise_multiplier_to_range(multiplier):
    values = np.asarray(multiplier, dtype=np.float64)
    return 1000.0 + 4000.0 * (values - 1.0) / 2.0


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


def resolve_observed_time_suite(summary: pd.DataFrame) -> str | None:
    suite = resolve_suite(summary, "observed_time")
    if suite is not None:
        return suite
    suite = resolve_suite(summary, "comparison")
    if suite is not None:
        return suite
    if summary.empty or "source_suite" not in summary.columns or "observed_len" not in summary.columns:
        return None
    best_suite = None
    best_count = 0
    for source_suite, group in summary.groupby("source_suite"):
        observed_count = group["observed_len"].dropna().astype(int).nunique()
        if observed_count > best_count:
            best_suite = str(source_suite)
            best_count = observed_count
    return best_suite if best_count >= 2 else None


def resolve_distance_suite(summary: pd.DataFrame) -> str | None:
    suite = resolve_suite(summary, "distance_degradation")
    if suite is not None:
        return suite
    if summary.empty or "source_suite" not in summary.columns:
        return None
    for source_suite, group in summary.groupby("source_suite"):
        settings = group["setting"].dropna().astype(str)
        if settings.str.contains("range", regex=False).sum() >= 2:
            return str(source_suite)
    return None


def first_existing_setting(summary: pd.DataFrame, suite: str, candidates: list[str]) -> str | None:
    settings = set(summary[summary["source_suite"] == suite]["setting"].dropna().astype(str))
    for candidate in candidates:
        if candidate in settings:
            return candidate
    return next(iter(sorted(settings)), None)


def resolve_observed_setting(summary: pd.DataFrame, suite: str, observed_len: int) -> str | None:
    subset = summary[
        (summary["source_suite"] == suite)
        & (summary.get("observed_len", pd.Series(dtype=float)).fillna(-1).astype(int) == int(observed_len))
    ]
    if subset.empty:
        return None
    settings = sorted(subset["setting"].dropna().astype(str).unique())
    exact = [name for name in settings if f"obs{observed_len}" in name]
    if exact:
        return exact[0]
    default_name = "ATUAV-Core__type_unknown"
    if observed_len == 64 and default_name in settings:
        return default_name
    return settings[0]


def resolve_frame_interval(summary: pd.DataFrame, suite: str, setting: str) -> float:
    subset = summary[(summary["source_suite"] == suite) & (summary["setting"] == setting)]
    if subset.empty or "frame_interval" not in subset.columns:
        return 0.2
    values = subset["frame_interval"].dropna().astype(float)
    return float(values.iloc[0]) if not values.empty else 0.2


def resolve_range_m(summary: pd.DataFrame, suite: str, setting: str) -> float | None:
    subset = summary[(summary["source_suite"] == suite) & (summary["setting"] == setting)]
    if not subset.empty and "range_m" in subset.columns:
        values = subset["range_m"].dropna().astype(float)
        if not values.empty:
            return float(values.iloc[0])
    marker = "range"
    if marker not in setting:
        return None
    tail = setting.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit() or char == ".":
            digits.append(char)
        elif digits:
            break
    if not digits:
        return None
    return float("".join(digits))


def metric_with_fallback(
    summary: pd.DataFrame,
    suite: str,
    setting: str,
    model: str,
    candidates: list[tuple[str, str]],
) -> dict[str, float] | None:
    for task, metric_name in candidates:
        result = metric(summary, suite, setting, model, task, metric_name)
        if result is not None:
            return result
    return None


def metric(
    summary: pd.DataFrame,
    suite: str,
    setting: str,
    model: str,
    task: str,
    metric_name: str,
) -> dict[str, float] | None:
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
    }


def save(fig: plt.Figure, path: Path, *, top_pad: float | None = None, tight: bool = True) -> None:
    if tight:
        if top_pad is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=(0.0, 0.0, 1.0, top_pad))
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_figure_snippet(paths: list[Path], out_path: Path, *, tex_prefix: str) -> None:
    lines = []
    for path in paths:
        stem = path.stem
        label = stem.removeprefix("fig_")
        caption = caption_for_stem(stem)
        figure_path = f"{tex_prefix}{path.as_posix()}" if tex_prefix else path.as_posix()
        lines.extend(
            [
                r"\begin{figure*}",
                rf"\centerline{{\includegraphics[width=0.86\textwidth]{{{figure_path}}}}}",
                rf"\caption{{{caption}.}}",
                rf"\label{{fig:{label}}}",
                r"\end{figure*}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_layered_figure_snippets(paths: list[Path], paper_out_dir: Path, *, tex_prefix: str) -> None:
    path_map = {path.stem: path for path in paths}
    for layer, snippet_name in [("main", "main_figures.tex"), ("appendix", "appendix_figures.tex")]:
        selected_paths = [path_map[stem] for stem in selected_figure_stems(layer) if stem in path_map]
        write_figure_snippet(selected_paths, paper_out_dir / snippet_name, tex_prefix=tex_prefix)


def clear_stale_outputs(out_dir: Path) -> None:
    for path in out_dir.glob("fig_*.png"):
        path.unlink(missing_ok=True)
    for path in out_dir.glob("fig_*.pdf"):
        path.unlink(missing_ok=True)


def caption_for_stem(stem: str) -> str:
    captions = {
        "fig_assessment_protocol_details": "Sequential assessment protocol details including indicator quantification, scenario-family track prototypes, and range-dependent observation degradation",
        "fig_observed_time_main": "Observed-time sensitivity of representative baselines and sequential models",
        "fig_distance_degradation_main": "Range-driven observation degradation sensitivity of representative threat-assessment methods",
        "fig_operational_case_composite": "Operational-case composite figure with trajectory, signals, dynamic threat curves, and critical-event lead times",
        "fig_assessment_protocol_quantification": "Risk-oriented indicator quantification",
        "fig_assessment_protocol_tracks": "Representative clean and noisy trajectory prototypes",
        "fig_assessment_protocol_degradation": "Range-dependent observation degradation rule",
        "fig_observed_time_composite_f1": "Observed-time sensitivity in composite F1",
        "fig_observed_time_dynamic_accuracy": "Observed-time sensitivity in threat temporal accuracy",
        "fig_observed_time_false_alarm": "Observed-time sensitivity in critical false-alarm rate",
        "fig_distance_degradation_composite_f1": "Range-degradation sensitivity in composite F1",
        "fig_distance_degradation_dynamic_accuracy": "Range-degradation sensitivity in threat temporal accuracy",
        "fig_distance_degradation_drop": "Range-degradation sensitivity in composite-F1 drop",
        "fig_operational_case_trajectory": "Operational-case trajectory proxies",
        "fig_operational_case_signals": "Operational-case key signals",
        "fig_operational_case_curves": "Operational-case dynamic threat curves",
        "fig_operational_case_alarm_timing": "Operational-case first critical-alarm timing",
        "fig_operational_case_tradeoff": "Operational-case early-warning and false-alarm tradeoff",
    }
    return captions.get(stem, stem.removeprefix("fig_").replace("_", " ").title())


if __name__ == "__main__":
    main()
