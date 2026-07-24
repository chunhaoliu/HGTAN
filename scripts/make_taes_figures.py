"""Generate the compact manuscript figures for the current paper stage."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t as student_t

def configure_publication_style() -> None:
    """Set final-size typography for IEEE Transactions figure assets."""
    sns.set_theme(style="whitegrid", context="paper")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


configure_publication_style()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sequence_generator import generate_uav_track_sequences
from models.traditional_baselines import get_traditional_models
from scripts.audit_lockbox_comparison import build_audit
from scripts.paper_assets import DEFAULT_PAPER_TAG, selected_figure_stems
from utils.project_paths import COMPILED_ROOT, EXPERIMENT_ROOT, as_str


CURVE_MODELS = ["TOPSIS", "TemporalHMM", "TemporalGRU", "TemporalHGTAN"]
ROBUSTNESS_MODEL_STYLES = {
    "TemporalGRU": {
        "color": "#555555",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.05,
        "markersize": 3.7,
        "alpha": 0.90,
        "zorder": 3,
    },
    "TemporalLSTM": {
        "color": "#8a8a8a",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.00,
        "markersize": 3.8,
        "alpha": 0.90,
        "zorder": 2,
    },
    "TemporalHGTAN": {
        "color": "#9e1b1f",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 4.2,
        "alpha": 1.0,
        "zorder": 6,
    },
}
OBSERVATION_MODEL_STYLES = {
    "MeanPoolMLP": {
        "color": "#8a8a8a",
        "marker": "s",
        "linestyle": ":",
        "linewidth": 1.00,
        "markersize": 3.7,
        "alpha": 0.90,
        "zorder": 2,
    },
    "TemporalGRU": {
        "color": "#555555",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 1.05,
        "markersize": 3.8,
        "alpha": 0.90,
        "zorder": 3,
    },
    "TemporalTransformer": {
        "color": "#4f78a8",
        "marker": "D",
        "linestyle": "-.",
        "linewidth": 1.05,
        "markersize": 3.6,
        "alpha": 0.92,
        "zorder": 4,
    },
    "TemporalHGTAN": {
        "color": "#9e1b1f",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.45,
        "markersize": 4.2,
        "alpha": 1.0,
        "zorder": 6,
    },
}
OPERATIONAL_MODEL_STYLES = {
    "TOPSIS": {"color": "#b0b0b0", "linewidth": 0.95, "alpha": 0.95, "linestyle": (0, (1.2, 1.2)), "zorder": 2},
    "TemporalHMM": {"color": "#777777", "linewidth": 1.05, "alpha": 0.95, "linestyle": (0, (4.0, 2.0)), "zorder": 3},
    "TemporalGRU": {"color": "#4f4f4f", "linewidth": 1.10, "alpha": 0.95, "linestyle": "-.", "zorder": 4},
    "TemporalLSTM": {"color": "#8c8c8c", "linewidth": 1.05, "alpha": 0.95, "linestyle": ":", "zorder": 4},
    "TemporalHGTAN": {"color": "#9e1b1f", "linewidth": 1.55, "alpha": 1.0, "linestyle": "-", "zorder": 7},
}

TRADEOFF_MODEL_STYLES = {
    "MeanPoolMLP": {"label": "Mean MLP", "color": "#777777", "marker": "s"},
    "FlatSequenceMLP": {"label": "Flat MLP", "color": "#303030", "marker": "D"},
    "TemporalGRU": {"label": "GRU", "color": "#555555", "marker": "o"},
    "TemporalLSTM": {"label": "LSTM", "color": "#777777", "marker": "^"},
    "TemporalTransformer": {"label": "Transformer", "color": "#555555", "marker": "P"},
    "TemporalTCN": {"label": "TCN", "color": "#777777", "marker": "v"},
    "TemporalHGTAN": {"label": "HGTAN", "color": "#8f1d1d", "marker": "o"},
}

BASELINE_ABBREVIATIONS = {
    "FlatSequenceMLP": "Flat MLP",
    "TemporalGRU": "GRU",
    "TemporalLSTM": "LSTM",
    "TemporalTransformer": "Transformer",
    "TemporalTCN": "TCN",
}

DIAGNOSTIC_MODELS = [
    "MeanPoolMLP",
    "TemporalGRU",
    "TemporalLSTM",
    "TemporalTransformer",
    "TemporalHGTAN",
]
DIAGNOSTIC_MODEL_STYLES = {
    "MeanPoolMLP": {"label": "Mean MLP", "color": "#4DBBD5", "marker": "s", "linestyle": "--"},
    "TemporalGRU": {"label": "GRU", "color": "#3C5488", "marker": "o", "linestyle": "-."},
    "TemporalLSTM": {"label": "LSTM", "color": "#00A087", "marker": "^", "linestyle": ":"},
    "TemporalTransformer": {"label": "Transformer", "color": "#7E57C2", "marker": "D", "linestyle": "--"},
    "TemporalHGTAN": {"label": "HGTAN", "color": "#E64B35", "marker": "o", "linestyle": "-"},
}
MISSING_MODEL_STYLES = {
    "MeanPoolMLP": {
        "label": "Mean MLP", "color": "#56B4E9", "marker": "s", "linestyle": "--",
        "linewidth": 1.00, "zorder": 3,
    },
    "TemporalGRU": {
        "label": "GRU", "color": "#0072B2", "marker": "o", "linestyle": "-.",
        "linewidth": 1.00, "zorder": 3,
    },
    "TemporalLSTM": {
        "label": "LSTM", "color": "#009E73", "marker": "^", "linestyle": ":",
        "linewidth": 1.05, "zorder": 3,
    },
    "TemporalTransformer": {
        "label": "Transformer", "color": "#CC79A7", "marker": "D", "linestyle": "--",
        "linewidth": 1.00, "zorder": 3,
    },
    "TemporalTCN": {
        "label": "TCN", "color": "#E69F00", "marker": "v", "linestyle": "-.",
        "linewidth": 1.00, "zorder": 3,
    },
    "TemporalHGTAN": {
        "label": "HGTAN", "color": "#D55E00", "marker": "o", "linestyle": "-",
        "linewidth": 1.55, "zorder": 7,
    },
}


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
    run_metrics = filter_suites(read_csv(compiled / f"{args.tag}_run_metrics.csv"), args)
    written = plot_assessment_protocol_details(out_dir)
    written.extend(plot_overall_tradeoff_figure(summary, out_dir))
    written.extend(plot_stability_paired_figure(experiment_root, out_dir))
    comparison_records = load_formal_comparison_records(experiment_root)
    transition_case = None
    if comparison_records:
        written.extend(plot_classwise_final_f1_figures(comparison_records, out_dir))
        written.extend(plot_critical_transition_figures(comparison_records, out_dir))
        transition_case = select_representative_transition_case(comparison_records)
    written.extend(plot_policy_holdout_comparison_figures(summary, run_metrics, out_dir))
    written.extend(plot_observed_time_main_figure(summary, out_dir))
    written.extend(plot_distance_degradation_figure(summary, out_dir))
    written.extend(plot_missing_robustness_figures(summary, out_dir))
    written.extend(plot_ablation_absolute_figures(run_metrics, out_dir))
    case = load_operational_case(experiment_root, summary, args) or generate_protocol_case()
    if case:
        written.extend(plot_operational_case_figure(case, out_dir, timeline_case=transition_case))

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


def new_panel_figure(
    *, width: float = 4.0, height: float = 3.2, journal: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    if journal:
        ax.grid(False)
        ax.grid(axis="y", linewidth=0.42, color="#e6e6e6", alpha=0.82)
        ax.tick_params(labelsize=7.2, length=2.4, width=0.6, color="#777777")
        ax.xaxis.label.set_size(8.2)
        ax.yaxis.label.set_size(8.2)
    else:
        ax.grid(True, linewidth=0.45, color="#e2e2e2", alpha=0.85)
        ax.tick_params(labelsize=9)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.65 if journal else 0.8)
        if journal:
            ax.spines[side].set_color("#777777")
    return fig, ax


def style_panel_axis(ax: plt.Axes) -> None:
    ax.grid(True, linewidth=0.55, color="#d9d9d9")
    ax.tick_params(labelsize=7.4, length=2.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.75)
        ax.spines[side].set_color("#777777")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )


def pretty_model_name(name: str) -> str:
    return {
        "TOPSIS": "TOPSIS",
        "TemporalHMM": "Temp. HMM",
        "MeanPoolMLP": "Mean MLP",
        "TemporalLSTM": "Temp. LSTM",
        "TemporalGRU": "Temp. GRU",
        "TemporalTransformer": "Temp. Transformer",
        "TemporalHGTAN": "Temp. HGTAN",
    }.get(name, name)


def compact_model_name(name: str) -> str:
    return {
        "TOPSIS": "TOPSIS",
        "TemporalHMM": "HMM",
        "TemporalLSTM": "LSTM",
        "TemporalGRU": "GRU",
        "TemporalHGTAN": "HGTAN",
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
    box: bool = False,
) -> None:
    bbox = (
        {"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": alpha}
        if box
        else None
    )
    ax.text(
        x + dx,
        y + dy,
        text,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va=va,
        clip_on=False,
        bbox=bbox,
    )


def annotate_right_label_stack(
    ax: plt.Axes,
    entries: list[dict[str, object]],
    *,
    x_label: float,
    connector_gap: float,
    fontsize: float = 7.4,
    min_gap_frac: float = 0.070,
) -> None:
    if not entries:
        return

    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    low = ymin + 0.055 * span
    high = ymax - 0.055 * span
    min_gap = min_gap_frac * span

    ordered = sorted(entries, key=lambda item: float(item["y"]))
    y_targets = [min(max(float(item["y"]), low), high) for item in ordered]
    for idx in range(1, len(y_targets)):
        if y_targets[idx] - y_targets[idx - 1] < min_gap:
            y_targets[idx] = y_targets[idx - 1] + min_gap
    if y_targets[-1] > high:
        shift = y_targets[-1] - high
        y_targets = [value - shift for value in y_targets]
    if y_targets[0] < low:
        shift = low - y_targets[0]
        y_targets = [value + shift for value in y_targets]

    for item, y_text in zip(ordered, y_targets):
        x_end = float(item["x"])
        y_end = float(item["y"])
        color = str(item["color"])
        if abs(y_text - y_end) > 0.02 * span:
            ax.plot(
                [x_end, x_label - connector_gap],
                [y_end, y_text],
                color=color,
                linewidth=0.55,
                alpha=0.50,
                clip_on=False,
            )
        annotate_series_label(
            ax,
            x=x_label,
            y=y_text,
            text=str(item["text"]),
            color=color,
            fontsize=fontsize,
            ha="left",
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
        "mission_as_input": False,
        "reference_policy_variant": "balanced",
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


def plot_overall_tradeoff_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Contrast terminal classification with prefix-level dynamic fidelity."""
    suite = resolve_suite(summary, "comparison")
    if suite is None or summary.empty:
        return []
    setting = first_existing_setting(summary, suite, ["ATUAV-Core__latent_state_masked"])
    if setting is None:
        return []

    rows: list[dict[str, float | str]] = []
    for model, style in TRADEOFF_MODEL_STYLES.items():
        final_f1 = metric(summary, suite, setting, model, "joint", "composite_f1")
        temporal_f1 = metric(summary, suite, setting, model, "threat_track", "temporal_macro_f1")
        if final_f1 is None or temporal_f1 is None:
            continue
        rows.append(
            {
                "Model": model,
                "Label": style["label"],
                "Composite F1 (%)": 100.0 * final_f1["mean"],
                "Composite F1 SD": 100.0 * final_f1["std"],
                "Threat temporal macro-F1 (%)": 100.0 * temporal_f1["mean"],
                "Threat temporal macro-F1 SD": 100.0 * temporal_f1["std"],
                "Source suite": suite,
                "Setting": setting,
                "Seeds": 3,
            }
        )
    if not rows:
        return []

    source = pd.DataFrame(rows)
    source.to_csv(out_dir / "fig_overall_tradeoff_source.csv", index=False)
    row_order = [
        "TemporalHGTAN",
        "MeanPoolMLP",
        "FlatSequenceMLP",
        "TemporalGRU",
        "TemporalLSTM",
        "TemporalTransformer",
        "TemporalTCN",
    ]
    row_lookup = {str(row["Model"]): row for row in rows}
    ordered_rows = [row_lookup[model] for model in row_order if model in row_lookup]
    y_positions = {
        "TemporalHGTAN": 0.0,
        "MeanPoolMLP": 1.25,
        "FlatSequenceMLP": 2.05,
        "TemporalGRU": 3.30,
        "TemporalLSTM": 4.10,
        "TemporalTransformer": 4.90,
        "TemporalTCN": 5.70,
    }
    proposed = "#1F5AA6"
    final_color = "#4E5965"
    temporal_color = "#4FA3B5"
    connector = "#CCD2D8"

    fig, ax = new_panel_figure(width=3.36, height=2.15, journal=True)
    ax.grid(False)
    ax.set_axisbelow(True)
    for tick in [60, 70, 80, 90]:
        ax.axvline(tick, color="#E3E6E9", linewidth=0.48, zorder=0)
    ax.axhspan(-0.42, 0.42, color="#EFF5FA", zorder=-1)

    for row in ordered_rows:
        model = str(row["Model"])
        y = y_positions[model]
        final_f1 = float(row["Composite F1 (%)"])
        temporal_f1 = float(row["Threat temporal macro-F1 (%)"])
        is_hgtan = model == "TemporalHGTAN"
        line_color = proposed if is_hgtan else connector
        ax.plot(
            [temporal_f1, final_f1],
            [y, y],
            color=line_color,
            linewidth=2.0 if is_hgtan else 1.25,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            final_f1,
            y,
            marker="o",
            s=35 if is_hgtan else 25,
            facecolor=proposed if is_hgtan else final_color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax.scatter(
            temporal_f1,
            y,
            marker="s",
            s=32 if is_hgtan else 23,
            facecolor=proposed if is_hgtan else temporal_color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )

    ax.set_yticks([y_positions[str(row["Model"])] for row in ordered_rows])
    ax.set_yticklabels([str(row["Label"]) for row in ordered_rows])
    for label, row in zip(ax.get_yticklabels(), ordered_rows):
        if str(row["Model"]) == "TemporalHGTAN":
            label.set_color(proposed)
            label.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(54.0, 91.0)
    ax.set_xticks([60, 70, 80, 90])
    ax.set_ylim(6.15, -0.55)
    ax.set_xlabel("Score (%)")
    ax.legend(
        handles=[
            Line2D([0], [0], linestyle="none", marker="o", markersize=5.0,
                   markerfacecolor=final_color, markeredgecolor="white", label="Final composite F1"),
            Line2D([0], [0], linestyle="none", marker="s", markersize=4.8,
                   markerfacecolor=temporal_color, markeredgecolor="white", label="Temporal macro-F1"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    path = out_dir / "fig_overall_final_dynamic_tradeoff.pdf"
    save(fig, path)
    return [path]


def resolve_stability_metrics_path(experiment_root: Path) -> Path | None:
    preferred = (
        experiment_root
        / "r3_stability_formal_c5_s10"
        / "ATUAV-Core__latent_state_masked"
        / "run_metrics.csv"
    )
    if preferred.exists():
        return preferred
    candidates = sorted(
        experiment_root.glob("*stability*/*latent_state_masked*/run_metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def plot_stability_paired_figure(experiment_root: Path, out_dir: Path) -> list[Path]:
    """Show ten-seed paired Composite-F1 differences without hiding overlap."""
    metrics_path = resolve_stability_metrics_path(experiment_root)
    if metrics_path is None:
        return []
    run_metrics = read_csv(metrics_path)
    subset = run_metrics[
        (run_metrics["task"] == "joint")
        & (run_metrics["metric"] == "composite_f1")
    ].copy()
    if subset.empty:
        return []
    baselines = [
        "FlatSequenceMLP",
        "TemporalGRU",
        "TemporalLSTM",
        "TemporalTransformer",
        "TemporalTCN",
    ]
    audit = build_audit(
        run_metrics,
        primary_model="TemporalHGTAN",
        baselines=baselines,
        task="joint",
        metric="composite_f1",
    ).set_index("baseline")
    pivot = subset.pivot_table(index="seed", columns="model", values="value", aggfunc="first")
    records: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str | np.ndarray]] = []
    for baseline in baselines:
        if "TemporalHGTAN" not in pivot or baseline not in pivot:
            continue
        paired = pivot[["TemporalHGTAN", baseline]].dropna()
        deltas = 100.0 * (paired["TemporalHGTAN"] - paired[baseline])
        if deltas.empty:
            continue
        mean = float(deltas.mean())
        sd = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        ci95 = float(student_t.ppf(0.975, len(deltas) - 1) * sd / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
        wins = int((deltas > 0).sum())
        wilcoxon_holm_p = float(audit.loc[baseline, "wilcoxon_holm_p"])
        summaries.append(
            {
                "Baseline": baseline,
                "Label": BASELINE_ABBREVIATIONS[baseline],
                "Mean delta (pp)": mean,
                "CI95 half-width (pp)": ci95,
                "Wins": wins,
                "N": len(deltas),
                "Wilcoxon Holm p": wilcoxon_holm_p,
                "Deltas": deltas.to_numpy(dtype=float),
            }
        )
        for seed, delta in deltas.items():
            records.append(
                {
                    "Baseline": baseline,
                    "Baseline label": BASELINE_ABBREVIATIONS[baseline],
                    "Seed": int(seed),
                    "HGTAN minus baseline Composite F1 (pp)": float(delta),
                    "Mean delta (pp)": mean,
                    "CI95 half-width (pp)": ci95,
                    "HGTAN wins": wins,
                    "Paired seeds": len(deltas),
                    "Wilcoxon Holm p": wilcoxon_holm_p,
                    "Source suite": metrics_path.parents[1].name,
                }
            )
    if not summaries:
        return []

    source = pd.DataFrame(records)
    source.to_csv(out_dir / "fig_stability_paired_source.csv", index=False)
    row_order = [str(item["Baseline"]) for item in summaries]
    seed_order = sorted(int(seed) for seed in source["Seed"].unique())
    delta_column = "HGTAN minus baseline Composite F1 (pp)"
    matrix = (
        source.pivot(index="Baseline", columns="Seed", values=delta_column)
        .reindex(index=row_order, columns=seed_order)
        .to_numpy(dtype=float)
    )
    all_values = matrix[np.isfinite(matrix)]
    color_limit = max(1.0, float(np.ceil(np.max(np.abs(all_values)) * 10.0) / 10.0))
    cmap = LinearSegmentedColormap.from_list(
        "paired_delta",
        ["#D97757", "#F7F7F5", "#1F5AA6"],
    )
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    fig, ax = plt.subplots(figsize=(3.36, 2.15))
    fig.subplots_adjust(left=0.20, right=0.99, top=0.80, bottom=0.30)
    ax.grid(False)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    n_rows, n_seeds = matrix.shape
    for boundary in np.arange(-0.5, n_seeds + 0.5, 1.0):
        ax.plot([boundary, boundary], [-0.5, n_rows - 0.5], color="white", linewidth=0.65, zorder=2)
    for boundary in np.arange(-0.5, n_rows + 0.5, 1.0):
        ax.plot([-0.5, n_seeds - 0.5], [boundary, boundary], color="white", linewidth=0.65, zorder=2)

    ax.set_xticks(np.arange(n_seeds))
    ax.set_xticklabels([f"S{index + 1}" for index in range(n_seeds)], fontsize=6.6)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([str(item["Label"]) for item in summaries], fontsize=7.2)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(-0.5, 14.15)
    ax.set_ylim(n_rows - 0.5, -1.02)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)

    header_y = -0.76
    mean_x, wins_x, p_x = 10.45, 12.00, 13.55
    ax.text(mean_x, header_y, r"Mean $\Delta$", ha="center", va="center", fontsize=6.7, color="#4E5965")
    ax.text(wins_x, header_y, "Wins", ha="center", va="center", fontsize=6.7, color="#4E5965")
    ax.text(p_x, header_y, "Holm $p$", ha="center", va="center", fontsize=6.7, color="#4E5965")
    for row_index, item in enumerate(summaries):
        mean = float(item["Mean delta (pp)"])
        p_value = float(item["Wilcoxon Holm p"])
        significant = p_value < 0.05
        ax.text(mean_x, row_index, f"{mean:+.2f}", ha="center", va="center", fontsize=7.0, color="#303840")
        ax.text(
            wins_x,
            row_index,
            f"{int(item['Wins'])}/{int(item['N'])}",
            ha="center",
            va="center",
            fontsize=7.0,
            color="#303840",
        )
        ax.text(
            p_x,
            row_index,
            f"{p_value:.3f}".lstrip("0"),
            ha="center",
            va="center",
            fontsize=7.0,
            color="#1F5AA6" if significant else "#59636D",
            fontweight="bold" if significant else "normal",
        )

    colorbar_axis = ax.inset_axes([0.0, -0.29, 0.66, 0.065])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([-color_limit, 0.0, color_limit])
    colorbar.ax.tick_params(labelsize=6.2, length=1.8, width=0.5, pad=1.5)
    colorbar.outline.set_linewidth(0.45)
    colorbar.outline.set_edgecolor("#A5ADB5")
    colorbar.set_label("HGTAN $-$ baseline Composite F1 (pp)", fontsize=6.8, labelpad=1.5)
    path = out_dir / "fig_stability_paired_delta.pdf"
    save(fig, path, tight=False)
    return [path]


def load_formal_comparison_records(experiment_root: Path) -> list[dict]:
    """Load the locked three-seed prediction payloads used by the main table."""
    preferred = (
        experiment_root
        / "r3_comparison_formal_c5_s3"
        / "ATUAV-Core__latent_state_masked"
        / "seed_checkpoints"
    )
    checkpoint_dirs = [preferred] if preferred.exists() else []
    if not checkpoint_dirs:
        checkpoint_dirs = sorted(
            experiment_root.glob("*comparison*/*latent_state_masked/seed_checkpoints"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if not checkpoint_dirs:
        return []

    records: list[dict] = []
    for path in sorted(checkpoint_dirs[0].glob("run_*_seed_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        predictions = record.get("predictions", {})
        if all(model in predictions for model in DIAGNOSTIC_MODELS):
            records.append(record)
    return sorted(records, key=lambda record: int(record.get("run_index", 0)))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[tuple[int, int, float]]:
    """Return 1-based class support and F1 without averaging away ordinal levels."""
    rows = []
    for label in range(1, n_classes + 1):
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2.0 * true_positive / denominator if denominator else 0.0
        rows.append((label, support, f1))
    return rows


def plot_classwise_final_f1_figures(records: list[dict], out_dir: Path) -> list[Path]:
    """Show whether aggregate gains persist across ordinal output levels."""
    source_rows: list[dict[str, float | int | str]] = []
    for record in records:
        seed = int(record["seed"])
        predictions = record["predictions"]
        for model in DIAGNOSTIC_MODELS:
            payload = predictions[model]
            for task, n_classes in [("Threat", 5), ("Urgency", 3)]:
                y_true = np.asarray(payload[f"{task.lower()}_true"], dtype=np.int64)
                y_pred = np.asarray(payload[f"{task.lower()}_pred"], dtype=np.int64)
                for label, support, f1 in per_class_f1(y_true, y_pred, n_classes):
                    source_rows.append(
                        {
                            "Seed": seed,
                            "Model": model,
                            "Model label": DIAGNOSTIC_MODEL_STYLES[model]["label"],
                            "Task": task,
                            "Level": label,
                            "Support": support,
                            "F1 (%)": 100.0 * f1,
                            "Source suite": "r3_comparison_formal_c5_s3",
                        }
                    )
    source = pd.DataFrame(source_rows)
    if source.empty:
        return []
    source.to_csv(out_dir / "fig_classwise_final_f1_source.csv", index=False)

    written: list[Path] = []
    for task, n_classes, y_limits, output_name in [
        ("Threat", 5, (72.0, 96.0), "fig_classwise_threat_f1.pdf"),
        ("Urgency", 3, (83.0, 96.0), "fig_classwise_urgency_f1.pdf"),
    ]:
        task_source = source[source["Task"] == task]
        summary = (
            task_source.groupby(["Model", "Level"], as_index=False)["F1 (%)"]
            .agg(["mean", "std"])
            .reset_index()
        )
        fig, ax = new_panel_figure(width=3.36, height=2.07, journal=True)
        ax.grid(False)
        ax.grid(axis="y", color="#E1E5E8", linewidth=0.48, zorder=0)
        for model in DIAGNOSTIC_MODELS:
            style = DIAGNOSTIC_MODEL_STYLES[model]
            rows = summary[summary["Model"] == model].sort_values("Level")
            if rows.empty:
                continue
            is_hgtan = model == "TemporalHGTAN"
            ax.errorbar(
                rows["Level"],
                rows["mean"],
                yerr=rows["std"].fillna(0.0),
                label=str(style["label"]),
                color=str(style["color"]),
                marker=str(style["marker"]),
                linestyle=str(style["linestyle"]),
                linewidth=1.65 if is_hgtan else 1.05,
                markersize=4.4 if is_hgtan else 3.6,
                markeredgecolor="white",
                markeredgewidth=0.45,
                elinewidth=0.65,
                capsize=1.6,
                alpha=1.0 if is_hgtan else 0.90,
                zorder=6 if is_hgtan else 3,
            )
        ax.set_xlabel(f"{task} level")
        ax.set_ylabel("Class-wise F1 (%)")
        ax.set_xticks(range(1, n_classes + 1))
        ax.set_xlim(0.75, n_classes + 0.25)
        ax.set_ylim(*y_limits)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            fontsize=6.4,
            handlelength=1.6,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
        path = out_dir / output_name
        save(fig, path, top_pad=0.82)
        written.append(path)
    return written


def first_critical_frames(sequences: np.ndarray) -> np.ndarray:
    """Return each track's first entry into threat levels 4--5, or -1."""
    critical = np.asarray(sequences, dtype=np.int64) >= 4
    has_event = critical.any(axis=1)
    first = np.full(len(critical), -1, dtype=np.int64)
    first[has_event] = critical[has_event].argmax(axis=1)
    return first


def select_representative_transition_case(records: list[dict]) -> dict | None:
    """Select a reproducible median-gain case from every eligible test track."""
    candidates: list[dict] = []
    for record in records:
        predictions = record.get("predictions", {})
        if "TemporalHGTAN" not in predictions or "TemporalGRU" not in predictions:
            continue
        reference = predictions["TemporalHGTAN"]
        threat_true = np.asarray(reference["threat_seq_true"], dtype=np.int64)
        urgency_true = np.asarray(reference["urgency_seq_true"], dtype=np.int64)
        true_first = first_critical_frames(threat_true)
        transition_indices = np.flatnonzero(
            (true_first >= 1) & (threat_true[:, -1] > threat_true[:, 0])
        )
        frame_interval = 0.2
        data_profile = record.get("data_profile", [])
        if data_profile:
            frame_interval = float(data_profile[0].get("frame_interval", frame_interval))

        for track_index in transition_indices:
            models = {}
            for model in CURVE_MODELS:
                payload = predictions.get(model)
                if payload is None:
                    continue
                models[model] = {
                    "threat_pred": np.asarray(payload["threat_seq_pred"], dtype=np.int64)[track_index],
                    "urgency_pred": np.asarray(payload["urgency_seq_pred"], dtype=np.int64)[track_index],
                }
            if "TemporalHGTAN" not in models or "TemporalGRU" not in models:
                continue

            target = threat_true[track_index]
            hgtan_pred = models["TemporalHGTAN"]["threat_pred"]
            gru_pred = models["TemporalGRU"]["threat_pred"]
            hgtan_mae = float(np.mean(np.abs(hgtan_pred - target)))
            gru_mae = float(np.mean(np.abs(gru_pred - target)))
            hgtan_first = first_critical(hgtan_pred, [4, 5])
            timing_error_frames = (
                abs(hgtan_first - int(true_first[track_index]))
                if hgtan_first >= 0
                else len(target)
            )
            candidates.append(
                {
                    "threat_true": target,
                    "urgency_true": urgency_true[track_index],
                    "frame_interval": frame_interval,
                    "models": models,
                    "selection": {
                        "seed": int(record["seed"]),
                        "track_index": int(track_index),
                        "gru_minus_hgtan_mae": gru_mae - hgtan_mae,
                        "hgtan_timing_error_frames": int(timing_error_frames),
                    },
                }
            )

    if not candidates:
        return None

    median_gain = float(
        np.median([case["selection"]["gru_minus_hgtan_mae"] for case in candidates])
    )
    candidates.sort(
        key=lambda case: (
            abs(case["selection"]["gru_minus_hgtan_mae"] - median_gain),
            case["selection"]["hgtan_timing_error_frames"],
            case["selection"]["seed"],
            case["selection"]["track_index"],
        )
    )
    selected = candidates[0]
    selected["selection"].update(
        {
            "candidate_tracks": len(candidates),
            "median_gru_minus_hgtan_mae": median_gain,
            "selection_rule": (
                "closest to the median GRU-minus-HGTAN sequence MAE gain; "
                "ties use the smallest HGTAN first-alarm timing error, seed, and track index"
            ),
        }
    )
    return selected


def plot_critical_transition_figures(records: list[dict], out_dir: Path) -> list[Path]:
    """Aggregate first-alarm timing and event-aligned critical activation."""
    timing_rows: list[dict[str, float | int | str]] = []
    aligned_rows: list[dict[str, float | int | str]] = []
    max_tolerance_frames = 20
    alignment_radius = 12

    for record in records:
        seed = int(record["seed"])
        predictions = record["predictions"]
        reference = predictions["TemporalHGTAN"]
        threat_true = np.asarray(reference["threat_seq_true"], dtype=np.int64)
        true_first = first_critical_frames(threat_true)
        transition_mask = (true_first >= 1) & (threat_true[:, -1] > threat_true[:, 0])
        transition_indices = np.flatnonzero(transition_mask)
        seq_len = threat_true.shape[1]

        aligned_mask = (
            transition_mask
            & (true_first >= alignment_radius)
            & (true_first <= seq_len - alignment_radius - 1)
        )
        aligned_indices = np.flatnonzero(aligned_mask)
        offsets = np.arange(-alignment_radius, alignment_radius + 1, dtype=np.int64)
        aligned_true = np.stack(
            [threat_true[index, true_first[index] + offsets] for index in aligned_indices]
        )
        reference_activation = (aligned_true >= 4).mean(axis=0)
        for offset, activation in zip(offsets, reference_activation):
            aligned_rows.append(
                {
                    "Seed": seed,
                    "Model": "CleanReference",
                    "Model label": "Clean reference",
                    "Relative frame": int(offset),
                    "Relative time (s)": 0.2 * int(offset),
                    "Critical activation rate": float(activation),
                    "Critical disagreement rate": 0.0,
                    "Threat-level MAE": 0.0,
                    "Event tracks": len(aligned_indices),
                    "Source suite": "r3_comparison_formal_c5_s3",
                }
            )

        for model in DIAGNOSTIC_MODELS:
            payload = predictions[model]
            threat_pred = np.asarray(payload["threat_seq_pred"], dtype=np.int64)
            pred_first = first_critical_frames(threat_pred)
            detected = pred_first[transition_indices] >= 0
            timing_error = np.full(len(transition_indices), np.inf, dtype=np.float64)
            timing_error[detected] = (
                pred_first[transition_indices][detected] - true_first[transition_indices][detected]
            )
            for tolerance in range(max_tolerance_frames + 1):
                timing_rows.append(
                    {
                        "Seed": seed,
                        "Model": model,
                        "Model label": DIAGNOSTIC_MODEL_STYLES[model]["label"],
                        "Tolerance frames": tolerance,
                        "Tolerance (s)": 0.2 * tolerance,
                        "Event-time agreement": float(np.mean(np.abs(timing_error) <= tolerance)),
                        "Premature alarm rate": float(np.mean(timing_error < -tolerance)),
                        "Late or missed rate": float(np.mean(timing_error > tolerance)),
                        "Detection rate": float(np.mean(detected)),
                        "Event tracks": len(transition_indices),
                        "Source suite": "r3_comparison_formal_c5_s3",
                    }
                )

            aligned_pred = np.stack(
                [threat_pred[index, true_first[index] + offsets] for index in aligned_indices]
            )
            activation = (aligned_pred >= 4).mean(axis=0)
            disagreement = ((aligned_pred >= 4) != (aligned_true >= 4)).mean(axis=0)
            mae = np.abs(aligned_pred - aligned_true).mean(axis=0)
            for offset, activation_value, disagreement_value, mae_value in zip(
                offsets, activation, disagreement, mae
            ):
                aligned_rows.append(
                    {
                        "Seed": seed,
                        "Model": model,
                        "Model label": DIAGNOSTIC_MODEL_STYLES[model]["label"],
                        "Relative frame": int(offset),
                        "Relative time (s)": 0.2 * int(offset),
                        "Critical activation rate": float(activation_value),
                        "Critical disagreement rate": float(disagreement_value),
                        "Threat-level MAE": float(mae_value),
                        "Event tracks": len(aligned_indices),
                        "Source suite": "r3_comparison_formal_c5_s3",
                    }
                )

    timing_source = pd.DataFrame(timing_rows)
    aligned_source = pd.DataFrame(aligned_rows)
    if timing_source.empty or aligned_source.empty:
        return []
    timing_source.to_csv(out_dir / "fig_event_timing_agreement_source.csv", index=False)
    aligned_source.to_csv(out_dir / "fig_event_aligned_disagreement_source.csv", index=False)
    window_rows: list[dict[str, float | int | str]] = []
    for (seed, model), rows in aligned_source[
        aligned_source["Model"] != "CleanReference"
    ].groupby(["Seed", "Model"]):
        pre_error = float(
            rows[rows["Relative time (s)"] < 0]["Critical disagreement rate"].mean()
        )
        post_error = float(
            rows[rows["Relative time (s)"] > 0]["Critical disagreement rate"].mean()
        )
        window_rows.append(
            {
                "Seed": int(seed),
                "Model": str(model),
                "Model label": DIAGNOSTIC_MODEL_STYLES[str(model)]["label"],
                "Pre-event disagreement (%)": 100.0 * pre_error,
                "Post-event disagreement (%)": 100.0 * post_error,
                "Balanced window disagreement (%)": 50.0 * (pre_error + post_error),
                "Source suite": "r3_comparison_formal_c5_s3",
            }
        )
    pd.DataFrame(window_rows).to_csv(
        out_dir / "fig_event_aligned_window_summary.csv",
        index=False,
    )

    written: list[Path] = []
    timing_summary = (
        timing_source.groupby(["Model", "Tolerance (s)"], as_index=False)["Event-time agreement"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = new_panel_figure(width=3.36, height=2.07, journal=True)
    ax.grid(False)
    ax.grid(axis="y", color="#E1E5E8", linewidth=0.48, zorder=0)
    for model in DIAGNOSTIC_MODELS:
        style = DIAGNOSTIC_MODEL_STYLES[model]
        rows = timing_summary[timing_summary["Model"] == model].sort_values("Tolerance (s)")
        x = rows["Tolerance (s)"].to_numpy(dtype=float)
        mean = 100.0 * rows["mean"].to_numpy(dtype=float)
        spread = 100.0 * rows["std"].fillna(0.0).to_numpy(dtype=float)
        is_hgtan = model == "TemporalHGTAN"
        ax.fill_between(x, mean - spread, mean + spread, color=str(style["color"]), alpha=0.08, linewidth=0)
        ax.plot(
            x,
            mean,
            label=str(style["label"]),
            color=str(style["color"]),
            marker=str(style["marker"]),
            markevery=4,
            linestyle=str(style["linestyle"]),
            linewidth=1.70 if is_hgtan else 1.05,
            markersize=4.2 if is_hgtan else 3.3,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=6 if is_hgtan else 3,
        )
    ax.set_xlabel(r"Allowed first-alarm error $\tau$ (s)")
    ax.set_ylabel(r"Events within $\pm\tau$ (\%)")
    ax.set_xlim(0.0, 4.0)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(0.0, 82.0)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.6,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    path = out_dir / "fig_event_timing_agreement.pdf"
    save(fig, path, top_pad=0.82)
    written.append(path)

    aligned_summary = (
        aligned_source[aligned_source["Model"] != "CleanReference"]
        .groupby(["Model", "Relative time (s)"], as_index=False)["Critical disagreement rate"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = new_panel_figure(width=3.36, height=2.07, journal=True)
    ax.grid(False)
    ax.grid(axis="y", color="#E1E5E8", linewidth=0.48, zorder=0)
    for model in DIAGNOSTIC_MODELS:
        style = DIAGNOSTIC_MODEL_STYLES[model]
        rows = aligned_summary[aligned_summary["Model"] == model].sort_values("Relative time (s)")
        x = rows["Relative time (s)"].to_numpy(dtype=float)
        mean = 100.0 * rows["mean"].to_numpy(dtype=float)
        spread = 100.0 * rows["std"].fillna(0.0).to_numpy(dtype=float)
        is_hgtan = model == "TemporalHGTAN"
        ax.fill_between(
            x,
            np.maximum(0.0, mean - spread),
            np.minimum(100.0, mean + spread),
            color=str(style["color"]),
            alpha=0.07,
            linewidth=0,
        )
        ax.plot(
            x,
            mean,
            label=str(style["label"]),
            color=str(style["color"]),
            marker=str(style["marker"]),
            markevery=4,
            linestyle=str(style["linestyle"]),
            linewidth=1.70 if is_hgtan else 1.05,
            markersize=4.2 if is_hgtan else 3.3,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=6 if is_hgtan else 3,
        )
    ax.axvspan(-0.025, 0.025, color="#777777", alpha=0.22, linewidth=0, zorder=1)
    ax.set_xlabel("Time relative to reference event (s)")
    ax.set_ylabel("Critical-state error (%)")
    ax.set_xlim(-2.4, 2.4)
    ax.set_xticks([-2.4, -1.2, 0.0, 1.2, 2.4])
    ax.set_ylim(0.0, 94.0)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=6.2,
        handlelength=1.6,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    path = out_dir / "fig_event_aligned_disagreement.pdf"
    save(fig, path, top_pad=0.82)
    written.append(path)
    return written


def policy_condition_label(setting: str) -> str:
    if "policy_balanced" in setting:
        return "Balanced"
    if "policy_consequence_first" in setting:
        return "Consequence-first"
    if "policy_access_first" in setting:
        return "Access-first"
    return setting


def holdout_condition_label(setting: str) -> str:
    labels = {
        "Probe_Surveillance": "Probe-Surveillance",
        "EW_Contested": "EW-Contested",
        "Strike_Penetration": "Strike-Penetration",
        "Saturation_Overload": "Saturation-Overload",
    }
    for token, label in labels.items():
        if token in setting:
            return label
    return setting


def compute_margin_rows(
    summary: pd.DataFrame,
    suite: str,
    settings: list[str],
    *,
    condition_labels: dict[str, str],
    include_mean: bool,
) -> list[dict[str, float | str]]:
    metric_specs = [
        ("Composite F1", "joint", "composite_f1", True, 100.0, "pp"),
        ("Temporal accuracy", "threat_track", "temporal_accuracy", True, 100.0, "pp"),
        ("Ordinal MAE", "threat_track", "mean_abs_ordinal_error", False, 1.0, "raw"),
    ]
    models = [
        "FlatSequenceMLP",
        "TemporalGRU",
        "TemporalLSTM",
        "TemporalTransformer",
        "TemporalTCN",
        "TemporalHGTAN",
    ]
    records: list[dict[str, float | str]] = []
    values_by_metric: dict[tuple[str, str], list[float]] = {}
    for setting in settings:
        for metric_label, task, metric_name, higher_better, scale, unit in metric_specs:
            values: dict[str, float] = {}
            for model in models:
                result = metric(summary, suite, setting, model, task, metric_name)
                if result is not None:
                    values[model] = float(result["mean"])
                    values_by_metric.setdefault((metric_name, model), []).append(float(result["mean"]))
            if "TemporalHGTAN" not in values or len(values) < 2:
                continue
            alternatives = {model: value for model, value in values.items() if model != "TemporalHGTAN"}
            best_model = (max if higher_better else min)(alternatives, key=alternatives.get)
            hgtan = values["TemporalHGTAN"]
            best = alternatives[best_model]
            margin = (hgtan - best) if higher_better else (best - hgtan)
            records.append(
                {
                    "Condition": condition_labels[setting],
                    "Metric": metric_label,
                    "HGTAN": scale * hgtan,
                    "Best alternative": scale * best,
                    "Alternative model": best_model,
                    "Alternative label": BASELINE_ABBREVIATIONS[best_model],
                    "HGTAN advantage": scale * margin,
                    "Unit": unit,
                    "Source suite": suite,
                    "Source setting": setting,
                }
            )

    if include_mean:
        for metric_label, _task, metric_name, higher_better, scale, unit in metric_specs:
            means = {
                model: float(np.mean(values))
                for model in models
                if (values := values_by_metric.get((metric_name, model)))
            }
            if "TemporalHGTAN" not in means or len(means) < 2:
                continue
            alternatives = {model: value for model, value in means.items() if model != "TemporalHGTAN"}
            best_model = (max if higher_better else min)(alternatives, key=alternatives.get)
            hgtan = means["TemporalHGTAN"]
            best = alternatives[best_model]
            margin = (hgtan - best) if higher_better else (best - hgtan)
            records.append(
                {
                    "Condition": "Mean",
                    "Metric": metric_label,
                    "HGTAN": scale * hgtan,
                    "Best alternative": scale * best,
                    "Alternative model": best_model,
                    "Alternative label": BASELINE_ABBREVIATIONS[best_model],
                    "HGTAN advantage": scale * margin,
                    "Unit": unit,
                    "Source suite": suite,
                    "Source setting": "mean_over_holdouts",
                }
            )
    return records


def margin_color(value: float, column_scale: float) -> tuple[float, float, float]:
    base = matplotlib.colors.to_rgb("#4f8665" if value >= 0 else "#b8644e")
    strength = 0.20 + 0.60 * min(abs(value) / max(column_scale, 1e-12), 1.0)
    return tuple((1.0 - strength) + strength * component for component in base)


def draw_margin_matrix(records: list[dict[str, float | str]], path: Path, condition_order: list[str]) -> None:
    metric_order = ["Composite F1", "Temporal accuracy", "Ordinal MAE"]
    lookup = {(str(row["Condition"]), str(row["Metric"])): row for row in records}
    scales = {
        metric_name: max(
            [abs(float(row["HGTAN advantage"])) for row in records if row["Metric"] == metric_name] or [1.0]
        )
        for metric_name in metric_order
    }
    image = np.ones((len(condition_order), len(metric_order), 3), dtype=float)
    for row_idx, condition in enumerate(condition_order):
        for col_idx, metric_name in enumerate(metric_order):
            row = lookup[(condition, metric_name)]
            image[row_idx, col_idx, :] = margin_color(float(row["HGTAN advantage"]), scales[metric_name])

    fig, ax = plt.subplots(figsize=(3.65, 1.78 + 0.26 * len(condition_order)))
    ax.imshow(image, aspect="auto", interpolation="none")
    ax.grid(False, which="major")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(["Composite F1\n(pp)", "Temporal accuracy\n(pp)", "Ordinal MAE"], fontsize=7.2)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks(np.arange(len(condition_order)))
    ax.set_yticklabels(condition_order, fontsize=7.2)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, len(metric_order), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(condition_order), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_idx, condition in enumerate(condition_order):
        for col_idx, metric_name in enumerate(metric_order):
            row = lookup[(condition, metric_name)]
            value = float(row["HGTAN advantage"])
            value_text = f"{value:+.3f}" if metric_name == "Ordinal MAE" else f"{value:+.2f}"
            ax.text(
                col_idx,
                row_idx,
                f"{value_text}\n{row['Alternative label']}",
                ha="center",
                va="center",
                fontsize=6.8,
                color="#232323",
                linespacing=1.18,
                fontweight="normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, path)


def plot_policy_holdout_margin_figures(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    policy_suite = resolve_suite(summary, "policy")
    holdout_suite = resolve_suite(summary, "holdout")
    if summary.empty or policy_suite is None or holdout_suite is None:
        return []

    policy_settings = sorted(summary[summary["source_suite"] == policy_suite]["setting"].dropna().astype(str).unique())
    holdout_settings = sorted(summary[summary["source_suite"] == holdout_suite]["setting"].dropna().astype(str).unique())
    policy_labels = {setting: policy_condition_label(setting) for setting in policy_settings}
    holdout_labels = {setting: holdout_condition_label(setting) for setting in holdout_settings}
    policy_records = compute_margin_rows(
        summary,
        policy_suite,
        policy_settings,
        condition_labels=policy_labels,
        include_mean=False,
    )
    holdout_records = compute_margin_rows(
        summary,
        holdout_suite,
        holdout_settings,
        condition_labels=holdout_labels,
        include_mean=True,
    )
    if not policy_records or not holdout_records:
        return []

    pd.DataFrame(policy_records + holdout_records).to_csv(
        out_dir / "fig_policy_holdout_margin_source.csv", index=False
    )
    policy_order = ["Balanced", "Consequence-first", "Access-first"]
    holdout_order = [
        "Probe-Surveillance",
        "EW-Contested",
        "Strike-Penetration",
        "Saturation-Overload",
        "Mean",
    ]
    policy_path = out_dir / "fig_policy_margin_matrix.pdf"
    holdout_path = out_dir / "fig_holdout_margin_matrix.pdf"
    draw_margin_matrix(policy_records, policy_path, policy_order)
    draw_margin_matrix(holdout_records, holdout_path, holdout_order)
    return [policy_path, holdout_path]


def compute_paired_comparison_rows(
    summary: pd.DataFrame,
    run_metrics: pd.DataFrame,
    suite: str,
    settings: list[str],
    *,
    condition_labels: dict[str, str],
) -> list[dict[str, float | str]]:
    metric_specs = [
        ("Composite F1", "joint", "composite_f1"),
        ("Temporal accuracy", "threat_track", "temporal_accuracy"),
    ]
    models = [
        "FlatSequenceMLP",
        "TemporalGRU",
        "TemporalLSTM",
        "TemporalTransformer",
        "TemporalTCN",
        "TemporalHGTAN",
    ]
    records: list[dict[str, float | str]] = []
    for setting in settings:
        for metric_label, task, metric_name in metric_specs:
            values: dict[str, dict[str, float]] = {}
            for model in models:
                result = metric(summary, suite, setting, model, task, metric_name)
                if result is not None:
                    values[model] = result
            if "TemporalHGTAN" not in values or len(values) < 2:
                continue
            alternatives = {model: value for model, value in values.items() if model != "TemporalHGTAN"}
            best_model = max(alternatives, key=lambda model: alternatives[model]["mean"])
            seed_rows = run_metrics[
                (run_metrics["source_suite"] == suite)
                & (run_metrics["setting"] == setting)
                & (run_metrics["task"] == task)
                & (run_metrics["metric"] == metric_name)
                & (run_metrics["model"].isin(["TemporalHGTAN", best_model]))
            ].copy()
            if seed_rows.empty:
                continue
            seed_rows["value"] = pd.to_numeric(seed_rows["value"], errors="coerce")
            paired = (
                seed_rows.dropna(subset=["seed", "value"])
                .groupby(["seed", "model"], as_index=False)["value"]
                .mean()
                .pivot(index="seed", columns="model", values="value")
                .dropna(subset=["TemporalHGTAN", best_model])
            )
            if paired.empty:
                continue
            paired["Delta (pp)"] = 100.0 * (paired["TemporalHGTAN"] - paired[best_model])
            mean_delta = float(paired["Delta (pp)"].mean())
            sd_delta = float(paired["Delta (pp)"].std(ddof=1)) if len(paired) > 1 else 0.0
            for seed, row in paired.iterrows():
                records.append(
                    {
                        "Stress family": "Policy" if "policy" in suite else "Held-out family",
                        "Condition": condition_labels[setting],
                        "Metric": metric_label,
                        "Seed": int(seed),
                        "HGTAN (%)": 100.0 * float(row["TemporalHGTAN"]),
                        "Alternative (%)": 100.0 * float(row[best_model]),
                        "Delta (pp)": float(row["Delta (pp)"]),
                        "Mean delta (pp)": mean_delta,
                        "SD delta (pp)": sd_delta,
                        "N": int(len(paired)),
                        "Alternative model": best_model,
                        "Alternative label": BASELINE_ABBREVIATIONS[best_model],
                        "Source suite": suite,
                        "Source setting": setting,
                    }
                )
    return records


def draw_paired_margin_panel(
    records: list[dict[str, float | str]],
    *,
    condition_order: list[str],
    path: Path,
) -> None:
    metric_styles = {
        "Composite F1": {"color": "#3F6FA0", "short": "F1", "legend": "Composite F1"},
        "Temporal accuracy": {
            "color": "#C65349",
            "short": "T-Acc.",
            "legend": "Temporal acc.",
        },
    }
    fig, ax = new_panel_figure(width=3.65, height=2.12, journal=True)
    ax.grid(False)
    ax.grid(axis="x", linewidth=0.42, color="#dedede", linestyle=":", alpha=0.95)
    ax.axvline(0.0, color="#8a8a8a", linewidth=0.70, zorder=3)
    y = np.arange(len(condition_order), dtype=float)
    offsets = {"Composite F1": -0.13, "Temporal accuracy": 0.13}
    tick_labels: list[str] = []
    interval_bounds: list[float] = [0.0]
    for idx, condition in enumerate(condition_order):
        alternative_labels: list[str] = []
        for metric_name, style in metric_styles.items():
            rows = [
                row for row in records if row["Condition"] == condition and row["Metric"] == metric_name
            ]
            if not rows:
                continue
            row_y = idx + offsets[metric_name]
            mean_delta = float(rows[0]["Mean delta (pp)"])
            sd_delta = float(rows[0]["SD delta (pp)"])
            ax.barh(
                row_y,
                2.0 * sd_delta,
                left=mean_delta - sd_delta,
                height=0.075,
                color=style["color"],
                edgecolor="none",
                alpha=0.28,
                zorder=1,
            )
            ax.barh(
                row_y,
                mean_delta,
                height=0.19,
                color=style["color"],
                edgecolor="none",
                alpha=0.94,
                zorder=2,
            )
            interval_bounds.extend([mean_delta - sd_delta, mean_delta + sd_delta])
            alternative_labels.append(str(rows[0]["Alternative label"]))
        tick_labels.append(f"{condition}\n[{' / '.join(alternative_labels)}]")
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=6.8, linespacing=1.05)
    ax.invert_yaxis()
    lower = min(interval_bounds)
    upper = max(interval_bounds)
    padding = max(0.7, 0.08 * (upper - lower))
    left = 2.0 * np.floor((min(0.0, lower) - padding) / 2.0)
    right = 2.0 * np.ceil((max(0.0, upper) + padding) / 2.0)
    ax.set_xlim(left, right)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("HGTAN - strongest alternative (percentage points)")
    handles = [
        Patch(facecolor=style["color"], edgecolor="none", label=style["legend"])
        for metric_name, style in metric_styles.items()
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=6.7,
        handlelength=1.25,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.23, top=0.82)
    save(fig, path, tight=False)


def plot_policy_holdout_comparison_figures(
    summary: pd.DataFrame,
    run_metrics: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    policy_suite = resolve_suite(summary, "policy")
    holdout_suite = resolve_suite(summary, "holdout")
    if summary.empty or run_metrics.empty or policy_suite is None or holdout_suite is None:
        return []
    policy_settings = sorted(summary[summary["source_suite"] == policy_suite]["setting"].dropna().astype(str).unique())
    holdout_settings = sorted(summary[summary["source_suite"] == holdout_suite]["setting"].dropna().astype(str).unique())
    policy_labels = {setting: policy_condition_label(setting) for setting in policy_settings}
    holdout_labels = {setting: holdout_condition_label(setting) for setting in holdout_settings}
    policy_records = compute_paired_comparison_rows(
        summary, run_metrics, policy_suite, policy_settings, condition_labels=policy_labels
    )
    holdout_records = compute_paired_comparison_rows(
        summary, run_metrics, holdout_suite, holdout_settings, condition_labels=holdout_labels
    )
    if not policy_records or not holdout_records:
        return []
    pd.DataFrame(policy_records + holdout_records).to_csv(
        out_dir / "fig_policy_holdout_paired_source.csv", index=False
    )
    specs = [
        (policy_records, ["Balanced", "Consequence-first", "Access-first"], "fig_policy_paired_margins.pdf"),
        (
            holdout_records,
            ["Probe-Surveillance", "EW-Contested", "Strike-Penetration", "Saturation-Overload"],
            "fig_holdout_paired_margins.pdf",
        ),
    ]
    written: list[Path] = []
    for records, order, filename in specs:
        path = out_dir / filename
        draw_paired_margin_panel(records, condition_order=order, path=path)
        written.append(path)
    return written


def plot_ablation_absolute_figures(run_metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Plot default module ablations and fixed-endpoint temporal controls."""
    if run_metrics.empty:
        return []

    default_suite = resolve_suite_excluding(run_metrics, "ablation", excluded=("fixed_endpoint",))
    fixed_suite = resolve_suite_containing(run_metrics, ("fixed_endpoint", "ablation"))
    if default_suite is None or fixed_suite is None:
        return []

    default_setting = first_existing_setting(
        run_metrics,
        default_suite,
        ["ATUAV-Core__latent_state_masked"],
    )
    fixed_setting = first_existing_setting(
        run_metrics,
        fixed_suite,
        ["ATUAV-Core__latent_state_masked__ablation_fixed_endpoint_obs32"],
    )
    if default_setting is None or fixed_setting is None:
        return []

    module_specs = [
        ("TemporalHGTAN", "Full", "#F04B3A", ""),
        ("TemporalHGTAN_MeanPool", "w/o T", "#22A7E0", ""),
        ("TemporalHGTAN_NoPrior", "w/o P", "#35C26B", ""),
        ("TemporalHGTAN_NoSynergy", "w/o S", "#8F5BD8", ""),
    ]
    temporal_summary_specs = [
        ("TemporalHGTAN", "Adaptive", "#F04B3A", ""),
        ("TemporalHGTAN_MeanPool", "Mean", "#22A7E0", ""),
        ("TemporalHGTAN_LastFrame", "Last", "#F2A900", ""),
    ]
    metric_specs = [
        ("Composite F1", "joint", "composite_f1", "composite_f1"),
        ("Temporal macro-F1", "threat_track", "temporal_macro_f1", "temporal_f1"),
    ]

    records: list[dict[str, float | int | str]] = []
    condition_specs = [
        ("Default modules", default_suite, default_setting, module_specs),
        ("Fixed-endpoint summaries", fixed_suite, fixed_setting, temporal_summary_specs),
    ]
    for condition, suite, setting, source_specs in condition_specs:
        suite_rows = run_metrics[run_metrics["source_suite"] == suite]
        for metric_label, task, metric_name, _metric_stem in metric_specs:
            metric_rows = suite_rows[
                (suite_rows["setting"] == setting)
                & (suite_rows["task"] == task)
                & (suite_rows["metric"] == metric_name)
            ]
            for model, short_label, _color, _hatch in source_specs:
                model_rows = metric_rows[metric_rows["model"] == model]
                for row in model_rows.itertuples(index=False):
                    records.append(
                        {
                            "source_suite": suite,
                            "source_setting": setting,
                            "condition": condition,
                            "metric": metric_label,
                            "model": model,
                            "model_label": short_label,
                            "seed": int(row.seed),
                            "value_percent": 100.0 * float(row.value),
                        }
                    )

    if not records:
        return []

    source = pd.DataFrame(records)
    source.to_csv(out_dir / "fig_ablation_absolute_source.csv", index=False)
    summary = (
        source.groupby(["condition", "metric", "model", "model_label"], as_index=False)["value_percent"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    written: list[Path] = []
    for metric_label, _task, _metric_name, metric_stem in metric_specs:
        panel_specs = [
            ("Default modules", "default", module_specs, True),
            ("Fixed-endpoint summaries", "fixed_summary", temporal_summary_specs, False),
        ]
        for condition, condition_stem, display_specs, show_ylabel in panel_specs:
            panel = summary[
                (summary["condition"] == condition) & (summary["metric"] == metric_label)
            ].set_index("model")
            if panel.empty:
                continue

            means = np.asarray([float(panel.loc[model, "mean"]) for model, *_rest in display_specs])
            value_span = float(np.max(means) - np.min(means))
            y_margin = max(0.18, 0.28 * value_span)
            y_min = 0.1 * np.floor(10.0 * (float(np.min(means)) - y_margin))
            y_max = 0.1 * np.ceil(10.0 * (float(np.max(means)) + y_margin))
            x_step = 0.88 if condition_stem == "fixed_summary" else 0.76
            x = x_step * np.arange(len(display_specs), dtype=float)
            fig, ax = new_panel_figure(width=2.28, height=1.38, journal=True)
            ax.grid(False)
            ax.grid(axis="y", linewidth=0.34, color="#D8D8D8", linestyle="-", alpha=0.75, zorder=0)
            bars = ax.bar(
                x,
                means,
                width=0.58,
                color=[spec[2] for spec in display_specs],
                edgecolor="#40505E",
                linewidth=0.62,
                zorder=3,
            )
            for bar, (_model, _label, _color, hatch) in zip(bars, display_specs):
                bar.set_hatch(hatch)
            ax.set_xlim(x[0] - 0.34, x[-1] + 0.34)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#6A6A6A")
                spine.set_linewidth(0.58)
            ax.set_xticks(x)
            tick_fontsize = 7.2 if condition_stem == "fixed_summary" else 7.5
            ax.set_xticklabels([spec[1] for spec in display_specs], fontsize=tick_fontsize)
            ax.tick_params(axis="x", length=0, pad=1.3)
            ax.tick_params(axis="y", labelsize=7.3, width=0.55, length=2.2)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.set_ylim(y_min, y_max)
            if show_ylabel:
                ax.set_ylabel(f"{metric_label} (%)", fontsize=8.4)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            path = out_dir / f"fig_ablation_{condition_stem}_{metric_stem}.pdf"
            save(fig, path)
            written.append(path)
    return written


def plot_observed_time_main_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    suite = resolve_observed_time_suite(summary)
    if suite is None or summary.empty:
        return []

    model_styles = OBSERVATION_MODEL_STYLES

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
                    "Composite F1 SD": 100.0 * comp["std"],
                    "Threat temporal accuracy (%)": 100.0 * tacc["mean"],
                    "Threat temporal accuracy SD": 100.0 * tacc["std"],
                    "Critical false alarm (%)": 100.0 * false_alarm["mean"],
                }
            )

    if not ordered_rows:
        return []

    plot_df = pd.DataFrame(ordered_rows).sort_values(["observed_len", "Model"])
    plot_df.to_csv(out_dir / "fig_observed_time_source.csv", index=False)
    metric_specs = [
        ("Composite F1", "Composite F1 SD", "fig_observed_time_composite_f1.pdf"),
        ("Threat temporal accuracy (%)", "Threat temporal accuracy SD", "fig_observed_time_dynamic_accuracy.pdf"),
    ]
    written: list[Path] = []
    for column, spread_column, filename in metric_specs:
        fig, ax = new_panel_figure(width=3.55, height=2.35, journal=True)
        for model, style in model_styles.items():
            sub = plot_df[plot_df["Model"] == model]
            if sub.empty:
                continue
            ax.errorbar(
                sub["seconds"],
                sub[column],
                yerr=sub[spread_column],
                label=pretty_model_name(model),
                color=style["color"],
                fmt=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markerfacecolor=style["color"] if model == "TemporalHGTAN" else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.7,
                elinewidth=0.65,
                capsize=1.6,
                alpha=style["alpha"],
                zorder=style["zorder"],
            )
        ax.set_xlabel("Available history (s)")
        ax.set_xticks(sorted(plot_df["seconds"].unique()))
        x_min = float(plot_df["seconds"].min())
        x_max = float(plot_df["seconds"].max())
        ax.set_xlim(x_min - 0.45, x_max + 0.45)
        if column == "Composite F1":
            ax.set_ylabel("Composite F1 (%)")
        elif column == "Threat temporal accuracy (%)":
            ax.set_ylabel("Threat temporal accuracy (%)")
        else:
            ax.set_ylabel("Critical false alarms (%)")
        lower = float((plot_df[column] - plot_df[spread_column]).min())
        upper = float((plot_df[column] + plot_df[spread_column]).max())
        pad = max(0.45, 0.04 * (upper - lower))
        ax.set_ylim(max(0.0, lower - pad), min(100.0, upper + pad))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            fontsize=6.2,
            handlelength=1.35,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
        save(fig, out_dir / filename, top_pad=0.86)
        written.append(out_dir / filename)
    return written


def plot_distance_degradation_figure(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    suite = resolve_distance_suite(summary)
    if suite is None or summary.empty:
        return []

    model_styles = ROBUSTNESS_MODEL_STYLES

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
                    "Composite F1 SD": 100.0 * comp["std"],
                    "Threat temporal accuracy (%)": 100.0 * tacc["mean"],
                    "Threat temporal accuracy SD": 100.0 * tacc["std"],
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
    plot_df.to_csv(out_dir / "fig_distance_degradation_source.csv", index=False)
    metric_specs = [
        ("Composite F1", "Composite F1 SD", "fig_distance_degradation_composite_f1.pdf"),
        ("Threat temporal accuracy (%)", "Threat temporal accuracy SD", "fig_distance_degradation_dynamic_accuracy.pdf"),
    ]
    written: list[Path] = []
    for column, spread_column, filename in metric_specs:
        fig, ax = new_panel_figure(width=3.55, height=2.35, journal=True)
        for model, style in model_styles.items():
            sub = plot_df[plot_df["Model"] == model]
            if sub.empty:
                continue
            ax.errorbar(
                sub["Range (m)"],
                sub[column],
                yerr=sub[spread_column],
                label=pretty_model_name(model),
                color=style["color"],
                fmt=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markerfacecolor=style["color"] if model == "TemporalHGTAN" else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.7,
                elinewidth=0.65,
                capsize=1.6,
                alpha=style["alpha"],
                zorder=style["zorder"],
            )
        ax.set_xlabel("Nominal sensing range (m)")
        ax.set_xticks([1000.0, 3000.0, 5000.0])
        x_min = float(plot_df["Range (m)"].min())
        x_max = float(plot_df["Range (m)"].max())
        ax.set_xlim(x_min - 100.0, x_max + 100.0)
        if column == "Composite F1":
            ax.set_ylabel("Composite F1 (%)")
        elif column == "Threat temporal accuracy (%)":
            ax.set_ylabel("Threat temporal accuracy (%)")
        lower = float((plot_df[column] - plot_df[spread_column]).min())
        upper = float((plot_df[column] + plot_df[spread_column]).max())
        pad = max(0.35, 0.04 * (upper - lower))
        ax.set_ylim(max(0.0, lower - pad), min(100.0, upper + pad))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
            fontsize=6.2,
            handlelength=1.35,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
        save(fig, out_dir / filename, top_pad=0.86)
        written.append(out_dir / filename)
    return written


def plot_missing_robustness_figures(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Plot frozen-model performance under random and contiguous frame loss."""
    suite = resolve_suite(summary, "missing")
    if suite is None or summary.empty:
        return []

    records: list[dict[str, float | str]] = []
    metric_specs = [
        ("Composite F1 (%)", "joint", "composite_f1"),
        ("Threat temporal macro-F1 (%)", "threat_track", "temporal_macro_f1"),
    ]
    suite_rows = summary[summary["source_suite"] == suite]
    for mode in ["random", "burst"]:
        for ratio in [0.0, 0.05, 0.10, 0.15, 0.20]:
            setting_rows = suite_rows[
                (suite_rows["test_missing_mode"] == mode)
                & np.isclose(suite_rows["test_missing_ratio"].astype(float), ratio)
            ]
            if setting_rows.empty:
                continue
            setting_name = str(setting_rows.iloc[0]["setting"])
            for model in MISSING_MODEL_STYLES:
                for label, task, metric_name in metric_specs:
                    result = metric(summary, suite, setting_name, model, task, metric_name)
                    if result is None:
                        continue
                    records.append(
                        {
                            "Missingness mode": mode,
                            "Missing-frame ratio (%)": 100.0 * ratio,
                            "Model": model,
                            "Model label": MISSING_MODEL_STYLES[model]["label"],
                            "Metric": label,
                            "Mean (%)": 100.0 * result["mean"],
                            "SD": 100.0 * result["std"],
                            "Source suite": suite,
                            "Source setting": setting_name,
                            "Seeds": 3,
                            "Training corruption": "none",
                        }
                    )
    if not records:
        return []

    source = pd.DataFrame(records)
    source.to_csv(out_dir / "fig_missing_robustness_source.csv", index=False)

    legend_fig = plt.figure(figsize=(7.05, 0.34))
    legend_ax = legend_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    legend_ax.axis("off")
    legend_handles = [
        Line2D(
            [0], [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=4.0,
            markerfacecolor=style["color"] if model == "TemporalHGTAN" else "white",
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
            label=style["label"],
        )
        for model, style in MISSING_MODEL_STYLES.items()
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        facecolor="white",
        framealpha=1.0,
        fontsize=7.0,
        handlelength=1.65,
        handletextpad=0.42,
        columnspacing=1.05,
        borderpad=0.35,
    )
    legend_path = out_dir / "fig_missing_robustness_legend.pdf"
    save(legend_fig, legend_path, tight=False)
    written = [legend_path]

    panel_specs = [
        ("random", "Composite F1 (%)", "fig_missing_random_composite_f1.pdf"),
        ("random", "Threat temporal macro-F1 (%)", "fig_missing_random_temporal_f1.pdf"),
        ("burst", "Composite F1 (%)", "fig_missing_burst_composite_f1.pdf"),
        ("burst", "Threat temporal macro-F1 (%)", "fig_missing_burst_temporal_f1.pdf"),
    ]
    for mode, metric_label, filename in panel_specs:
        panel = source[(source["Missingness mode"] == mode) & (source["Metric"] == metric_label)]
        if panel.empty:
            continue
        fig, ax = new_panel_figure(width=3.48, height=2.18, journal=True)
        for model, style in MISSING_MODEL_STYLES.items():
            sub = panel[panel["Model"] == model].sort_values("Missing-frame ratio (%)")
            if sub.empty:
                continue
            ax.errorbar(
                sub["Missing-frame ratio (%)"],
                sub["Mean (%)"],
                yerr=sub["SD"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=3.8 if model != "TemporalHGTAN" else 4.2,
                markerfacecolor=style["color"] if model == "TemporalHGTAN" else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.70,
                elinewidth=0.55,
                capsize=1.35,
                alpha=1.0,
                zorder=style["zorder"],
            )
        ax.set_xlabel("Missing-frame ratio (%)")
        ax.set_ylabel(metric_label)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(-0.6, 20.6)
        lower = float((panel["Mean (%)"] - panel["SD"]).min())
        upper = float((panel["Mean (%)"] + panel["SD"]).max())
        pad = max(0.30, 0.045 * (upper - lower))
        ax.set_ylim(max(0.0, lower - pad), min(100.0, upper + pad))
        path = out_dir / filename
        save(fig, path)
        written.append(path)
    return written


def plot_operational_case_composite(
    *,
    features: np.ndarray,
    clean_features: np.ndarray,
    threat_true: np.ndarray,
    time_axis: np.ndarray,
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    noisy_x: np.ndarray,
    noisy_y: np.ndarray,
    true_first: int,
    timing_rows: list[dict[str, float | int | str]],
    model_curves: dict,
    keep_models: list[str],
    out_dir: Path,
) -> Path:
    event_time = float(time_axis[true_first]) if true_first >= 0 else float("nan")
    time_end = float(time_axis[-1])
    fig = plt.figure(figsize=(7.20, 6.05))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.02, 1.00, 0.96],
        width_ratios=[1.00, 1.08],
        hspace=0.66,
        wspace=0.34,
    )
    ax_traj = fig.add_subplot(grid[0, 0])
    ax_signal = fig.add_subplot(grid[0, 1])
    ax_curve = fig.add_subplot(grid[1, :])
    ax_alarm = fig.add_subplot(grid[2, :])
    for ax in [ax_traj, ax_signal, ax_curve, ax_alarm]:
        style_panel_axis(ax)

    ax_traj.plot(x_pos, y_pos, linewidth=1.75, color="#2f4b66", label="Clean reference", zorder=3)
    ax_traj.scatter(noisy_x, noisy_y, s=9, color="#b46a5a", alpha=0.42, label="Observed samples", zorder=2)
    ax_traj.scatter(x_pos[0], y_pos[0], color="#5d8a6a", s=28, edgecolor="#222222", linewidth=0.35, zorder=5, label="Start")
    ax_traj.scatter(x_pos[-1], y_pos[-1], color="#222222", s=34, edgecolor="white", linewidth=0.35, zorder=5, label="Final")
    if true_first >= 0:
        ax_traj.scatter(
            x_pos[true_first],
            y_pos[true_first],
            marker="*",
            color="#b64c4c",
            edgecolor="#222222",
            linewidth=0.40,
            s=92,
            zorder=6,
            label="_nolegend_",
        )
        ax_traj.annotate(
            f"Critical event\n{event_time:.1f} s",
            xy=(x_pos[true_first], y_pos[true_first]),
            xytext=(8, -21),
            textcoords="offset points",
            fontsize=6.9,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.55},
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
    ax_traj.set_xlabel("Relative x", fontsize=8.0)
    ax_traj.set_ylabel("Relative y", fontsize=8.0)
    ax_traj.legend(
        loc="lower right",
        fontsize=6.2,
        frameon=False,
        handlelength=1.3,
        borderaxespad=0.1,
        ncol=2,
        columnspacing=0.75,
    )
    add_panel_label(ax_traj, "(a)")

    signal_specs = [
        ("Heading", 6, "#6f8f78", 0.035),
        ("Distance", 8, "#4d7fa8", 0.000),
        ("Time-to-arrival", 11, "#b65f4f", -0.035),
    ]
    for feature_name, feature_idx, color, label_offset in signal_specs:
        ax_signal.plot(time_axis, features[:, feature_idx], linewidth=1.65, color=color)
        ax_signal.plot(time_axis, clean_features[:, feature_idx], linewidth=0.85, color=color, alpha=0.35, linestyle="--")
        annotate_series_label(
            ax_signal,
            x=time_end,
            y=float(features[-1, feature_idx]),
            text=feature_name,
            color=color,
            dx=0.18,
            dy=label_offset,
            fontsize=6.8,
            alpha=0.80,
        )
    if true_first >= 0:
        ax_signal.axvline(event_time, color="#333333", linewidth=0.9, linestyle="--")
        ax_signal.text(
            event_time - 0.12,
            0.96,
            "Reference event",
            ha="right",
            va="top",
            fontsize=6.7,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax_signal.set_xlabel("Time (s)", fontsize=8.0)
    ax_signal.set_ylabel("Normalized value", fontsize=8.0)
    ax_signal.set_xlim(float(time_axis[0]) - 0.2, time_end + 1.55)
    ax_signal.set_ylim(0, 1)
    add_panel_label(ax_signal, "(b)")

    ax_curve.axhspan(4, 5.2, color="#b64c4c", alpha=0.075, zorder=0)
    ax_curve.step(time_axis, threat_true, where="post", label="Clean reference", linewidth=2.45, color="#222222", zorder=8)
    for model in keep_models:
        style = OPERATIONAL_MODEL_STYLES.get(model, {"linewidth": 1.7, "alpha": 0.9, "linestyle": "-", "zorder": 3})
        ax_curve.step(
            time_axis,
            model_curves[model]["threat_pred"],
            where="post",
            label=pretty_model_name(model),
            linewidth=style.get("linewidth", 1.7),
            color=style.get("color", None),
            alpha=style.get("alpha", 0.9),
            linestyle=style.get("linestyle", "-"),
            zorder=style.get("zorder", 3),
        )
    if true_first >= 0:
        ax_curve.axvline(event_time, color="#333333", linewidth=0.9, linestyle="--")
    ax_curve.text(time_end - 0.05, 4.82, "Critical zone", ha="right", va="center", fontsize=6.9, color="#7f3f3f")
    ax_curve.set_xlabel("Time (s)", fontsize=8.0)
    ax_curve.set_ylabel("Threat level", fontsize=8.0)
    ax_curve.set_yticks([1, 2, 3, 4, 5])
    ax_curve.set_ylim(0.8, 5.2)
    ax_curve.set_xlim(float(time_axis[0]) - 0.15, time_end + 0.35)
    ax_curve.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        fontsize=6.7,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.95,
        borderaxespad=0.0,
    )
    add_panel_label(ax_curve, "(c)")

    timing_df = pd.DataFrame(timing_rows)
    if not timing_df.empty:
        y_positions = np.arange(len(timing_df))
        if true_first >= 0:
            ax_alarm.axvspan(0.0, event_time, color="#b64c4c", alpha=0.060, zorder=0)
            ax_alarm.axvline(event_time, color="#222222", linewidth=1.0, linestyle="--", zorder=2)
            ax_alarm.text(event_time + 0.10, -0.42, f"Reference {event_time:.1f} s", fontsize=6.8, color="#222222")
        metric_x = time_end + 1.35
        for y_pos_i, row in zip(y_positions, timing_rows):
            model_name = str(row["Model"])
            alarm_t = float(row["Alarm time (s)"])
            false_frames = int(row["False frames"])
            mae = float(row["MAE"])
            ax_alarm.hlines(y_pos_i, 0.0, time_end, color="#d9d9d9", linewidth=3.1, zorder=1)
            if np.isfinite(alarm_t):
                if model_name == "Clean reference":
                    color = "#222222"
                    marker = "D"
                    size = 38
                else:
                    color = OPERATIONAL_MODEL_STYLES.get(model_name, {}).get("color", "#405d7d")
                    marker = "o" if false_frames == 0 else "s"
                    size = 42 if false_frames == 0 else 48
                ax_alarm.scatter(alarm_t, y_pos_i, s=size, color=color, marker=marker, edgecolor="#222222", linewidth=0.40, zorder=4)
                if model_name != "Clean reference":
                    label_x = min(alarm_t + 0.16, time_end - 0.55)
                    label_ha = "left"
                    if alarm_t > time_end - 1.0:
                        label_x = alarm_t - 0.16
                        label_ha = "right"
                    ax_alarm.text(
                        label_x,
                        y_pos_i + 0.20,
                        f"{alarm_t:.1f}s",
                        ha=label_ha,
                        va="bottom",
                        fontsize=6.7,
                        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                    )
            metric_text = "-- / --" if model_name == "Clean reference" else f"{false_frames:d} / {mae:.2f}"
            ax_alarm.text(metric_x, y_pos_i, metric_text, ha="left", va="center", fontsize=6.8, color="#333333")
        ax_alarm.text(metric_x, -0.62, "False frames / MAE", ha="left", va="bottom", fontsize=6.8, color="#333333")
        ax_alarm.set_yticks(y_positions)
        ax_alarm.set_yticklabels([pretty_model_name(name) if name != "Clean reference" else name for name in timing_df["Model"]])
        ax_alarm.set_xlim(-0.45, time_end + 2.55)
        ax_alarm.set_xlabel("First critical-alarm time (s)", fontsize=8.0)
        ax_alarm.set_ylim(len(timing_df) - 0.45, -0.85)
    else:
        ax_alarm.text(0.5, 0.5, "No critical event in this sequence", transform=ax_alarm.transAxes, ha="center", va="center", fontsize=8.0)
        ax_alarm.set_axis_off()
    add_panel_label(ax_alarm, "(d)")

    path = out_dir / "fig_operational_case_composite.pdf"
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.070, top=0.965)
    save(fig, path, tight=False)
    return path


def plot_operational_case_figure(
    case: dict,
    out_dir: Path,
    *,
    timeline_case: dict | None = None,
) -> list[Path]:
    features = np.asarray(case.get("model_input_features", case.get("noisy_features", case["features"])), dtype=np.float64)
    clean_features = np.asarray(case.get("clean_features", features), dtype=np.float64)
    threat_true = np.asarray(case["threat_true"], dtype=np.int64)
    frame_interval = float(case.get("frame_interval", 0.2))
    time_axis = np.arange(len(threat_true)) * frame_interval

    written: list[Path] = []
    model_curves = case.get("models", {})
    keep_models = [model for model in CURVE_MODELS if model in model_curves]
    true_first = first_critical(threat_true, [4, 5])
    source = pd.DataFrame(
        {
            "time_s": time_axis,
            "clean_threat": threat_true,
            "observed_heading": features[:, 6],
            "clean_heading": clean_features[:, 6],
            "observed_distance": features[:, 8],
            "clean_distance": clean_features[:, 8],
            "observed_time_to_arrival": features[:, 11],
            "clean_time_to_arrival": clean_features[:, 11],
        }
    )
    for model in keep_models:
        source[f"{model}_threat"] = np.asarray(model_curves[model]["threat_pred"], dtype=np.int64)
        source[f"{model}_critical"] = (
            np.asarray(model_curves[model]["threat_pred"], dtype=np.int64) >= 4
        ).astype(np.int64)
    source["clean_critical"] = (threat_true >= 4).astype(np.int64)
    source.to_csv(out_dir / "fig_operational_case_source.csv", index=False)

    fig, ax = new_panel_figure(width=3.55, height=2.15, journal=True)
    for feature_name, feature_idx, color, linestyle in [
        ("Distance", 8, "#303030", "-"),
        ("Heading", 6, "#6f6f6f", "--"),
        ("Time-to-arrival", 11, "#9a9a9a", "-."),
    ]:
        ax.plot(
            time_axis,
            features[:, feature_idx],
            label=feature_name,
            linewidth=1.10,
            color=color,
            linestyle=linestyle,
            zorder=3,
        )
    if true_first >= 0:
        ax.axvline(time_axis[true_first], color="#8f1d1d", linewidth=0.85, linestyle=":", zorder=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value")
    ax.set_ylim(0, 1)
    ax.set_xlim(float(time_axis[0]) - 0.2, float(time_axis[-1]) + 0.2)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.3,
        ncol=1,
        handlelength=1.8,
        borderaxespad=0.25,
    )
    path = out_dir / "fig_operational_case_signals.pdf"
    save(fig, path)
    written.append(path)

    timeline_case = timeline_case or case
    timeline_threat_true = np.asarray(timeline_case["threat_true"], dtype=np.int64)
    timeline_frame_interval = float(timeline_case.get("frame_interval", 0.2))
    timeline_time_axis = np.arange(len(timeline_threat_true)) * timeline_frame_interval
    timeline_model_curves = timeline_case.get("models", {})
    timeline_models = [model for model in CURVE_MODELS if model in timeline_model_curves]
    timeline_true_first = first_critical(timeline_threat_true, [4, 5])
    timing_rows = operational_timing_rows(timeline_case)
    selection = timeline_case.get("selection", {})
    timing_source = pd.DataFrame(timing_rows)
    for key, value in selection.items():
        timing_source[f"Selection {key.replace('_', ' ')}"] = value
    timing_source.to_csv(out_dir / "fig_operational_case_timing_source.csv", index=False)

    timeline_source = pd.DataFrame(
        {
            "time_s": timeline_time_axis,
            "clean_threat": timeline_threat_true,
            "clean_critical": (timeline_threat_true >= 4).astype(np.int64),
        }
    )
    for model in timeline_models:
        model_pred = np.asarray(timeline_model_curves[model]["threat_pred"], dtype=np.int64)
        timeline_source[f"{model}_threat"] = model_pred
        timeline_source[f"{model}_critical"] = (model_pred >= 4).astype(np.int64)
    timeline_source.to_csv(out_dir / "fig_operational_case_timeline_source.csv", index=False)

    timeline_rows: list[tuple[str, np.ndarray, str]] = [
        ("Clean reference", timeline_threat_true >= 4, "#303030")
    ]
    for model in timeline_models:
        style = OPERATIONAL_MODEL_STYLES.get(model, {})
        timeline_rows.append(
            (
                pretty_model_name(model),
                np.asarray(timeline_model_curves[model]["threat_pred"], dtype=np.int64) >= 4,
                str(style.get("color", "#707070")),
            )
        )

    def critical_intervals(mask: np.ndarray) -> list[tuple[float, float]]:
        binary = np.asarray(mask, dtype=np.int8)
        changes = np.diff(np.pad(binary, (1, 1), constant_values=0))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        return [
            (
                float(start * timeline_frame_interval),
                float((stop - start) * timeline_frame_interval),
            )
            for start, stop in zip(starts, stops)
        ]

    fig, ax = new_panel_figure(width=7.10, height=2.15, journal=True)
    ax.grid(False)
    ax.grid(axis="x", linewidth=0.42, color="#e3e3e3", alpha=0.85, zorder=0)
    y_positions = np.arange(len(timeline_rows), dtype=float)
    time_end = float(timeline_time_axis[-1] + timeline_frame_interval)
    for y_pos, (label, critical_mask, color) in zip(y_positions, timeline_rows):
        ax.hlines(y_pos, 0.0, time_end, color="#e0e0e0", linewidth=2.0, zorder=1)
        intervals = critical_intervals(critical_mask)
        if intervals:
            ax.broken_barh(
                intervals,
                (y_pos - 0.18, 0.36),
                facecolors=color,
                edgecolors="none",
                alpha=0.98,
                zorder=3,
            )
            first_start = intervals[0][0]
            ax.scatter(
                first_start,
                y_pos,
                s=16,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
    if timeline_true_first >= 0:
        event_time = float(timeline_time_axis[timeline_true_first])
        ax.axvspan(event_time - 0.04, event_time + 0.04, color="#777777", alpha=0.30, linewidth=0, zorder=2)
        ax.text(
            event_time,
            -0.62,
            f"Reference {event_time:.1f} s",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )
    ax.set_xlabel("Time (s)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in timeline_rows], fontsize=7.1)
    ax.set_ylim(len(timeline_rows) - 0.55, -0.78)
    ax.set_xlim(0.0, time_end)
    path = out_dir / "fig_operational_case_timeline.pdf"
    save(fig, path)
    written.append(path)

    if timing_rows:
        timing_df = pd.DataFrame(timing_rows)
        timing_df = timing_df[timing_df["Model"] != "Clean reference"].copy()
        y_positions = np.arange(len(timing_df))

        fig, ax = new_panel_figure(width=3.55, height=2.15, journal=True)
        ax.grid(False)
        ax.grid(axis="x", linewidth=0.42, color="#e5e5e5", alpha=0.85)
        reference_time = float(time_axis[true_first]) if true_first >= 0 else float("nan")
        if true_first >= 0:
            ax.axvline(reference_time, color="#222222", linewidth=1.0, linestyle="--", zorder=5)
        for y_pos_i, row in zip(y_positions, timing_df.to_dict("records")):
            alarm_t = row["Alarm time (s)"]
            model_name = str(row["Model"])
            if np.isfinite(alarm_t):
                color = OPERATIONAL_MODEL_STYLES.get(model_name, {}).get("color", "#405d7d")
                marker = "o" if model_name == "TemporalHGTAN" else "s"
                size = 34 if model_name == "TemporalHGTAN" else 27
                if np.isfinite(reference_time):
                    ax.hlines(
                        y_pos_i,
                        min(float(alarm_t), reference_time),
                        max(float(alarm_t), reference_time),
                        color="#8f8f8f" if model_name != "TemporalHGTAN" else color,
                        linewidth=1.15 if model_name == "TemporalHGTAN" else 0.75,
                        alpha=1.0,
                        zorder=2,
                    )
                ax.scatter(
                    alarm_t,
                    y_pos_i,
                    s=size,
                    facecolor=color if model_name == "TemporalHGTAN" else "white",
                    edgecolor=color,
                    marker=marker,
                    linewidth=0.75,
                    zorder=4,
                )
        ax.set_yticks(y_positions)
        ax.set_yticklabels([compact_model_name(name) for name in timing_df["Model"]])
        ax.set_xlim(float(time_axis[0]) - 0.2, float(time_axis[-1]) + 0.25)
        ax.set_xlabel("First critical-alarm time (s)")
        ax.invert_yaxis()
        path = out_dir / "fig_operational_case_alarm_timing.pdf"
        save(fig, path, top_pad=0.88)
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
    return select_representative_operational_case(cases)


def select_representative_operational_case(cases: list[dict]) -> dict:
    """Select the median HGTAN-versus-GRU MAE gain rather than the best-looking case."""
    ranked: list[tuple[float, dict]] = []
    for case in cases:
        models = case.get("models", {})
        if "TemporalHGTAN" not in models or "TemporalGRU" not in models:
            continue
        threat_true = np.asarray(case["threat_true"], dtype=np.int64)
        if first_critical(threat_true, [4, 5]) < 0:
            continue
        frame_interval = float(case.get("frame_interval", 0.2))
        hgtan_stats = case_alarm_stats(
            threat_true,
            np.asarray(models["TemporalHGTAN"]["threat_pred"], dtype=np.int64),
            frame_interval=frame_interval,
        )
        gru_stats = case_alarm_stats(
            threat_true,
            np.asarray(models["TemporalGRU"]["threat_pred"], dtype=np.int64),
            frame_interval=frame_interval,
        )
        ranked.append((float(gru_stats["mae"] - hgtan_stats["mae"]), case))
    if not ranked:
        return sorted(cases, key=score_operational_case)[len(cases) // 2]
    ranked.sort(key=lambda item: item[0])
    return ranked[len(ranked) // 2][1]


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

    for baseline_name in ["TemporalGRU", "TemporalLSTM", "TemporalHMM", "TOPSIS"]:
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
        "mission_as_input": False,
        "reference_policy_variant": "balanced",
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
            "TemporalGRU": {
                "threat_pred": delayed_curve(threat_seq[case_idx], delay=3),
                "urgency_pred": delayed_curve(urgency_seq[case_idx], delay=3),
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
            "Model": "Clean reference",
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


def resolve_suite_containing(summary: pd.DataFrame, tokens: tuple[str, ...]) -> str | None:
    if summary.empty or "source_suite" not in summary.columns:
        return None
    sources = sorted(str(value) for value in summary["source_suite"].dropna().unique())
    return next((source for source in sources if all(token in source for token in tokens)), None)


def resolve_suite_excluding(
    summary: pd.DataFrame,
    token: str,
    *,
    excluded: tuple[str, ...],
) -> str | None:
    if summary.empty or "source_suite" not in summary.columns:
        return None
    sources = sorted(str(value) for value in summary["source_suite"].dropna().unique())
    return next(
        (
            source
            for source in sources
            if token in source and not any(blocked in source for blocked in excluded)
        ),
        None,
    )


def resolve_observed_time_suite(summary: pd.DataFrame) -> str | None:
    suite = resolve_suite_containing(summary, ("fixed_endpoint", "window"))
    if suite is not None:
        return suite
    suite = resolve_suite_containing(summary, ("fixed_endpoint", "observed"))
    if suite is not None:
        return suite
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
    best_suite = None
    best_count = 0
    for source_suite, group in summary.groupby("source_suite"):
        if "range_m" in group.columns:
            range_values = pd.to_numeric(group["range_m"], errors="coerce").dropna().unique()
            count = len(range_values)
        else:
            settings = group["setting"].dropna().astype(str)
            count = settings[settings.str.contains("range", regex=False)].nunique()
        if count > best_count:
            best_suite = str(source_suite)
            best_count = count
    return best_suite if best_count >= 2 else None


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
        "std": float(row["std"]) if "std" in row and pd.notna(row["std"]) else 0.0,
        "ci95": float(row["ci95"]) if "ci95" in row and pd.notna(row["ci95"]) else 0.0,
    }


def save(fig: plt.Figure, path: Path, *, top_pad: float | None = None, tight: bool = True) -> None:
    if tight:
        if top_pad is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=(0.0, 0.0, 1.0, top_pad))
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
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
    for pattern in ["fig_*.png", "fig_*.pdf", "fig_*.svg", "fig_*_source.csv"]:
        for path in out_dir.glob(pattern):
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
        "fig_overall_final_dynamic_tradeoff": "Final classification versus prefix-level dynamic fidelity",
        "fig_stability_paired_delta": "Paired Composite-F1 differences over ten matched random seeds",
        "fig_classwise_threat_f1": "Class-wise final threat F1 under the default protocol",
        "fig_classwise_urgency_f1": "Class-wise final urgency F1 under the default protocol",
        "fig_policy_paired_margins": "Paired HGTAN margins under three reference policies",
        "fig_holdout_paired_margins": "Paired HGTAN margins under leave-one-scenario-family-out evaluation",
        "fig_observed_time_composite_f1": "Fixed-endpoint history sensitivity in composite F1",
        "fig_observed_time_dynamic_accuracy": "Fixed-endpoint history sensitivity in threat temporal accuracy",
        "fig_distance_degradation_composite_f1": "Range-degradation sensitivity in composite F1",
        "fig_distance_degradation_dynamic_accuracy": "Range-degradation sensitivity in threat temporal accuracy",
        "fig_ablation_default_composite_f1": "Default-condition composite F1 ablation",
        "fig_ablation_fixed_summary_composite_f1": "Fixed-endpoint temporal-summary composite F1 control",
        "fig_ablation_default_temporal_f1": "Default-condition temporal macro-F1 ablation",
        "fig_ablation_fixed_summary_temporal_f1": "Fixed-endpoint temporal-summary temporal macro-F1 control",
        "fig_event_timing_agreement": "First-critical-alarm agreement over clean-reference transition tracks",
        "fig_event_aligned_disagreement": "Critical-decision disagreement aligned to the clean-reference transition",
        "fig_operational_case_signals": "Operational-case key signals",
        "fig_operational_case_timeline": "Representative critical-decision timelines",
        "fig_operational_case_alarm_timing": "Operational-case first critical-alarm timing",
    }
    return captions.get(stem, stem.removeprefix("fig_").replace("_", " ").title())


if __name__ == "__main__":
    main()
