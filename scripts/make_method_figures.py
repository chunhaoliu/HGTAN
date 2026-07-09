"""Generate method figures for the ATUAV threat-assessment manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
MANUSCRIPT_DIR = PROJECT_ROOT / "IEEE_TAES_Manuscript"
METHOD_DIR = PROJECT_ROOT / "IEEE_TAES_Manuscript" / "figures" / "method"

COLORS = {
    "clean": "#dbe9f6",
    "clean_edge": "#2f5f8f",
    "obs": "#fbe3cf",
    "obs_edge": "#a75f23",
    "cap": "#5b8cc0",
    "intent": "#6aa77a",
    "opp": "#d09b45",
    "ctx": "#b86a6a",
    "neutral": "#f4f4f4",
    "dark": "#303030",
}


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pipeline(MANUSCRIPT_DIR / "overall_pipeline_revised.pdf")
    plot_prior_fusion(MANUSCRIPT_DIR / "innovation1_prior_fusion_revised.pdf")


def box(ax, xy, w, h, text, *, fc, ec="#444444", fontsize=8.5, lw=1.25, rounded=True):
    patch_cls = patches.FancyBboxPatch if rounded else patches.Rectangle
    kwargs = {
        "xy": xy,
        "width": w,
        "height": h,
        "facecolor": fc,
        "edgecolor": ec,
        "linewidth": lw,
    }
    if rounded:
        kwargs.update({"boxstyle": "round,pad=0.018,rounding_size=0.025"})
    patch = patch_cls(**kwargs)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, *, color="#333333", lw=1.2, style="-|>"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": style, "lw": lw, "color": color, "shrinkA": 4, "shrinkB": 4},
    )


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=500, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.95, "Sequential Multi-Feature UAV Threat-Assessment Pipeline", fontsize=12, weight="bold")

    clean_y = 0.66
    obs_y = 0.24
    clean_steps_x = [0.035, 0.205, 0.375, 0.545]
    obs_steps_x = [0.035, 0.180, 0.325, 0.470, 0.650, 0.805, 0.920]
    w, h = 0.115, 0.12

    box(ax, (clean_steps_x[0], clean_y), w, h, "Operational\nscenario", fc=COLORS["clean"], ec=COLORS["clean_edge"])
    box(ax, (clean_steps_x[1], clean_y), w, h, "Clean\ntrajectory", fc=COLORS["clean"], ec=COLORS["clean_edge"])
    box(ax, (clean_steps_x[2], clean_y), w, h, "Fixed label\nrules", fc=COLORS["clean"], ec=COLORS["clean_edge"])
    box(ax, (clean_steps_x[3], clean_y), w, h, "Threat / urgency\nlabels", fc=COLORS["clean"], ec=COLORS["clean_edge"])
    for i in range(3):
        arrow(ax, (clean_steps_x[i] + w, clean_y + h / 2), (clean_steps_x[i + 1], clean_y + h / 2), color=COLORS["clean_edge"])

    box(ax, (obs_steps_x[0], obs_y), w, h, "Operational\nscenario", fc=COLORS["obs"], ec=COLORS["obs_edge"])
    box(ax, (obs_steps_x[1], obs_y), w, h, "Range/environment\nsensor degradation", fc=COLORS["obs"], ec=COLORS["obs_edge"], fontsize=7.8)
    box(ax, (obs_steps_x[2], obs_y), w, h, "Noisy observed\ntrack", fc=COLORS["obs"], ec=COLORS["obs_edge"])
    box(ax, (obs_steps_x[3], obs_y), w, h, "Reliability-gated\nprior fusion", fc="#f2efe8", ec="#7a6a3d", fontsize=7.8)
    box(ax, (obs_steps_x[5], obs_y), w, h, "Adaptive temporal\nevidence fusion", fc="#eef4fb", ec="#4a6f93", fontsize=7.6)
    box(ax, (obs_steps_x[6], obs_y), 0.070, h, "Threat /\nurgency", fc="#eef0f6", ec="#4d5676", fontsize=7.4)
    for i in range(3):
        arrow(ax, (obs_steps_x[i] + w, obs_y + h / 2), (obs_steps_x[i + 1], obs_y + h / 2), color=COLORS["obs_edge"])

    group_y = 0.095
    group_x = [0.565, 0.645, 0.725, 0.805]
    group_labels = [
        ("Capability", COLORS["cap"]),
        ("Intent", COLORS["intent"]),
        ("Opportunity", COLORS["opp"]),
        ("Context", COLORS["ctx"]),
    ]
    for x, (label, color) in zip(group_x, group_labels):
        box(ax, (x, group_y), 0.072, 0.080, label, fc=color, ec="#333333", fontsize=6.9)
    ax.text(0.575, 0.193, "group-specific encoding", fontsize=8, color="#333333")
    for x in group_x:
        arrow(ax, (obs_steps_x[3] + w, obs_y + h / 2), (x + 0.036, group_y + 0.080), color="#555555", lw=0.85)
        arrow(ax, (x + 0.036, group_y + 0.080), (obs_steps_x[5], obs_y + h / 2), color="#555555", lw=0.85)
    box(ax, (0.785, 0.49), 0.105, 0.105, "Group-synergy\nreasoning", fc="#f4e9e9", ec="#8c4d4d", fontsize=7.5)
    arrow(ax, (obs_steps_x[5] + w, obs_y + h / 2), (0.785, 0.535), color="#555555")
    arrow(ax, (0.890, 0.535), (obs_steps_x[6], obs_y + h / 2), color="#4d5676")

    ax.plot([0.03, 0.95], [0.54, 0.54], color="#bbbbbb", lw=0.8, linestyle="--")
    ax.text(0.03, 0.80, "Clean state: used only for labels and event sequence", fontsize=8.5, color=COLORS["clean_edge"])
    ax.text(0.03, 0.405, "Observed state: decision-model input; target-type feature is masked", fontsize=8.5, color=COLORS["obs_edge"])
    ax.text(0.055, 0.53, "64 frames, 0.2 s interval; prior fusion occurs before group-specific encoding", fontsize=8.2)
    save(fig, path)


def plot_prior_fusion(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.93, "Reliability-Gated Prior Fusion", fontsize=11.5, weight="bold")

    box(ax, (0.05, 0.58), 0.18, 0.16, "Noisy observed\nindicator $x_{t,f}$", fc="#fbe3cf", ec="#a75f23")
    box(ax, (0.05, 0.24), 0.18, 0.16, "Expert prior\nweight $\\bar{w}_f$", fc="#f2efe8", ec="#7a6a3d")
    box(ax, (0.33, 0.50), 0.18, 0.20, "Reliability gate\n$\\lambda_{t,f}$", fc="#eef4fb", ec="#4a6f93")
    box(ax, (0.33, 0.18), 0.18, 0.18, "Weighted prior\nform $\\bar{w}_f x_{t,f}$", fc="#f7f1de", ec="#7a6a3d", fontsize=8.1)
    box(ax, (0.61, 0.37), 0.19, 0.19, "Residual prior\ncorrection", fc="#e7f0e9", ec="#4c7a57")
    box(ax, (0.86, 0.39), 0.11, 0.15, "Corrected\nfeature\n$\\tilde{x}_{t,f}$", fc="#f6f6f6", ec="#444444", fontsize=7.8)

    arrow(ax, (0.23, 0.66), (0.33, 0.60), color="#a75f23")
    arrow(ax, (0.23, 0.32), (0.33, 0.27), color="#7a6a3d")
    arrow(ax, (0.51, 0.60), (0.61, 0.50), color="#4a6f93")
    arrow(ax, (0.51, 0.27), (0.61, 0.43), color="#7a6a3d")
    arrow(ax, (0.80, 0.47), (0.86, 0.465), color="#4c7a57")

    ax.text(0.61, 0.21, r"$\tilde{x}_{t,f}=x_{t,f}+\rho_f\lambda_{t,f}(\bar{w}_fx_{t,f}-x_{t,f})$", fontsize=9.3)
    ax.text(0.06, 0.08, "The prior is a gated residual correction, not a hard feature reweighting.", fontsize=8.3, color="#333333")
    save(fig, path)


def plot_static_temporal(path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 6.9), sharex=True)
    t = np.arange(64) * 0.2
    true_score = 0.32 + 0.12 / (1 + np.exp(-(t - 5.4) * 1.8)) + 0.24 / (1 + np.exp(-(t - 7.1) * 2.1))
    rng = np.random.default_rng(7)
    noisy = true_score + 0.035 * rng.standard_normal(len(t))
    thresholds = [0.44, 0.64]

    panels = [
        ("A. Static MADM/TOPSIS: frame-wise thresholding", noisy, "#9b9b9b", "threshold flutter"),
        ("B. HMM: one-step temporal inertia", smooth(noisy, 0.72), "#c7833d", "limited memory"),
        ("C. Temporal HGTAN: multi-frame semantic evidence", smooth(true_score + 0.010 * rng.standard_normal(len(t)), 0.88), "#b34d4d", "persistent maneuver evidence"),
    ]
    for ax, (title, curve, color, callout) in zip(axes, panels):
        ax.plot(t, noisy, color="#c7c7c7", lw=1.0, alpha=0.75, label="Noisy observed score")
        ax.plot(t, curve, color=color, lw=2.3, label=title.split(":")[0])
        for thr in thresholds:
            ax.axhline(thr, color="#444444", linestyle="--", lw=0.8)
        ax.axvspan(5.6, 7.4, color="#f3d7d7", alpha=0.45)
        ax.text(0.01, 0.86, title, transform=ax.transAxes, fontsize=10, weight="bold")
        ax.text(0.72, 0.16, callout, transform=ax.transAxes, fontsize=8.6, bbox={"fc": "white", "ec": "#bbbbbb"})
        ax.set_ylim(0.25, 0.82)
        ax.set_ylabel("Threat score")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Why Sequential Evidence Is Needed Under Noisy Tracks", fontsize=12, weight="bold", y=0.995)
    save(fig, path)


def smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1 - alpha) * x[i]
    return y


def plot_hgtan_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.94, "Temporal Heterogeneous Threat Assessment Network", fontsize=12, weight="bold")

    box(ax, (0.03, 0.43), 0.10, 0.16, "$T\\times16$\nnoisy indicators", fc="#f7f7f7", ec="#333333")
    box(ax, (0.17, 0.75), 0.11, 0.095, "Capability", fc=COLORS["cap"], ec="#333333", fontsize=8)
    box(ax, (0.17, 0.60), 0.11, 0.095, "Intent", fc=COLORS["intent"], ec="#333333", fontsize=8)
    box(ax, (0.17, 0.45), 0.11, 0.095, "Opportunity", fc=COLORS["opp"], ec="#333333", fontsize=8)
    box(ax, (0.17, 0.30), 0.11, 0.095, "Context", fc=COLORS["ctx"], ec="#333333", fontsize=8)
    for y in [0.797, 0.647, 0.497, 0.347]:
        arrow(ax, (0.13, 0.51), (0.17, y), color="#555555", lw=0.9)

    box(ax, (0.33, 0.54), 0.13, 0.19, "Reliability-gated\nprior fusion\n$residual$", fc="#f2efe8", ec="#7a6a3d")
    box(ax, (0.33, 0.29), 0.13, 0.14, "Group-specific\nencoder", fc="#eeeeee", ec="#555555")
    for y in [0.797, 0.647, 0.497, 0.347]:
        arrow(ax, (0.28, y), (0.33, 0.635), color="#555555", lw=0.9)
    arrow(ax, (0.395, 0.54), (0.395, 0.43), color="#555555")

    box(ax, (0.52, 0.69), 0.11, 0.10, "Attention\n evidence", fc="#eef4fb", ec="#4a6f93", fontsize=8)
    box(ax, (0.52, 0.52), 0.11, 0.10, "Mean trend\n evidence", fc="#eef4fb", ec="#4a6f93", fontsize=8)
    box(ax, (0.52, 0.35), 0.11, 0.10, "Last-frame\n evidence", fc="#eef4fb", ec="#4a6f93", fontsize=8)
    box(ax, (0.67, 0.50), 0.11, 0.13, "Adaptive\n evidence gate\n$\\pi_g$", fc="#e7f0e9", ec="#4c7a57", fontsize=8)
    for y in [0.74, 0.57, 0.40]:
        arrow(ax, (0.46, 0.50), (0.52, y), color="#555555", lw=0.9)
        arrow(ax, (0.63, y), (0.67, 0.565), color="#555555", lw=0.9)

    box(ax, (0.81, 0.55), 0.105, 0.17, "Group-synergy\nattention\n$4\\times4$", fc="#f4e9e9", ec="#8c4d4d", fontsize=8)
    box(ax, (0.81, 0.32), 0.105, 0.13, "Shared state\n$\\mathbf{r}$", fc="#f6f6f6", ec="#555555", fontsize=8)
    arrow(ax, (0.78, 0.565), (0.81, 0.635), color="#555555")
    arrow(ax, (0.862, 0.55), (0.862, 0.45), color="#555555")

    box(ax, (0.94, 0.60), 0.055, 0.10, "Threat\n5 levels", fc="#f7eeee", ec="#9a4c4c", fontsize=7.3)
    box(ax, (0.94, 0.37), 0.055, 0.10, "Urgency\n3 levels", fc="#eef3f7", ec="#4d6a8c", fontsize=7.3)
    arrow(ax, (0.915, 0.385), (0.94, 0.42), color="#555555")
    arrow(ax, (0.915, 0.385), (0.94, 0.65), color="#555555")

    ax.text(0.34, 0.79, "soft expert prior, not hard weighting", fontsize=8.3, color="#6a5c35")
    ax.text(0.52, 0.83, "history, trend, and current engagement state", fontsize=8.3, color="#416783")
    ax.text(0.79, 0.76, "capability x intent x opportunity x context", fontsize=8.3, color="#784848")
    save(fig, path)


if __name__ == "__main__":
    main()
