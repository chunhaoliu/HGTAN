"""External ground-target validation for the Air Target UAV manuscript.

This script intentionally reads the copied dataset inside the Air Target UAV
project tree. It does not depend on any original source directory.
The experiment is a compact engineering sanity check, not a replacement for
the air-target sequential evaluation.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DATA_PATH = CODE_ROOT / "data" / "external" / "ground_target_glass2.xlsx"
OUT_DIR = PROJECT_ROOT / "Outputs" / "atuav_assessment" / "external_ground_validation"
FIG_PATH = (
    PROJECT_ROOT
    / "IEEE_TAES_Manuscript"
    / "figures"
    / "experiments"
    / "fig_external_ground_confusion.pdf"
)


SEEDS = (42, 123, 456, 789, 2026)
LABELS = np.arange(1, 7)
GROUP_SLICES = ((0, 3), (3, 6), (6, 8), (8, 9))
PAIR_INDICES = tuple((i, j) for i in range(9) for j in range(i + 1, 9))
STATIC_REFERENCE_CONFIG_ID = 0
STATIC_TUNING_MARGIN = 0.03
PRIOR_WEIGHTS = np.array([0.14, 0.11, 0.10, 0.12, 0.15, 0.10, 0.10, 0.08, 0.10])
PRIOR_WEIGHTS = PRIOR_WEIGHTS / PRIOR_WEIGHTS.sum()


@dataclass(frozen=True)
class StaticHGTANConfig:
    hidden_dim: int = 24
    dropout: float = 0.08
    lr: float = 1.5e-3
    weight_decay: float = 8e-3
    ordinal_weight: float = 0.05
    input_noise: float = 0.012
    epochs: int = 420
    patience: int = 60


def load_ground_target_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing copied dataset: {path}. Copy the external validation workbook into the Air Target UAV data directory first."
        )
    df = pd.read_excel(path, header=None).dropna(axis=0, how="any")
    if df.shape[1] != 10:
        raise ValueError(f"Expected 10 columns: one label plus nine indicators, got {df.shape[1]}.")
    y = df.iloc[:, 0].astype(int).to_numpy()
    x = df.iloc[:, 1:].astype(float).to_numpy()
    return x, y


def prior_gated_features(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a static prior-gated representation from nine threat indicators.

    The copied ground-target dataset contains static normalized indicators, so
    this adapter keeps the proposed paper's prior-fusion idea but removes the
    temporal blocks that require a track sequence.
    """
    train01, eval01 = unit_interval_features(x_train, x_eval)

    # Distance, velocity, heading, equipment range, firepower, detection,
    # electronic countermeasure, communication, survivability.

    def enrich(x_raw: np.ndarray, x01: np.ndarray) -> np.ndarray:
        confidence = np.clip(np.abs(x01 - 0.5) * 2.0, 0.0, 1.0)
        weighted = x01 * PRIOR_WEIGHTS * x01.shape[1]
        gated = (1.0 - 0.35 * confidence) * x01 + (0.35 * confidence) * weighted
        group_means = np.column_stack(
            [
                x01[:, 0:3].mean(axis=1),
                x01[:, 3:6].mean(axis=1),
                x01[:, 6:8].mean(axis=1),
                x01[:, 8],
            ]
        )
        prior_score = x01 @ PRIOR_WEIGHTS
        reliability = confidence.mean(axis=1)
        return np.column_stack([x_raw, gated, group_means, prior_score, reliability])

    return enrich(x_train, train01), enrich(x_eval, eval01)


def unit_interval_features(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = x_train.min(axis=0)
    hi = x_train.max(axis=0)
    span = np.where(np.abs(hi - lo) < 1e-8, 1.0, hi - lo)

    def transform(x: np.ndarray) -> np.ndarray:
        return np.clip((x - lo) / span, 0.0, 1.0).astype(np.float32)

    return transform(x_train), transform(x_eval)


class StaticHGTAN(nn.Module):
    """Static adaptation of Temporal HGTAN for non-sequential indicator samples."""

    def __init__(self, hidden_dim: int = 24, dropout: float = 0.08) -> None:
        super().__init__()
        self.register_buffer(
            "prior_weights",
            torch.tensor(PRIOR_WEIGHTS, dtype=torch.float32).view(1, -1),
        )
        self.reliability_gate = nn.Sequential(
            nn.Linear(18, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 9),
            nn.Sigmoid(),
        )
        self.group_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(end - start, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
                for start, end in GROUP_SLICES
            ]
        )
        self.group_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.group_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 14 + len(PAIR_INDICES), 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_prior = x * self.prior_weights * x.shape[-1]
        gate = self.reliability_gate(torch.cat([x, weighted_prior], dim=-1))
        corrected = (1.0 - gate) * x + gate * weighted_prior

        group_tokens = []
        group_scores = []
        for encoder, (start, end) in zip(self.group_encoders, GROUP_SLICES):
            group_x = corrected[:, start:end]
            group_tokens.append(encoder(group_x))
            group_scores.append(group_x.mean(dim=1, keepdim=True))
        tokens = torch.stack(group_tokens, dim=1)
        attended, _ = self.group_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.group_norm(tokens + attended)
        pooled_mean = tokens.mean(dim=1)
        pooled_max = tokens.max(dim=1).values
        prior_score = (corrected * self.prior_weights).sum(dim=1, keepdim=True)
        aux = torch.cat(group_scores + [prior_score], dim=1)
        pairwise = torch.cat(
            [corrected[:, i : i + 1] * corrected[:, j : j + 1] for i, j in PAIR_INDICES],
            dim=1,
        )
        return self.head(torch.cat([pooled_mean, pooled_max, corrected, aux, pairwise], dim=1))


def static_validation_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred)
    return float(macro_f1 + 0.08 * qwk - 0.03 * mae)


def fit_static_hgtan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    config: StaticHGTANConfig,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> StaticHGTAN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    model = StaticHGTAN(hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train - 1, dtype=torch.long, device=device)
    x_val_tensor = None if x_val is None else torch.tensor(x_val, dtype=torch.float32, device=device)

    counts = np.bincount(y_train - 1, minlength=6).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = None
    best_score = -float("inf")
    best_loss = float("inf")
    stale = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if config.input_noise > 0:
            x_epoch = torch.clamp(x_tensor + torch.randn_like(x_tensor) * config.input_noise, 0.0, 1.0)
        else:
            x_epoch = x_tensor
        logits = model(x_epoch)
        ce_loss = F.cross_entropy(logits, y_tensor, weight=weight_tensor)
        probs = F.softmax(logits, dim=1)
        expected = torch.sum(probs * torch.arange(6, dtype=torch.float32, device=device), dim=1)
        ordinal_loss = F.smooth_l1_loss(expected, y_tensor.float())
        loss = ce_loss + config.ordinal_weight * ordinal_loss
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        improved = False
        if x_val_tensor is not None and y_val is not None and (epoch % 5 == 0 or epoch == config.epochs - 1):
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_tensor).argmax(dim=1).cpu().numpy() + 1
            score = static_validation_score(y_val, val_pred)
            if score > best_score + 1e-5:
                best_score = score
                improved = True
        elif x_val_tensor is None:
            if loss_value + 1e-5 < best_loss:
                best_loss = loss_value
                improved = True
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 160 and stale >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_static_hgtan(model: StaticHGTAN, x_eval: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x_eval_tensor).argmax(dim=1).cpu().numpy() + 1
    return pred.astype(int)


def train_static_hgtan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    config: StaticHGTANConfig,
) -> np.ndarray:
    model = fit_static_hgtan(x_train, y_train, seed=seed, config=config)
    return predict_static_hgtan(model, x_test)


def static_hgtan_search_space() -> list[StaticHGTANConfig]:
    base = StaticHGTANConfig()
    return [
        base,
        replace(base, hidden_dim=16, dropout=0.05, lr=2.0e-3, weight_decay=5e-3, ordinal_weight=0.03, input_noise=0.008),
        replace(base, hidden_dim=16, dropout=0.08, lr=1.5e-3, weight_decay=8e-3, ordinal_weight=0.05, input_noise=0.0),
        replace(base, hidden_dim=24, dropout=0.05, lr=1.5e-3, weight_decay=5e-3, ordinal_weight=0.05, input_noise=0.010),
        replace(base, hidden_dim=24, dropout=0.03, lr=1.0e-3, weight_decay=3e-3, ordinal_weight=0.03, input_noise=0.0),
        replace(base, hidden_dim=24, dropout=0.12, lr=1.0e-3, weight_decay=1.0e-2, ordinal_weight=0.05, input_noise=0.015),
        replace(base, hidden_dim=32, dropout=0.05, lr=1.0e-3, weight_decay=8e-3, ordinal_weight=0.03, input_noise=0.010),
        replace(base, hidden_dim=32, dropout=0.08, lr=1.5e-3, weight_decay=8e-3, ordinal_weight=0.05, input_noise=0.012),
        replace(base, hidden_dim=32, dropout=0.12, lr=1.5e-3, weight_decay=1.2e-2, ordinal_weight=0.08, input_noise=0.010),
        replace(base, hidden_dim=48, dropout=0.08, lr=1.0e-3, weight_decay=1.0e-2, ordinal_weight=0.03, input_noise=0.012),
    ]


def tune_static_hgtan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[StaticHGTANConfig, list[dict[str, float | int]]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=seed + 1009)
    inner_train_idx, val_idx = next(splitter.split(x_train, y_train))
    x_inner, y_inner = x_train[inner_train_idx], y_train[inner_train_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    best_config: StaticHGTANConfig | None = None
    best_score = -float("inf")
    tuning_rows: list[dict[str, float | int]] = []

    for config_id, config in enumerate(static_hgtan_search_space()):
        model = fit_static_hgtan(x_inner, y_inner, seed=seed + config_id * 17, config=config, x_val=x_val, y_val=y_val)
        pred = predict_static_hgtan(model, x_val)
        row = {
            "seed": seed,
            "config_id": config_id,
            "val_score": static_validation_score(y_val, pred),
            "val_accuracy": accuracy_score(y_val, pred),
            "val_macro_f1": f1_score(y_val, pred, labels=LABELS, average="macro", zero_division=0),
            "val_ordinal_mae": mean_absolute_error(y_val, pred),
            "val_quadratic_kappa": cohen_kappa_score(y_val, pred, labels=LABELS, weights="quadratic"),
            **asdict(config),
        }
        tuning_rows.append(row)
        if row["val_score"] > best_score:
            best_score = float(row["val_score"])
            best_config = config

    if best_config is None:
        best_config = StaticHGTANConfig()
    final_config = replace(best_config, epochs=760, patience=90)
    return final_config, tuning_rows


def build_models(seed: int) -> dict[str, object]:
    return {
        "Logistic regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "SVM-RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=8.0, gamma="scale", class_weight="balanced")),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Gaussian NB": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        ),
        "LDA": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="lbfgs",
                        alpha=5e-3,
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Prior-gated MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(48, 16),
                        activation="relu",
                        solver="lbfgs",
                        alpha=5e-3,
                        max_iter=2500,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def metric_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, seed: int) -> dict[str, float | int | str]:
    critical_mask = y_true >= 5
    critical_recall = float(np.mean(y_pred[critical_mask] >= 5)) if critical_mask.any() else float("nan")
    severe_under = float(np.mean(y_pred[critical_mask] <= 3)) if critical_mask.any() else float("nan")
    return {
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "ordinal_mae": mean_absolute_error(y_true, y_pred),
        "quadratic_kappa": cohen_kappa_score(y_true, y_pred, labels=LABELS, weights="quadratic"),
        "critical_recall": critical_recall,
        "severe_underestimate": severe_under,
    }


def summarize(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    metrics = [
        "accuracy",
        "macro_f1",
        "ordinal_mae",
        "quadratic_kappa",
        "critical_recall",
        "severe_underestimate",
    ]
    parts = []
    for model, group in raw.groupby("model", sort=False):
        row: dict[str, float | str] = {"model": model}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        parts.append(row)
    return pd.DataFrame(parts), raw


def plot_confusion(y_true: list[int], y_pred: list[int], model_name: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(3.85, 3.25))
    im = ax.imshow(cm_norm, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels([f"L{i}" for i in LABELS])
    ax.set_yticklabels([f"L{i}" for i in LABELS])
    ax.set_xlabel("Predicted level", fontsize=9)
    ax.set_ylabel("True level", fontsize=9)
    ax.tick_params(labelsize=8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.55 else "#1f2933"
            value = cm_norm[i, j] * 100.0
            ax.text(
                j,
                i,
                f"{value:.0f}" if value >= 1.0 else "0",
                ha="center",
                va="center",
                fontsize=7.3,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row recall", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.text(
        0.0,
        1.04,
        "Cell values are row percentages",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#4a4a4a",
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run(data_path: Path, out_dir: Path, fig_path: Path) -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    x, y = load_ground_target_dataset(data_path)

    rows: list[dict[str, float | int | str]] = []
    tuning_rows: list[dict[str, float | int]] = []
    selected_rows: list[dict[str, float | int]] = []
    static_splits: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    predictions: dict[str, dict[str, list[int]]] = {}

    for seed in SEEDS:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, test_idx = next(splitter.split(x, y))
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        x_train_pg, x_test_pg = prior_gated_features(x_train, x_test)
        x_train_static, x_test_static = unit_interval_features(x_train, x_test)

        for model_name, model in build_models(seed).items():
            if model_name == "Prior-gated MLP":
                model.fit(x_train_pg, y_train)
                pred = model.predict(x_test_pg)
            else:
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
            predictions.setdefault(model_name, {"true": [], "pred": []})
            predictions[model_name]["true"].extend(y_test.tolist())
            predictions[model_name]["pred"].extend(pred.tolist())
            rows.append(metric_row(model_name, y_test, pred, seed))

        _, seed_tuning_rows = tune_static_hgtan(x_train_static, y_train, seed)
        tuning_rows.extend(seed_tuning_rows)
        static_splits.append((seed, x_train_static, y_train, x_test_static, y_test))

    tuning_df = pd.DataFrame(tuning_rows)
    mean_scores = tuning_df.groupby("config_id")["val_score"].mean()
    reference_score = float(mean_scores.loc[STATIC_REFERENCE_CONFIG_ID])
    candidate_config_id = int(mean_scores.idxmax())
    candidate_score = float(mean_scores.loc[candidate_config_id])
    if candidate_score >= reference_score + STATIC_TUNING_MARGIN:
        global_config_id = candidate_config_id
        selection_strategy = "global_inner_validation_mean"
    else:
        global_config_id = STATIC_REFERENCE_CONFIG_ID
        selection_strategy = "reference_guarded_inner_validation"
    global_config = replace(static_hgtan_search_space()[global_config_id], epochs=760, patience=90)

    for seed, x_train_static, y_train, x_test_static, y_test in static_splits:
        model_name = "HGTAN-Static (Ours)"
        seed_selected = tuning_df[(tuning_df["seed"] == seed) & (tuning_df["config_id"] == global_config_id)]
        if not seed_selected.empty:
            selected_row = seed_selected.iloc[0].to_dict()
            selected_row["selection_strategy"] = selection_strategy
            selected_row["reference_config_id"] = STATIC_REFERENCE_CONFIG_ID
            selected_row["reference_mean_val_score"] = reference_score
            selected_row["candidate_config_id"] = candidate_config_id
            selected_row["candidate_mean_val_score"] = candidate_score
            selected_row["tuning_margin"] = STATIC_TUNING_MARGIN
            selected_rows.append(selected_row)
        pred = train_static_hgtan(x_train_static, y_train, x_test_static, seed, global_config)
        predictions.setdefault(model_name, {"true": [], "pred": []})
        predictions[model_name]["true"].extend(y_test.tolist())
        predictions[model_name]["pred"].extend(pred.tolist())
        rows.append(metric_row(model_name, y_test, pred, seed))

    summary, raw = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "external_ground_validation_summary.csv", index=False)
    raw.to_csv(out_dir / "external_ground_validation_runs.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out_dir / "external_ground_validation_static_tuning.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_dir / "external_ground_validation_static_selected.csv", index=False)
    figure_model = "HGTAN-Static (Ours)"
    if figure_model not in predictions:
        figure_model = str(summary.sort_values("macro_f1_mean", ascending=False).iloc[0]["model"])
    figure_predictions = predictions[figure_model]
    plot_confusion(figure_predictions["true"], figure_predictions["pred"], figure_model, fig_path)

    print(f"Dataset: {data_path}")
    print(f"Samples: {len(y)}, indicators: {x.shape[1]}, labels: {sorted(np.unique(y).tolist())}")
    print(f"Summary: {out_dir / 'external_ground_validation_summary.csv'}")
    print(f"Figure: {fig_path}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fig", type=Path, default=FIG_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.out_dir, args.fig)
