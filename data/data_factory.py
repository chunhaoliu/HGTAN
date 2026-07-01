"""Data provider factory, following the Autoformer/FEDformer style."""

from __future__ import annotations

from typing import Any

from data.data_loader import DataBundle
from data.experiment_pipeline import prepare_experiment_data


def data_provider(
    config: dict[str, Any],
    seed: int,
    *,
    split_strategy: str | None = None,
    noise_level: float | None = None,
    missing_ratio: float | None = None,
    enforce_min_class_samples: bool = True,
) -> DataBundle:
    """Build the complete train/validation/test bundle for one run."""
    data_cfg = config["data"]
    payload = prepare_experiment_data(
        seed,
        config,
        split_strategy=split_strategy or data_cfg.get("split_strategy", "stratified"),
        noise_level=data_cfg.get("noise_level", 0.0) if noise_level is None else noise_level,
        missing_ratio=data_cfg.get("missing_ratio", 0.0) if missing_ratio is None else missing_ratio,
        enforce_min_class_samples=enforce_min_class_samples,
    )
    return DataBundle(payload)
