"""Model registry and construction helpers for ATUAV threat assessment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models.baselines import CNNBaseline, LSTMBaseline, MLPBaseline, ResNetBaseline, TransformerBaseline
from models.graph_baselines import GATBaseline, GCNBaseline, GraphSAGEBaseline
from models.hgtan import (
    HGTAN,
    TemporalHGTAN,
    TemporalHGTANLastFrame,
    TemporalHGTANMeanPool,
    TemporalHGTANNoPrior,
    TemporalHGTANNoSynergy,
)
from models.hgtan_v2 import (
    TemporalHGTANV2,
    TemporalHGTANV2Core,
    TemporalHGTANV2CoreNoGrouping,
    TemporalHGTANV2CoreNoTemporal,
    TemporalHGTANV2NoReliability,
    TemporalHGTANV2NoSynergy,
    TemporalHGTANV2NoTemporal,
)
from models.temporal_baselines import (
    FlatSequenceMLPBaseline,
    LastFrameMLPBaseline,
    MeanPoolMLPBaseline,
    TemporalGRUBaseline,
    TemporalLSTMBaseline,
    TemporalTCNBaseline,
    TemporalTransformerBaseline,
)
from models.traditional_baselines import TRADITIONAL_MODEL_NAMES, get_traditional_models
from utils.config import N_FEATURES

ModelBuilder = Callable[[dict[str, Any]], Any]

STANDARD_MODEL_REGISTRY = {
    "MLP": MLPBaseline,
    "Transformer": TransformerBaseline,
    "LSTM": LSTMBaseline,
    "CNN": CNNBaseline,
    "ResNet": ResNetBaseline,
    "GCN": GCNBaseline,
    "GAT": GATBaseline,
    "GraphSAGE": GraphSAGEBaseline,
}

SEQUENTIAL_MODEL_REGISTRY = {
    "LastFrameMLP": LastFrameMLPBaseline,
    "MeanPoolMLP": MeanPoolMLPBaseline,
    "FlatSequenceMLP": FlatSequenceMLPBaseline,
    "TemporalGRU": TemporalGRUBaseline,
    "TemporalLSTM": TemporalLSTMBaseline,
    "TemporalTransformer": TemporalTransformerBaseline,
    "TemporalTCN": TemporalTCNBaseline,
}

HGTAN_MODEL_REGISTRY = {
    "HGTAN": HGTAN,
    "TemporalHGTAN": TemporalHGTAN,
    "TemporalHGTAN_LastFrame": TemporalHGTANLastFrame,
    "TemporalHGTAN_MeanPool": TemporalHGTANMeanPool,
    "TemporalHGTAN_NoPrior": TemporalHGTANNoPrior,
    "TemporalHGTAN_NoSynergy": TemporalHGTANNoSynergy,
    "TemporalHGTANV2": TemporalHGTANV2,
    "TemporalHGTANV2_Core": TemporalHGTANV2Core,
    "TemporalHGTANV2_CoreNoTemporal": TemporalHGTANV2CoreNoTemporal,
    "TemporalHGTANV2_CoreNoGrouping": TemporalHGTANV2CoreNoGrouping,
    "TemporalHGTANV2_NoReliability": TemporalHGTANV2NoReliability,
    "TemporalHGTANV2_NoTemporal": TemporalHGTANV2NoTemporal,
    "TemporalHGTANV2_NoSynergy": TemporalHGTANV2NoSynergy,
}

SEQUENTIAL_MODELS = frozenset(SEQUENTIAL_MODEL_REGISTRY) | frozenset(
    name for name in HGTAN_MODEL_REGISTRY if name.startswith("Temporal")
)


def _build_standard_model(model_class: type[Any], model_cfg: dict[str, Any]) -> Any:
    del model_cfg
    return model_class(num_features=N_FEATURES)


def _build_hgtan_model(model_class: type[Any], model_cfg: dict[str, Any]) -> Any:
    return model_class(
        num_features=N_FEATURES,
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        use_prior_weights=model_cfg.get("use_prior_weights", False),
        prior_alpha=model_cfg.get("prior_weight_alpha", 0.3),
    )


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    **{
        name: (lambda model_cfg, model_class=model_class: _build_standard_model(model_class, model_cfg))
        for name, model_class in STANDARD_MODEL_REGISTRY.items()
    },
    **{
        name: (lambda model_cfg, model_class=model_class: _build_standard_model(model_class, model_cfg))
        for name, model_class in SEQUENTIAL_MODEL_REGISTRY.items()
    },
    **{
        name: (lambda model_cfg, model_class=model_class: _build_hgtan_model(model_class, model_cfg))
        for name, model_class in HGTAN_MODEL_REGISTRY.items()
    },
}
TRAINABLE_MODELS = frozenset(MODEL_BUILDERS)


def build_model(name: str, model_cfg: dict[str, Any]):
    """Construct a neural assessment model by name."""
    if name not in MODEL_BUILDERS:
        valid_text = ", ".join(sorted(TRAINABLE_MODELS))
        raise ValueError(f"Unknown trainable model={name!r}. Valid trainable models: {valid_text}")
    return MODEL_BUILDERS[name](model_cfg)


def get_selected_traditional_models(model_names: list[str]) -> dict[str, Any]:
    """Return instantiated traditional baselines selected by name."""
    traditional_models = get_traditional_models()
    return {name: model for name, model in traditional_models.items() if name in model_names}


def get_selected_trainable_models(model_names: list[str]) -> list[str]:
    """Return selected trainable model names while preserving CLI order."""
    return _select_known_names(model_names, TRAINABLE_MODELS)


def get_selected_sequential_models(model_names: list[str]) -> list[str]:
    """Return selected models that support sequence tensors."""
    return _select_known_names(model_names, SEQUENTIAL_MODELS)


def validate_model_names(model_names: list[str]) -> list[str]:
    """Validate model names after model-group expansion."""
    valid = set(TRADITIONAL_MODEL_NAMES) | set(TRAINABLE_MODELS)
    unknown = [name for name in model_names if name not in valid]
    if unknown:
        valid_text = ", ".join(sorted(valid))
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}. Valid options: {valid_text}")
    return model_names


def _select_known_names(model_names: list[str], valid_names: set[str] | frozenset[str]) -> list[str]:
    return [name for name in model_names if name in valid_names]
