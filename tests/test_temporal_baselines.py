import torch

from exp.registry import MODEL_GROUPS
from models.model_factory import build_model
from utils.config import HGTANConfig, N_CLASSES, N_FEATURES, N_URGENCY


def test_reviewer_requested_temporal_baselines_are_registered_and_forward():
    cfg = HGTANConfig.get_config()["model"]
    x = torch.randn(3, 64, N_FEATURES)

    for name in ["FlatSequenceMLP", "TemporalTransformer", "TemporalTCN"]:
        model = build_model(name, cfg)
        threat_logits, urgency_logits = model(x)

        assert threat_logits.shape == (3, N_CLASSES)
        assert urgency_logits.shape == (3, N_URGENCY)


def test_flat_sequence_mlp_accepts_prefix_windows_for_dynamic_metrics():
    cfg = HGTANConfig.get_config()["model"]
    model = build_model("FlatSequenceMLP", cfg)

    for frames in [1, 32, 64, 80]:
        x = torch.randn(3, frames, N_FEATURES)
        threat_logits, urgency_logits = model(x)

        assert threat_logits.shape == (3, N_CLASSES)
        assert urgency_logits.shape == (3, N_URGENCY)


def test_seq_main_contains_reviewer_requested_baselines():
    seq_main = MODEL_GROUPS["seq_main"]

    assert "FlatSequenceMLP" in seq_main
    assert "TemporalTransformer" in seq_main
    assert "TemporalTCN" in seq_main
