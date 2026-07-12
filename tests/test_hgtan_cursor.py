"""Unit checks for TemporalHGTAN_Cursor forward shapes."""

from models.hgtan_cursor import (
    TemporalHGTANCursor,
    TemporalHGTANCursorDeep,
    TemporalHGTANCursorNoReliability,
    TemporalHGTANCursorWithMix,
)
import torch


def test_cursor_variants_return_dual_logits() -> None:
    x = torch.randn(4, 32, 16)
    for cls in (
        TemporalHGTANCursor,
        TemporalHGTANCursorNoReliability,
        TemporalHGTANCursorWithMix,
        TemporalHGTANCursorDeep,
    ):
        model = cls(embed_dim=32, hidden_dim=64, dropout=0.0)
        threat, urgency = model(x)
        assert threat.shape == (4, 5)
        assert urgency.shape == (4, 3)


def test_cursor_exposes_gate_weights() -> None:
    model = TemporalHGTANCursor(embed_dim=32, hidden_dim=64, dropout=0.0)
    _ = model(torch.randn(2, 16, 16))
    assert model.gate_weights is not None
    assert model.gate_weights.shape == (2, 4)
