from __future__ import annotations

import torch

from models.hgtan_v2 import (
    TemporalHGTANV2,
    TemporalHGTANV2Core,
    TemporalHGTANV2CoreNoGrouping,
    TemporalHGTANV2CoreNoTemporal,
    TemporalHGTANV2NoReliability,
    TemporalHGTANV2NoSynergy,
    TemporalHGTANV2NoTemporal,
)


def test_hgtan_v2_and_ablations_return_dual_logits() -> None:
    inputs = torch.rand(3, 12, 16)
    for model_class in (
        TemporalHGTANV2,
        TemporalHGTANV2Core,
        TemporalHGTANV2CoreNoTemporal,
        TemporalHGTANV2CoreNoGrouping,
        TemporalHGTANV2NoReliability,
        TemporalHGTANV2NoTemporal,
        TemporalHGTANV2NoSynergy,
    ):
        model = model_class(embed_dim=32, hidden_dim=64, dropout=0.0)
        threat, urgency = model(inputs)
        assert threat.shape == (3, 5)
        assert urgency.shape == (3, 3)


def test_hgtan_v2_exposes_featurewise_reliability_and_group_relations() -> None:
    model = TemporalHGTANV2(embed_dim=32, hidden_dim=64, dropout=0.0)
    model(torch.rand(2, 10, 16))

    assert model.reliability_weights is not None
    assert model.reliability_weights.shape == (2, 10, 16)
    assert model.attention_weights is not None
    assert model.attention_weights.shape == (2, 4, 4)
    torch.testing.assert_close(model.attention_weights.sum(dim=-1), torch.ones(2, 4))
