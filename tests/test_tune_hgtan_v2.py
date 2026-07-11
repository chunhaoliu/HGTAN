from __future__ import annotations

import torch

from scripts.tune_hgtan_v2 import PrefixDataset, _ordinal_loss


def test_prefix_dataset_returns_all_frame_labels() -> None:
    dataset = PrefixDataset(
        x=torch.zeros(2, 4, 16).numpy(),
        threat_seq=torch.tensor([[1, 2, 3, 4], [2, 2, 3, 5]]).numpy(),
        urgency_seq=torch.tensor([[1, 1, 2, 3], [1, 2, 2, 3]]).numpy(),
    )
    x, threat, urgency = dataset[0]
    assert x.shape == (4, 16)
    assert threat.tolist() == [0, 1, 2, 3]
    assert urgency.tolist() == [0, 0, 1, 2]


def test_ordinal_loss_is_lower_for_correct_ordered_prediction() -> None:
    labels = torch.tensor([3])
    correct = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]])
    distant = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])
    assert _ordinal_loss(correct, labels) < _ordinal_loss(distant, labels)
