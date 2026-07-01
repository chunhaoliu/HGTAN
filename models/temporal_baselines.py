"""Temporal baselines for sequential ATUAV assessment protocols."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.config import N_CLASSES, N_FEATURES, N_URGENCY


def _check_sequence_input(x: torch.Tensor, model_name: str) -> None:
    if x.dim() != 3:
        raise ValueError(f"{model_name} expects input shape (batch, time, features), got {tuple(x.shape)}")


def _safe_lstm_forward(lstm: nn.LSTM, inputs: torch.Tensor):
    """Avoid intermittent Windows CUDA/cuDNN LSTM shutdown faults."""
    if inputs.is_cuda and torch.backends.cudnn.enabled:
        with torch.backends.cudnn.flags(enabled=False):
            return lstm(inputs)
    return lstm(inputs)


class LastFrameMLPBaseline(nn.Module):
    """Non-temporal reference baseline that only uses the final observed frame."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_norm = nn.LayerNorm(num_features)
        self.shared = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_head = nn.Linear(hidden_dim, N_CLASSES)
        self.urgency_head = nn.Linear(hidden_dim, N_URGENCY)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "LastFrameMLPBaseline")
        hidden = self.shared(self.input_norm(x[:, -1, :]))
        return self.threat_head(hidden), self.urgency_head(hidden)


class MeanPoolMLPBaseline(nn.Module):
    """Order-agnostic temporal baseline using mean pooling over observed frames."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_norm = nn.LayerNorm(num_features)
        self.shared = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_head = nn.Linear(hidden_dim, N_CLASSES)
        self.urgency_head = nn.Linear(hidden_dim, N_URGENCY)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "MeanPoolMLPBaseline")
        pooled = self.input_norm(x).mean(dim=1)
        hidden = self.shared(pooled)
        return self.threat_head(hidden), self.urgency_head(hidden)


class TemporalGRUBaseline(nn.Module):
    """GRU sequence baseline with attention pooling over track time steps."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_norm = nn.LayerNorm(num_features)
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttentionTemporalPool(out_dim)
        self.decoder = TemporalDecoder(out_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "TemporalGRUBaseline")
        outputs, _ = self.gru(self.input_norm(x))
        return self.decoder(self.pool(outputs))


class TemporalLSTMBaseline(nn.Module):
    """LSTM sequence baseline with attention pooling over track time steps."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_norm = nn.LayerNorm(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.pool = AttentionTemporalPool(out_dim)
        self.decoder = TemporalDecoder(out_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "TemporalLSTMBaseline")
        outputs, _ = _safe_lstm_forward(self.lstm, self.input_norm(x))
        return self.decoder(self.pool(outputs))


class AttentionTemporalPool(nn.Module):
    """Learned attention pooling over temporal hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(states), dim=1)
        return (states * weights).sum(dim=1)


class TemporalDecoder(nn.Module):
    """Shared dual-task decoder for temporal recurrent baselines."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_head = nn.Linear(hidden_dim, N_CLASSES)
        self.urgency_head = nn.Linear(hidden_dim, N_URGENCY)

    def forward(self, pooled: torch.Tensor):
        hidden = self.shared(pooled)
        return self.threat_head(hidden), self.urgency_head(hidden)
