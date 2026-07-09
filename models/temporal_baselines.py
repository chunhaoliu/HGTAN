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


class FlatSequenceMLPBaseline(nn.Module):
    """Reviewer-requested MLP over the full 64 x 16 observed sequence."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        seq_len: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.seq_len = seq_len
        self.num_features = num_features
        input_dim = seq_len * num_features
        self.input_norm = nn.LayerNorm(num_features)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_head = nn.Linear(hidden_dim // 2, N_CLASSES)
        self.urgency_head = nn.Linear(hidden_dim // 2, N_URGENCY)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "FlatSequenceMLPBaseline")
        x = self._fit_fixed_window(x)
        flat = self.input_norm(x).reshape(x.size(0), self.seq_len * self.num_features)
        hidden = self.shared(flat)
        return self.threat_head(hidden), self.urgency_head(hidden)

    def _fit_fixed_window(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == self.seq_len:
            return x
        if x.size(1) > self.seq_len:
            return x[:, -self.seq_len :, :]
        padded = x.new_zeros(x.size(0), self.seq_len, x.size(2))
        padded[:, : x.size(1), :] = x
        return padded


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


class TemporalTransformerBaseline(nn.Module):
    """Transformer encoder sequence classifier for stronger temporal comparison."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.max_seq_len = max_seq_len
        self.input_norm = nn.LayerNorm(num_features)
        self.input_projection = nn.Linear(num_features, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionTemporalPool(hidden_dim)
        self.decoder = TemporalDecoder(hidden_dim, hidden_dim, dropout)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "TemporalTransformerBaseline")
        if x.size(1) > self.max_seq_len:
            raise ValueError(
                f"TemporalTransformerBaseline supports at most {self.max_seq_len} frames, got {x.size(1)}"
            )
        tokens = self.input_projection(self.input_norm(x))
        tokens = tokens + self.position_embedding[:, : x.size(1), :]
        states = self.encoder(tokens)
        return self.decoder(self.pool(states))


class TemporalTCNBaseline(nn.Module):
    """Temporal convolutional sequence classifier for non-recurrent comparison."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_norm = nn.LayerNorm(num_features)
        self.stem = nn.Conv1d(num_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TemporalTCNBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.pool = AttentionTemporalPool(hidden_dim)
        self.decoder = TemporalDecoder(hidden_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor):
        _check_sequence_input(x, "TemporalTCNBaseline")
        states = self.input_norm(x).transpose(1, 2)
        states = self.blocks(self.stem(states)).transpose(1, 2)
        return self.decoder(self.pool(states))


class TemporalTCNBlock(nn.Module):
    """Residual same-length temporal convolution block."""

    def __init__(self, *, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


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
