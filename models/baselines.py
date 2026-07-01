"""Neural baseline models for ATUAV threat assessment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import (
    MLP_DROPOUT,
    MLP_HIDDEN_DIM,
    N_CLASSES,
    N_FEATURES,
    N_URGENCY,
    TRANS_DROPOUT,
    TRANS_EMBED_DIM,
    TRANS_NUM_HEADS,
    TRANS_NUM_LAYERS,
)


def _safe_lstm_forward(lstm: nn.LSTM, inputs: torch.Tensor):
    """Avoid intermittent Windows CUDA/cuDNN LSTM shutdown faults."""
    if inputs.is_cuda and torch.backends.cudnn.enabled:
        with torch.backends.cudnn.flags(enabled=False):
            return lstm(inputs)
    return lstm(inputs)


class MLPBaseline(nn.Module):
    """Feed-forward MLP baseline."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = MLP_HIDDEN_DIM,
        num_layers: int = 3,
        dropout: float = MLP_DROPOUT,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        layers = []
        in_dim = num_features
        for _ in range(max(num_layers - 1, 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.threat_head = nn.Linear(hidden_dim, N_CLASSES)
        self.urgency_head = nn.Linear(hidden_dim, N_URGENCY)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.threat_head(h), self.urgency_head(h)


class TransformerBaseline(nn.Module):
    """Transformer encoder over feature tokens."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        embed_dim: int = TRANS_EMBED_DIM,
        num_heads: int = TRANS_NUM_HEADS,
        num_layers: int = TRANS_NUM_LAYERS,
        dropout: float = TRANS_DROPOUT,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.feature_embed = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_features + 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.threat_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, N_CLASSES))
        self.urgency_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, N_URGENCY))

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.feature_embed(x.unsqueeze(-1))
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        cls_out = self.transformer(x)[:, 0, :]
        return self.threat_head(cls_out), self.urgency_head(cls_out)


class LSTMBaseline(nn.Module):
    """LSTM baseline treating the feature vector as a short sequence."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del num_features, kwargs
        self.feature_embed = nn.Linear(1, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.threat_head = _head(hidden_dim * 2, hidden_dim, N_CLASSES, dropout)
        self.urgency_head = _head(hidden_dim * 2, hidden_dim, N_URGENCY, dropout)

    def forward(self, x: torch.Tensor):
        x = self.feature_embed(x.unsqueeze(-1))
        _, (h_n, _) = _safe_lstm_forward(self.lstm, x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.threat_head(h), self.urgency_head(h)


class CNNBaseline(nn.Module):
    """1D CNN baseline over feature tokens."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del num_features, kwargs
        self.feature_embed = nn.Linear(1, hidden_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.threat_head = _head(hidden_dim * 2, hidden_dim, N_CLASSES, dropout)
        self.urgency_head = _head(hidden_dim * 2, hidden_dim, N_URGENCY, dropout)

    def forward(self, x: torch.Tensor):
        x = self.feature_embed(x.unsqueeze(-1)).permute(0, 2, 1)
        h = self.conv_layers(x).squeeze(-1)
        return self.threat_head(h), self.urgency_head(h)


class ResBlock(nn.Module):
    """Residual MLP block."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(F.gelu(x + self.block(x)))


class ResNetBaseline(nn.Module):
    """Residual feed-forward baseline."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.threat_head = _head(hidden_dim, hidden_dim // 2, N_CLASSES, dropout)
        self.urgency_head = _head(hidden_dim, hidden_dim // 2, N_URGENCY, dropout)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.threat_head(h), self.urgency_head(h)


class AttentionMLPBaseline(nn.Module):
    """MLP baseline with learned feature attention."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.feature_attn = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_features),
        )
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_head = nn.Linear(hidden_dim, N_CLASSES)
        self.urgency_head = nn.Linear(hidden_dim, N_URGENCY)

    def forward(self, x: torch.Tensor):
        attn = F.softmax(self.feature_attn(x), dim=-1)
        h = self.encoder(x * attn)
        return self.threat_head(h), self.urgency_head(h)


def get_baseline_models() -> dict[str, type[nn.Module]]:
    """Return neural baseline classes."""
    return {
        "MLP": MLPBaseline,
        "Transformer": TransformerBaseline,
        "LSTM": LSTMBaseline,
        "CNN": CNNBaseline,
        "ResNet": ResNetBaseline,
        "AttentionMLP": AttentionMLPBaseline,
    }


def _head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )
