"""HGTAN-Cursor: causal group encoders with length-aware local/global pooling.

Designed to compete on default, short-history, and far-range axes by combining:
- per-group causal GRU states (strong on short windows),
- global temporal pooling (competitive on full-window final labels),
- a length/confidence gate that mixes the two,
- optional light confidence reweighting for far-range noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hgtan_layers import DualTaskDecoder
from models.hgtan import _require_sequence_input
from utils.config import (
    DROPOUT,
    EMBED_DIM,
    GROUP_DIMS,
    HIDDEN_DIM,
    N_FEATURES,
    NUM_GROUPS,
    NUM_HEADS,
    NUM_LAYERS,
)


class LightConfidenceReweight(nn.Module):
    """Softly reweight features when track confidence is low."""

    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid(),
        )
        self.strength = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        confidence = x[:, :, 15:16]
        gate = self.net(torch.cat([x, confidence], dim=-1))
        # Emphasize reweighting when confidence is low.
        low_conf = (1.0 - confidence).clamp(0.0, 1.0)
        alpha = torch.sigmoid(self.strength) * low_conf
        return x * (1.0 + alpha * (2.0 * gate - 1.0))


class GroupCausalEncoder(nn.Module):
    """Encode each semantic group with an independent causal GRU."""

    def __init__(self, embed_dim: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.input_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(group_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group_dim in GROUP_DIMS
            ]
        )
        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    embed_dim,
                    embed_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                for _ in GROUP_DIMS
            ]
        )
        self.group_embeddings = nn.Parameter(torch.randn(NUM_GROUPS, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns (batch, time, groups, embed)
        states = []
        offset = 0
        for group_idx, (group_dim, proj, gru) in enumerate(
            zip(GROUP_DIMS, self.input_projs, self.grus)
        ):
            group_x = proj(x[:, :, offset : offset + group_dim])
            group_x = group_x + self.group_embeddings[group_idx].view(1, 1, -1)
            encoded, _ = gru(group_x)
            states.append(self.dropout(encoded))
            offset += group_dim
        return torch.stack(states, dim=2)


class LocalGlobalGatePool(nn.Module):
    """Mix last-step local state with full-window global evidence per group."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, max(embed_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(embed_dim // 2, 1), 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self, group_states: torch.Tensor, confidence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # group_states: (B, T, G, D); confidence: (B, T, 1)
        batch, time_steps, num_groups, embed_dim = group_states.shape
        flat = group_states.permute(0, 2, 1, 3).reshape(batch * num_groups, time_steps, embed_dim)
        local = flat[:, -1, :]
        weights = torch.softmax(self.attn(flat), dim=1)
        global_pool = (flat * weights).sum(dim=1)
        mean_conf = confidence.mean(dim=1).expand(batch, num_groups).reshape(batch * num_groups, 1)
        length_feat = torch.full(
            (batch * num_groups, 1),
            math.log(max(time_steps, 1)),
            device=flat.device,
            dtype=flat.dtype,
        )
        gate_logit = self.gate(torch.cat([local, global_pool, mean_conf, length_feat], dim=-1))
        # Short windows favor local; longer windows can admit global pooling.
        length_bias = 0.35 * math.tanh(math.log(max(time_steps, 1)) / math.log(64.0))
        pi = torch.sigmoid(gate_logit - length_bias)
        mixed = pi * local + (1.0 - pi) * global_pool
        mixed = mixed.reshape(batch, num_groups, embed_dim)
        pi = pi.reshape(batch, num_groups)
        return mixed, pi


class LightGroupMix(nn.Module):
    """Optional shallow residual group interaction."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.residual_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        mixed, _ = self.attn(groups, groups, groups, need_weights=False)
        updated = self.norm(groups + mixed)
        updated = self.norm2(updated + self.ffn(updated))
        strength = torch.sigmoid(self.residual_logit)
        return groups + strength * (updated - groups)


class TemporalHGTANCursor(nn.Module):
    """Cursor redesign targeting default / short-history / far-range leadership."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        use_prior_weights: bool | None = None,
        prior_alpha: float | None = None,
        use_reliability: bool = True,
        use_group_mix: bool = False,
        gru_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        del num_features, num_heads, num_layers, use_prior_weights, prior_alpha, kwargs
        self.use_reliability = use_reliability
        self.use_group_mix = use_group_mix
        self.reliability = LightConfidenceReweight(N_FEATURES, max(embed_dim // 2, 16))
        self.encoder = GroupCausalEncoder(embed_dim, dropout, num_layers=gru_layers)
        self.pool = LocalGlobalGatePool(embed_dim)
        self.group_mix = LightGroupMix(embed_dim, dropout) if use_group_mix else None
        self.decoder = DualTaskDecoder(embed_dim, dropout=dropout, gradient_isolation=False)
        self.gate_weights = None

    def forward(self, x: torch.Tensor):
        _require_sequence_input(x, "TemporalHGTANCursor")
        if self.use_reliability:
            x = self.reliability(x)
        states = self.encoder(x)
        pooled, gate = self.pool(states, x[:, :, 15:16])
        self.gate_weights = gate.detach()
        if self.group_mix is not None:
            pooled = self.group_mix(pooled)
        return self.decoder(pooled)


class TemporalHGTANCursorNoReliability(TemporalHGTANCursor):
    def __init__(self, *args, **kwargs):
        kwargs["use_reliability"] = False
        super().__init__(*args, **kwargs)


class TemporalHGTANCursorWithMix(TemporalHGTANCursor):
    def __init__(self, *args, **kwargs):
        kwargs["use_group_mix"] = True
        super().__init__(*args, **kwargs)


class TemporalHGTANCursorDeep(TemporalHGTANCursor):
    def __init__(self, *args, **kwargs):
        kwargs["gru_layers"] = 2
        super().__init__(*args, **kwargs)
