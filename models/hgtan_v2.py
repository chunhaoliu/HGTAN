"""Experimental HGTAN-v2 with clean temporal and relation boundaries."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hgtan_layers import DualTaskDecoder
from models.hgtan import TemporalEvidencePool, _require_sequence_input
from utils.config import (
    DROPOUT,
    EMBED_DIM,
    GROUP_DIMS,
    HIDDEN_DIM,
    N_FEATURES,
    NUM_GROUPS,
    NUM_HEADS,
    NUM_LAYERS,
    get_prior_weights_tensor,
)


class FeaturewiseReliabilityPrior(nn.Module):
    """Fuse data-driven and expert importance using per-feature reliability."""

    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.num_features = num_features
        prior = get_prior_weights_tensor().float()
        self.register_buffer("prior_importance", prior / prior.sum() * num_features)
        context_dim = num_features * 2 + 1
        self.reliability = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
        )
        self.data_importance = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_features),
        )
        self.residual_logit = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        confidence = x[:, :, 15:16]
        context = torch.cat([x, delta.abs(), confidence], dim=-1)
        reliability = torch.sigmoid(self.reliability(context))
        data_importance = F.softmax(self.data_importance(context), dim=-1) * self.num_features
        prior = self.prior_importance.view(1, 1, -1)
        importance = reliability * data_importance + (1.0 - reliability) * prior
        strength = torch.sigmoid(self.residual_logit)
        return x + strength * (x * importance - x), reliability


class IndependentGroupEncoder(nn.Module):
    """Encode semantic groups without performing premature cross-group mixing."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(group_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for group_dim in GROUP_DIMS
            ]
        )
        self.group_embeddings = nn.Parameter(torch.randn(NUM_GROUPS, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = []
        offset = 0
        for group_idx, (group_dim, encoder) in enumerate(zip(GROUP_DIMS, self.encoders)):
            group = encoder(x[:, :, offset : offset + group_dim])
            groups.append(group + self.group_embeddings[group_idx].view(1, 1, -1))
            offset += group_dim
        return torch.stack(groups, dim=2)


class GroupwiseTemporalEvidence(nn.Module):
    """Encode causal group histories before adaptive evidence pooling."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = TemporalEvidencePool(embed_dim)

    def forward(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, num_groups, embed_dim = groups.shape
        sequence = groups.permute(0, 2, 1, 3).reshape(batch_size * num_groups, time_steps, embed_dim)
        states, _ = self.gru(sequence)
        pooled, weights, mixture = self.pool(self.dropout(states))
        return (
            pooled.reshape(batch_size, num_groups, embed_dim),
            weights.reshape(batch_size, num_groups, time_steps),
            mixture.reshape(batch_size, num_groups, -1),
        )


class DynamicGroupRelation(nn.Module):
    """Infer sample-adaptive directed relations among the four semantic groups."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, max(embed_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(embed_dim // 2, 1), 1),
        )
        self.message = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(groups)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)
        left = normalized.unsqueeze(2).expand(-1, -1, NUM_GROUPS, -1)
        right = normalized.unsqueeze(1).expand(-1, NUM_GROUPS, -1, -1)
        pair = torch.cat([left, right, left * right, (left - right).abs()], dim=-1)
        gate = torch.sigmoid(self.edge_gate(pair).squeeze(-1))
        logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.size(-1))
        relation = torch.softmax(logits + torch.log(gate.clamp_min(1e-6)), dim=-1)
        messages = self.message(torch.matmul(relation, value))
        updated = groups + messages
        return updated + self.ffn(self.norm2(updated)), relation


class TemporalHGTANV2(nn.Module):
    """Experimental reliability-temporal-relation HGTAN."""

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
        use_temporal: bool = True,
        use_synergy: bool = True,
        **kwargs,
    ):
        super().__init__()
        del num_heads, num_layers, use_prior_weights, prior_alpha, kwargs
        self.use_reliability = use_reliability
        self.use_temporal = use_temporal
        self.use_synergy = use_synergy
        self.reliability_prior = FeaturewiseReliabilityPrior(num_features, max(embed_dim // 2, 16))
        self.group_encoder = IndependentGroupEncoder(embed_dim, dropout)
        self.temporal = GroupwiseTemporalEvidence(embed_dim, dropout)
        self.relation = DynamicGroupRelation(embed_dim, hidden_dim, dropout)
        self.decoder = DualTaskDecoder(embed_dim, dropout=dropout, gradient_isolation=False)
        self.reliability_weights = None
        self.temporal_weights = None
        self.temporal_mix_weights = None
        self.attention_weights = None

    def forward(self, x: torch.Tensor):
        _require_sequence_input(x, "TemporalHGTANV2")
        if self.use_reliability:
            x, reliability = self.reliability_prior(x)
            self.reliability_weights = reliability.detach()
        else:
            self.reliability_weights = None
        groups = self.group_encoder(x)
        if self.use_temporal:
            pooled, temporal_weights, temporal_mix = self.temporal(groups)
            self.temporal_weights = temporal_weights.detach()
            self.temporal_mix_weights = temporal_mix.detach()
        else:
            pooled = groups.mean(dim=1)
            self.temporal_weights = None
            self.temporal_mix_weights = None
        if self.use_synergy:
            pooled, relation = self.relation(pooled)
            self.attention_weights = relation.detach()
        else:
            self.attention_weights = None
        return self.decoder(pooled)


class TemporalHGTANV2NoReliability(TemporalHGTANV2):
    def __init__(self, *args, **kwargs):
        kwargs["use_reliability"] = False
        super().__init__(*args, **kwargs)


class TemporalHGTANV2NoTemporal(TemporalHGTANV2):
    def __init__(self, *args, **kwargs):
        kwargs["use_temporal"] = False
        super().__init__(*args, **kwargs)


class TemporalHGTANV2NoSynergy(TemporalHGTANV2):
    def __init__(self, *args, **kwargs):
        kwargs["use_synergy"] = False
        super().__init__(*args, **kwargs)
