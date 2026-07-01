"""HGTAN models and assessment ablation variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from layers import DualTaskDecoder, HierarchicalFeatureEncoder, SynergyAttentionModule
from utils.config import (
    DROPOUT,
    EMBED_DIM,
    HGTANConfig,
    HIDDEN_DIM,
    N_CLASSES,
    N_FEATURES,
    N_URGENCY,
    NUM_GROUPS,
    NUM_HEADS,
    NUM_LAYERS,
)


def _resolve_prior_settings(
    use_prior_weights: bool | None,
    prior_alpha: float | None,
) -> tuple[bool, float]:
    if use_prior_weights is None:
        use_prior_weights = HGTANConfig.MODEL.get("use_prior_weights", True)
    if prior_alpha is None:
        prior_alpha = HGTANConfig.MODEL.get("prior_weight_alpha", 0.3)
    return bool(use_prior_weights), float(prior_alpha)


def _require_sequence_input(x: torch.Tensor, model_name: str) -> None:
    if x.dim() != 3:
        raise ValueError(f"{model_name} expects (batch, time, features), got {tuple(x.shape)}")


class HGTAN(nn.Module):
    """Hierarchical Group Threat Attention Network."""

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
        gradient_isolation: bool = True,
    ):
        super().__init__()
        del num_features
        use_prior_weights, prior_alpha = _resolve_prior_settings(use_prior_weights, prior_alpha)

        self.encoder = HierarchicalFeatureEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
            use_prior_weights=use_prior_weights,
            prior_alpha=prior_alpha,
        )
        self.sam_layers = nn.ModuleList(
            [
                SynergyAttentionModule(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = DualTaskDecoder(
            embed_dim=embed_dim,
            dropout=dropout,
            gradient_isolation=gradient_isolation,
        )
        self.attention_weights = None
        self.learned_weights = None
        self._init_weights()

    def forward(self, x: torch.Tensor):
        group_features, weights_info = self.encoder(x)
        self.learned_weights = weights_info

        for sam_layer in self.sam_layers:
            group_features, attn_weights = sam_layer(group_features)

        self.attention_weights = attn_weights
        return self.decoder(group_features)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_attention_weights(self):
        return self.attention_weights

    def get_learned_weights(self):
        return self.learned_weights


class HGTAN_NoHierarchical(nn.Module):
    """Ablation without hierarchical group encoding."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.num_groups = NUM_GROUPS
        self.flat_encoder = nn.Sequential(
            nn.Linear(num_features, embed_dim * NUM_GROUPS),
            nn.LayerNorm(embed_dim * NUM_GROUPS),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sam_layers = nn.ModuleList(
            [
                SynergyAttentionModule(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder = DualTaskDecoder(embed_dim, num_groups=NUM_GROUPS, dropout=dropout)
        self.attention_weights = None

    def forward(self, x: torch.Tensor):
        group_features = self.flat_encoder(x).view(x.size(0), self.num_groups, -1)
        for sam_layer in self.sam_layers:
            group_features, attn_weights = sam_layer(group_features)
        self.attention_weights = attn_weights
        return self.decoder(group_features)


class HGTAN_NoSynergy(nn.Module):
    """Ablation replacing synergy attention with standard transformer layers."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        **kwargs,
    ):
        super().__init__()
        del num_features, kwargs
        self.encoder = HierarchicalFeatureEncoder(embed_dim, dropout, use_prior_weights=False)
        self.attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = DualTaskDecoder(embed_dim, num_groups=NUM_GROUPS, dropout=dropout)

    def forward(self, x: torch.Tensor):
        group_features, _ = self.encoder(x)
        for layer in self.attn_layers:
            group_features = layer(group_features)
        return self.decoder(group_features)


class HGTAN_NoDualTask(nn.Module):
    """Ablation with independent flat heads for threat and urgency."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        **kwargs,
    ):
        super().__init__()
        del num_features, kwargs
        self.encoder = HierarchicalFeatureEncoder(embed_dim, dropout, use_prior_weights=False)
        self.sam_layers = nn.ModuleList(
            [
                SynergyAttentionModule(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        flat_dim = embed_dim * NUM_GROUPS
        self.threat_decoder = nn.Sequential(
            nn.Linear(flat_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_CLASSES),
        )
        self.urgency_decoder = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_URGENCY),
        )
        self.attention_weights = None

    def forward(self, x: torch.Tensor):
        group_features, _ = self.encoder(x)
        for sam_layer in self.sam_layers:
            group_features, attn_weights = sam_layer(group_features)
        self.attention_weights = attn_weights
        flat = group_features.view(x.size(0), -1)
        return self.threat_decoder(flat), self.urgency_decoder(flat)


class HGTAN_NoPrior(nn.Module):
    """Ablation without prior-weight fusion."""

    def __init__(
        self,
        num_features: int = N_FEATURES,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        **kwargs,
    ):
        super().__init__()
        del num_features, kwargs
        self.encoder = HierarchicalFeatureEncoder(embed_dim, dropout, use_prior_weights=False)
        self.sam_layers = nn.ModuleList(
            [
                SynergyAttentionModule(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder = DualTaskDecoder(embed_dim, dropout=dropout)
        self.attention_weights = None

    def forward(self, x: torch.Tensor):
        group_features, _ = self.encoder(x)
        for sam_layer in self.sam_layers:
            group_features, attn_weights = sam_layer(group_features)
        self.attention_weights = attn_weights
        return self.decoder(group_features)


class TemporalAttentionPool(nn.Module):
    """Attention pooling over time for each semantic group."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, max(embed_dim // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(embed_dim // 2, 1), 1),
        )

    def forward(self, group_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.scorer(group_sequence), dim=1)
        return (group_sequence * weights).sum(dim=1), weights.squeeze(-1)


class TemporalEvidencePool(nn.Module):
    """Blend attention, mean, and final-frame temporal evidence."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = TemporalAttentionPool(embed_dim)
        self.mix_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )
        self.recency_logit = nn.Parameter(torch.tensor(0.2))

    def forward(self, group_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_pooled, attn_weights = self.attention(group_sequence)
        time_steps = group_sequence.size(1)
        if time_steps > 1:
            positions = torch.linspace(
                -1.0,
                0.0,
                steps=time_steps,
                device=group_sequence.device,
                dtype=group_sequence.dtype,
            )
            recency = torch.softmax(self.recency_logit * positions, dim=0)
            recency_pooled = (group_sequence * recency.view(1, time_steps, 1)).sum(dim=1)
        else:
            recency_pooled = group_sequence[:, -1, :]
        mean_pooled = group_sequence.mean(dim=1)
        last_pooled = group_sequence[:, -1, :]
        gate = torch.softmax(
            self.mix_gate(torch.cat([attn_pooled, mean_pooled, last_pooled], dim=-1)),
            dim=-1,
        )
        pooled = (
            gate[:, 0:1] * attn_pooled
            + gate[:, 1:2] * mean_pooled
            + gate[:, 2:3] * 0.5 * (last_pooled + recency_pooled)
        )
        return pooled, attn_weights, gate


class TemporalHGTAN(nn.Module):
    """Temporal HGTAN for track sequences shaped as (batch, time, features)."""

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
        gradient_isolation: bool = True,
        pooling: str = "attention",
    ):
        super().__init__()
        del num_features
        use_prior_weights, prior_alpha = _resolve_prior_settings(use_prior_weights, prior_alpha)

        self.pooling = pooling
        self.encoder = HierarchicalFeatureEncoder(
            embed_dim=embed_dim,
            dropout=dropout,
            use_prior_weights=use_prior_weights,
            prior_alpha=prior_alpha,
        )
        self.temporal_pool = TemporalEvidencePool(embed_dim)
        self.sam_layers = nn.ModuleList(
            [
                SynergyAttentionModule(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder = DualTaskDecoder(
            embed_dim=embed_dim,
            dropout=dropout,
            gradient_isolation=gradient_isolation,
        )
        self.attention_weights = None
        self.temporal_weights = None
        self.temporal_mix_weights = None
        self.learned_weights = None

    def forward(self, x: torch.Tensor):
        _require_sequence_input(x, "TemporalHGTAN")

        batch_size, time_steps, n_features = x.shape
        flat_x = x.reshape(batch_size * time_steps, n_features)
        group_features, weights_info = self.encoder(flat_x)
        self.learned_weights = weights_info
        group_features = group_features.reshape(batch_size, time_steps, NUM_GROUPS, -1)

        if self.pooling == "last":
            pooled_groups = group_features[:, -1, :, :]
            self.temporal_weights = torch.zeros(batch_size, NUM_GROUPS, time_steps, device=x.device)
            self.temporal_weights[:, :, -1] = 1.0
            self.temporal_mix_weights = None
        elif self.pooling == "mean":
            pooled_groups = group_features.mean(dim=1)
            self.temporal_weights = torch.full(
                (batch_size, NUM_GROUPS, time_steps),
                1.0 / max(time_steps, 1),
                device=x.device,
            )
            self.temporal_mix_weights = None
        else:
            grouped = group_features.permute(0, 2, 1, 3).reshape(batch_size * NUM_GROUPS, time_steps, -1)
            pooled, temporal_weights, temporal_mix = self.temporal_pool(grouped)
            pooled_groups = pooled.reshape(batch_size, NUM_GROUPS, -1)
            self.temporal_weights = temporal_weights.reshape(batch_size, NUM_GROUPS, time_steps).detach()
            self.temporal_mix_weights = temporal_mix.reshape(batch_size, NUM_GROUPS, 3).detach()

        for sam_layer in self.sam_layers:
            pooled_groups, attn_weights = sam_layer(pooled_groups)
        self.attention_weights = attn_weights
        return self.decoder(pooled_groups)

    def get_attention_weights(self):
        return self.attention_weights

    def get_temporal_weights(self):
        return self.temporal_weights

    def get_temporal_mix_weights(self):
        return self.temporal_mix_weights

    def get_learned_weights(self):
        return self.learned_weights


class TemporalHGTANLastFrame(TemporalHGTAN):
    """Temporal HGTAN ablation that only uses the final observed frame."""

    def __init__(self, *args, **kwargs):
        kwargs["pooling"] = "last"
        super().__init__(*args, **kwargs)


class TemporalHGTANMeanPool(TemporalHGTAN):
    """Temporal HGTAN ablation with mean pooling over frames."""

    def __init__(self, *args, **kwargs):
        kwargs["pooling"] = "mean"
        super().__init__(*args, **kwargs)


class TemporalHGTANNoPrior(TemporalHGTAN):
    """Temporal HGTAN ablation without prior fusion."""

    def __init__(self, *args, **kwargs):
        kwargs["use_prior_weights"] = False
        super().__init__(*args, **kwargs)


class TemporalHGTANNoSynergy(TemporalHGTAN):
    """Temporal HGTAN ablation replacing synergy attention with transformer layers."""

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
        gradient_isolation: bool = True,
        pooling: str = "attention",
    ):
        super().__init__(
            num_features=num_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=0,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_prior_weights=use_prior_weights,
            prior_alpha=prior_alpha,
            gradient_isolation=gradient_isolation,
            pooling=pooling,
        )
        self.attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        _require_sequence_input(x, "TemporalHGTANNoSynergy")

        batch_size, time_steps, n_features = x.shape
        flat_x = x.reshape(batch_size * time_steps, n_features)
        group_features, weights_info = self.encoder(flat_x)
        self.learned_weights = weights_info
        group_features = group_features.reshape(batch_size, time_steps, NUM_GROUPS, -1)
        grouped = group_features.permute(0, 2, 1, 3).reshape(batch_size * NUM_GROUPS, time_steps, -1)
        pooled, temporal_weights, temporal_mix = self.temporal_pool(grouped)
        pooled_groups = pooled.reshape(batch_size, NUM_GROUPS, -1)
        self.temporal_weights = temporal_weights.reshape(batch_size, NUM_GROUPS, time_steps).detach()
        self.temporal_mix_weights = temporal_mix.reshape(batch_size, NUM_GROUPS, 3).detach()

        for layer in self.attn_layers:
            pooled_groups = layer(pooled_groups)
        return self.decoder(pooled_groups)
