"""Neural building blocks for HGTAN."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import (
    DROPOUT,
    EMBED_DIM,
    GROUP_DIMS,
    HIDDEN_DIM,
    N_CLASSES,
    N_FEATURES,
    N_URGENCY,
    NUM_GROUPS,
    NUM_HEADS,
    get_prior_weights_tensor,
)


class AttentionPriorFusion(nn.Module):
    """Fuse expert prior weights with sample-adaptive feature weights."""

    def __init__(self, n_features: int = N_FEATURES, embed_dim: int = 64, prior_alpha: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.register_buffer("prior_weights", get_prior_weights_tensor())

        self.query_proj = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.key_proj = nn.Linear(n_features, embed_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid(),
        )
        self.learnable_weights = nn.Parameter(torch.ones(n_features) * 0.5)
        initial_gate = min(max(float(prior_alpha), 1e-4), 1.0 - 1e-4)
        self.prior_logit = nn.Parameter(torch.tensor(math.log(initial_gate / (1.0 - initial_gate))))
        self.residual_logit = nn.Parameter(torch.tensor(-1.1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size = x.size(0)
        query = self.query_proj(x)
        prior = self.prior_weights.unsqueeze(0).expand(batch_size, -1)
        key = self.key_proj(prior)

        content_gate = torch.sigmoid((query * key).sum(dim=-1, keepdim=True))
        global_gate = torch.sigmoid(self.prior_logit).view(1, 1)
        attn_score = 0.5 * content_gate + 0.5 * global_gate
        learnable = F.softmax(self.learnable_weights, dim=0)
        combined_weights = attn_score * self.prior_weights + (1.0 - attn_score) * learnable

        importance = combined_weights * float(self.n_features)
        weighted_x = x * importance
        gate = self.fusion_gate(torch.cat([x, weighted_x], dim=-1))
        residual_strength = torch.sigmoid(self.residual_logit)
        output = x + residual_strength * gate * (weighted_x - x)

        weights_info = {
            "attn_score": attn_score.mean().item(),
            "combined_weights": combined_weights[0].detach(),
            "residual_strength": residual_strength.detach(),
        }
        return output, weights_info


class HierarchicalFeatureEncoder(nn.Module):
    """Encode the 16 indicators as four semantic feature groups."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        dropout: float = DROPOUT,
        use_prior_weights: bool = True,
        prior_alpha: float = 0.3,
    ):
        super().__init__()
        self.group_dims = GROUP_DIMS
        self.num_groups = NUM_GROUPS
        self.embed_dim = embed_dim
        self.use_prior_weights = use_prior_weights

        if use_prior_weights:
            self.weight_fusion = AttentionPriorFusion(N_FEATURES, max(embed_dim // 2, 1), prior_alpha)

        self.group_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for dim in self.group_dims
            ]
        )
        self.group_embeddings = nn.Parameter(torch.randn(self.num_groups, embed_dim) * 0.02)
        self.cross_group_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
        weights_info = None
        if self.use_prior_weights:
            x, weights_info = self.weight_fusion(x)

        group_features = []
        start_idx = 0
        for group_idx, (dim, encoder) in enumerate(zip(self.group_dims, self.group_encoders)):
            group_x = x[:, start_idx:start_idx + dim]
            group_features.append(encoder(group_x) + self.group_embeddings[group_idx])
            start_idx += dim

        stacked = torch.stack(group_features, dim=1)
        residual = stacked
        stacked = self.cross_norm(stacked)
        stacked, _ = self.cross_group_attn(stacked, stacked, stacked)
        return residual + 0.5 * stacked, weights_info


class SynergyAttentionModule(nn.Module):
    """Model conditional swarm synergy through group-level self-attention."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
        swarm_group_idx: int = 2,
    ):
        super().__init__()
        self.swarm_group_idx = swarm_group_idx

        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.synergy_detector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.synergy_amplifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.synergy_strength = nn.Parameter(torch.tensor(0.3))

    def forward(self, group_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = group_features
        x = self.norm1(group_features)
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = residual + attn_output

        swarm_features = x[:, self.swarm_group_idx, :]
        synergy_score = torch.sigmoid(self.synergy_detector(swarm_features))
        synergy_factor = self.synergy_amplifier(swarm_features).unsqueeze(1)
        strength = torch.sigmoid(self.synergy_strength)
        x = x * (1.0 + synergy_score.unsqueeze(-1) * synergy_factor * strength)

        residual = x
        x = self.norm2(x)
        return residual + self.ffn(x), attn_weights


class DualTaskDecoder(nn.Module):
    """Decoupled decoder for threat level and urgency degree."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_groups: int = NUM_GROUPS,
        num_threat_classes: int = N_CLASSES,
        num_urgency_classes: int = N_URGENCY,
        dropout: float = DROPOUT,
        gradient_isolation: bool = True,
    ):
        super().__init__()
        del num_groups
        self.gradient_isolation = gradient_isolation

        self.threat_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.urgency_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

        pool_dim = embed_dim * 2
        self.threat_encoder = nn.Sequential(
            nn.Linear(pool_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_head = nn.Linear(embed_dim, num_threat_classes)

        self.urgency_encoder = nn.Sequential(
            nn.Linear(pool_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threat_to_urgency = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_urgency_classes),
        )

    def forward(self, group_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        threat_features = self._pool_and_encode(
            group_features,
            pool=self.threat_pool,
            encoder=self.threat_encoder,
        )
        threat_logits = self.threat_head(threat_features)

        urgency_input_features = group_features.detach() if self.gradient_isolation else group_features
        urgency_features = self._pool_and_encode(
            urgency_input_features,
            pool=self.urgency_pool,
            encoder=self.urgency_encoder,
        )
        threat_aux = self.threat_to_urgency(threat_features.detach())
        urgency_logits = self.urgency_head(torch.cat([urgency_features, threat_aux], dim=-1))
        return threat_logits, urgency_logits

    @staticmethod
    def _pool_and_encode(group_features: torch.Tensor, pool: nn.Module, encoder: nn.Module) -> torch.Tensor:
        attn_scores = pool(group_features)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_pooled = (group_features * attn_weights).sum(dim=1)
        mean_pooled = group_features.mean(dim=1)
        return encoder(torch.cat([attn_pooled, mean_pooled], dim=-1))


class PriorWeightFusion(nn.Module):
    """Legacy simple prior-weight fusion used for comparison."""

    def __init__(self, n_features: int = N_FEATURES, alpha: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.register_buffer("prior_weights", get_prior_weights_tensor())
        self.data_weights = nn.Parameter(torch.ones(n_features))
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.alpha)
        data_weights = F.softmax(self.data_weights, dim=0)
        combined_weights = alpha * self.prior_weights + (1.0 - alpha) * data_weights
        return x * combined_weights.unsqueeze(0), combined_weights
