"""
Feature-group Graph baselines v2.1

在特征组图上应用图神经网络:
- GCNBaseline    (经典GCN)
- GATBaseline    (图注意力网络)
- GraphSAGEBaseline (GraphSAGE聚合)

v2.1 更新:
- 适配16维正式特征体系
- 4组: capability(4) + intent(4) + opportunity(4) + context(4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import (
    N_FEATURES, N_CLASSES, N_URGENCY,
    GROUP_DIMS, NUM_GROUPS,
    EMBED_DIM, DROPOUT,
)


# ======================= 分组投影 ==========================================
class GroupProjector(nn.Module):
    """将16维特征按组投影到统一维度"""

    def __init__(self, group_dims=GROUP_DIMS, embed_dim=EMBED_DIM):
        super().__init__()
        self.group_dims = group_dims
        self.embed_dim = embed_dim

        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            ) for dim in group_dims
        ])

    def forward(self, x):
        """
        x: (batch, 16)
        返回: (batch, 4, embed_dim)
        """
        group_features = []
        start_idx = 0
        for i, (dim, projector) in enumerate(zip(self.group_dims, self.projectors)):
            group_x = x[:, start_idx:start_idx + dim]
            group_encoded = projector(group_x)
            group_features.append(group_encoded)
            start_idx += dim
        return torch.stack(group_features, dim=1)


# ======================= 全连接邻接矩阵 ====================================
def get_full_adjacency(num_nodes=NUM_GROUPS, device='cpu'):
    """返回全连接邻接矩阵（带自环）"""
    adj = torch.ones(num_nodes, num_nodes, device=device)
    return adj


def normalize_adjacency(adj):
    """对称归一化邻接矩阵"""
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


# ======================= GCN 基线 ==========================================
class GCNLayer(nn.Module):
    """单层GCN"""

    def __init__(self, in_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_dim)
        adj: (num_nodes, num_nodes)
        """
        x = self.fc(x)
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return F.gelu(self.dropout(x))


class GCNBaseline(nn.Module):
    """GCN基线模型"""

    def __init__(self, num_features=N_FEATURES, embed_dim=EMBED_DIM,
                 num_layers=2, dropout=DROPOUT, **kwargs):
        super().__init__()
        self.num_groups = NUM_GROUPS

        self.projector = GroupProjector(GROUP_DIMS, embed_dim)

        self.gcn_layers = nn.ModuleList([
            GCNLayer(embed_dim, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        flat_dim = embed_dim * self.num_groups
        self.threat_head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_CLASSES),
        )

        self.urgency_head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_URGENCY),
        )

    def forward(self, x):
        device = x.device
        adj = normalize_adjacency(get_full_adjacency(self.num_groups, device))

        node_features = self.projector(x)

        for layer in self.gcn_layers:
            node_features = layer(node_features, adj)

        flat = node_features.view(x.size(0), -1)
        return self.threat_head(flat), self.urgency_head(flat)


# ======================= GAT 基线 ==========================================
class GATLayer(nn.Module):
    """单层GAT"""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=DROPOUT):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))

    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_dim)
        adj: (num_nodes, num_nodes)
        """
        batch, num_nodes, _ = x.shape

        h = self.W(x).view(batch, num_nodes, self.num_heads, self.head_dim)

        e_src = (h * self.a_src).sum(dim=-1)
        e_dst = (h * self.a_dst).sum(dim=-1)

        e = e_src.unsqueeze(3) + e_dst.unsqueeze(2)
        e = self.leaky_relu(e)

        adj_mask = adj.unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(adj_mask == 0, float('-inf'))

        attn = F.softmax(e, dim=2)
        attn = self.dropout(attn)

        out = torch.einsum('bnih,bnhd->bihd', attn, h)
        out = out.reshape(batch, num_nodes, -1)

        return F.gelu(out)


class GATBaseline(nn.Module):
    """GAT基线模型"""

    def __init__(self, num_features=N_FEATURES, embed_dim=EMBED_DIM,
                 num_heads=4, num_layers=2, dropout=DROPOUT, **kwargs):
        super().__init__()
        self.num_groups = NUM_GROUPS

        self.projector = GroupProjector(GROUP_DIMS, embed_dim)

        self.gat_layers = nn.ModuleList([
            GATLayer(embed_dim, embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        flat_dim = embed_dim * self.num_groups
        self.threat_head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_CLASSES),
        )

        self.urgency_head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_URGENCY),
        )

    def forward(self, x):
        device = x.device
        adj = get_full_adjacency(self.num_groups, device)

        node_features = self.projector(x)

        for layer in self.gat_layers:
            node_features = layer(node_features, adj)

        flat = node_features.view(x.size(0), -1)
        return self.threat_head(flat), self.urgency_head(flat)


# ======================= GraphSAGE 基线 ====================================
class GraphSAGELayer(nn.Module):
    """GraphSAGE聚合层"""

    def __init__(self, in_dim, out_dim, dropout=DROPOUT):
        super().__init__()
        self.self_fc = nn.Linear(in_dim, out_dim)
        self.neigh_fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_dim)
        adj: (num_nodes, num_nodes)
        """
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        neigh_agg = torch.bmm(
            adj.unsqueeze(0).expand(x.size(0), -1, -1),
            x
        ) / degree.unsqueeze(0)

        self_features = self.self_fc(x)
        neigh_features = self.neigh_fc(neigh_agg)

        out = self.norm(self.dropout(self_features + neigh_features))
        return F.gelu(out)


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE基线模型"""

    def __init__(self, num_features=N_FEATURES, embed_dim=EMBED_DIM,
                 num_layers=2, dropout=DROPOUT, **kwargs):
        super().__init__()
        self.num_groups = NUM_GROUPS

        self.projector = GroupProjector(GROUP_DIMS, embed_dim)

        self.sage_layers = nn.ModuleList([
            GraphSAGELayer(embed_dim, embed_dim, dropout)
            for _ in range(num_layers)
        ])

        flat_dim = embed_dim * self.num_groups
        self.threat_head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_CLASSES),
        )

        self.urgency_head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, N_URGENCY),
        )

    def forward(self, x):
        device = x.device
        adj = get_full_adjacency(self.num_groups, device)

        node_features = self.projector(x)

        for layer in self.sage_layers:
            node_features = layer(node_features, adj)

        flat = node_features.view(x.size(0), -1)
        return self.threat_head(flat), self.urgency_head(flat)
