import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from features.drift import compute_drift

POOL_DIM = 5 * 3 + 2  # mean/std/max of 5 drift dims + log n_nodes + max depth


def pool_drift_features(drift, hop_depths):
    """Graph-level set-pooling of per-node drift: the 12 RF statistics + max moments.

    Returns [POOL_DIM] vector: mean, std, max over nodes for each of the 5 drift
    dimensions, plus log(num_nodes) and max hop depth.
    """
    mean = drift.mean(dim=0)
    std = drift.std(dim=0, unbiased=False)
    mx = drift.amax(dim=0)
    n = drift.size(0)
    extras = torch.stack(
        [
            torch.tensor(math.log(max(n, 1)), dtype=drift.dtype, device=drift.device),
            hop_depths.max().to(drift.dtype),
        ]
    )
    return torch.cat([mean, std, mx, extras], dim=0)


def get_parent_and_children(data):
    x = data.x
    edge_index = data.edge_index
    parent = getattr(data, "parent", None)
    n_children = getattr(data, "n_children", None)
    if parent is None or n_children is None:
        parent = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        n_children = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        src, dst = edge_index[:, edge_index[0] != edge_index[1]]
        for u, v in zip(src.tolist(), dst.tolist()):
            if parent[v] == -1:
                parent[v] = u
            n_children[u] += 1
    return parent, n_children


def build_pool_mlp(pool_dim, dropout, pool_out_dim=64):
    return nn.Sequential(
        nn.Linear(POOL_DIM, pool_out_dim),
        nn.LayerNorm(pool_out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(pool_out_dim, pool_dim),
        nn.ReLU(),
    )


class DriftPoolOnly(nn.Module):
    """Learned version of the drift-only discriminator (random forest analogue).

    Consumes the same aggregate drift statistics as the RF (plus max moments)
    through an MLP. No text content, no graph structure, no sequence model.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.pool_mlp = build_pool_mlp(hidden_dim, dropout, pool_out_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data, return_drift=False):
        x = data.x
        hop_depths = data.hop_depth
        parent, n_children = get_parent_and_children(data)

        drift_features = compute_drift(x, parent, hop_depths, n_children)
        pooled = pool_drift_features(drift_features, hop_depths)
        logits = self.classifier(self.pool_mlp(pooled).unsqueeze(0))

        if return_drift:
            return logits, drift_features
        return logits


class DriftPoolHybrid(nn.Module):
    """HybridDegradLSTM variant that consumes drift ONLY at graph level.

    Branches:
      1. BiLSTM over hop-sorted nodes (cascade order).
      2. Tree message passing over projected features + hop-depth embeddings
         (structure; no per-node drift concatenation — per-node drift hurt
         in the ablation study).
      3. Drift set-pooling: mean/std/max moments of per-node drift + size/depth
         scalars through an MLP (the aggregate signal that made the drift-only
         random forest discriminative).

    This directly tests the Section 6.3 hypothesis: aggregate drift helps,
    per-node drift hurts.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pool_dim: int = 64,
        use_pool: bool = True,
        use_tree: bool = True,
    ):
        super().__init__()
        self.use_pool = use_pool
        self.use_tree = use_tree

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        lstm_out_dim = hidden_dim * 2

        self.hop_embed = nn.Embedding(50, 16)
        aux_dim = 16
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Linear(2 * (hidden_dim + aux_dim), hidden_dim))

        if use_pool:
            self.pool_mlp = build_pool_mlp(pool_dim, dropout)

        fusion_dim = lstm_out_dim + hidden_dim + (pool_dim if use_pool else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data, return_drift=False):
        x = data.x
        edge_index = data.edge_index
        hop_depths = data.hop_depth
        parent, n_children = get_parent_and_children(data)

        h = self.input_proj(x)

        order = hop_depths.argsort()
        lstm_in = h[order].unsqueeze(0)
        _, (h_n, _) = self.lstm(lstm_in)
        lstm_h = torch.cat([h_n[-2], h_n[-1]], dim=-1).squeeze(0)

        if self.use_tree:
            h_depth = self.hop_embed(hop_depths.clamp(0, 49))
            for conv in self.convs:
                h_aug = torch.cat([h, h_depth], dim=-1)
                src, dst = edge_index
                neighbor_h = h_aug[src]
                aggr = torch.zeros_like(h_aug).index_add(0, dst, neighbor_h)
                counts = torch.zeros(h_aug.shape[0], 1, device=h_aug.device)
                counts = counts.index_add(
                    0, dst, torch.ones_like(src).float().unsqueeze(1)
                )
                aggr = aggr / counts.clamp(min=1)
                h = F.relu(conv(torch.cat([h_aug, aggr], dim=-1)))
            graph_h = h.mean(dim=0)
        else:
            graph_h = h.mean(dim=0)
            graph_h = F.relu(self.convs[0](torch.cat([graph_h, graph_h], dim=-1)))

        parts = [lstm_h, graph_h]
        drift_features = None
        if self.use_pool:
            drift_features = compute_drift(x, parent, hop_depths, n_children)
            pooled = pool_drift_features(drift_features, hop_depths)
            parts.append(self.pool_mlp(pooled))

        combined = torch.cat(parts, dim=-1)
        logits = self.classifier(combined.unsqueeze(0))

        if return_drift:
            return logits, drift_features
        return logits