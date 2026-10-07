import torch
import torch.nn as nn
import torch.nn.functional as F

from features.drift import compute_drift


class DegradationGNN(nn.Module):
    """
    BiGCN-style propagation GNN with temporal degradation encoding.

    Uses cached parent/n_children from data loader for performance.
    Processes propagation trees by:
    1. Message passing over graph structure (like BiGCN)
    2. Temporal encoding of hop depths
    3. Drift features between parent-child pairs
    4. Multi-task: veracity classification + drift prediction
    """
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        drift_dim: int = 5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hop_embed = nn.Embedding(50, 16)
        self.drift_proj = nn.Linear(drift_dim, 16)

        self.convs = nn.ModuleList()
        aux_dim = 32
        for _ in range(num_layers):
            self.convs.append(nn.Linear(2 * (hidden_dim + aux_dim), hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.drift_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, drift_dim),
        )

    def forward(self, data, return_drift=False):
        edge_index = data.edge_index
        hop_depths = data.hop_depth
        x = data.x

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

        drift_features = compute_drift(x, parent, hop_depths, n_children)

        h = self.input_proj(x)
        h_depth = self.hop_embed(hop_depths.clamp(0, 49))
        h_drift = self.drift_proj(drift_features)

        for conv in self.convs:
            h_aug = torch.cat([h, h_depth, h_drift], dim=-1)
            src, dst = edge_index
            neighbor_h = h_aug[src]
            aggr = torch.zeros_like(h_aug)
            aggr = aggr.index_add(0, dst, neighbor_h)
            counts = torch.zeros(h_aug.shape[0], 1, device=h_aug.device)
            counts = counts.index_add(0, dst, torch.ones_like(src).float().unsqueeze(1))
            aggr = aggr / counts.clamp(min=1)
            h_new = conv(torch.cat([h_aug, aggr], dim=-1))
            h = F.relu(h_new)

        graph_h = h.mean(dim=0, keepdim=True)
        logits = self.classifier(graph_h)

        if return_drift:
            drift_pred = self.drift_head(h)
            return logits, drift_pred

        return logits
