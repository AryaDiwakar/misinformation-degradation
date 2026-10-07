import torch
import torch.nn as nn
import torch.nn.functional as F

from features.drift import compute_drift


class AblationGNN(nn.Module):
    """DegradGNN with ablation flags for component analysis.

    use_drift:   include drift features (parent-child degradation)
    use_hop:     include hop-depth temporal embedding
    use_tree:    message passing over tree structure; if False, MLP over mean of node embeddings
    """
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_drift: bool = True,
        use_hop: bool = True,
        use_tree: bool = True,
    ):
        super().__init__()
        self.use_drift = use_drift
        self.use_hop = use_hop
        self.use_tree = use_tree

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hop_embed = nn.Embedding(50, 16)
        self.drift_proj = nn.Linear(5, 16)

        aux_dim = (16 if use_hop else 0) + (16 if use_drift else 0)
        self.aux_dim = aux_dim
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Linear(2 * (hidden_dim + aux_dim), hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        hop_depths = data.hop_depth

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

        h = self.input_proj(x)
        aux_parts = []
        if self.use_hop:
            aux_parts.append(self.hop_embed(hop_depths.clamp(0, 49)))
        if self.use_drift:
            drift_features = compute_drift(x, parent, hop_depths, n_children)
            aux_parts.append(self.drift_proj(drift_features))

        if self.use_tree:
            for conv in self.convs:
                if aux_parts:
                    h_aug = torch.cat([h] + aux_parts, dim=-1)
                else:
                    h_aug = h
                src, dst = edge_index
                neighbor_h = h_aug[src]
                aggr = torch.zeros_like(h_aug)
                aggr = aggr.index_add(0, dst, neighbor_h)
                counts = torch.zeros(h_aug.shape[0], 1, device=h_aug.device)
                counts = counts.index_add(0, dst, torch.ones_like(src).float().unsqueeze(1))
                aggr = aggr / counts.clamp(min=1)
                h_new = conv(torch.cat([h_aug, aggr], dim=-1))
                h = F.relu(h_new)
        else:
            for conv in self.convs:
                if aux_parts:
                    h_aug = torch.cat([h] + aux_parts, dim=-1)
                else:
                    h_aug = h
                h_new = conv(torch.cat([h_aug, h_aug], dim=-1))
                h = F.relu(h_new)

        graph_h = h.mean(dim=0, keepdim=True)
        return self.classifier(graph_h)


class BERTMLP(nn.Module):
    """Mean-pooled BERT features -> MLP. Ignores graph structure entirely."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        h = data.x.mean(dim=0, keepdim=True)
        return self.net(h)
