import torch
import torch.nn as nn
import torch.nn.functional as F

from features.drift import compute_drift


class HybridDegradLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True, batch_first=True,
        )
        lstm_out_dim = hidden_dim * 2

        self.hop_embed = nn.Embedding(50, 16)
        self.drift_proj = nn.Linear(5, 16)
        self.convs = nn.ModuleList()
        aux_dim = 32
        for _ in range(num_layers):
            self.convs.append(nn.Linear(2 * (hidden_dim + aux_dim), hidden_dim))

        fusion_dim = lstm_out_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.drift_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, data, return_drift=False):
        x = data.x
        edge_index = data.edge_index
        hop_depths = data.hop_depth

        h = self.input_proj(x)

        order = hop_depths.argsort()
        lstm_in = h[order].unsqueeze(0)
        _, (h_n, _) = self.lstm(lstm_in)
        if self.lstm.bidirectional:
            lstm_h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            lstm_h = h_n[-1]
        lstm_h = lstm_h.squeeze(0)

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

        graph_h = h.mean(dim=0)

        combined = torch.cat([lstm_h, graph_h], dim=-1)
        logits = self.classifier(combined.unsqueeze(0))

        if return_drift:
            drift_pred = self.drift_head(h)
            return logits, drift_pred

        return logits
