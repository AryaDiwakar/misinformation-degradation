import torch
import torch.nn as nn
import torch.nn.functional as F

from features.drift import compute_drift


class DegradationTrajectoryNetwork(nn.Module):
    """
    Trajectory-aware fake news detector.

    Core idea: given a partial propagation cascade, predict how the
    degradation trajectory will evolve and classify veracity based on
    both the cascade structure and the predicted trajectory.

    Fake news maintains artificially stable coherence (flat trajectory).
    Real news degrades naturally (steep trajectory).

    Architecture:
        1. GNN encodes cascade structure + drift features
        2. Transformer decodes the full trajectory from partial observation
        3. Coherence score derived from trajectory prediction
        4. Veracity classification from graph repr + trajectory repr
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        drift_dim: int = 5,
        max_seq_len: int = 50,
        nhead: int = 4,
        traj_layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hop_embed = nn.Embedding(50, 16)
        self.drift_proj = nn.Linear(drift_dim, 16)
        self.drift_dim = drift_dim

        gnn_aux_dim = 32
        self.gnns = nn.ModuleList()
        for _ in range(num_layers):
            self.gnns.append(nn.Linear(2 * (hidden_dim + gnn_aux_dim), hidden_dim))

        self.gnn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.trajectory_encoder = nn.Linear(drift_dim, hidden_dim)
        self.trajectory_pos = nn.Embedding(max_seq_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.trajectory_transformer = nn.TransformerEncoder(enc_layer, num_layers=traj_layers)
        self.trajectory_projector = nn.Linear(hidden_dim, drift_dim)

        self.coherence_scorer = nn.Sequential(
            nn.Linear(drift_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        fusion_dim = hidden_dim + hidden_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.drift_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, drift_dim),
        )

    def encode_cascade(self, x, edge_index, hop_depths, drift_features):
        h = self.input_proj(x)
        h_depth = self.hop_embed(hop_depths.clamp(0, 49))
        h_drift = self.drift_proj(drift_features)

        for gnn in self.gnns:
            h_aug = torch.cat([h, h_depth, h_drift], dim=-1)
            src, dst = edge_index
            neighbor_h = h_aug[src]
            aggr = torch.zeros_like(h_aug)
            aggr = aggr.index_add(0, dst, neighbor_h)
            counts = torch.zeros(h_aug.shape[0], 1, device=h_aug.device)
            counts = counts.index_add(0, dst, torch.ones_like(src, dtype=torch.float).unsqueeze(1))
            aggr = aggr / counts.clamp(min=1)
            h = F.relu(gnn(torch.cat([h_aug, aggr], dim=-1)))

        return h

    def predict_trajectory(self, hop_drift_stats):
        batch_size = hop_drift_stats.shape[0]
        seq_len = hop_drift_stats.shape[1]

        x = self.trajectory_encoder(hop_drift_stats)
        positions = self.trajectory_pos(
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = x + positions
        hidden = self.trajectory_transformer(x)
        trajectory_pred = self.trajectory_projector(hidden)
        return trajectory_pred, hidden

    def forward(self, data, return_trajectory=False):
        x = data.x
        edge_index = data.edge_index
        hop_depths = data.hop_depth
        n_nodes = data.x.size(0)

        parent = getattr(data, "parent", None)
        n_children = getattr(data, "n_children", None)
        if parent is None or n_children is None:
            parent = torch.full((n_nodes,), -1, dtype=torch.long, device=x.device)
            n_children = torch.zeros(n_nodes, dtype=torch.long, device=x.device)
            src, dst = edge_index[:, edge_index[0] != edge_index[1]]
            for u, v in zip(src.tolist(), dst.tolist()):
                if parent[v] == -1:
                    parent[v] = u
                n_children[u] += 1

        drift_features = compute_drift(x, parent, hop_depths, n_children)

        h = self.encode_cascade(x, edge_index, hop_depths, drift_features)
        graph_repr = h.mean(dim=0)

        max_hop = hop_depths.max().item() + 1
        hop_drift_stats = []
        for hop in range(max_hop):
            mask = hop_depths == hop
            if mask.sum() > 0:
                hop_mean = drift_features[mask].mean(dim=0)
            else:
                hop_mean = torch.zeros(self.drift_dim, device=x.device)
            hop_drift_stats.append(hop_mean)
        hop_drift_stats = torch.stack(hop_drift_stats, dim=0).unsqueeze(0)

        trajectory_pred, traj_hidden = self.predict_trajectory(hop_drift_stats)
        trajectory_pred = trajectory_pred.squeeze(0)
        traj_hidden = traj_hidden.squeeze(0)

        mean_trajectory = trajectory_pred.mean(dim=0)
        coherence = self.coherence_scorer(mean_trajectory)

        traj_repr = traj_hidden.mean(dim=0)
        combined = torch.cat([graph_repr, traj_repr, coherence], dim=-1)
        logits = self.classifier(combined.unsqueeze(0))

        drift_pred = self.drift_head(h)

        if return_trajectory:
            return logits, drift_pred, trajectory_pred, coherence

        return logits, drift_pred
