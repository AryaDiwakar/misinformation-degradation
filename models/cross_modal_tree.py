"""Cross-modal, edge-conditioned tree model for UPFD cascades.

The model keeps text and user-profile features in separate encoders, fuses them
with a learned gate, and passes information in both directions along each
parent/child relation. Edge messages are conditioned on the text and profile
change across that relation rather than on node features alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax


def _parent_indices(data):
    parent = getattr(data, "parent", None)
    if parent is None:
        raise ValueError("CrossModalTreeNet requires parent indices on each cascade")
    return parent


def _modal_edge_drift(text, profile, parent):
    valid = parent >= 0
    safe_parent = parent.clamp(min=0)
    text_drift = 1 - F.cosine_similarity(text, text[safe_parent], dim=-1)
    profile_drift = 1 - F.cosine_similarity(profile, profile[safe_parent], dim=-1)
    drift = torch.stack([text_drift, profile_drift], dim=-1)
    return drift * valid.unsqueeze(-1).to(drift.dtype)


class EdgeConditionedTreeLayer(nn.Module):
    """Bidirectional parent/child message passing with relation-conditioned gates."""

    def __init__(self, hidden_dim, hop_dim, dropout):
        super().__init__()
        edge_dim = 4 * hidden_dim + 2
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direction_gate = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_dim + hop_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, parent, hop_embedding, edge_drift):
        valid = parent >= 0
        if not valid.any():
            return self.norm(h + self.update(torch.cat(
                [h, torch.zeros_like(h), torch.zeros_like(h), hop_embedding], dim=-1
            )))

        child = valid.nonzero(as_tuple=True)[0]
        ancestor = parent[child]
        parent_h = h[ancestor]
        child_h = h[child]
        relation = torch.cat(
            [parent_h, child_h, (parent_h - child_h).abs(), parent_h * child_h,
             edge_drift[child]],
            dim=-1,
        )
        parent_to_child, child_to_parent = self.edge_encoder(relation).chunk(2, dim=-1)
        gate = torch.sigmoid(self.direction_gate(relation))
        parent_to_child = parent_to_child * gate
        child_to_parent = child_to_parent * (1 - gate)

        child_messages = torch.zeros_like(h).index_add(0, child, parent_to_child)
        parent_messages = torch.zeros_like(h).index_add(0, ancestor, child_to_parent)
        child_counts = torch.zeros(h.size(0), 1, device=h.device, dtype=h.dtype)
        child_counts = child_counts.index_add(
            0, child, torch.ones((child.numel(), 1), device=h.device, dtype=h.dtype)
        )
        parent_counts = torch.zeros_like(child_counts).index_add(
            0, ancestor,
            torch.ones((ancestor.numel(), 1), device=h.device, dtype=h.dtype),
        )
        child_messages = child_messages / child_counts.clamp(min=1)
        parent_messages = parent_messages / parent_counts.clamp(min=1)
        update = self.update(
            torch.cat([h, child_messages, parent_messages, hop_embedding], dim=-1)
        )
        return self.norm(h + self.dropout(update))


class CrossModalTreeNet(nn.Module):
    """Text/profile fusion with edge-aware, bidirectional tree propagation.

    ``data.x`` must contain text embeddings and ``data.profile_x`` must contain
    the aligned UPFD profile features. The classifier uses attention, mean, and
    max readouts so it can retain sparse cascade evidence as well as global
    trends.
    """

    def __init__(
        self,
        input_dim=768,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        profile_dim=10,
        hop_dim=16,
        modality_dropout=0.1,
    ):
        super().__init__()
        self.supports_batch = True
        self.profile_dim = int(profile_dim)
        self.modality_dropout = float(modality_dropout)
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(self.profile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.hop_embedding = nn.Embedding(64, hop_dim)
        self.hop_projection = nn.Linear(hop_dim, hidden_dim)
        self.layers = nn.ModuleList(
            EdgeConditionedTreeLayer(hidden_dim, hop_dim, dropout)
            for _ in range(num_layers)
        )
        self.readout_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _apply_modality_dropout(self, text, profile):
        if not self.training or self.modality_dropout <= 0:
            return text, profile
        if torch.rand((), device=text.device) < self.modality_dropout:
            if torch.rand((), device=text.device) < 0.5:
                text = torch.zeros_like(text)
            else:
                profile = torch.zeros_like(profile)
        return text, profile

    def forward(self, data):
        profile_x = getattr(data, "profile_x", None)
        if profile_x is None:
            raise ValueError(
                "CrossModalTreeNet requires profile_x; load cascades with "
                "feature_type='bert+profile'"
            )
        if profile_x.ndim != 2 or profile_x.size(0) != data.x.size(0):
            raise ValueError(
                "profile_x must be a [num_nodes, profile_dim] matrix aligned with data.x"
            )
        if profile_x.size(-1) != self.profile_dim:
            raise ValueError(
                f"Expected {self.profile_dim} profile features, got {profile_x.size(-1)}"
            )

        parent = _parent_indices(data)
        graph_ids = getattr(data, "batch", None)
        ptr = getattr(data, "ptr", None)
        if graph_ids is None:
            graph_ids = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            num_graphs = 1
        else:
            num_graphs = int(graph_ids.max().item()) + 1
            # PyG does not increment the custom ``parent`` field when batching;
            # convert its per-cascade indices to the concatenated node space.
            if ptr is not None:
                valid = parent >= 0
                parent = parent.clone()
                parent[valid] += ptr[graph_ids[valid]]
        text = self.text_encoder(data.x)
        profile = self.profile_encoder(profile_x)
        text, profile = self._apply_modality_dropout(text, profile)
        fusion_input = torch.cat(
            [text, profile, (text - profile).abs(), text * profile], dim=-1
        )
        gate = torch.sigmoid(self.fusion_gate(fusion_input))
        h = self.fusion_norm(gate * text + (1 - gate) * profile)

        hop = data.hop_depth.clamp(0, self.hop_embedding.num_embeddings - 1)
        hop_embedding = self.hop_embedding(hop)
        h = h + self.hop_projection(hop_embedding)
        edge_drift = _modal_edge_drift(data.x, profile_x, parent)
        for layer in self.layers:
            h = layer(h, parent, hop_embedding, edge_drift)

        attention = softmax(self.readout_score(h).squeeze(-1), graph_ids)
        attended = global_add_pool(h * attention.unsqueeze(-1), graph_ids, size=num_graphs)
        mean = global_mean_pool(h, graph_ids, size=num_graphs)
        maximum = global_max_pool(h, graph_ids, size=num_graphs)
        pooled = torch.cat([attended, mean, maximum], dim=-1)
        return self.classifier(pooled)
