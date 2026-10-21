import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_drift(
    node_features: torch.Tensor,
    parent_indices: torch.Tensor,
    hop_depths: torch.Tensor,
    n_children: torch.Tensor,
) -> torch.Tensor:
    """
    Computes per-node drift features relative to parent.

    Args:
        node_features: [num_nodes, feat_dim] — text or joint embeddings
        parent_indices: [num_nodes] — index of parent for each node (-1 for root)
        hop_depths: [num_nodes] — depth in propagation tree
        n_children: [num_nodes] — number of child nodes

    Returns:
        drift_features: [num_nodes, drift_dim]
            - text_cosine_dist_to_parent
            - l2_dist_to_parent
            - depth (normalized)
            - n_children (normalized)
            - is_root (binary)
    """
    num_nodes = node_features.shape[0]
    device = node_features.device

    parent_features = node_features[parent_indices.clamp(min=0)]
    parent_mask = (parent_indices >= 0).float().unsqueeze(1)

    cosine_dist = 1 - F.cosine_similarity(node_features, parent_features, dim=-1)
    cosine_dist = cosine_dist.unsqueeze(1) * parent_mask

    l2_dist = torch.norm(node_features - parent_features, dim=-1, keepdim=True)
    l2_dist = l2_dist * parent_mask

    depth_norm = (hop_depths / hop_depths.max().clamp(min=1)).unsqueeze(1)
    children_norm = (n_children / n_children.max().clamp(min=1)).unsqueeze(1)
    is_root = (hop_depths == 0).float().unsqueeze(1)

    drift = torch.cat(
        [cosine_dist, l2_dist, depth_norm, children_norm, is_root], dim=-1
    )
    return drift


class DriftProjector(nn.Module):
    """Projects raw drift features into a learned drift embedding.

    Not used by any current model: the DegradGNN, HybridDegradLSTM and
    DriftPool variants project the drift vector inline. Kept for reuse by
    future architectures.
    """

    def __init__(self, drift_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(drift_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, drift: torch.Tensor) -> torch.Tensor:
        return self.net(drift)
