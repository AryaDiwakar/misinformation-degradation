"""Correct, shared cascade truncation for early-detection evaluation."""

import torch
from torch_geometric.data import Data

from data.upfd.loader import build_parent_index, compute_hop_depths, count_children


def truncate_cascade(graph: Data, fraction: float) -> Data:
    """Keep the shallowest fraction of nodes and rebuild graph metadata."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    if graph.x.size(0) == 0:
        raise ValueError("cannot truncate an empty cascade")

    order = graph.hop_depth.argsort(stable=True)
    keep_count = max(1, int(graph.x.size(0) * fraction))
    keep_nodes = order[:keep_count]

    node_map = torch.full(
        (graph.x.size(0),), -1, dtype=torch.long, device=graph.x.device
    )
    node_map[keep_nodes] = torch.arange(keep_count, device=graph.x.device)
    edge_mask = (
        (node_map[graph.edge_index[0]] >= 0)
        & (node_map[graph.edge_index[1]] >= 0)
    )
    edge_index = node_map[graph.edge_index[:, edge_mask]]
    hop_depth = compute_hop_depths(edge_index.cpu(), keep_count).to(graph.x.device)
    parent = build_parent_index(edge_index.cpu(), keep_count).to(graph.x.device)
    n_children = count_children(edge_index.cpu(), keep_count).to(graph.x.device)

    result = Data(
        x=graph.x[keep_nodes],
        edge_index=edge_index,
        y=graph.y.clone() if getattr(graph, "y", None) is not None else None,
        hop_depth=hop_depth,
        parent=parent,
        n_children=n_children,
        root_idx=torch.tensor([0], device=graph.x.device),
        n_nodes=torch.tensor([keep_count], device=graph.x.device),
    )
    return result
