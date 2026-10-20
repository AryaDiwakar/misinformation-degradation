"""Input loading and prediction helpers for one propagation cascade."""

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch_geometric.data import Data

from data.upfd.loader import build_parent_index, compute_hop_depths, count_children
from features.drift import compute_drift


def cascade_from_arrays(arrays: Mapping[str, Any]) -> Data:
    """Build a validated PyG cascade from NumPy-like arrays."""
    if "x" not in arrays or "edge_index" not in arrays:
        raise ValueError("cascade input must contain x and edge_index arrays")

    x = torch.as_tensor(np.asarray(arrays["x"]), dtype=torch.float32)
    edge_index = torch.as_tensor(np.asarray(arrays["edge_index"]), dtype=torch.long)
    if x.ndim != 2 or x.size(0) == 0:
        raise ValueError("x must be a non-empty [num_nodes, feature_dim] matrix")
    if edge_index.ndim != 2:
        raise ValueError("edge_index must be a rank-2 array")
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.numel() and (
        edge_index.min() < 0 or edge_index.max() >= x.size(0)
    ):
        raise ValueError("edge_index contains an invalid node ID")

    hop_depth = torch.as_tensor(
        np.asarray(arrays["hop_depth"]), dtype=torch.long
    ) if "hop_depth" in arrays else compute_hop_depths(edge_index, x.size(0))
    if hop_depth.ndim != 1 or hop_depth.numel() != x.size(0):
        raise ValueError("hop_depth must contain one value per node")

    parent = (
        torch.as_tensor(np.asarray(arrays["parent"]), dtype=torch.long)
        if "parent" in arrays
        else build_parent_index(edge_index, x.size(0), hop_depth)
    )
    n_children = (
        torch.as_tensor(np.asarray(arrays["n_children"]), dtype=torch.long)
        if "n_children" in arrays
        else count_children(edge_index, x.size(0))
    )
    if parent.numel() != x.size(0) or n_children.numel() != x.size(0):
        raise ValueError("parent and n_children must contain one value per node")

    data = Data(
        x=x,
        edge_index=edge_index,
        hop_depth=hop_depth,
        parent=parent,
        n_children=n_children,
        root_idx=torch.tensor([0]),
        n_nodes=torch.tensor([x.size(0)]),
    )
    if "y" in arrays:
        data.y = torch.as_tensor(np.asarray(arrays["y"]), dtype=torch.long).reshape(-1)
    return data


def load_cascade_npz(path: str | Path) -> Data:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("cascade inputs must be .npz files")
    with np.load(path, allow_pickle=False) as arrays:
        return cascade_from_arrays({key: arrays[key] for key in arrays.files})


def predict_cascade(model, graph: Data, device: torch.device) -> Dict[str, Any]:
    """Return a model prediction and drift explanation for one cascade."""
    model.eval()
    graph_device = graph.to(device)
    with torch.no_grad():
        logits = model(graph_device)
        probability_fake = float(torch.sigmoid(logits).reshape(-1)[0].cpu())

    drift = compute_drift(
        graph.x,
        graph.parent,
        graph.hop_depth,
        graph.n_children,
    ).cpu().numpy()
    by_hop = {}
    for hop in np.unique(graph.hop_depth.cpu().numpy()):
        mask = graph.hop_depth.cpu().numpy() == hop
        by_hop[str(int(hop))] = float(np.linalg.norm(drift[mask, :2], axis=1).mean())

    return {
        "label": "fake" if probability_fake >= 0.5 else "real",
        "probability_fake": probability_fake,
        "confidence": max(probability_fake, 1.0 - probability_fake),
        "cascade_size": int(graph.x.size(0)),
        "feature_dim": int(graph.x.size(1)),
        "max_depth": int(graph.hop_depth.max().item()),
        "mean_cosine_drift": float(drift[:, 0].mean()),
        "mean_l2_drift": float(drift[:, 1].mean()),
        "mean_content_drift": float(np.linalg.norm(drift[:, :2], axis=1).mean()),
        "content_drift_by_hop": by_hop,
    }
