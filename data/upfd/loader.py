import os
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, coalesce
from collections import deque
from typing import List, Tuple, Optional


def read_txt_array(path, sep=",", dtype=torch.long):
    with open(path) as f:
        lines = f.readlines()
    items = [list(map(int, line.strip().replace(" ", "").split(sep))) for line in lines]
    return torch.tensor(items, dtype=dtype)


def compute_hop_depths(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)
    depths = torch.full((num_nodes,), -1, dtype=torch.long)
    depths[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if depths[v] == -1:
                depths[v] = depths[u] + 1
                q.append(v)
    return depths


def build_parent_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    hop_depth: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Assign each node its parent in the propagation tree.

    The UPFD edge list is directed (parent -> child), so the first incoming
    edge of a node identifies its parent. When ``hop_depth`` is supplied the
    assignment is checked against the depths and repaired where it disagrees:
    the parent must sit exactly one hop above its child, and the root has no
    parent at all. The repair pass only runs for nodes that disagree, so the
    common case costs one extra scan of the node array.
    """
    parent = torch.full((num_nodes,), -1, dtype=torch.long)
    src, dst = edge_index[:, edge_index[0] != edge_index[1]]
    src_list, dst_list = src.tolist(), dst.tolist()
    for u, v in zip(src_list, dst_list):
        if parent[v] == -1:
            parent[v] = u

    if hop_depth is None:
        return parent

    hop = hop_depth.tolist()
    mismatched = [
        v
        for v in range(num_nodes)
        if parent[v].item() != -1
        and (hop[v] == 0 or hop[parent[v].item()] != hop[v] - 1)
    ]
    if not mismatched:
        return parent

    needed = set(mismatched)
    candidates: dict = {}
    for u, v in zip(src_list, dst_list):
        if v in needed:
            candidates.setdefault(v, []).append(u)

    for v in mismatched:
        if hop[v] == 0:
            # The root has no parent, by definition.
            parent[v] = -1
            continue
        expected = hop[v] - 1
        options = candidates.get(v, [])
        matching = [u for u in options if hop[u] == expected]
        parent[v] = matching[0] if matching else options[0]
    return parent


def count_children(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    n_children = torch.zeros(num_nodes, dtype=torch.long)
    for u, v in zip(*edge_index[:, edge_index[0] != edge_index[1]]):
        n_children[u] += 1
    return n_children


class UPFDGraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str = "gossipcop",
        feature_type: str = "bert",
        split: str = "train",
    ):
        self.root = root
        self.subset = subset
        self.feature_type = feature_type
        self.split = split
        data_dir = os.path.join(root, subset)

        feature_types = [name.strip() for name in feature_type.split("+")]
        if not feature_types or any(not name for name in feature_types):
            raise ValueError(f"Invalid feature_type {feature_type!r}")
        if len(set(feature_types)) != len(feature_types):
            raise ValueError(f"Duplicate feature types in {feature_type!r}")

        edge_index = read_txt_array(os.path.join(data_dir, "A.txt"), sep=",").t()
        node_graph_id = torch.from_numpy(
            np.load(os.path.join(data_dir, "node_graph_id.npy"))
        ).long()
        graph_labels = torch.from_numpy(
            np.load(os.path.join(data_dir, "graph_labels.npy"))
        ).long()
        _, graph_labels = graph_labels.unique(sorted=True, return_inverse=True)

        feature_matrices = {}
        for name in feature_types:
            feature_path = os.path.join(data_dir, f"new_{name}_feature.npz")
            feature_matrices[name] = torch.from_numpy(
                sp.load_npz(feature_path).toarray()
            ).float()
        x = feature_matrices[feature_types[0]]
        for name, features in feature_matrices.items():
            if features.size(0) != x.size(0):
                raise ValueError(
                    f"Feature rows for {name!r} do not align with {feature_types[0]!r}"
                )

        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

        num_graphs = graph_labels.size(0)
        node_graph_np = node_graph_id.numpy()
        if node_graph_np.size != num_nodes:
            raise ValueError(
                f"Node graph IDs ({node_graph_np.size}) do not align with features ({num_nodes})"
            )
        if node_graph_np.size and (
            node_graph_np.min() < 0 or node_graph_np.max() >= num_graphs
        ):
            raise ValueError("Node graph IDs reference a missing graph")

        # Group nodes and edges once. The former implementation built a full
        # node mask and scanned the full edge list separately for every graph,
        # making dataset construction O(num_graphs * (nodes + edges)).
        node_order = np.argsort(node_graph_np, kind="stable")
        node_counts = np.bincount(node_graph_np, minlength=num_graphs)
        node_offsets = np.concatenate([[0], node_counts.cumsum()])

        edge_np = edge_index.numpy()
        source_graph = node_graph_np[edge_np[0]]
        same_graph = source_graph == node_graph_np[edge_np[1]]
        edge_index = edge_index[:, torch.from_numpy(same_graph)]
        source_graph = source_graph[same_graph]
        edge_order = np.argsort(source_graph, kind="stable")
        edge_counts = np.bincount(source_graph, minlength=num_graphs)
        edge_offsets = np.concatenate([[0], edge_counts.cumsum()])

        global_to_local = torch.empty(num_nodes, dtype=torch.long)
        graphs = []
        for g in range(num_graphs):
            node_start, node_end = node_offsets[g : g + 2]
            nodes = torch.from_numpy(node_order[node_start:node_end].copy())
            if nodes.numel() == 0:
                continue
            global_to_local[nodes] = torch.arange(nodes.numel())
            edge_start, edge_end = edge_offsets[g : g + 2]
            selected_edges = torch.from_numpy(edge_order[edge_start:edge_end].copy())
            g_edge_index = global_to_local[edge_index[:, selected_edges]]
            g_x = x[nodes]
            hop_depth = compute_hop_depths(g_edge_index, nodes.size(0))
            parent = build_parent_index(g_edge_index, nodes.size(0), hop_depth)
            n_children = count_children(g_edge_index, nodes.size(0))
            extra_features = {
                f"{name}_x": features[nodes]
                for name, features in feature_matrices.items()
                if name != feature_types[0]
            }
            data = Data(
                x=g_x,
                edge_index=g_edge_index,
                y=graph_labels[g].unsqueeze(0),
                hop_depth=hop_depth,
                parent=parent,
                n_children=n_children,
                root_idx=torch.tensor([0]),
                n_nodes=torch.tensor([nodes.size(0)]),
                **extra_features,
            )
            graphs.append(data)

        split_idx = torch.from_numpy(
            np.load(os.path.join(data_dir, f"{split}_idx.npy"))
        ).long()
        self.graphs = [graphs[i] for i in split_idx]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_fn(batch):
    return Batch.from_data_list(batch)


def get_upfd_loaders(
    root: str = "./data/upfd/raw",
    subset: str = "gossipcop",
    feature_type: str = "bert",
    batch_size: int = 32,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = UPFDGraphDataset(root, subset, feature_type, "train")
    val_ds = UPFDGraphDataset(root, subset, feature_type, "val")
    test_ds = UPFDGraphDataset(root, subset, feature_type, "test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
