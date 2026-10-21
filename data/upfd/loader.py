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

        edge_index = read_txt_array(os.path.join(data_dir, "A.txt"), sep=",").t()
        node_graph_id = torch.from_numpy(
            np.load(os.path.join(data_dir, "node_graph_id.npy"))
        ).long()
        graph_labels = torch.from_numpy(
            np.load(os.path.join(data_dir, "graph_labels.npy"))
        ).long()
        _, graph_labels = graph_labels.unique(sorted=True, return_inverse=True)

        feature_path = os.path.join(data_dir, f"new_{feature_type}_feature.npz")
        x = torch.from_numpy(sp.load_npz(feature_path).todense()).float()

        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

        num_graphs = graph_labels.size(0)
        graphs = []
        for g in range(num_graphs):
            nodes = (node_graph_id == g).nonzero(as_tuple=True)[0]
            if nodes.numel() == 0:
                continue
            node_offset = nodes[0].item()
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[nodes] = True
            mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            g_edge_index = edge_index[:, mask].clone()
            g_edge_index -= node_offset
            g_x = x[nodes]
            hop_depth = compute_hop_depths(g_edge_index, nodes.size(0))
            parent = build_parent_index(g_edge_index, nodes.size(0), hop_depth)
            n_children = count_children(g_edge_index, nodes.size(0))
            data = Data(
                x=g_x,
                edge_index=g_edge_index,
                y=graph_labels[g].unsqueeze(0),
                hop_depth=hop_depth,
                parent=parent,
                n_children=n_children,
                root_idx=torch.tensor([0]),
                n_nodes=torch.tensor([nodes.size(0)]),
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
