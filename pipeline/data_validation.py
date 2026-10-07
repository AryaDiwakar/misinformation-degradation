"""Validation for the on-disk UPFD dataset layout."""

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import scipy.sparse as sp


class DataValidationError(ValueError):
    """Raised when the local dataset cannot be safely used."""


REQUIRED_FILES = (
    "A.txt",
    "node_graph_id.npy",
    "graph_labels.npy",
    "train_idx.npy",
    "val_idx.npy",
    "test_idx.npy",
)


def validate_upfd_dataset(
    root: str | Path,
    subset: str,
    feature_type: str = "bert",
) -> Dict[str, Any]:
    """Validate one UPFD subset and return a machine-readable summary."""
    data_dir = Path(root) / subset
    if not data_dir.is_dir():
        raise DataValidationError(f"Missing dataset directory: {data_dir}")

    required = [*REQUIRED_FILES, f"new_{feature_type}_feature.npz"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise DataValidationError(
            f"{subset}: missing required files: {', '.join(missing)}"
        )

    edges = np.loadtxt(data_dir / "A.txt", delimiter=",", dtype=np.int64, ndmin=2)
    if edges.shape[1] != 2:
        raise DataValidationError(f"{subset}: A.txt must have two columns")
    if edges.size and edges.min() < 0:
        raise DataValidationError(f"{subset}: A.txt contains negative node IDs")

    node_graph_id = np.load(data_dir / "node_graph_id.npy")
    graph_labels = np.load(data_dir / "graph_labels.npy")
    features = sp.load_npz(data_dir / f"new_{feature_type}_feature.npz")

    if node_graph_id.ndim != 1:
        raise DataValidationError(f"{subset}: node_graph_id.npy must be one-dimensional")
    if graph_labels.ndim != 1:
        raise DataValidationError(f"{subset}: graph_labels.npy must be one-dimensional")
    if features.shape[0] != node_graph_id.size:
        raise DataValidationError(
            f"{subset}: feature rows ({features.shape[0]}) do not match "
            f"node IDs ({node_graph_id.size})"
        )
    if edges.size and edges.max() >= features.shape[0]:
        raise DataValidationError(f"{subset}: A.txt references a missing node")
    if node_graph_id.size and (
        node_graph_id.min() < 0 or node_graph_id.max() >= graph_labels.size
    ):
        raise DataValidationError(f"{subset}: node graph IDs are out of range")

    split_sizes = {}
    split_sets = {}
    for split in ("train", "val", "test"):
        values = np.load(data_dir / f"{split}_idx.npy")
        if values.ndim != 1:
            raise DataValidationError(f"{subset}: {split}_idx.npy must be one-dimensional")
        if values.size and (values.min() < 0 or values.max() >= graph_labels.size):
            raise DataValidationError(f"{subset}: {split} indices are out of range")
        split_sets[split] = set(values.tolist())
        split_sizes[split] = int(values.size)

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_sets[first] & split_sets[second]
        if overlap:
            raise DataValidationError(
                f"{subset}: {first}/{second} split overlap ({len(overlap)} graphs)"
            )

    graph_counts = np.bincount(node_graph_id.astype(np.int64), minlength=graph_labels.size)
    if np.any(graph_counts == 0):
        empty = np.flatnonzero(graph_counts == 0)
        raise DataValidationError(f"{subset}: empty graphs: {empty[:5].tolist()}")

    return {
        "subset": subset,
        "feature_type": feature_type,
        "num_graphs": int(graph_labels.size),
        "num_nodes": int(features.shape[0]),
        "num_edges": int(edges.shape[0]),
        "feature_dim": int(features.shape[1]),
        "splits": split_sizes,
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(graph_labels, return_counts=True))
        },
    }


def validate_subsets(
    root: str | Path,
    subsets: Iterable[str],
    feature_type: str = "bert",
) -> Dict[str, Dict[str, Any]]:
    return {
        subset: validate_upfd_dataset(root, subset, feature_type)
        for subset in subsets
    }
