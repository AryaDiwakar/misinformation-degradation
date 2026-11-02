import numpy as np
import scipy.sparse as sp

from pipeline.data_validation import validate_upfd_dataset
from data.upfd.loader import UPFDGraphDataset


def test_validate_upfd_subset(tmp_path):
    subset_dir = tmp_path / "toy"
    subset_dir.mkdir()
    np.savetxt(
        subset_dir / "A.txt",
        np.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 3]]),
        fmt="%d",
        delimiter=",",
    )
    np.save(subset_dir / "node_graph_id.npy", np.array([0, 0, 1, 2]))
    np.save(subset_dir / "graph_labels.npy", np.array([0, 1, 0]))
    np.save(subset_dir / "train_idx.npy", np.array([0]))
    np.save(subset_dir / "val_idx.npy", np.array([1]))
    np.save(subset_dir / "test_idx.npy", np.array([2]))
    sp.save_npz(subset_dir / "new_bert_feature.npz", sp.eye(4, 8, format="csr"))

    summary = validate_upfd_dataset(tmp_path, "toy")

    assert summary["num_graphs"] == 3
    assert summary["num_nodes"] == 4
    assert summary["feature_dim"] == 8
    assert summary["splits"] == {"train": 1, "val": 1, "test": 1}


def test_multiview_features_are_validated_and_attached_to_graphs(tmp_path):
    subset_dir = tmp_path / "toy"
    subset_dir.mkdir(parents=True)
    np.savetxt(
        subset_dir / "A.txt",
        np.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 3]]),
        fmt="%d",
        delimiter=",",
    )
    np.save(subset_dir / "node_graph_id.npy", np.array([0, 0, 1, 2]))
    np.save(subset_dir / "graph_labels.npy", np.array([0, 1, 0]))
    np.save(subset_dir / "train_idx.npy", np.array([0]))
    np.save(subset_dir / "val_idx.npy", np.array([1]))
    np.save(subset_dir / "test_idx.npy", np.array([2]))
    sp.save_npz(subset_dir / "new_bert_feature.npz", sp.eye(4, 8, format="csr"))
    sp.save_npz(subset_dir / "new_profile_feature.npz", sp.eye(4, 3, format="csr"))

    summary = validate_upfd_dataset(tmp_path, "toy", "bert+profile")
    assert summary["feature_dims"] == {"bert": 8, "profile": 3}

    dataset = UPFDGraphDataset(tmp_path, "toy", "bert+profile", "train")
    graph = dataset[0]
    assert graph.x.shape == (2, 8)
    assert graph.profile_x.shape == (2, 3)
