import numpy as np
import scipy.sparse as sp

from pipeline.data_validation import validate_upfd_dataset


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
