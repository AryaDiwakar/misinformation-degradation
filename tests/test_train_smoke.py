import json
import sys

import numpy as np
import scipy.sparse as sp


def test_canonical_train_writes_checkpoint_and_results(tmp_path, monkeypatch):
    data_root = tmp_path / "raw"
    subset_dir = data_root / "toy"
    subset_dir.mkdir(parents=True)

    edges = []
    graph_ids = []
    for graph_id in range(6):
        root = graph_id * 2
        child = root + 1
        edges.extend([(root, root), (child, child), (root, child)])
        graph_ids.extend([graph_id, graph_id])
    np.savetxt(subset_dir / "A.txt", np.array(edges), fmt="%d", delimiter=",")
    np.save(subset_dir / "node_graph_id.npy", np.array(graph_ids))
    np.save(subset_dir / "graph_labels.npy", np.array([0, 1, 0, 1, 0, 1]))
    np.save(subset_dir / "train_idx.npy", np.array([0, 1, 2]))
    np.save(subset_dir / "val_idx.npy", np.array([3]))
    np.save(subset_dir / "test_idx.npy", np.array([4, 5]))
    sp.save_npz(
        subset_dir / "new_bert_feature.npz",
        sp.eye(12, 8, format="csr"),
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""data:\n  root: {data_root}\n  subset: toy\n  feature_type: bert\n  batch_size: 2\nmodel:\n  name: hybrid\n  hidden_dim: 8\n  num_layers: 1\n  dropout: 0.0\ntraining:\n  epochs: 1\n  lr: 0.001\n  weight_decay: 0.0\n  early_stop_patience: 2\n  drift_weight: 0.1\nlogging:\n  log_dir: {tmp_path / 'logs'}\n  checkpoint_dir: {tmp_path / 'checkpoints'}\n  result_dir: {tmp_path / 'results'}\n"""
    )

    import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--seed",
            "3",
        ],
    )
    train.main()

    # The config-driven trainer uses a distinct filename so it can never
    # overwrite the canonical multiseed checkpoint for the same model/seed.
    checkpoint = tmp_path / "checkpoints" / "hybrid_toy_seed3_trainaug.pt"
    result = tmp_path / "results" / "hybrid_toy_seed3_trainaug.json"
    assert checkpoint.is_file()
    assert result.is_file()
    saved = json.loads(result.read_text())
    assert saved["model"] == "hybrid"
    assert saved["dataset"] == "toy"
    assert saved["checkpoint"] == str(checkpoint)
    assert saved["protocol"] == "train_config"


def test_cross_modal_tree_trains_with_feature_override(tmp_path, monkeypatch):
    data_root = tmp_path / "raw"
    subset_dir = data_root / "toy"
    subset_dir.mkdir(parents=True)
    edges = []
    graph_ids = []
    for graph_id in range(6):
        root = graph_id * 2
        child = root + 1
        edges.extend([(root, root), (child, child), (root, child)])
        graph_ids.extend([graph_id, graph_id])
    np.savetxt(subset_dir / "A.txt", np.array(edges), fmt="%d", delimiter=",")
    np.save(subset_dir / "node_graph_id.npy", np.array(graph_ids))
    np.save(subset_dir / "graph_labels.npy", np.array([0, 1, 0, 1, 0, 1]))
    np.save(subset_dir / "train_idx.npy", np.array([0, 1, 2]))
    np.save(subset_dir / "val_idx.npy", np.array([3]))
    np.save(subset_dir / "test_idx.npy", np.array([4, 5]))
    sp.save_npz(subset_dir / "new_bert_feature.npz", sp.eye(12, 8, format="csr"))
    sp.save_npz(subset_dir / "new_profile_feature.npz", sp.eye(12, 10, format="csr"))

    config_path = tmp_path / "cross_modal.yaml"
    config_path.write_text(
        f"""data:\n  root: {data_root}\n  subset: toy\n  feature_type: bert\n  batch_size: 2\nmodel:\n  name: cross_modal_tree\n  hidden_dim: 8\n  num_layers: 1\n  dropout: 0.0\ntraining:\n  epochs: 1\n  lr: 0.001\n  weight_decay: 0.0\n  early_stop_patience: 2\n  use_drift_aux_loss: false\nlogging:\n  log_dir: {tmp_path / 'logs'}\n  checkpoint_dir: {tmp_path / 'checkpoints'}\n  result_dir: {tmp_path / 'results'}\n"""
    )

    import train

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--features",
            "bert+profile",
            "--device",
            "cpu",
            "--seed",
            "5",
        ],
    )
    train.main()

    assert (tmp_path / "checkpoints" / "cross_modal_tree_toy_seed5_trainaug.pt").is_file()
    result = json.loads(
        (tmp_path / "results" / "cross_modal_tree_toy_seed5_trainaug.json").read_text()
    )
    assert result["model"] == "cross_modal_tree"


def test_train_and_multiseed_use_distinct_checkpoint_filenames(tmp_path):
    """Regression test: both runners used to write the same filename.

    Running the config-driven trainer silently replaced the canonical paper
    checkpoint with the weaker drift-auxiliary-loss variant.
    """
    from pathlib import Path

    train_source = Path("train.py").read_text()
    multiseed_source = Path("multiseed.py").read_text()
    assert "_trainaug" in train_source
    assert "_trainaug" not in multiseed_source
