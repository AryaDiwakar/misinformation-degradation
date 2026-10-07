import torch
from torch_geometric.data import Data

from pipeline.checkpoints import load_checkpoint, save_checkpoint
from pipeline.models import build_model


def make_graph():
    return Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long
        ),
        y=torch.tensor([1]),
        hop_depth=torch.tensor([0, 1, 2, 3]),
        parent=torch.tensor([-1, 0, 1, 2]),
        n_children=torch.tensor([1, 1, 1, 0]),
    )


def test_registered_models_return_one_logit():
    graph = make_graph()
    config = {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    for name in (
        "baseline_lstm",
        "degradation_gnn",
        "hybrid",
        "drift_pool",
        "drift_pool_only",
        "gcn",
        "gat",
        "bigcn",
    ):
        model = build_model(name, 8, config).eval()
        with torch.no_grad():
            output = model(graph)
        assert output.shape == (1, 1), name


def test_checkpoint_round_trip(tmp_path):
    config = {"input_dim": 8, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    model = build_model("hybrid", 8, config).eval()
    path = tmp_path / "hybrid.pt"
    save_checkpoint(
        path,
        model,
        "hybrid",
        config,
        {"data": {"subset": "synthetic"}},
        epoch=3,
        best_val_loss=0.25,
        seed=7,
    )

    loaded, metadata = load_checkpoint(path, torch.device("cpu"))
    assert metadata["model_name"] == "hybrid"
    assert metadata["epoch"] == 3
    with torch.no_grad():
        assert loaded(make_graph()).shape == (1, 1)
