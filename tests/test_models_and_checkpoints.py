import json

import torch
from torch_geometric.data import Data

from pipeline.checkpoints import load_checkpoint, save_checkpoint
from pipeline.models import build_model, canonical_model_name


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


def test_dtn_returns_logits_and_trajectory():
    graph = make_graph()
    model = build_model("dtn", 8, {"hidden_dim": 16, "dropout": 0.0}).eval()
    with torch.no_grad():
        output = model(graph)
    assert isinstance(output, tuple)
    assert output[0].shape == (1, 1)


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


def test_drift_pool_only_checkpoint_reloads_with_its_own_hidden_dim(tmp_path):
    """Regression test: multiseed.py recorded hidden_dim=128 for every model.

    DriftPoolOnly defaults to hidden_dim=64, so the recorded config did not
    match the trained architecture and every DriftPool checkpoint failed to
    reload with a size mismatch.
    """
    from models.drift_pool import DriftPoolOnly

    trained = DriftPoolOnly(8)  # default hidden_dim=64
    path = tmp_path / "drift_pool_only.pt"
    save_checkpoint(
        path,
        trained,
        "drift_pool_only",
        {"input_dim": 8, "hidden_dim": 64, "dropout": 0.3, "name": "drift_pool_only"},
        {"protocol": "multiseed"},
        epoch=1,
        best_val_loss=0.5,
        seed=1,
    )
    loaded, metadata = load_checkpoint(path, torch.device("cpu"))
    assert metadata["model_name"] == "drift_pool_only"
    assert loaded.pool_mlp[-2].weight.shape == trained.pool_mlp[-2].weight.shape


def test_multiseed_model_configs_match_constructor_defaults():
    """Every per-model checkpoint config must reproduce the trained shapes."""
    import multiseed

    for name, model_cls in multiseed.MODELS.items():
        config = multiseed.MODEL_CONFIGS[name]
        trained = multiseed.build_model_for(name, 8)
        rebuilt = build_model(canonical_model_name(name), 8, config)
        trained_state = trained.state_dict()
        rebuilt_state = rebuilt.state_dict()
        assert set(trained_state) == set(rebuilt_state), name
        for key in trained_state:
            assert (
                trained_state[key].shape == rebuilt_state[key].shape
            ), f"{name}.{key}: {trained_state[key].shape} vs {rebuilt_state[key].shape}"


def test_load_checkpoint_reports_shape_mismatch_clearly(tmp_path):
    path = tmp_path / "bad.pt"
    model = build_model("hybrid", 8, {"hidden_dim": 16, "num_layers": 1}).eval()
    save_checkpoint(
        path,
        model,
        "hybrid",
        {"input_dim": 8, "hidden_dim": 16, "num_layers": 1},
        {},
        epoch=1,
        best_val_loss=0.1,
        seed=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_config"]["hidden_dim"] = 32
    torch.save(payload, path)

    try:
        load_checkpoint(path, torch.device("cpu"))
    except RuntimeError as exc:
        message = str(exc)
        assert "hidden_dim" in message
        assert "32" in message
    else:  # pragma: no cover
        raise AssertionError("expected a RuntimeError for a mismatched checkpoint")


def test_model_aliases_are_stable():
    assert canonical_model_name("HybridDegradLSTM") == "hybrid"
    assert canonical_model_name("drift-pool-only") == "drift_pool_only"
    assert canonical_model_name("LSTM") == "baseline_lstm"
