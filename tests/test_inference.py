import numpy as np
import torch

from pipeline.inference import cascade_from_arrays, predict_cascade
from pipeline.models import build_model


def test_inference_builds_graph_and_returns_explanation():
    graph = cascade_from_arrays(
        {
            "x": np.ones((3, 8), dtype="float32"),
            "edge_index": np.array([[0, 1, 2, 0, 0], [0, 1, 2, 1, 2]]),
            "hop_depth": np.array([0, 1, 1]),
        }
    )
    model = build_model("hybrid", 8, {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0})
    result = predict_cascade(model, graph, torch.device("cpu"))

    assert result["label"] in {"fake", "real"}
    assert 0 <= result["probability_fake"] <= 1
    assert result["cascade_size"] == 3
    assert set(result["content_drift_by_hop"]) == {"0", "1"}
