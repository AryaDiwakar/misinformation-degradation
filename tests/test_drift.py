import torch

from features.drift import compute_drift


def test_root_drift_is_masked_and_depth_is_normalized():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    parent = torch.tensor([-1, 0, 1])
    depth = torch.tensor([0, 1, 2])
    children = torch.tensor([1, 1, 0])

    drift = compute_drift(features, parent, depth, children)

    assert drift.shape == (3, 5)
    assert torch.allclose(drift[0, :2], torch.zeros(2))
    assert torch.allclose(drift[:, 2], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.equal(drift[:, 4], torch.tensor([1.0, 0.0, 0.0]))
