import torch
from torch_geometric.data import Data

from pipeline.truncation import truncate_cascade


def make_graph():
    # Root -> two children, with self-loops matching the loader output.
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 0], [0, 1, 2, 1, 2]], dtype=torch.long
    )
    return Data(
        x=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        edge_index=edge_index,
        y=torch.tensor([1]),
        hop_depth=torch.tensor([0, 1, 1]),
        parent=torch.tensor([-1, 0, 0]),
        n_children=torch.tensor([2, 0, 0]),
    )


def test_truncation_rebuilds_parent_and_children():
    truncated = truncate_cascade(make_graph(), 0.67)

    assert truncated.x.shape == (2, 4)
    assert torch.equal(truncated.hop_depth, torch.tensor([0, 1]))
    assert torch.equal(truncated.parent, torch.tensor([-1, 0]))
    assert torch.equal(truncated.n_children, torch.tensor([1, 0]))
    assert int(truncated.edge_index.max()) < 2


def test_truncation_rejects_invalid_fraction():
    for fraction in (0.0, -0.1, 1.1):
        try:
            truncate_cascade(make_graph(), fraction)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid truncation fraction was accepted")
