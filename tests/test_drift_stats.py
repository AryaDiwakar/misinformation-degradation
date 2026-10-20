import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from pipeline.drift_stats import (
    build_drift_table,
    cascade_level_stats,
    hop_composition,
    hop_standardized_ratio,
    node_level_regression,
    per_hop_table,
    pooled_ratio,
    size_matched_ratio,
)
from pipeline.truncation import is_root_only, truncate_cascade


def make_cascade(size=6, feat_dim=8, label=0, symmetrise=False):
    """A simple chain cascade: 0 -> 1 -> 2 -> ... (node 0 is the root).

    The UPFD edge list is directed (parent -> child); ``symmetrise=True``
    builds the harder case where every edge appears in both directions.
    """
    x = torch.randn(size, feat_dim)
    src = torch.arange(size - 1, dtype=torch.long)
    dst = torch.arange(1, size, dtype=torch.long)
    if symmetrise:
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])]
        )
    else:
        edge_index = torch.stack([src, dst])
    hop_depth = torch.arange(size, dtype=torch.long)
    parent = torch.tensor([-1] + list(range(size - 1)), dtype=torch.long)
    n_children = torch.tensor([1] * (size - 1) + [0], dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([label]),
        hop_depth=hop_depth,
        parent=parent,
        n_children=n_children,
    )


def test_truncate_cascade_rebuilds_metadata():
    graph = make_cascade(size=10)
    truncated = truncate_cascade(graph, 0.5)
    assert truncated.x.size(0) == 5
    assert truncated.hop_depth.tolist() == [0, 1, 2, 3, 4]
    assert truncated.parent.tolist() == [-1, 0, 1, 2, 3]
    assert truncated.n_children.tolist() == [1, 1, 1, 1, 0]


def test_parent_assignment_survives_a_symmetrised_edge_list():
    """The root must never adopt a child as its parent."""
    from data.upfd.loader import build_parent_index

    graph = make_cascade(size=5, symmetrise=True)
    parent = build_parent_index(graph.edge_index, 5, graph.hop_depth)
    assert parent.tolist() == [-1, 0, 1, 2, 3]


def test_truncate_cascade_rejects_invalid_fractions():
    graph = make_cascade()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            truncate_cascade(graph, bad)


def test_root_only_detection():
    graph = make_cascade(size=10)
    assert not is_root_only(graph)
    assert not is_root_only(truncate_cascade(graph, 0.5))
    # A 10-node cascade truncated to 10% keeps a single node.
    assert is_root_only(truncate_cascade(graph, 0.1))


def test_truncation_is_shared_across_scripts():
    """The ablation and DTN scripts must use the same truncation helper."""
    from pipeline import truncation

    for module_name in (
        "scripts.run_ablation_10pct",
        "scripts.train_dtn",
        "scripts.run_dtn_comparison",
    ):
        module = __import__(module_name, fromlist=["truncate_graph"])
        graph = make_cascade(size=8)
        expected = truncation.truncate_cascade(graph, 0.5)
        actual = module.truncate_graph(graph, 0.5)
        assert actual.x.size(0) == expected.x.size(0)
        assert actual.parent.tolist() == expected.parent.tolist()
        assert actual.hop_depth.tolist() == expected.hop_depth.tolist()


def _synthetic_table(magnitudes, hops, labels, cascades, cascade_labels=None):
    from pipeline.drift_stats import DriftTable

    unique = sorted(set(cascades))
    if cascade_labels is None:
        # First half real, second half fake.
        midpoint = len(unique) // 2
        cascade_labels = [0] * midpoint + [1] * (len(unique) - midpoint)
    return DriftTable(
        magnitude=np.asarray(magnitudes, dtype=float),
        hop=np.asarray(hops, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        cascade=np.asarray(cascades, dtype=np.int64),
        cascade_size=np.array([4] * len(unique), dtype=np.int64),
        cascade_label=np.asarray(cascade_labels, dtype=np.int64),
        cascade_hop1_frac=np.array([0.5] * len(unique)),
    )


def test_hop_standardisation_removes_the_compositional_confound():
    """Two classes with identical per-hop drift must have ratio 1.0.

    The pooled ratio differs because the classes have different hop mixes.
    """
    # Real: 3 nodes at hop 1, 1 at hop 2. Fake: 1 at hop 1, 3 at hop 2.
    # The per-hop drift is identical for both classes (10 at hop 1, 2 at
    # hop 2), so any class difference in the pooled mean is pure composition.
    magnitudes = [10.0, 10.0, 10.0, 2.0] + [10.0, 2.0, 2.0, 2.0]
    hops = [1, 1, 1, 2] + [1, 2, 2, 2]
    labels = [0, 0, 0, 0] + [1, 1, 1, 1]
    cascades = [0, 0, 0, 0] + [1, 1, 1, 1]
    table = _synthetic_table(
        magnitudes, hops, labels, cascades, cascade_labels=[0, 1]
    )

    pooled = pooled_ratio(table)
    standardized = hop_standardized_ratio(table, min_nodes=1)

    assert pooled["ratio_b_over_a"] < 0.7  # confounded
    assert abs(standardized["ratio_fake_over_real"] - 1.0) < 1e-9  # controlled


def test_cascade_level_stats_uses_one_number_per_cascade():
    magnitudes = [10.0, 1.0, 2.0, 2.0]
    hops = [1, 2, 1, 2]
    labels = [0, 0, 1, 1]
    cascades = [0, 0, 1, 1]
    table = _synthetic_table(
        magnitudes, hops, labels, cascades, cascade_labels=[0, 1]
    )
    stats = cascade_level_stats(table, min_hop=2)
    assert stats["n_cascades"] == 2
    # real cascade mean over hop>=2 = 1.0 ; fake = 2.0
    assert stats["mean_a"] == pytest.approx(1.0)
    assert stats["mean_b"] == pytest.approx(2.0)


def test_node_level_regression_recovers_a_known_effect():
    """With a constant per-hop offset the regression coefficient must find it."""
    rng = np.random.default_rng(0)
    n_cascades = 200
    magnitudes, hops, labels, cascades = [], [], [], []
    for cascade in range(n_cascades):
        label = cascade % 2
        offset = -0.5 if label else 0.0
        for hop in (1, 2):
            magnitudes.append(5.0 + offset + rng.normal(0, 0.01))
            hops.append(hop)
            labels.append(label)
            cascades.append(cascade)
    table = _synthetic_table(magnitudes, hops, labels, cascades)
    result = node_level_regression(table)
    assert result["fake_coefficient"] == pytest.approx(-0.5, abs=0.05)
    assert result["p_value"] < 1e-6


def test_per_hop_and_composition_helpers_run():
    # Both classes present at every hop.
    magnitudes = [10.0, 10.0, 2.0] + [10.0, 2.0, 2.0]
    hops = [1, 1, 2] + [1, 2, 2]
    labels = [0, 0, 0] + [1, 1, 1]
    cascades = [0, 0, 0] + [1, 1, 1]
    table = _synthetic_table(
        magnitudes, hops, labels, cascades, cascade_labels=[0, 1]
    )
    rows = per_hop_table(table, min_nodes=1)
    assert {row["hop"] for row in rows} == {1, 2}
    composition = hop_composition(table)
    assert set(composition) == {"real", "fake"}
    assert isinstance(size_matched_ratio(table), list)


def test_drift_stats_module_exposes_the_report():
    from pipeline.drift_stats import full_report

    assert callable(full_report)
