"""Shared drift statistics used by the analysis scripts and the paper.

Drift magnitude follows the paper's definition exactly: the L2 norm of the
(cosine distance to parent, L2 distance to parent) pair, i.e.

    m_i = || [1 - cos(x_i, x_p), ||x_i - x_p||_2] ||_2

Root nodes have no parent, so their content drift is zero *by construction*.
Every content-drift comparison in this module therefore excludes hop-0 nodes;
`hop_standardized_ratio` and `cascade_level_stats` additionally control for
the propagation-depth composition of each class, which is the confound that
makes the naive pooled ratio misleading (real cascades are hop-1-heavy and
hop-1 nodes carry far more drift than deep nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy import stats

from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift

REAL = 0
FAKE = 1


def drift_magnitude(
    x: torch.Tensor,
    parent: torch.Tensor,
    hop_depth: torch.Tensor,
    n_children: torch.Tensor,
) -> np.ndarray:
    """Per-node content drift magnitude as a NumPy array."""
    drift = compute_drift(x, parent, hop_depth, n_children)
    return torch.norm(drift[:, :2], dim=-1).cpu().numpy()


@dataclass
class DriftTable:
    """Flattened per-node drift table for one dataset split."""

    magnitude: np.ndarray  # [n_nodes] content drift magnitude
    hop: np.ndarray  # [n_nodes] propagation depth
    label: np.ndarray  # [n_nodes] 0 = real, 1 = fake
    cascade: np.ndarray  # [n_nodes] cascade index
    cascade_size: np.ndarray  # [n_cascades]
    cascade_label: np.ndarray  # [n_cascades]
    cascade_hop1_frac: np.ndarray  # [n_cascades] share of nodes at hop 1

    @property
    def n_cascades(self) -> int:
        return int(self.cascade_size.size)

    @property
    def n_nodes(self) -> int:
        return int(self.magnitude.size)


def build_drift_table(
    root: str = "./data/upfd/raw",
    subset: str = "gossipcop",
    split: str = "test",
    feature_type: str = "bert",
) -> DriftTable:
    """Compute the per-node drift table for a dataset split."""
    dataset = UPFDGraphDataset(
        root=root, subset=subset, feature_type=feature_type, split=split
    )

    magnitudes: List[np.ndarray] = []
    hops: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    cascades: List[np.ndarray] = []
    sizes: List[int] = []
    cascade_labels: List[int] = []
    hop1_fracs: List[float] = []

    for index, graph in enumerate(dataset):
        magnitude = drift_magnitude(
            graph.x, graph.parent, graph.hop_depth, graph.n_children
        )
        hop = graph.hop_depth.cpu().numpy()
        label = int(graph.y.item())

        magnitudes.append(magnitude)
        hops.append(hop)
        labels.append(np.full(magnitude.size, label, dtype=np.int64))
        cascades.append(np.full(magnitude.size, index, dtype=np.int64))
        sizes.append(int(graph.x.size(0)))
        cascade_labels.append(label)
        hop1_fracs.append(float((hop == 1).mean()))

    return DriftTable(
        magnitude=np.concatenate(magnitudes),
        hop=np.concatenate(hops),
        label=np.concatenate(labels),
        cascade=np.concatenate(cascades),
        cascade_size=np.array(sizes, dtype=np.int64),
        cascade_label=np.array(cascade_labels, dtype=np.int64),
        cascade_hop1_frac=np.array(hop1_fracs, dtype=np.float64),
    )


def _welch(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Welch t-test of ``a`` vs ``b`` plus the mean ratio b/a."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "ratio_b_over_a": float(b.mean() / a.mean()) if a.mean() else float("nan"),
        "t": float(t),
        "p": float(p),
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


def pooled_ratio(table: DriftTable) -> Dict[str, float]:
    """The naive pooled statistic: every non-root node contributes equally.

    This is the number the paper historically reported (ratio 0.890 on
    GossipCop). It is reported here for continuity, but it is *not* the
    recommended effect size: the two classes have different propagation-depth
    compositions, so pooling mixes a content effect with a structure effect.
    """
    keep = table.hop > 0
    real = table.magnitude[keep & (table.label == REAL)]
    fake = table.magnitude[keep & (table.label == FAKE)]
    return _welch(real, fake)


def per_hop_table(
    table: DriftTable, max_hop: int = 8, min_nodes: int = 10
) -> List[Dict[str, float]]:
    """Per-depth drift statistics for hops 1..max_hop."""
    rows: List[Dict[str, float]] = []
    keep = table.hop > 0
    for hop in range(1, max_hop + 1):
        sel = keep & (table.hop == hop)
        real = table.magnitude[sel & (table.label == REAL)]
        fake = table.magnitude[sel & (table.label == FAKE)]
        if real.size < min_nodes or fake.size < min_nodes:
            continue
        row = _welch(real, fake)
        row["hop"] = hop
        rows.append(row)
    return rows


def hop_standardized_ratio(
    table: DriftTable, max_hop: int = 8, min_nodes: int = 10
) -> Dict[str, float]:
    """Direct standardisation of the drift ratio over propagation depth.

    Each hop's per-class mean drift is re-weighted by the *combined* hop
    distribution of both classes before the classes are compared. This removes
    the compositional confound: if fake cascades happen to sit at deeper hops
    (where drift is naturally smaller), the pooled ratio would attribute that
    structure to content.
    """
    keep = table.hop > 0
    rows = []
    for hop in range(1, max_hop + 1):
        sel = keep & (table.hop == hop)
        real = table.magnitude[sel & (table.label == REAL)]
        fake = table.magnitude[sel & (table.label == FAKE)]
        if real.size < min_nodes or fake.size < min_nodes:
            continue
        rows.append(
            {
                "hop": hop,
                "real_mean": float(real.mean()),
                "fake_mean": float(fake.mean()),
                "n_real": int(real.size),
                "n_fake": int(fake.size),
            }
        )
    if not rows:
        return {}

    total = sum(r["n_real"] + r["n_fake"] for r in rows)
    real_std = sum((r["n_real"] + r["n_fake"]) / total * r["real_mean"] for r in rows)
    fake_std = sum((r["n_real"] + r["n_fake"]) / total * r["fake_mean"] for r in rows)

    return {
        "max_hop": max_hop,
        "real_mean_standardized": float(real_std),
        "fake_mean_standardized": float(fake_std),
        "ratio_fake_over_real": float(fake_std / real_std) if real_std else float("nan"),
        "relative_reduction_pct": float((1 - fake_std / real_std) * 100)
        if real_std
        else float("nan"),
        "per_hop": rows,
    }


def cascade_level_stats(table: DriftTable, min_hop: int = 2) -> Dict[str, float]:
    """Cascade-level analysis: the unit a detector actually sees.

    Each cascade contributes one number (its mean drift over nodes at depth
    >= ``min_hop``), so the comparison is not driven by how many nodes a
    cascade contributes to the node-level pool.
    """
    per_cascade: Dict[int, List[float]] = {}
    for cascade in np.unique(table.cascade):
        sel = (table.cascade == cascade) & (table.hop >= min_hop)
        if sel.any():
            per_cascade[int(cascade)] = [float(table.magnitude[sel].mean())]

    indices = np.array(sorted(per_cascade))
    values = np.array([per_cascade[i][0] for i in indices])
    labels = table.cascade_label[indices]

    real = values[labels == REAL]
    fake = values[labels == FAKE]
    result = _welch(real, fake)
    result["min_hop"] = min_hop
    result["n_cascades"] = int(indices.size)
    return result


def hop_composition(table: DriftTable, max_hop: int = 5) -> Dict[str, Dict[str, float]]:
    """Share of nodes sitting at each propagation depth, per class."""
    keep = table.hop > 0
    out: Dict[str, Dict[str, float]] = {}
    for label, name in ((REAL, "real"), (FAKE, "fake")):
        sel = keep & (table.label == label)
        total = int(sel.sum())
        comp = {}
        for hop in range(1, max_hop + 1):
            comp[f"hop{hop}"] = float(((table.hop == hop) & sel).sum() / total)
        comp["mean_cascade_size"] = float(
            table.cascade_size[table.cascade_label == label].mean()
        )
        out[name] = comp
    return out


def node_level_regression(table: DriftTable) -> Dict[str, float]:
    """OLS of node drift on a fake indicator with hop fixed effects.

    Standard errors are clustered by cascade, so the reported p-value respects
    the fact that nodes inside a cascade are not independent. This is the
    strictest version of the claim: it asks whether fake news drifts less
    *at the same propagation depth*.
    """
    import statsmodels.api as sm

    keep = table.hop > 0
    y = table.magnitude[keep]
    fake = (table.label[keep] == FAKE).astype(float)
    hop = table.hop[keep]
    clusters = table.cascade[keep]

    # Explicit intercept + hop dummies for every hop except the first, which
    # the intercept absorbs as the reference level.
    hops = sorted(np.unique(hop))
    hop_dummies = np.eye(len(hops))[np.searchsorted(hops, hop)][:, 1:]
    design = np.column_stack([np.ones(y.size), fake, hop_dummies])

    model = sm.OLS(y, design, hasconst=True).fit(
        cov_type="cluster", cov_kwds={"groups": clusters}
    )
    coef = float(model.params[1])
    se = float(model.bse[1])
    lo, hi = model.conf_int()[1]
    baseline = float(y[fake == 0].mean())

    return {
        "n_nodes": int(y.size),
        "n_cascades": int(clusters.max() + 1),
        "fake_coefficient": coef,
        "std_error": se,
        "p_value": float(model.pvalues[0]),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "baseline_real_mean": baseline,
        "adjusted_relative_reduction_pct": float(coef / baseline * 100)
        if baseline
        else float("nan"),
    }


def size_matched_ratio(table: DriftTable, bucket: int = 10) -> List[Dict[str, float]]:
    """Drift ratio within cascade-size buckets (controls for cascade length)."""
    rows: List[Dict[str, float]] = []
    per_cascade: Dict[int, List[float]] = {}
    for cascade in np.unique(table.cascade):
        sel = (table.cascade == cascade) & (table.hop > 0)
        if sel.any():
            per_cascade[int(cascade)] = [float(table.magnitude[sel].mean())]

    indices = np.array(sorted(per_cascade))
    values = np.array([per_cascade[i][0] for i in indices])
    labels = table.cascade_label[indices]
    sizes = table.cascade_size[indices]

    for lo in range(0, int(sizes.max()) + bucket, bucket):
        sel = (sizes >= lo) & (sizes < lo + bucket)
        real = values[sel & (labels == REAL)]
        fake = values[sel & (labels == FAKE)]
        if real.size < 5 or fake.size < 5:
            continue
        row = _welch(real, fake)
        row["size_bucket"] = f"{lo}-{lo + bucket - 1}"
        rows.append(row)
    return rows


def full_report(table: DriftTable, max_hop: int = 8) -> Dict[str, object]:
    """Every statistic the paper quotes about the drift phenomenon."""
    return {
        "n_cascades": table.n_cascades,
        "n_nodes": table.n_nodes,
        "pooled": pooled_ratio(table),
        "hop_standardized": hop_standardized_ratio(table, max_hop=max_hop),
        "per_hop": per_hop_table(table, max_hop=max_hop),
        "cascade_level": cascade_level_stats(table),
        "cascade_level_hop1plus": cascade_level_stats(table, min_hop=1),
        "hop_composition": hop_composition(table),
        "node_level_regression": node_level_regression(table),
        "size_matched": size_matched_ratio(table),
    }
