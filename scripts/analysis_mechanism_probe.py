"""Probe the mechanism behind the drift effect.

The paper's explanation for "fake news drifts less" is that coordinated or
templated spreaders push homogeneous, near-duplicate reposts, while real news
accrues heterogeneous commentary. That explanation is currently a narrative
with no direct test. If it is right, fake cascades should show:

  1. a higher near-duplicate rate among sibling nodes (copied content), and
  2. lower embedding dispersion among siblings (homogeneous content).

Both are directly measurable from the precomputed BERT features. If the
mechanism holds, this is the strongest evidence in the paper; if it does not,
the phenomenon is an unexplained statistical regularity and the paper should
say so.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift

DUPLICATE_THRESHOLDS = (0.90, 0.95, 0.98)


def cascade_homogeneity(graph):
    """Sibling similarity and within-cascade embedding dispersion."""
    x = graph.x
    n = x.size(0)

    # Normalise so similarity is not driven by embedding norm.
    xn = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Siblings: nodes sharing a parent. Group the children by parent with a
    # single sort, then generate each group's pairs vectorised — looping over
    # every parent separately costs O(n^2) torch dispatches per cascade.
    parent = graph.parent
    valid = parent >= 0
    child_idx = valid.nonzero(as_tuple=True)[0]
    child_parent = parent[valid]
    order = torch.argsort(child_parent, stable=True)
    child_idx = child_idx[order]
    child_parent = child_parent[order]

    unique_parents, counts = torch.unique_consecutive(
        child_parent, return_counts=True
    )
    starts = torch.cat(
        [torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]]
    )

    pair_i, pair_j = [], []
    group_sizes = []
    for gi in range(int(unique_parents.numel())):
        k = int(counts[gi])
        if k < 2:
            continue
        start = int(starts[gi])
        group = child_idx[start : start + k]
        tri = torch.triu_indices(k, k, offset=1)
        pair_i.append(group[tri[0]])
        pair_j.append(group[tri[1]])
        group_sizes.append(k)

    duplicate_rates = {t: float("nan") for t in DUPLICATE_THRESHOLDS}
    if pair_i:
        pi = torch.cat(pair_i)
        pj = torch.cat(pair_j)
        pair_cos = (xn[pi] * xn[pj]).sum(dim=1)
        # Group id per pair, so each sibling group's duplicate rate can be
        # averaged and then averaged over groups.
        group_id = torch.repeat_interleave(
            torch.arange(len(group_sizes)),
            torch.tensor(
                [g * (g - 1) // 2 for g in group_sizes], dtype=torch.long
            ),
        )
        for threshold in DUPLICATE_THRESHOLDS:
            is_dup = (pair_cos >= threshold).float()
            totals = torch.zeros(len(group_sizes)).index_add_(0, group_id, is_dup)
            counts_per_group = torch.zeros(len(group_sizes)).index_add_(
                0, group_id, torch.ones_like(is_dup)
            )
            rates = totals / counts_per_group.clamp(min=1)
            duplicate_rates[threshold] = float(rates.mean())

    # Within-cascade dispersion: mean pairwise cosine distance across all
    # node pairs (a cascade-level "how varied is this conversation" measure).
    if n > 1:
        sim = xn @ xn.T
        iu = torch.triu_indices(n, n, offset=1)
        mean_pairwise_cos = float(sim[iu[0], iu[1]].mean())
    else:
        mean_pairwise_cos = float("nan")

    return {
        "n_sibling_groups": len(group_sizes),
        "duplicate_rate": duplicate_rates,
        "mean_pairwise_cosine": mean_pairwise_cos,
        "n_nodes": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset", nargs="?", default="gossipcop")
    parser.add_argument("--root", default="./data/upfd/raw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    subset = args.subset
    ds = UPFDGraphDataset(
        root=args.root, subset=subset, feature_type="bert", split="test"
    )

    per_class = {0: [], 1: []}
    total = len(ds)
    for index, graph in enumerate(ds):
        per_class[int(graph.y.item())].append(cascade_homogeneity(graph))
        if (index + 1) % 500 == 0:
            print(f"  processed {index + 1}/{total}", flush=True)

    out = {"subset": subset, "n_cascades": {}}
    print(f"== {subset} ==")
    for label, name in ((0, "real"), (1, "fake")):
        records = per_class[label]
        out["n_cascades"][name] = len(records)
        print(f"  {name}: n={len(records)}")
        for threshold in DUPLICATE_THRESHOLDS:
            values = np.array(
                [r["duplicate_rate"][threshold] for r in records], dtype=float
            )
            values = values[~np.isnan(values)]
            mean = float(values.mean()) if values.size else float("nan")
            std = float(values.std()) if values.size else float("nan")
            out.setdefault("duplicate_rate", {}).setdefault(
                str(threshold), {}
            )[name] = {"mean": mean, "std": std, "n": int(values.size)}
            print(f"    sibling near-duplicate rate @cos>={threshold}: {mean:.4f} ± {std:.4f}")

        values = np.array(
            [r["mean_pairwise_cosine"] for r in records], dtype=float
        )
        values = values[~np.isnan(values)]
        out.setdefault("mean_pairwise_cosine", {})[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "n": int(values.size),
        }
        print(f"    within-cascade mean pairwise cosine: {values.mean():.4f} ± {values.std():.4f}")

    print("\n  Class comparison (Mann-Whitney U):")
    for threshold in DUPLICATE_THRESHOLDS:
        r = np.array(
            [x["duplicate_rate"][threshold] for x in per_class[0]], dtype=float
        )
        f = np.array(
            [x["duplicate_rate"][threshold] for x in per_class[1]], dtype=float
        )
        r = r[~np.isnan(r)]
        f = f[~np.isnan(f)]
        stat, p = stats.mannwhitneyu(r, f, alternative="two-sided")
        print(
            f"    near-duplicate @>={threshold}: real={r.mean():.4f} fake={f.mean():.4f} "
            f"U={stat:.0f} p={p:.3g}"
        )
        out.setdefault("tests", {}).setdefault(f"duplicate_{threshold}", {})[
            "p_value"
        ] = float(p)

    r = np.array([x["mean_pairwise_cosine"] for x in per_class[0]], dtype=float)
    f = np.array([x["mean_pairwise_cosine"] for x in per_class[1]], dtype=float)
    r = r[~np.isnan(r)]
    f = f[~np.isnan(f)]
    stat, p = stats.mannwhitneyu(r, f, alternative="two-sided")
    print(
        f"    within-cascade dispersion: real={r.mean():.4f} fake={f.mean():.4f} "
        f"U={stat:.0f} p={p:.3g}"
    )
    out.setdefault("tests", {})["mean_pairwise_cosine"] = {"p_value": float(p)}

    path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        f"mechanism_probe_{subset}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"\nSaved {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
