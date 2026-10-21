"""Link the mechanism to the effect: does near-duplication explain drift?

`scripts/analysis_mechanism_probe.py` shows that fake-news cascades contain
markedly more near-duplicate *sibling* reposts than real-news cascades. This
script asks the follow-up question that turns a correlation into a mechanism:

  1. Is the near-duplication rate negatively correlated with drift?
  2. Does that correlation hold *within* each class (so it is not just a
     composition effect)?
  3. Does it survive controlling for cascade size?
  4. Does near-duplication alone discriminate fake from real?

Outputs ``results/mechanism_linkage_gossipcop.json``.
"""

import json
import os
import sys

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift
from scripts.analysis_mechanism_probe import cascade_homogeneity

THRESHOLD = 0.95


def residualise(values, controls):
    """Remove the linear effect of ``controls`` from ``values``."""
    design = np.column_stack([np.ones(len(controls)), controls])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "gossipcop"
    ds = UPFDGraphDataset(
        root="./data/upfd/raw", subset=subset, feature_type="bert", split="test"
    )

    duplicate, drift, labels, sizes = [], [], [], []
    for graph in ds:
        homogeneity = cascade_homogeneity(graph)
        features = compute_drift(
            graph.x, graph.parent, graph.hop_depth, graph.n_children
        )
        magnitude = torch.norm(features[:, :2], dim=-1).numpy()
        non_root = graph.hop_depth.numpy() > 0
        duplicate.append(homogeneity["duplicate_rate"][THRESHOLD])
        drift.append(
            float(magnitude[non_root].mean()) if non_root.any() else np.nan
        )
        labels.append(int(graph.y.item()))
        sizes.append(int(graph.x.size(0)))

    duplicate = np.array(duplicate)
    drift = np.array(drift)
    labels = np.array(labels)
    sizes = np.array(sizes)
    usable = ~np.isnan(drift) & ~np.isnan(duplicate)

    results = {
        "subset": subset,
        "threshold": THRESHOLD,
        "n_cascades": int(len(labels)),
        "n_usable": int(usable.sum()),
        "duplicate_rate_mean": {
            "real": float(duplicate[usable & (labels == 0)].mean()),
            "fake": float(duplicate[usable & (labels == 1)].mean()),
        },
    }

    print(f"== {subset} (n={usable.sum()} cascades with a sibling group) ==")
    print(
        "  sibling near-duplicate rate @cos>={}: real={:.4f} fake={:.4f}".format(
            THRESHOLD,
            results["duplicate_rate_mean"]["real"],
            results["duplicate_rate_mean"]["fake"],
        )
    )

    # 1. overall correlation
    pearson_r, pearson_p = stats.pearsonr(duplicate[usable], drift[usable])
    spearman_r, spearman_p = stats.spearmanr(duplicate[usable], drift[usable])
    results["pearson_r"] = float(pearson_r)
    results["pearson_p"] = float(pearson_p)
    results["spearman_r"] = float(spearman_r)
    results["spearman_p"] = float(spearman_p)
    print(
        f"\n  corr(near-duplication, drift): r={pearson_r:+.4f} p={pearson_p:.3g} "
        "(Spearman rho="
        f"{spearman_r:+.4f})"
    )

    # 2. within each class
    results["within_class"] = {}
    for label, name in ((0, "real"), (1, "fake")):
        mask = usable & (labels == label)
        r, p = stats.pearsonr(duplicate[mask], drift[mask])
        results["within_class"][name] = {"r": float(r), "p": float(p)}
        print(f"    within {name:<4}: r={r:+.4f} p={p:.3g}")

    # 3. partial correlation controlling for cascade size
    log_size = np.log(sizes[usable])
    dup_resid = residualise(duplicate[usable], log_size)
    drift_resid = residualise(drift[usable], log_size)
    partial_r, partial_p = stats.pearsonr(dup_resid, drift_resid)
    results["partial_r_given_size"] = float(partial_r)
    results["partial_p_given_size"] = float(partial_p)
    print(
        f"    partial corr | log size: r={partial_r:+.4f} p={partial_p:.3g}"
    )

    # 4. discriminative power of near-duplication alone
    results["auc_duplicate_alone"] = float(
        roc_auc_score(labels[usable], duplicate[usable])
    )
    results["auc_size_alone"] = float(
        roc_auc_score(labels[usable], log_size)
    )
    print(
        f"\n  AUC near-duplication alone = {results['auc_duplicate_alone']:.4f}"
        f"   (log size alone = {results['auc_size_alone']:.4f})"
    )

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        f"mechanism_linkage_{subset}.json",
    )
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
