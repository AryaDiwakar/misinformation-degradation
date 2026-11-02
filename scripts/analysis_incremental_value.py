"""Does content drift carry information that cascade size does not?

This is the question that decides whether the work is worth a journal paper.
The reframed claim would not be "drift is a strong signal" (it is not) but
"drift adds information beyond what cascade shape already gives you".

Method:
  * Nested feature sets: size only, structural only, content only, and the
    combinations.
  * Paired bootstrap over the *same* evaluation folds, so the difference in
    AUC between two feature sets is tested on identical data.
  * A joint control: cascade-level drift compared within matched
    (cascade size x hop depth) strata, which removes both composition
    effects at once.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift
from pipeline.drift_stats import build_drift_table, cascade_level_stats

RNG = np.random.default_rng(0)


def feature_blocks(graph):
    drift = compute_drift(
        graph.x, graph.parent, graph.hop_depth, graph.n_children
    ).numpy()

    def block(dims):
        return np.array(
            [v for dim in dims for v in (drift[:, dim].mean(), drift[:, dim].std())]
        )

    return {
        "content": block((0, 1)),
        "structural": block((2, 3, 4)),
        "size": np.array(
            [np.log(max(int(graph.x.size(0)), 1)), float(graph.hop_depth.max().item())]
        ),
    }


SETS = {
    "size": ("size",),
    "structural": ("structural",),
    "content": ("content",),
    "size+content": ("size", "content"),
    "size+structural": ("size", "structural"),
    "all 12": ("content", "structural", "size"),
}


def build_matrices(graphs):
    blocks = [feature_blocks(g) for g in graphs]
    out = {}
    for name, keys in SETS.items():
        out[name] = np.stack(
            [
                np.concatenate([b[key] for key in keys])
                for b in blocks
            ]
        )
    return out


def paired_bootstrap_auc(X_a, X_b, y, n_splits=5, n_boot=1000, seed=0):
    """Paired bootstrap of AUC(a) - AUC(b) on identical folds."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y))

    def oof_auc(X):
        probs = np.zeros(len(y))
        for train_idx, test_idx in folds:
            model = RandomForestClassifier(n_estimators=200, random_state=seed)
            model.fit(X[train_idx], y[train_idx])
            probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        return roc_auc_score(y, probs), probs

    auc_a, probs_a = oof_auc(X_a)
    auc_b, probs_b = oof_auc(X_b)

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y[idx], probs_a[idx]) - roc_auc_score(y[idx], probs_b[idx])
        )
    diffs = np.array(diffs)
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "diff": float(auc_a - auc_b),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }


def joint_size_hop_control(table, size_bucket=10, hop=2):
    """Cascade-level drift within matched (size, hop) strata.

    Removes both composition effects at once: cascades are compared only
    against other cascades of a similar length, and only nodes at the same
    propagation depth contribute.
    """
    per_cascade = {}
    for cascade in np.unique(table.cascade):
        sel = (table.cascade == cascade) & (table.hop >= hop)
        if sel.any():
            per_cascade[int(cascade)] = float(table.magnitude[sel].mean())

    indices = np.array(sorted(per_cascade))
    values = np.array([per_cascade[i] for i in indices])
    labels = table.cascade_label[indices]
    sizes = table.cascade_size[indices]

    strata = {}
    for lo in range(0, int(sizes.max()) + size_bucket, size_bucket):
        sel = (sizes >= lo) & (sizes < lo + size_bucket)
        real = values[sel & (labels == 0)]
        fake = values[sel & (labels == 1)]
        if len(real) >= 5 and len(fake) >= 5:
            strata[f"{lo}-{lo + size_bucket - 1}"] = {
                "real_mean": float(real.mean()),
                "fake_mean": float(fake.mean()),
                "ratio": float(fake.mean() / real.mean()),
                "n_real": int(len(real)),
                "n_fake": int(len(fake)),
            }
    ratios = [v["ratio"] for v in strata.values()]
    return {
        "per_stratum": strata,
        "n_strata": len(strata),
        "n_strata_below_one": int(sum(1 for r in ratios if r < 1)),
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "median_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset", nargs="?", default="gossipcop")
    parser.add_argument("--root", default="./data/upfd/raw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    subset = args.subset
    root = args.root

    ds = UPFDGraphDataset(
        root=root, subset=subset, feature_type="bert", split="test"
    )
    graphs = [g for g in ds]
    y = np.array([g.y.item() for g in graphs])
    matrices = build_matrices(graphs)

    print(f"== {subset} (n={len(y)} cascades) ==")
    results = {}
    for name, X in matrices.items():
        auc = np.mean(
            [
                roc_auc_score(
                    y[te],
                    RandomForestClassifier(n_estimators=200, random_state=s)
                    .fit(X[tr], y[tr])
                    .predict_proba(X[te])[:, 1],
                )
                for s, (tr, te) in enumerate(
                    StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
                )
            ]
        )
        results[name] = {"auc": float(auc)}
        print(f"  {name:<16} AUC = {auc:.4f}")

    print("\n  Paired bootstrap (5-fold OOF, 1000 resamples):")
    comparisons = [
        ("content", "size", "Does content drift beat cascade size alone?"),
        ("size+content", "size", "Does adding content drift to size help?"),
        ("size+content", "size+structural", "Does content beat structure at equal size?"),
        ("all 12", "size", "Does the full set beat size alone?"),
    ]
    for a, b, question in comparisons:
        stat = paired_bootstrap_auc(matrices[a], matrices[b], y)
        results[f"cmp__{a}_vs__{b}"] = stat
        print(
            f"    {a} - {b}: {stat['diff']:+.4f} "
            f"[{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}] "
            f"p={stat['p_two_sided']:.4f}   ({question})"
        )

    print("\n  Joint (cascade size x hop depth) control, cascade-level drift:")
    table = build_drift_table(root=root, subset=subset, split="test")
    control = joint_size_hop_control(table)
    results["joint_size_hop_control"] = control
    print(
        f"    {control['n_strata_below_one']}/{control['n_strata']} size strata "
        f"show fake < real; mean ratio {control['mean_ratio']:.4f}, "
        f"median {control['median_ratio']:.4f}"
    )
    for name, rec in control["per_stratum"].items():
        print(
            f"      size {name:<8} real={rec['real_mean']:.3f} "
            f"fake={rec['fake_mean']:.3f} ratio={rec['ratio']:.3f} "
            f"(n={rec['n_real']}/{rec['n_fake']})"
        )

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        f"incremental_value_{subset}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
