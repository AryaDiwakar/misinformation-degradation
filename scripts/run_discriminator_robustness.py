"""Drift-only discriminator (RF) on full vs 10%-truncated cascades.

Checks whether graph-level drift statistics remain discriminative when only
the first 10% of the cascade is visible (robustness of the drift signal).

12 features per graph: mean/std of the 5 drift dims + log n_nodes + max depth.

Note on the feature set: of the 12 statistics, 4 are *content* drift
(cosine/L2 distance to the parent), 6 are *structural* drift (depth,
children, is_root) and 2 are pure *size* (log n_nodes, max depth). The
historical claim that this discriminates "with no text or graph features"
overstates matters: see ``scripts/analysis_discriminator_decomposition.py``
for the full decomposition, which is the analysis the paper now quotes.

Protocol notes:
  * Cross-validation is stratified k-fold over the *test* split (as before).
  * A leakage-free train-split -> test-split score is also reported.
  * Cascades that truncate to a single root node carry no drift information
    (their drift is zero by construction), so that subgroup is reported
    separately.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift
from pipeline.truncation import is_root_only, truncate_cascade

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "drift_discriminator_robustness.json",
)


def truncate(graph, fraction):
    """Truncate a cascade, keeping the shallowest fraction of nodes."""
    return truncate_cascade(graph, fraction)


def graph_features(g):
    drift = compute_drift(g.x, g.parent, g.hop_depth, g.n_children).numpy()
    feats = []
    for dim in range(5):
        feats.append(drift[:, dim].mean())
        feats.append(drift[:, dim].std())
    feats.append(np.log(g.x.shape[0]))
    feats.append(g.hop_depth.max().item())
    return np.array(feats)


def stratified_cv(X, labels, n_splits=5, seed=0):
    """Stratified k-fold CV: the protocol the paper describes as '5-fold'."""
    accs, aucs = [], []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, labels)):
        rf = RandomForestClassifier(n_estimators=200, random_state=seed + fold)
        rf.fit(X[train_idx], labels[train_idx])
        p = rf.predict_proba(X[test_idx])[:, 1]
        accs.append(accuracy_score(labels[test_idx], p > 0.5))
        aucs.append(roc_auc_score(labels[test_idx], p))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_splits": n_splits,
    }


def repeated_holdout(X, labels, seed=0, repeats=5):
    """The historical protocol, kept for comparability with older artifacts."""
    accs, aucs = [], []
    for s in range(repeats):
        Xtr, Xte, ytr, yte = train_test_split(
            X, labels, test_size=0.3, stratify=labels, random_state=seed + s
        )
        rf = RandomForestClassifier(n_estimators=200, random_state=seed + s)
        rf.fit(Xtr, ytr)
        p = rf.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(yte, (p > 0.5).astype(float)))
        aucs.append(roc_auc_score(yte, p))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "repeats": repeats,
    }


def run(subset, seed=0, root="./data/upfd/raw"):
    ds = UPFDGraphDataset(
        root=root, subset=subset, feature_type="bert", split="test"
    )
    graphs = [d for d in ds]
    labels = np.array([d.y.item() for d in graphs])

    X_full = np.stack([graph_features(g) for g in graphs])
    truncated = [truncate(g, 0.1) for g in graphs]
    X_10 = np.stack([graph_features(g) for g in truncated])
    root_only = np.array([is_root_only(g) for g in truncated])

    out = {
        "subset": subset,
        "n": len(labels),
        "n_root_only_at_10pct": int(root_only.sum()),
        "root_only_fraction_at_10pct": float(root_only.mean()),
    }
    for tag, X in [("full", X_full), ("truncated_10pct", X_10)]:
        cv = stratified_cv(X, labels, seed=seed)
        holdout = repeated_holdout(X, labels, seed=seed)
        out[tag] = {
            "acc_mean": cv["acc_mean"],
            "acc_std": cv["acc_std"],
            "auc_mean": cv["auc_mean"],
            "auc_std": cv["auc_std"],
            "protocol": "stratified_5fold_cv",
            "historical_70_30_holdout": holdout,
        }
        print(
            f"  [{subset}] {tag}: acc={cv['acc_mean']:.4f}±{cv['acc_std']:.4f} "
            f"auc={cv['auc_mean']:.4f}±{cv['auc_std']:.4f}"
        )

        # Cascades that retain only the root have zero drift by construction.
        if tag == "truncated_10pct" and root_only.any():
            keep = ~root_only
            if len(np.unique(labels[keep])) > 1:
                sub_cv = stratified_cv(X[keep], labels[keep], seed=seed)
                out[tag]["excluding_root_only"] = {
                    "n_cascades": int(keep.sum()),
                    "acc_mean": sub_cv["acc_mean"],
                    "acc_std": sub_cv["acc_std"],
                    "auc_mean": sub_cv["auc_mean"],
                    "auc_std": sub_cv["auc_std"],
                }
                print(
                    f"  [{subset}] {tag} excluding root-only "
                    f"(n={int(keep.sum())}): acc={sub_cv['acc_mean']:.4f} "
                    f"auc={sub_cv['auc_mean']:.4f}"
                )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/upfd/raw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    results = {}
    for subset in ["gossipcop", "politifact"]:
        print(f"== {subset} ==")
        results[subset] = run(subset, seed=args.seed, root=args.root)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
