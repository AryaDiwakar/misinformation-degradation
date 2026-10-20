"""Decompose the drift-only discriminator into content vs. structure vs. size.

The historical claim is that a 12-statistic random forest reaches ~0.87 AUC on
GossipCop "with no text or graph features". That framing does not hold up
under decomposition, because the 12 statistics are not all content drift:

  * content drift  : mean/std of the cosine and L2 distances to the parent (4)
  * structural     : mean/std of depth, children and is_root (6)
  * size           : log(n_nodes) and max depth (2)

The last two groups are properties of the cascade's *shape*, not of content
degradation, and the two classes differ a lot in shape (real GossipCop
cascades average 76 nodes, fake ones 39).

This script therefore reports, for the full cascade and for the 10% truncated
cascade:

  * each feature group on its own,
  * the full 12-feature set,
  * the effect of excluding cascades that truncate to a single root node
    (whose drift is zero by construction and therefore carries no drift
    information at all),
  * a leakage-free protocol that fits on the train split and scores the test
    split, next to the historical in-test-split cross-validation.

Outputs ``results/drift_discriminator_decomposition.json``.
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
from pipeline.truncation import truncate_cascade

CONTENT_DIMS = (0, 1)  # cosine distance, L2 distance
STRUCTURAL_DIMS = (2, 3, 4)  # depth, children, is_root
GROUPS = ("content", "structural", "size", "all")


def truncate(graph, fraction):
    """Truncate a cascade, keeping the shallowest fraction of nodes."""
    return truncate_cascade(graph, fraction)


def feature_groups(graph):
    """Return the (content, structural, size) feature blocks for one cascade."""
    drift = compute_drift(
        graph.x, graph.parent, graph.hop_depth, graph.n_children
    ).numpy()

    def block(dims):
        return np.array(
            [v for dim in dims for v in (drift[:, dim].mean(), drift[:, dim].std())]
        )

    content = block(CONTENT_DIMS)
    structural = block(STRUCTURAL_DIMS)
    size = np.array(
        [np.log(max(int(graph.x.size(0)), 1)), float(graph.hop_depth.max().item())]
    )
    return content, structural, size


def assemble(graphs, group):
    """Stack the requested feature group across cascades."""
    rows = []
    for graph in graphs:
        content, structural, size = feature_groups(graph)
        if group == "content":
            rows.append(content)
        elif group == "structural":
            rows.append(structural)
        elif group == "size":
            rows.append(size)
        else:
            rows.append(np.concatenate([content, structural, size]))
    return np.stack(rows)


def cross_validate(X, y, n_splits=5, seed=0):
    """Stratified k-fold cross-validation (the paper's stated protocol)."""
    accs, aucs = [], []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        model = RandomForestClassifier(n_estimators=200, random_state=seed + fold)
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        accs.append(accuracy_score(y[test_idx], probs > 0.5))
        aucs.append(roc_auc_score(y[test_idx], probs))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_splits": n_splits,
    }


def repeated_holdout(X, y, seed=0, repeats=5):
    """The historical protocol: repeated random 70/30 splits."""
    accs, aucs = [], []
    for split in range(repeats):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=seed + split
        )
        model = RandomForestClassifier(n_estimators=200, random_state=seed + split)
        model.fit(x_train, y_train)
        probs = model.predict_proba(x_test)[:, 1]
        accs.append(accuracy_score(y_test, probs > 0.5))
        aucs.append(roc_auc_score(y_test, probs))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "repeats": repeats,
    }


def fit_train_score_test(X_train, y_train, X_test, y_test, seed=0):
    """Leakage-free protocol: fit on the train split, score the test split."""
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, probs > 0.5)),
        "auc_roc": float(roc_auc_score(y_test, probs)),
    }


def analyse_subset(root, subset, seed=0):
    print(f"== {subset} ==", flush=True)
    splits = {
        name: UPFDGraphDataset(
            root=root, subset=subset, feature_type="bert", split=name
        )
        for name in ("train", "test")
    }
    train_graphs = [g for g in splits["train"]]
    test_graphs = [g for g in splits["test"]]
    y_test = np.array([g.y.item() for g in test_graphs])
    y_train = np.array([g.y.item() for g in train_graphs])

    truncated_test = [truncate(g, 0.1) for g in test_graphs]
    truncated_train = [truncate(g, 0.1) for g in train_graphs]
    kept = np.array([int(g.x.size(0)) for g in truncated_test])
    root_only = kept == 1

    result = {
        "subset": subset,
        "n_test_cascades": len(test_graphs),
        "n_train_cascades": len(train_graphs),
        "n_test_nodes": int(sum(int(g.x.size(0)) for g in test_graphs)),
        "mean_cascade_size": {
            "real": float(
                np.mean([g.x.size(0) for g in test_graphs if g.y.item() == 0])
            ),
            "fake": float(
                np.mean([g.x.size(0) for g in test_graphs if g.y.item() == 1])
            ),
        },
        "truncated_10pct": {
            "n_root_only_cascades": int(root_only.sum()),
            "root_only_fraction": float(root_only.mean()),
        },
    }

    print(
        f"  10% truncation leaves a single root node for {root_only.sum()}"
        f"/{len(test_graphs)} cascades ({root_only.mean() * 100:.1f}%)"
    )

    for setting, graphs, labels, train_graphs_for_setting in (
        ("full", test_graphs, y_test, train_graphs),
        ("truncated_10pct", truncated_test, y_test, truncated_train),
    ):
        result[setting] = {}
        for group in GROUPS:
            X = assemble(graphs, group)
            cv = cross_validate(X, labels, seed=seed)
            holdout = repeated_holdout(X, labels, seed=seed)
            result[setting][group] = {
                "n_features": int(X.shape[1]),
                "stratified_5fold_cv": cv,
                "repeated_70_30_holdout": holdout,
            }
            print(
                f"  [{setting:>14}] {group:<11} acc={cv['acc_mean']:.4f} "
                f"auc={cv['auc_mean']:.4f}  ({X.shape[1]} feats)"
            )

            # Excluding root-only cascades: the drift signal must be present
            # in cascades that actually retained a parent-child pair.
            if setting == "truncated_10pct" and root_only.any():
                keep_mask = ~root_only
                if len(np.unique(labels[keep_mask])) > 1:
                    Xs = X[keep_mask]
                    cv_sub = cross_validate(Xs, labels[keep_mask], seed=seed)
                    result[setting][group]["excluding_root_only"] = {
                        "n_cascades": int(keep_mask.sum()),
                        "stratified_5fold_cv": cv_sub,
                    }
                    print(
                        f"                    {group:<11} excluding root-only: "
                        f"acc={cv_sub['acc_mean']:.4f} auc={cv_sub['auc_mean']:.4f}"
                    )

        # Leakage-free protocol.
        result[setting]["leakage_free"] = {}
        for group in GROUPS:
            X_train = assemble(train_graphs_for_setting, group)
            X_test = assemble(graphs, group)
            if len(np.unique(y_train)) < 2:
                continue
            score = fit_train_score_test(X_train, y_train, X_test, labels, seed=seed)
            result[setting]["leakage_free"][group] = score
            print(
                f"  [{setting:>14}] {group:<11} train->test "
                f"acc={score['accuracy']:.4f} auc={score['auc_roc']:.4f}"
            )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/upfd/raw")
    parser.add_argument("--subsets", default="gossipcop,politifact")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "results",
            "drift_discriminator_decomposition.json",
        ),
    )
    args = parser.parse_args()

    results = {}
    for subset in args.subsets.split(","):
        subset = subset.strip()
        if subset:
            results[subset] = analyse_subset(args.root, subset, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
