"""Drift-only discriminator (RF) on full vs 10%-truncated cascades.

Checks whether graph-level drift statistics remain discriminative when only the
first 10% of the cascade is visible (robustness of the drift signal).
12 features per graph: mean/std of the 5 drift dims + log n_nodes + max depth.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.upfd.loader import UPFDGraphDataset, build_parent_index, count_children
from features.drift import compute_drift


def truncate(graph, fraction):
    order = graph.hop_depth.argsort()
    n = graph.x.size(0)
    keep = max(1, int(n * fraction))
    keep_nodes = order[:keep]
    node_map = {old.item(): i for i, old in enumerate(keep_nodes)}
    mask = torch.zeros(n, dtype=torch.bool)
    mask[keep_nodes] = True
    edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
    ei = graph.edge_index[:, edge_mask].clone()
    for i in range(ei.size(1)):
        ei[0, i] = node_map[ei[0, i].item()]
        ei[1, i] = node_map[ei[1, i].item()]
    return {
        "x": graph.x[keep_nodes],
        "edge_index": ei,
        "hop_depth": graph.hop_depth[keep_nodes],
        "parent": build_parent_index(ei, keep),
        "n_children": count_children(ei, keep),
    }


def graph_features(g):
    drift = compute_drift(g["x"], g["parent"], g["hop_depth"], g["n_children"]).numpy()
    feats = []
    for dim in range(5):
        feats.append(drift[:, dim].mean())
        feats.append(drift[:, dim].std())
    feats.append(np.log(g["x"].shape[0]))
    feats.append(g["hop_depth"].max().item())
    return np.array(feats)


def run(subset, seed=0):
    ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="test")
    graphs = [d for d in ds]
    labels = np.array([d.y.item() for d in graphs])

    X_full = np.stack([graph_features(g) for g in graphs])
    X_10 = np.stack([graph_features(truncate(g, 0.1)) for g in graphs])

    out = {"subset": subset, "n": len(labels)}
    for tag, X in [("full", X_full), ("truncated_10pct", X_10)]:
        accs, aucs = [], []
        for s in range(5):
            Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, stratify=labels, random_state=seed + s)
            rf = RandomForestClassifier(n_estimators=200, random_state=seed + s)
            rf.fit(Xtr, ytr)
            p = rf.predict_proba(Xte)[:, 1]
            accs.append(accuracy_score(yte, (p > 0.5).astype(float)))
            aucs.append(roc_auc_score(yte, p))
        out[tag] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        }
        print(f"  [{subset}] {tag}: acc={out[tag]['acc_mean']:.4f}±{out[tag]['acc_std']:.4f} "
              f"auc={out[tag]['auc_mean']:.4f}±{out[tag]['auc_std']:.4f}")
    return out


def main():
    results = {}
    for subset in ["gossipcop", "politifact"]:
        print(f"== {subset} ==")
        results[subset] = run(subset)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "drift_discriminator_robustness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
