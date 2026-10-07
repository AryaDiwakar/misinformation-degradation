"""Seed-level significance stats + reproducible drift-ratio statistics for the paper.

Reads results/multiseed_all.json and recomputes drift-ratio significance from raw
test-set features (GossipCop + Politifact).
"""
import json
import os
import sys

import numpy as np
import torch
from scipy import stats as sps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift

DEVICE = "cpu"


def paired_test(name, a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    t, p = sps.ttest_rel(a, b)
    sd = np.std(d, ddof=1) if n > 1 else 0.0
    cohen = d.mean() / sd if sd > 0 else 0.0
    print(f"\n[{name}] n={n}")
    print(f"  A: {np.mean(a):.4f}±{np.std(a):.4f}   B: {np.mean(b):.4f}±{np.std(b):.4f}")
    print(f"  mean diff = {d.mean():+.4f}")
    print(f"  paired t = {t:.3f}, p = {p:.4f}")
    print(f"  Cohen's d = {cohen:+.3f}")
    print(f"  per-seed diffs = {[f'{x:+.4f}' for x in d]}")
    return {"t": t, "p": p, "cohen_d": cohen, "mean_diff": float(d.mean())}


def drift_ratio_stats(subset):
    ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="test")
    real_mag, fake_mag = [], []
    for i in range(len(ds)):
        d = ds[i]
        drift = compute_drift(d.x, d.parent, d.hop_depth, d.n_children).numpy()
        mag = np.linalg.norm(drift[:, :2], axis=1)
        if d.y.item() == 1:
            fake_mag.append(mag)
        else:
            real_mag.append(mag)
    real_mag = np.concatenate(real_mag)
    fake_mag = np.concatenate(fake_mag)
    ratio = fake_mag.mean() / real_mag.mean()
    t, p = sps.ttest_ind(fake_mag, real_mag, equal_var=False)
    print(f"\n[{subset}] drift ratio fake/real = {ratio:.4f} (real {real_mag.mean():.4f} vs fake {fake_mag.mean():.4f})")
    print(f"  n nodes: real {len(real_mag):,} fake {len(fake_mag):,}")
    print(f"  Welch t = {t:.2f}, p = {p:.3e}")
    return {"ratio": float(ratio), "real_mean": float(real_mag.mean()), "fake_mean": float(fake_mag.mean()),
            "t": float(t), "p": float(p), "n_real": int(len(real_mag)), "n_fake": int(len(fake_mag))}


def main():
    out = {}
    ms = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "multiseed_all.json")))
    gc = ms["gossipcop"]

    out["hybrid_vs_lstm_gossipcop_acc"] = paired_test(
        "GossipCop Hybrid vs LSTM (acc)",
        [m["accuracy"] for m in gc["Hybrid"]["per_seed"]],
        [m["accuracy"] for m in gc["BaselineLSTM"]["per_seed"]],
    )
    out["hybrid_vs_lstm_gossipcop_auc"] = paired_test(
        "GossipCop Hybrid vs LSTM (AUC)",
        [m["auc_roc"] for m in gc["Hybrid"]["per_seed"]],
        [m["auc_roc"] for m in gc["BaselineLSTM"]["per_seed"]],
    )
    out["hybrid_vs_degradgnn_gossipcop_acc"] = paired_test(
        "GossipCop Hybrid vs DegradGNN (acc)",
        [m["accuracy"] for m in gc["Hybrid"]["per_seed"]],
        [m["accuracy"] for m in gc["DegradGNN"]["per_seed"]],
    )

    out["drift_ratio_gossipcop"] = drift_ratio_stats("gossipcop")
    out["drift_ratio_politifact"] = drift_ratio_stats("politifact")

    with open(os.path.join(os.path.dirname(__file__), "..", "results", "significance.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/significance.json")


if __name__ == "__main__":
    main()
