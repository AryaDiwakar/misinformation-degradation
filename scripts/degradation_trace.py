"""
LIVE DEMO: show how information DEGRADES as it propagates.

Prints, for one REAL and one FAKE GossipCop cascade:
  - hop-by-hop mean drift  -> the degradation curve
  - per-hop drift magnitude table (parent -> child)

The dataset-level line reports both the naive pooled ratio and the
hop-standardised ratio, because the pooled figure mixes a content effect
with a difference in propagation-depth composition between the classes.

Run:  python scripts/degradation_trace.py [--max-nodes 60]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift
from pipeline.drift_stats import build_drift_table


def pick(ds, label, lo=25, hi=90):
    for d in ds:
        if d.y.item() == label and lo <= d.x.size(0) <= hi:
            return d
    raise RuntimeError(f"no graph with label={label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=60)
    ap.add_argument("--subset", default="gossipcop")
    args = ap.parse_args()

    ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                          feature_type="bert", split="test")
    fake = pick(ds, 1)
    real = pick(ds, 0)

    for label, d in [("REAL", real), ("FAKE", fake)]:
        diff = compute_drift(d.x, d.parent, d.hop_depth, d.n_children).numpy()
        mag = np.linalg.norm(diff[:, :2], axis=1)
        hd = d.hop_depth.numpy()
        parent = d.parent.numpy()
        n = d.x.size(0)

        print("=" * 72)
        print(f"  {label} NEWS  |  {n} nodes, max depth {hd.max()}  ({args.subset} test)")
        print("  seq | hop | parent | cos_dist ->parent | L2 ->parent | drift mag")
        print("-" * 72)
        shown = 0
        order = hd.argsort()
        for i in order:
            if shown >= args.max_nodes:
                break
            shown += 1
            pr = "-r-" if parent[i] < 0 else f"{parent[i]:>4d}"
            print(f"  {i:>3d} | {hd[i]:>2d} | {pr:>5s} | {diff[i,0]:>16.4f} | "
                  f"{diff[i,1]:>13.4f} | {mag[i]:>6.2f}")
        print()
        print("  Degradation curve (mean drift per hop):")
        for h in range(hd.max() + 1):
            m = mag[hd == h]
            if len(m):
                print(f"    hop {h}: n={len(m):>4d}  mean drift={m.mean():.3f} "
                      f"  cum_dist={diff[hd == h, 0].mean():.3f}  cum_l2={diff[hd == h, 1].mean():.3f}")
        print()

    # Dataset-level aggregate.
    table = build_drift_table(subset=args.subset, split="test")
    from pipeline.drift_stats import (
        cascade_level_stats,
        hop_standardized_ratio,
        pooled_ratio,
    )

    pooled = pooled_ratio(table)
    standardized = hop_standardized_ratio(table)
    cascade = cascade_level_stats(table)

    print("=" * 72)
    print("  DATASET-LEVEL DEGRADATION (all test nodes)")
    print(f"    pooled (all hops pooled)   real={pooled['mean_a']:.3f}  "
          f"fake={pooled['mean_b']:.3f}  ratio={pooled['ratio_b_over_a']:.3f}  "
          f"t={pooled['t']:.1f}  p={pooled['p']:.2g}")
    print(f"    hop-standardised           real={standardized['real_mean_standardized']:.3f}  "
          f"fake={standardized['fake_mean_standardized']:.3f}  "
          f"ratio={standardized['ratio_fake_over_real']:.3f}  "
          f"({standardized['relative_reduction_pct']:+.1f}% vs real)")
    print(f"    cascade-level (hop>=2)     ratio={cascade['ratio_b_over_a']:.3f}  "
          f"t={cascade['t']:.2f}  p={cascade['p']:.2g}  n={cascade['n_cascades']}")
    print("    NOTE: the pooled ratio mixes the content effect with the classes'")
    print("          different propagation-depth compositions; the hop-standardised")
    print("          and cascade-level figures isolate the content effect.")
    print("=" * 72)


if __name__ == "__main__":
    main()
