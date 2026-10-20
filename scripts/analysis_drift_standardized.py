"""Recompute the drift phenomenon with the compositional confound removed.

The historical headline ("fake news drifts ~11% less") is a *pooled node-level*
statistic: every non-root node contributes equally. Real and fake cascades
have different propagation-depth compositions (real cascades are hop-1-heavy,
and hop-1 nodes carry ~2.4x the drift of deep nodes), so the pooled ratio
mixes a content effect with a structure effect.

This script reports the pooled number for continuity, and alongside it the
statistics that actually isolate the content effect:

  * hop-standardised ratio   (direct standardisation over propagation depth)
  * cascade-level ratio      (one number per cascade, the unit a detector sees)
  * node-level regression    (hop fixed effects, cascade-clustered SEs)
  * size-matched ratio       (within cascade-length buckets)

Outputs ``results/drift_standardized.json``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline.drift_stats import build_drift_table, full_report

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "drift_standardized.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data/upfd/raw")
    parser.add_argument(
        "--subsets", default="gossipcop,politifact", help="comma-separated subsets"
    )
    parser.add_argument("--max-hop", type=int, default=8)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    results = {}
    for subset in args.subsets.split(","):
        subset = subset.strip()
        if not subset:
            continue
        print(f"== {subset} ==", flush=True)
        table = build_drift_table(root=args.root, subset=subset, split="test")
        report = full_report(table, max_hop=args.max_hop)
        results[subset] = report

        pooled = report["pooled"]
        std = report["hop_standardized"]
        casc = report["cascade_level"]
        reg = report["node_level_regression"]

        print(f"  pooled (historical)      ratio={pooled['ratio_b_over_a']:.4f} "
              f"t={pooled['t']:.1f} p={pooled['p']:.3g}")
        print(f"  hop-standardised         ratio={std['ratio_fake_over_real']:.4f} "
              f"({std['relative_reduction_pct']:+.1f}% vs real)")
        print(f"  cascade-level (hop>=2)   ratio={casc['ratio_b_over_a']:.4f} "
              f"t={casc['t']:.2f} p={casc['p']:.3g} n={casc['n_cascades']}")
        print(f"  node-level, hop FE       coef={reg['fake_coefficient']:+.4f} "
              f"se={reg['std_error']:.4f} p={reg['p_value']:.3g} "
              f"({reg['adjusted_relative_reduction_pct']:+.1f}% vs real)")
        print(f"  hop composition          real={report['hop_composition']['real']}")
        print(f"                          fake={report['hop_composition']['fake']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
