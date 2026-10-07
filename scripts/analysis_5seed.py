"""Extended significance analysis: 5-seed Hybrid vs LSTM, DriftPool comparisons.

Reads:
  results/multiseed_all.json        (canonical 3-seed results, seeds 1-3)
  results/multiseed_seeds45.json   (extension seeds 4,5 for LSTM/Hybrid)
  results/multiseed_driftpool.json (new DriftPoolHybrid, seeds 1-3)

Writes results/significance_5seed.json with n=3 and n=5 paired statistics.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def model_snapshot(subset, model, files):
    per_seed = {}
    for f in files:
        d = json.load(open(os.path.join(R, f)))
        rec = d.get(subset, d).get(model, {})
        for m in rec.get("per_seed", []):
            s = m.get("seed")
            per_seed[s] = m
    seeds = sorted(
        s for s in per_seed if s is not None
    )
    return per_seed, seeds


def paired(a, b):
    a, b = np.array(a, float), np.array(b, float)
    d = a - b
    n = len(d)
    if n < 2 or np.std(d, ddof=1) == 0:
        return {"n": int(n), "mean_diff": float(d.mean()), "t": None, "p": None,
                "cohen_d": 0.0, "positive_out_of_n": int((d > 0).sum())}
    sd = np.std(d, ddof=1)
    from scipy import stats as sps
    t, p = sps.ttest_rel(a, b)
    return {"n": int(n), "mean_a": float(a.mean()), "std_a": float(a.std()),
            "mean_b": float(b.mean()), "std_b": float(b.std()),
            "mean_diff": float(d.mean()), "t": float(t), "p": float(p),
            "cohen_d": float(d.mean() / sd), "positive_out_of_n": int((d > 0).sum()),
            "per_seed_diff": [float(x) for x in d]}


def summary(model, per_seed, seeds):
    acc = [per_seed[s]["accuracy"] for s in seeds]
    auc = [per_seed[s]["auc_roc"] for s in seeds]
    return f"{model:<14} acc {np.mean(acc):.4f}±{np.std(acc):.4f}  auc {np.mean(auc):.4f}±{np.std(auc):.4f}  (seeds {seeds})"


def main():
    out = {}

    for tag, files in [
        ("n3", ["multiseed_all.json", "multiseed_driftpool.json"]),
        ("n5", ["multiseed_all.json", "multiseed_seeds45.json"]),
    ]:
        snap = {m: model_snapshot("gossipcop", m, files) for m in
                ["BaselineLSTM", "DegradGNN", "Hybrid", "DriftPool"]}
        print(f"\n== {tag} ==")
        for m, (ps, seeds) in snap.items():
            print("  " + summary(m, ps, seeds))

        ps, seeds = snap["Hybrid"]
        lstm, lseeds = snap["BaselineLSTM"]
        # align on common seeds
        common = sorted(set(seeds) & set(lseeds))
        out[f"{tag}_hybrid_vs_lstm_acc"] = paired(
            [ps[s]["accuracy"] for s in common], [lstm[s]["accuracy"] for s in common])
        out[f"{tag}_hybrid_vs_lstm_auc"] = paired(
            [ps[s]["auc_roc"] for s in common], [lstm[s]["auc_roc"] for s in common])

        dp, dseeds = snap["DriftPool"]
        common = sorted(set(seeds) & set(dseeds))
        out[f"{tag}_hybrid_vs_driftpool_acc"] = paired(
            [ps[s]["accuracy"] for s in common], [dp[s]["accuracy"] for s in common])
        out[f"{tag}_driftpool_vs_lstm_acc"] = paired(
            [dp[s]["accuracy"] for s in common], [lstm[s]["accuracy"] for s in common])

        dpd, ddseeds = snap["DriftPool"]
        degg, dgseeds = snap["DegradGNN"]
        common = sorted(set(ddseeds) & set(dgseeds))
        out[f"{tag}_driftpool_vs_degradgnn_acc"] = paired(
            [dpd[s]["accuracy"] for s in common], [degg[s]["accuracy"] for s in common])

    with open(os.path.join(R, "significance_5seed.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/significance_5seed.json")


if __name__ == "__main__":
    main()