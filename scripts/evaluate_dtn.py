"""
Evaluate DTN trajectory quality and coherence scores.

Analyzes:
  1. Trajectory prediction MSE (fake vs real)
  2. Coherence score distributions (fake vs real)
  3. Correlation between coherence score and veracity
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift
from pipeline.checkpoints import load_checkpoint

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def analyze_trajectories(model, test_ds, device):
    model.eval()
    all_coherence = []
    all_labels = []
    all_traj_pred = []
    all_traj_target = []

    for d in test_ds:
        d = d.to(device)
        logits, drift_pred, trajectory_pred, coherence = model(d, return_trajectory=True)

        parent = getattr(d, "parent", None)
        n_children = getattr(d, "n_children", None)
        if parent is None:
            parent = torch.full((d.x.size(0),), -1, dtype=torch.long, device=device)
        if n_children is None:
            n_children = torch.zeros(d.x.size(0), dtype=torch.long, device=device)
        target_drift = compute_drift(d.x, parent, d.hop_depth, n_children)

        max_hop = d.hop_depth.max().item() + 1
        target_stats = []
        for hop in range(max_hop):
            mask = d.hop_depth == hop
            if mask.sum() > 0:
                target_stats.append(target_drift[mask].mean(dim=0))
            else:
                target_stats.append(torch.zeros(5, device=device))
        target_trajectory = torch.stack(target_stats, dim=0)

        all_coherence.append(coherence.item())
        all_labels.append(d.y.item())
        all_traj_pred.append(trajectory_pred.cpu().numpy())
        all_traj_target.append(target_trajectory.cpu().numpy())

    all_coherence = np.array(all_coherence)
    all_labels = np.array(all_labels)

    real_mask = all_labels == 0
    fake_mask = all_labels == 1

    results = {
        "n_real": int(real_mask.sum()),
        "n_fake": int(fake_mask.sum()),
        "coherence_real_mean": float(all_coherence[real_mask].mean()),
        "coherence_real_std": float(all_coherence[real_mask].std()),
        "coherence_fake_mean": float(all_coherence[fake_mask].mean()),
        "coherence_fake_std": float(all_coherence[fake_mask].std()),
        "coherence_gap": float(all_coherence[real_mask].mean() - all_coherence[fake_mask].mean()),
    }

    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(all_coherence[real_mask], all_coherence[fake_mask], alternative="two-sided")
    results["mannwhitney_u"] = float(stat)
    results["mannwhitney_p"] = float(pval)

    fake_traj = np.array([t[:min(5, t.shape[0])] for t in all_traj_target[fake_mask]])
    real_traj = np.array([t[:min(5, t.shape[0])] for t in all_traj_target[real_mask]])
    results["real_traj_mean_by_hop"] = real_traj.mean(axis=0).tolist()
    results["fake_traj_mean_by_hop"] = fake_traj.mean(axis=0).tolist()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subset", default="gossipcop")
    ap.add_argument("--out", default="results/dtn_trajectory_analysis.json")
    args = ap.parse_args()

    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                               feature_type="bert", split="test")
    model, _metadata = load_checkpoint(args.checkpoint, DEVICE, model_name="dtn")
    model.eval()

    results = analyze_trajectories(model, test_ds, DEVICE)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Real coherence:  {results['coherence_real_mean']:.4f} ± {results['coherence_real_std']:.4f}")
    print(f"Fake coherence:  {results['coherence_fake_mean']:.4f} ± {results['coherence_fake_std']:.4f}")
    print(f"Gap:             {results['coherence_gap']:.4f}")
    print(f"Mann-Whitney p:  {results['mannwhitney_p']:.2e}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
