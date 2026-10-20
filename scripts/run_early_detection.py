"""Multi-seed early detection: train on FULL cascades, evaluate at 10/25/50/100% truncation.

Clean, reproducible protocol via EarlyDetector. 3 seeds x (BaselineLSTM, DegradGNN, Hybrid).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN
from models.hybrid import HybridDegradLSTM
from models.drift_pool import DriftPoolHybrid, DriftPoolOnly
from models.early_detector import EarlyDetector

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODELS = {
    "BaselineLSTM": BaselineLSTM,
    "DegradGNN": DegradationGNN,
    "Hybrid": HybridDegradLSTM,
    "DriftPool": DriftPoolHybrid,
    "DriftPoolOnly": DriftPoolOnly,
}
TRUNCS = [0.1, 0.25, 0.5, 1.0]


def train_and_eval(model_cls, train_ds, val_ds, test_ds, seed, epochs=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = model_cls(768).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()
    best_acc, best_state, bad = 0.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch in tl:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            opt.zero_grad()
            logits = torch.cat([model(d) for d in batch.to_data_list()], dim=0)
            crit(logits, labels).backward()
            opt.step()
        model.eval()
        vc = vs = 0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                labels = batch.y.float().unsqueeze(1)
                logits = torch.cat([model(d) for d in batch.to_data_list()], dim=0)
                vc += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
                vs += labels.size(0)
        vacc = vc / max(vs, 1)
        if vacc > best_acc:
            best_acc, bad = vacc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()

    graphs = [d for d in test_ds]
    labels = [d.y.item() for d in graphs]
    ed = EarlyDetector(model)
    curve = ed.evaluate(graphs, labels, truncations=TRUNCS)

    out = {}
    for frac in TRUNCS:
        out[f"pct_{int(frac * 100)}"] = {
            "accuracy": curve[frac]["accuracy"],
            "f1": curve[frac]["f1"],
            "auc_roc": curve[frac].get("auc_roc", 0.0),
        }
    out["full_acc"] = curve[1.0]["accuracy"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--models", default="BaselineLSTM,DegradGNN,Hybrid")
    ap.add_argument("--out", default="results/early_detection_multiseed.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    model_names = args.models.split(",")

    train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="train")
    val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="val")
    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="test")

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    for name in model_names:
        done_seeds = {m["seed"] for m in results.get(name, {}).get("per_seed", [])}
        missing = [s for s in seeds if s not in done_seeds]
        if not missing:
            print(f"Skipping {name} (complete)", flush=True)
            continue
        metrics = results.get(name, {}).get("per_seed", [])
        for seed in missing:
            t0 = time.time()
            m = train_and_eval(MODELS[name], train_ds, val_ds, test_ds, seed)
            m["seed"] = seed
            metrics.append(m)
            print(f"[{name}] seed={seed} full={m['full_acc']:.4f} "
                  f"10%={m['pct_10']['accuracy']:.4f} 25%={m['pct_25']['accuracy']:.4f} "
                  f"50%={m['pct_50']['accuracy']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            # Preserve any previously stored aggregate keys so a resumed run
            # does not silently drop them.
            results[name] = {**results.get(name, {}), "per_seed": metrics}
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
        for pct in [10, 25, 50, 100]:
            vals = [m[f"pct_{pct}"]["accuracy"] for m in metrics]
            results[name][f"acc_{pct}pct_mean"] = float(np.mean(vals))
            results[name][f"acc_{pct}pct_std"] = float(np.std(vals))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== FINAL (multi-seed early detection) ===")
    for name in model_names:
        r = results.get(name, {})
        if not r:
            print(f"{name:<12} (no results)")
            continue
        row = "  ".join(f"{pct}%:{r.get(f'acc_{pct}pct_mean', float('nan')):.4f}±{r.get(f'acc_{pct}pct_std', float('nan')):.4f}"
                        for pct in [10, 25, 50, 100])
        print(f"{name:<12} {row}")


if __name__ == "__main__":
    main()
