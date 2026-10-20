"""Full-cascade component ablation for DegradGNN (GossipCop).

Produces ``results/ablations.json``: the 100%-cascade ablation table that the
paper reports in Section 5.4 and that the ablation figure reads.

The historical ``results/ablations.json`` had no producing script in the
repository, so the table could not be regenerated. This script reproduces it
with the same protocol as ``scripts/run_ablation_10pct.py``: one model per
variant, trained on the GossipCop train split, selected on validation
accuracy, evaluated on the test split.

Variants:
  Full (drift+hop+tree)  the complete DegradGNN-style model
  w/o drift              drop the per-node drift features
  w/o hop                drop the hop-depth embedding
  w/o drift+hop          drop both (tree message passing only)
  w/o tree (MLP)         mean-pooled MLP over node embeddings
  BERT-MLP               text-only baseline, ignores the graph entirely

Run:
    python3 scripts/run_ablation_100pct.py --seed 1
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
from models.ablation_gnn import BERTMLP, AblationGNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

VARIANTS = {
    "Full (drift+hop+tree)": dict(use_drift=True, use_hop=True, use_tree=True),
    "w/o drift": dict(use_drift=False, use_hop=True, use_tree=True),
    "w/o hop": dict(use_drift=True, use_hop=False, use_tree=True),
    "w/o drift+hop": dict(use_drift=False, use_hop=False, use_tree=True),
    "w/o tree (MLP)": dict(use_drift=False, use_hop=False, use_tree=False),
}


@torch.no_grad()
def evaluate(model, graphs):
    model.eval()
    probs, ys = [], []
    for g in graphs:
        logits = model(g.to(DEVICE))
        probs.append(torch.sigmoid(logits.reshape(-1)[0]).item())
        ys.append(g.y.item())
    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(float)
    result = {
        "accuracy": float(accuracy_score(ys, preds)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
    }
    if len(np.unique(ys)) > 1:
        result["auc_roc"] = float(roc_auc_score(ys, probs))
        result["auc_pr"] = float(roc_auc_score(ys, probs))
    return result


def train(model, tl, vl, epochs=20, lr=1e-3, patience=8):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()
    best_acc, best_state, bad = 0.0, None, 0
    for _ in range(1, epochs + 1):
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
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--root", default="./data/upfd/raw")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results", "ablations.json"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = UPFDGraphDataset(root=args.root, subset="gossipcop",
                                feature_type="bert", split="train")
    val_ds = UPFDGraphDataset(root=args.root, subset="gossipcop",
                              feature_type="bert", split="val")
    test_ds = UPFDGraphDataset(root=args.root, subset="gossipcop",
                               feature_type="bert", split="test")
    tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_graphs = [d for d in test_ds]

    results = {}
    for name, flags in VARIANTS.items():
        t0 = time.time()
        model = AblationGNN(input_dim=768, hidden_dim=128, num_layers=2,
                            dropout=0.3, **flags).to(DEVICE)
        model = train(model, tl, vl, epochs=args.epochs)
        metrics = evaluate(model, test_graphs)
        results[name] = metrics
        print(f"{name:<24} acc={metrics['accuracy']:.4f} "
              f"auc={metrics.get('auc_roc', 0):.4f}  [{time.time() - t0:.0f}s]", flush=True)
        with open(os.path.abspath(args.out), "w") as f:
            json.dump(results, f, indent=2)

    # Text-only baseline: mean-pooled BERT features through an MLP.
    t0 = time.time()
    mlp = BERTMLP(input_dim=768, hidden_dim=128, dropout=0.3).to(DEVICE)
    mlp = train(mlp, tl, vl, epochs=args.epochs)
    results["BERT-MLP"] = evaluate(mlp, test_graphs)
    print(f"{'BERT-MLP':<24} acc={results['BERT-MLP']['accuracy']:.4f} "
          f"auc={results['BERT-MLP'].get('auc_roc', 0):.4f}  [{time.time() - t0:.0f}s]",
          flush=True)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
