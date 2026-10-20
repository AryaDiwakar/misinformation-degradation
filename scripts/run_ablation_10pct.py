"""Train AblationGNN variants on full GossipCop data, evaluate at 100% and 10% truncation.

Tests the hypothesis: drift/hop (structural degradation) features help in the
early-detection regime even though they hurt at full cascade size.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.ablation_gnn import AblationGNN
from pipeline.truncation import truncate_cascade

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def truncate_graph(graph, fraction):
    """Truncate a cascade using the shared pipeline implementation.

    The shared helper rebuilds hop depth, parent and child counts for the
    retained subgraph, so truncated drift targets stay consistent with the
    full-cascade ones.
    """
    return truncate_cascade(graph, fraction)


@torch.no_grad()
def evaluate(model, graphs):
    model.eval()
    probs, ys = [], []
    for g in graphs:
        x = g["x"].to(DEVICE)
        ei = g["edge_index"].to(DEVICE)
        hd = g["hop_depth"].to(DEVICE)
        parent = g["parent"].to(DEVICE)
        nc = g["n_children"].to(DEVICE)
        logit = model(DataLike(x, ei, hd, parent, nc))
        probs.append(torch.sigmoid(logit).item())
        ys.append(g["y"])
    ys = np.array(ys)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(float)
    r = {"accuracy": float(accuracy_score(ys, preds))}
    if len(np.unique(ys)) > 1:
        r["auc_roc"] = float(roc_auc_score(ys, probs))
    return r


class DataLike:
    def __init__(self, x, edge_index, hop_depth, parent, n_children):
        self.x = x
        self.edge_index = edge_index
        self.hop_depth = hop_depth
        self.parent = parent
        self.n_children = n_children


def train(model, tl, vl, epochs=20, lr=1e-3, patience=8):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
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
            best_acc = vacc
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="results/ablations_10pct.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="train")
    val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="val")
    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="test")
    tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    test_graphs = [d for d in test_ds]
    trunc10 = []
    for d in test_ds:
        t = truncate_graph(d, 0.1)
        t["y"] = d.y.item()
        trunc10.append(t)

    variants = {
        "Full": {"use_drift": True, "use_hop": True},
        "w/o drift": {"use_drift": False, "use_hop": True},
        "w/o hop": {"use_drift": True, "use_hop": False},
    }

    results = {}
    for name, flags in variants.items():
        t0 = time.time()
        model = AblationGNN(input_dim=768, hidden_dim=128, num_layers=2, dropout=0.3, **flags).to(DEVICE)
        model = train(model, tl, vl, epochs=args.epochs)
        full = evaluate(model, test_graphs)
        early = evaluate(model, trunc10)
        results[name] = {"full": full, "truncated_10pct": early, "train_time_s": round(time.time() - t0, 1)}
        print(f"{name}: full={full['accuracy']:.4f} (auc {full.get('auc_roc', 0):.4f}) | 10pct={early['accuracy']:.4f} (auc {early.get('auc_roc', 0):.4f}) [{time.time()-t0:.0f}s]", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== FINAL (10% truncation) ===")
    for name, r in results.items():
        e = r["truncated_10pct"]
        print(f"{name:<10} 10pct Acc={e['accuracy']:.4f} AUC={e.get('auc_roc', 0):.4f} | full Acc={r['full']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
