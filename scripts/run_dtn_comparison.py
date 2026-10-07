"""
Full pipeline: train DTN + baselines, compare at all truncation levels.

Produces results/dtn_comparison.json with DTN vs BaselineLSTM vs DegradGNN vs Hybrid.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN
from models.hybrid import HybridDegradLSTM
from models.degradation_trajectory import DegradationTrajectoryNetwork
from features.drift import compute_drift

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")


def truncate_graph(graph, fraction):
    device = graph.x.device
    order = graph.hop_depth.argsort()
    n = graph.x.size(0)
    keep = max(2, int(n * fraction))
    keep_nodes = order[:keep]
    node_map = {old.item(): i for i, old in enumerate(keep_nodes)}
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[keep_nodes] = True
    edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
    new_edge_index = graph.edge_index[:, edge_mask].clone()
    for i in range(new_edge_index.size(1)):
        new_edge_index[0, i] = node_map[new_edge_index[0, i].item()]
        new_edge_index[1, i] = node_map[new_edge_index[1, i].item()]

    from torch_geometric.data import Data
    truncated = Data(
        x=graph.x[keep_nodes],
        edge_index=new_edge_index,
        y=graph.y,
        hop_depth=graph.hop_depth[keep_nodes],
    )
    if hasattr(graph, "parent") and graph.parent is not None:
        truncated.parent = graph.parent[keep_nodes]
    return truncated


def move_graph_to_device(graph, device):
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.hop_depth = graph.hop_depth.to(device)
    graph.y = graph.y.to(device)
    if hasattr(graph, "parent") and graph.parent is not None:
        graph.parent = graph.parent.to(device)
    return graph


def train_baseline(model_cls, train_ds, val_ds, seed, epochs=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = model_cls(768).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
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
    return model


def train_dtn(train_ds, val_ds, seed, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = DegradationTrajectoryNetwork(input_dim=768, hidden_dim=128,
                                          num_layers=2, dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    best_acc, best_state, bad = 0.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch in tl:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            opt.zero_grad()

            total_loss = torch.tensor(0.0, device=DEVICE)
            for i, data in enumerate(batch.to_data_list()):
                logits, drift_pred = model(data)
                parent = getattr(data, "parent", None)
                n_children = getattr(data, "n_children", None)
                if parent is None:
                    parent = torch.full((data.x.size(0),), -1, dtype=torch.long, device=DEVICE)
                if n_children is None:
                    n_children = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
                target_drift = compute_drift(data.x, parent, data.hop_depth, n_children)

                cls_loss = bce(logits, labels[i].unsqueeze(0))
                drift_loss = mse(drift_pred, target_drift.detach())
                total_loss = total_loss + cls_loss + 0.3 * drift_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vc = vs = 0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                labels = batch.y.float().unsqueeze(1)
                logits_list = []
                for data in batch.to_data_list():
                    logits, _ = model(data)
                    logits_list.append(logits)
                logits = torch.cat(logits_list, dim=0)
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
    return model


@torch.no_grad()
def evaluate_model(model, test_ds, fractions, is_dtn=False):
    results = {}
    for frac in fractions:
        probs, ys = [], []
        for d in test_ds:
            if frac < 1.0:
                d_eval = truncate_graph(d, frac)
            else:
                d_eval = d
            d_eval = move_graph_to_device(d_eval, DEVICE)
            if is_dtn:
                logits, _ = model(d_eval)
            else:
                logits = model(d_eval)
            probs.append(torch.sigmoid(logits).item())
            ys.append(d_eval.y.item())
        probs = np.array(probs)
        ys = np.array(ys)
        preds = (probs > 0.5).astype(float)
        pct = int(frac * 100)
        r = {"accuracy": float((preds == ys).mean())}
        if len(np.unique(ys)) > 1:
            r["auc_roc"] = float(roc_auc_score(ys, probs))
            r["f1"] = float(f1_score(ys, preds))
        results[f"pct_{pct}"] = r
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="gossipcop")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dtn-epochs", type=int, default=30)
    ap.add_argument("--out", default="results/dtn_comparison.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    fractions = [0.1, 0.25, 0.5, 1.0]

    print(f"Loading {args.subset}...", flush=True)
    train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                                feature_type="bert", split="train")
    val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                              feature_type="bert", split="val")
    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                               feature_type="bert", split="test")
    print(f"  Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}\n", flush=True)

    all_results = {
        "BaselineLSTM": {"per_seed": []},
        "DegradGNN": {"per_seed": []},
        "Hybrid": {"per_seed": []},
        "DTN": {"per_seed": []},
    }
    model_map = {
        "BaselineLSTM": BaselineLSTM,
        "DegradGNN": DegradGNN,
        "Hybrid": HybridDegradLSTM,
    }

    for seed in seeds:
        print(f"--- Seed {seed} ---", flush=True)
        for name, model_cls in model_map.items():
            t0 = time.time()
            model = train_baseline(model_cls, train_ds, val_ds, seed, epochs=args.epochs)
            metrics = evaluate_model(model, test_ds, fractions, is_dtn=False)
            elapsed = time.time() - t0
            metrics["train_time_s"] = round(elapsed, 1)
            all_results[name]["per_seed"].append(metrics)
            acc10 = metrics["pct_10"]["accuracy"]
            acc100 = metrics["pct_100"]["accuracy"]
            print(f"  {name:<15s} 10%={acc10:.4f} 100%={acc100:.4f}  ({elapsed:.0f}s)", flush=True)

        t0 = time.time()
        dtn_model = train_dtn(train_ds, val_ds, seed, epochs=args.dtn_epochs)
        dtn_metrics = evaluate_model(dtn_model, test_ds, fractions, is_dtn=True)
        dtn_metrics["train_time_s"] = round(time.time() - t0, 1)
        all_results["DTN"]["per_seed"].append(dtn_metrics)
        print(f"  {'DTN':<15s} 10%={dtn_metrics['pct_10']['accuracy']:.4f} 100%={dtn_metrics['pct_100']['accuracy']:.4f}  ({dtn_metrics['train_time_s']:.0f}s)", flush=True)

    for name in all_results:
        for pct in [int(f * 100) for f in fractions]:
            accs = [s[f"pct_{pct}"]["accuracy"] for s in all_results[name]["per_seed"]]
            aucs = [s[f"pct_{pct}"].get("auc_roc", 0) for s in all_results[name]["per_seed"]]
            all_results[name][f"acc_{pct}pct_mean"] = float(np.mean(accs))
            all_results[name][f"acc_{pct}pct_std"] = float(np.std(accs))
            all_results[name][f"auc_{pct}pct_mean"] = float(np.mean(aucs))
            all_results[name][f"auc_{pct}pct_std"] = float(np.std(aucs))

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Model':<15s}", end="")
    for pct in [int(f * 100) for f in fractions]:
        print(f"  {pct:>3d}%", end="")
    print()
    print(f"{'-'*70}")
    for name in all_results:
        print(f"{name:<15s}", end="")
        for pct in [int(f * 100) for f in fractions]:
            m = all_results[name].get(f"acc_{pct}pct_mean", 0)
            s = all_results[name].get(f"acc_{pct}pct_std", 0)
            print(f"  {m:.3f}±{s:.2f}", end="")
        print()
    print(f"{'='*70}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
