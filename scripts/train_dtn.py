"""
Train the DegradationTrajectoryNetwork (DTN).

Multi-task training:
  1. Veracity classification (main)
  2. Per-node drift prediction (auxiliary)
  3. Hop-level trajectory prediction (auxiliary)

Evaluation at 10/25/50/100% cascade truncation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset, collate_fn
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


def compute_hop_trajectory_targets(drift_features, hop_depths):
    max_hop = hop_depths.max().item() + 1
    stats = []
    for hop in range(max_hop):
        mask = hop_depths == hop
        if mask.sum() > 0:
            stats.append(drift_features[mask].mean(dim=0))
        else:
            stats.append(torch.zeros(drift_features.shape[1], device=drift_features.device))
    return torch.stack(stats, dim=0)


@torch.no_grad()
def evaluate_at_fraction(model, test_ds, fraction, device):
    model.eval()
    probs, ys = [], []
    for d in test_ds:
        if fraction < 1.0:
            d = truncate_graph(d, fraction)
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        d.hop_depth = d.hop_depth.to(device)
        d.y = d.y.to(device)
        if hasattr(d, "parent") and d.parent is not None:
            d.parent = d.parent.to(device)
        logits, _ = model(d)
        probs.append(torch.sigmoid(logits).item())
        ys.append(d.y.item())
    probs = np.array(probs)
    ys = np.array(ys)
    preds = (probs > 0.5).astype(float)
    acc = float.accuracy_score(ys, preds) if hasattr(float, "accuracy_score") else float((preds == ys).mean())
    acc = float((preds == ys).mean())
    result = {"accuracy": acc}
    if len(np.unique(ys)) > 1:
        result["auc_roc"] = float(roc_auc_score(ys, probs))
        result["f1"] = float(f1_score(ys, preds))
    return result


def train_one_seed(seed, train_ds, val_ds, test_ds, epochs=30, lr=1e-3, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = DegradationTrajectoryNetwork(input_dim=768, hidden_dim=128, num_layers=2,
                                          dropout=0.3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    best_val_acc = 0.0
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_graphs = 0

        for batch in tl:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            opt.zero_grad()

            total_loss = torch.tensor(0.0, device=DEVICE)
            for data in batch.to_data_list():
                logits, drift_pred = model(data, return_trajectory=False)

                parent = getattr(data, "parent", None)
                n_children = getattr(data, "n_children", None)
                if parent is None:
                    parent = torch.full((data.x.size(0),), -1, dtype=torch.long, device=DEVICE)
                if n_children is None:
                    n_children = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
                target_drift = compute_drift(data.x, parent, data.hop_depth, n_children)

                cls_loss = bce(logits, labels[data.y.item() if hasattr(data.y, 'item') else 0].unsqueeze(0))
                drift_loss = mse(drift_pred, target_drift.detach())

                max_hop = data.hop_depth.max().item() + 1
                hop_stats = []
                for hop in range(max_hop):
                    mask = data.hop_depth == hop
                    if mask.sum() > 0:
                        hop_stats.append(target_drift[mask].mean(dim=0))
                    else:
                        hop_stats.append(torch.zeros(5, device=DEVICE))
                target_trajectory = torch.stack(hop_stats, dim=0)

                hop_drift_input = []
                for hop in range(max_hop):
                    mask = data.hop_depth == hop
                    if mask.sum() > 0:
                        hop_drift_input.append(drift_pred[mask].mean(dim=0))
                    else:
                        hop_drift_input.append(torch.zeros(5, device=DEVICE))
                hop_drift_input = torch.stack(hop_drift_input, dim=0).unsqueeze(0)

                pred_trajectory, _ = model.predict_trajectory(hop_drift_input)
                pred_trajectory = pred_trajectory.squeeze(0)
                traj_loss = mse(pred_trajectory, target_trajectory.detach())

                total_loss = total_loss + cls_loss + 0.3 * drift_loss + 0.3 * traj_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += total_loss.item()
            n_graphs += len(batch.to_data_list())

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
        val_acc = vc / max(vs, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | loss={epoch_loss / n_graphs:.4f} val_acc={val_acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.to(DEVICE)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="gossipcop")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/dtn_early_detection.json")
    ap.add_argument("--truncations", default="0.1,0.25,0.5,1.0")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    truncations = [float(t) for t in args.truncations.split(",")]

    print(f"Loading {args.subset}...", flush=True)
    train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                                feature_type="bert", split="train")
    val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                              feature_type="bert", split="val")
    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset,
                               feature_type="bert", split="test")
    print(f"  Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}", flush=True)

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    for seed in seeds:
        seed_key = f"seed_{seed}"
        if seed_key in results:
            print(f"  Skipping seed {seed} (already done)", flush=True)
            continue

        print(f"\n=== Seed {seed} ===", flush=True)
        t0 = time.time()
        model = train_one_seed(seed, train_ds, val_ds, test_ds, epochs=args.epochs)

        seed_results = {}
        for frac in truncations:
            pct = int(frac * 100)
            m = evaluate_at_fraction(model, test_ds, frac, DEVICE)
            seed_results[f"pct_{pct}"] = m
            print(f"    {pct:>3d}%: acc={m['accuracy']:.4f} auc={m.get('auc_roc', 0):.4f}", flush=True)

        elapsed = time.time() - t0
        results[seed_key] = seed_results
        results[seed_key]["train_time_s"] = round(elapsed, 1)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    all_metrics = {}
    for pct in [int(t * 100) for t in truncations]:
        accs = [results[f"seed_{s}"][f"pct_{pct}"]["accuracy"] for s in seeds
                if f"seed_{s}" in results]
        aucs = [results[f"seed_{s}"][f"pct_{pct}"].get("auc_roc", 0) for s in seeds
                if f"seed_{s}" in results]
        all_metrics[f"acc_{pct}pct_mean"] = float(np.mean(accs))
        all_metrics[f"acc_{pct}pct_std"] = float(np.std(accs))
        if aucs:
            all_metrics[f"auc_{pct}pct_mean"] = float(np.mean(aucs))
            all_metrics[f"auc_{pct}pct_std"] = float(np.std(aucs))
    results["summary"] = all_metrics

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== DTN Summary ===")
    for pct in [int(t * 100) for t in truncations]:
        am = all_metrics.get(f"acc_{pct}pct_mean", 0)
        as_ = all_metrics.get(f"acc_{pct}pct_std", 0)
        au = all_metrics.get(f"auc_{pct}pct_mean", 0)
        aus = all_metrics.get(f"auc_{pct}pct_std", 0)
        print(f"  {pct:>3d}%: acc={am:.4f}±{as_:.4f}  auc={au:.4f}±{aus:.4f}")


if __name__ == "__main__":
    main()
