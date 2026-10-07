"""Run experiments: politifact (50ep) then gossipcop (10ep)."""
import torch, os, sys, time, json, numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN, _build_parent_index
from models.baselines import GCNModel, GATModel, BiGCNModel
from models.early_detector import EarlyDetector
from features.drift import compute_drift

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

def metrics(y_true, y_pred, y_prob):
    r = {}
    try:
        r["accuracy"] = float(accuracy_score(y_true, y_pred))
        r["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        r["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        r["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if len(np.unique(y_true)) > 1:
            r["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            r["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except:
        pass
    return r

def ece(y_true, y_prob, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    e = 0.0
    for i in range(nb):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if m.sum() == 0: continue
        e += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(e / max(len(y_true), 1))

def run(model, name, subset, epochs, is_gnn=False, dw=0.0):
    print(f"\n--- {name} ({subset}, {epochs}ep) ---", flush=True)
    train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="train")
    val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="val")
    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="test")
    tl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader_l = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    tel = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()
    dc = torch.nn.MSELoss()
    best_v = float("inf")
    best_s = None
    pat, bad = 10, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tloss = tc = ts = 0
        for batch in tl:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            opt.zero_grad()
            llist = []
            dl = torch.tensor(0.0, device=DEVICE)
            for data in batch.to_data_list():
                if is_gnn:
                    logits, dp = model(data, return_drift=True)
                    llist.append(logits)
                    parent = _build_parent_index(data.edge_index, data.x.size(0))
                    nc = torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE)
                    for u, v in zip(*data.edge_index[:, data.edge_index[0] != data.edge_index[1]]):
                        nc[u] += 1
                    dl = dl + dc(dp, compute_drift(data.x, parent, data.hop_depth, nc).detach())
                else:
                    llist.append(model(data))
            logits = torch.cat(llist, dim=0)
            loss = crit(logits, labels)
            if is_gnn and dw > 0:
                loss = loss + dw * dl
            loss.backward()
            opt.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            tc += (preds == labels).sum().item()
            ts += labels.size(0)
            tloss += loss.item() * labels.size(0)

        model.eval()
        vloss = vc = vs = 0
        with torch.no_grad():
            for batch in val_loader_l:
                batch = batch.to(DEVICE)
                labels = batch.y.float().unsqueeze(1)
                llist = []
                for data in batch.to_data_list():
                    if is_gnn:
                        logits, _ = model(data, return_drift=True)
                    else:
                        logits = model(data)
                    llist.append(logits)
                logits = torch.cat(llist, dim=0)
                loss = crit(logits, labels)
                preds = (torch.sigmoid(logits) > 0.5).float()
                vc += (preds == labels).sum().item()
                vs += labels.size(0)
                vloss += loss.item() * labels.size(0)
        vl = vloss / max(vs, 1)
        va = vc / max(vs, 1)

        if vl < best_v:
            best_v = vl; bad = 0
            best_s = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= pat: break
        if ep % 10 == 0:
            print(f"  Ep {ep:3d} | Train: {tloss/max(ts,1):.4f}/{tc/max(ts,1):.4f} | Val: {vl:.4f}/{va:.4f}", flush=True)

    model.load_state_dict(best_s)
    model = model.to(DEVICE)
    model.eval()
    all_p, all_prob, all_l = [], [], []
    with torch.no_grad():
        for batch in tel:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            llist = []
            for data in batch.to_data_list():
                if is_gnn:
                    logits, _ = model(data, return_drift=True)
                else:
                    logits = model(data)
                llist.append(logits)
            logits = torch.cat(llist, dim=0)
            probs = torch.sigmoid(logits)
            all_prob.extend(probs.cpu().numpy().flatten())
            all_l.extend(labels.cpu().numpy().flatten())
    all_l = np.array(all_l)
    all_prob = np.array(all_prob)
    all_p = (all_prob > 0.5).astype(float)
    m = metrics(all_l, all_p, all_prob)
    m["ece"] = ece(all_l, all_prob)

    ag = [d for batch in tel for d in batch.to_data_list()]
    al2 = [d.y.item() for d in ag]
    ed = EarlyDetector(model).evaluate(ag, al2, truncations=[0.1, 0.25, 0.5, 0.75, 1.0])

    r = {"model": name, "subset": subset, "test_metrics": m,
         "early_detection": {f"{int(k*100)}%": v for k, v in ed.items()},
         "train_time_s": round(time.time() - t0, 1)}
    print(f"  >> Test: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUCROC={m['auc_roc']:.4f} ECE={m['ece']:.4f} | {r['train_time_s']:.0f}s", flush=True)
    return r

results = []

subset = "politifact"
for cls, name, kw in [
    (GCNModel(768), "GCN", {}),
    (GATModel(768), "GAT", {}),
    (BiGCNModel(768), "BiGCN", {}),
    (BaselineLSTM(768), "BaselineLSTM", {}),
    (DegradationGNN(768), "DegradGNN (no drift)", {"is_gnn": True, "dw": 0.0}),
    (DegradationGNN(768), "DegradGNN (multi-task)", {"is_gnn": True, "dw": 0.5}),
]:
    r = run(cls, name, subset, 50, **kw)
    results.append(r)

subset = "gossipcop"
for cls, name, kw in [
    (GCNModel(768), "GCN", {}),
    (BaselineLSTM(768), "BaselineLSTM", {}),
    (DegradationGNN(768), "DegradGNN (no drift)", {"is_gnn": True, "dw": 0.0}),
    (DegradationGNN(768), "DegradGNN (multi-task)", {"is_gnn": True, "dw": 0.5}),
]:
    r = run(cls, name, subset, 10, **kw)
    results.append(r)

os.makedirs("results", exist_ok=True)
with open("results/all_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print()
print("=" * 110)
print(f"{'Model':<28} {'Subset':<14} {'Acc':>7} {'F1':>7} {'AUC-ROC':>8} {'AUC-PR':>8} {'ECE':>7} {'Time':>7}")
print("=" * 110)
for r in results:
    m = r["test_metrics"]
    print(
        f"{r['model']:<28} {r['subset']:<14} "
        f"{m.get('accuracy',0):>.4f}  {m.get('f1',0):>.4f}  "
        f"{m.get('auc_roc',0):>.4f}   {m.get('auc_pr',0):>.4f}   "
        f"{m.get('ece',0):>.4f}   {r['train_time_s']:>.0f}s"
    )
print()
print("Done. Results saved to results/all_results.json")
