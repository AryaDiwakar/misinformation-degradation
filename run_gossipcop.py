"""GossipCop experiments — 3 models, 20 epochs each."""
import torch, os, sys, json, time, numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.baselines import GCNModel
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN
from features.drift import compute_drift

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
os.makedirs("results", exist_ok=True)

def run(model, name, epochs=20, is_gnn=False, dw=0.0):
    ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="train")
    ds2 = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="val")
    ds3 = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="test")
    tl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(ds2, batch_size=32, shuffle=False, collate_fn=collate_fn)
    tel = DataLoader(ds3, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"{name}: {len(ds)}/{len(ds2)}/{len(ds3)} graphs", flush=True)

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()
    dc = torch.nn.MSELoss()
    bv, bs, pat, bad = float("inf"), None, 10, 0
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
            for d in batch.to_data_list():
                if is_gnn:
                    logits, dp = model(d, return_drift=True)
                    llist.append(logits)
                    p = getattr(d, "parent", None)
                    nc = getattr(d, "n_children", None)
                    if p is None: p = torch.full((d.x.size(0),), -1, device=DEVICE)
                    if nc is None: nc = torch.zeros(d.x.size(0), dtype=torch.long, device=DEVICE)
                    dl = dl + dc(dp, compute_drift(d.x, p, d.hop_depth, nc).detach())
                else:
                    llist.append(model(d))
            logits = torch.cat(llist, dim=0)
            loss = crit(logits, labels)
            if is_gnn and dw > 0: loss = loss + dw * dl
            loss.backward()
            opt.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            tc += (preds == labels).sum().item()
            ts += labels.size(0)
            tloss += loss.item() * labels.size(0)

        model.eval()
        vloss = vc = vs = 0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                labels = batch.y.float().unsqueeze(1)
                llist = []
                for d in batch.to_data_list():
                    if is_gnn: logits, _ = model(d, return_drift=True)
                    else: logits = model(d)
                    llist.append(logits)
                logits = torch.cat(llist, dim=0)
                loss = crit(logits, labels)
                preds = (torch.sigmoid(logits) > 0.5).float()
                vc += (preds == labels).sum().item()
                vs += labels.size(0)
                vloss += loss.item() * labels.size(0)

        v_loss = vloss / max(vs, 1)
        v_acc = vc / max(vs, 1)
        if v_loss < bv: bv = v_loss; bad = 0; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if bad >= pat: break
        if ep % 5 == 0:
            print(f"  {name} Ep {ep:3d} | Train: {tloss/max(ts,1):.4f}/{tc/max(ts,1):.4f} | Val: {v_loss:.4f}/{v_acc:.4f}", flush=True)

    model.load_state_dict(bs)
    model.eval()
    ap, app, al = [], [], []
    with torch.no_grad():
        for batch in tel:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            llist = []
            for d in batch.to_data_list():
                if is_gnn: logits, _ = model(d, return_drift=True)
                else: logits = model(d)
                llist.append(logits)
            logits = torch.cat(llist, dim=0)
            probs = torch.sigmoid(logits)
            app.extend(probs.cpu().numpy().flatten())
            al.extend(labels.cpu().numpy().flatten())

    al = np.array(al)
    app = np.array(app)
    ap = (app > 0.5).astype(float)
    m = {
        "accuracy": float(accuracy_score(al, ap)),
        "f1": float(f1_score(al, ap, zero_division=0)),
        "auc_roc": float(roc_auc_score(al, app)) if len(np.unique(al)) > 1 else 0.0,
    }
    t = round(time.time() - t0, 1)
    print(f"  >> {name}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUCROC={m['auc_roc']:.4f} | {t}s", flush=True)
    r = {"model": name, "subset": "gossipcop", "test_metrics": m, "train_time_s": t}
    with open("results/gossipcop_partial.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    return r

results = []
results.append(run(GCNModel(768), "GCN"))
print("GCN done", flush=True)
results.append(run(BaselineLSTM(768), "BaselineLSTM"))
print("BaselineLSTM done", flush=True)
results.append(run(DegradationGNN(768), "DegradGNN (no drift)", is_gnn=True))
print("DegradGNN done", flush=True)

print("\nALL DONE", flush=True)
print("="*80)
for r in results:
    m = r["test_metrics"]
    print(f"{r['model']:<24} {r['subset']:<12} Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUCROC={m['auc_roc']:.4f} | {r['train_time_s']}s")
