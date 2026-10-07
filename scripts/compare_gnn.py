"""Controlled: train DegradGNN vs AblationGNN(Full) identically, eval both harnesses."""
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.degradation_gnn import DegradationGNN
from models.ablation_gnn import AblationGNN
from models.early_detector import EarlyDetector
from scripts.run_ablation_10pct import truncate_graph, evaluate, DataLike

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def train(model, tl, vl):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()
    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(1, 21):
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
    return model


def my_eval(model, graphs):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for g in graphs:
            x = g["x"].to(DEVICE); ei = g["edge_index"].to(DEVICE)
            hd = g["hop_depth"].to(DEVICE); parent = g["parent"].to(DEVICE); nc = g["n_children"].to(DEVICE)
            logit = model(DataLike(x, ei, hd, parent, nc))
            probs.append(torch.sigmoid(logit).item()); ys.append(g["y"])
    probs = np.array(probs); ys = np.array(ys)
    return float((((probs > 0.5).astype(float)) == ys).mean())


for name, cls in [("DegradGNN", DegradationGNN), ("AblationGNN_Full", lambda d: AblationGNN(d, use_drift=True, use_hop=True))]:
    torch.manual_seed(1); np.random.seed(1)
    tr = UPFDGraphDataset(root='./data/upfd/raw', subset='gossipcop', feature_type='bert', split='train')
    va = UPFDGraphDataset(root='./data/upfd/raw', subset='gossipcop', feature_type='bert', split='val')
    te = UPFDGraphDataset(root='./data/upfd/raw', subset='gossipcop', feature_type='bert', split='test')
    tl = DataLoader(tr, batch_size=64, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(va, batch_size=64, shuffle=False, collate_fn=collate_fn)
    model = cls(768).to(DEVICE)
    model = train(model, tl, vl)
    model.eval()

    full_graphs = [{"x": d.x, "edge_index": d.edge_index, "hop_depth": d.hop_depth,
                    "parent": d.parent, "n_children": d.n_children, "y": d.y.item()} for d in te]
    trunc10 = [truncate_graph(d, 0.1) for d in te]
    for g in trunc10:
        g["y"] = g.pop("y") if "y" in g else None
    trunc10 = [dict(truncate_graph(d, 0.1), y=d.y.item()) for d in te]

    my_full = my_eval(model, full_graphs)
    my_10 = my_eval(model, trunc10)
    labels = [d.y.item() for d in te]
    eds = EarlyDetector(model).evaluate(te, labels, truncations=[0.1, 1.0])
    print(f"{name}: my_eval full={my_full:.4f} 10pct={my_10:.4f} | EarlyDetector 10pct={eds[0.1]['accuracy']:.4f} 100pct={eds[1.0]['accuracy']:.4f}", flush=True)
