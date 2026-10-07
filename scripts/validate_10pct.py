"""Validate the 10%-truncation evaluation path against early_detection.json numbers.

Trains DegradGNN with standard multiseed settings (seed 1), then evaluates at
10% and 100% truncation using the exact EarlyDetector code path that produced
the ~90% @10% numbers. Detects whether my run_ablation_10pct truncation differs.
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.degradation_gnn import DegradationGNN
from models.early_detector import EarlyDetector

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1)
np.random.seed(1)

train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="train")
val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="val")
test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="test")
tl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
vl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

model = DegradationGNN(768).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
crit = torch.nn.BCEWithLogitsLoss()
best_acc, best_state, bad = 0.0, None, 0
for ep in range(1, 21):
    model.train()
    for batch in tl:
        batch = batch.to(DEVICE)
        labels = batch.y.float().unsqueeze(1)
        opt.zero_grad()
        logits = torch.cat([model(d, return_drift=True)[0] for d in batch.to_data_list()], dim=0)
        crit(logits, labels).backward()
        opt.step()
    model.eval()
    vc = vs = 0
    with torch.no_grad():
        for batch in vl:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            logits = torch.cat([model(d, return_drift=True)[0] for d in batch.to_data_list()], dim=0)
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

ag = [d for batch in test_ds for d in [batch]]
# rebuild graphs via loader test split
test_graphs = [d for d in test_ds]
labels = [d.y.item() for d in test_graphs]

ed = EarlyDetector(model)
curve = ed.evaluate(test_graphs, labels, truncations=[0.1, 0.25, 0.5, 1.0])
for frac, m in curve.items():
    print(f"truncation={frac}: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")

with open("results/validation_degradgnn_10pct.json", "w") as f:
    json.dump(curve, f, indent=2, default=str)
