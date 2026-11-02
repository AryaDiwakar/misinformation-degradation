import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.baselines import GCNModel, GATModel, BiGCNModel
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN
from models.hybrid import HybridDegradLSTM
from models.drift_pool import DriftPoolHybrid, DriftPoolOnly
from models.cross_modal_tree import CrossModalTreeNet
from pipeline.checkpoints import save_checkpoint
from pipeline.models import canonical_model_name

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODELS = {
    "GCN": GCNModel,
    "GAT": GATModel,
    "BiGCN": BiGCNModel,
    "BaselineLSTM": BaselineLSTM,
    "DegradGNN": DegradationGNN,
    "Hybrid": HybridDegradLSTM,
    "DriftPool": DriftPoolHybrid,
    "DriftPoolOnly": DriftPoolOnly,
    "CrossModalTree": CrossModalTreeNet,
}

DEFAULT_MODELS = tuple(name for name in MODELS if name != "CrossModalTree")

# Constructor arguments recorded in each checkpoint. These must match the
# arguments the model is actually built with, otherwise the checkpoint cannot
# be reloaded (shape mismatches). DriftPoolOnly defaults to hidden_dim=64
# while the other models default to 128, so the per-model value matters.
MODEL_CONFIGS = {
    "GCN": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "GAT": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "BiGCN": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "BaselineLSTM": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "DegradGNN": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "Hybrid": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "DriftPool": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
    "DriftPoolOnly": {"hidden_dim": 64, "dropout": 0.3},
    "CrossModalTree": {
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "profile_dim": 10,
        "hop_dim": 16,
        "modality_dropout": 0.1,
    },
}


def build_model_for(name, input_dim):
    """Instantiate a model with the exact configuration stored in checkpoints."""
    cls = MODELS[name]
    kwargs = dict(MODEL_CONFIGS[name])
    if name == "BaselineLSTM":
        kwargs["bidirectional"] = True
    return cls(input_dim, **kwargs)

EPOCHS = {"politifact": 50, "gossipcop": 20}
BS = {"politifact": 32, "gossipcop": 64}


def train_and_eval(model_cls, train_ds, val_ds, test_ds, subset, seed, epochs,
                   model_config=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    bs = BS[subset]
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    tel = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)

    if model_config is None:
        model = model_cls(768).to(DEVICE)
    else:
        kwargs = {k: v for k, v in model_config.items() if k != "input_dim"}
        model = model_cls(768, **kwargs).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.BCEWithLogitsLoss()
    best_acc, best_state, patience, bad = 0.0, None, 8, 0
    best_epoch, best_vloss = 0, float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        for batch in tl:
            batch = batch.to(DEVICE)
            labels = batch.y.float().unsqueeze(1)
            opt.zero_grad()
            if getattr(model, "supports_batch", False):
                logits = model(batch)
            else:
                logits = torch.cat([model(d) for d in batch.to_data_list()], dim=0)
            crit(logits, labels).backward()
            opt.step()

        model.eval()
        vc, vs, vloss = 0, 0, 0.0
        with torch.no_grad():
            for batch in vl:
                batch = batch.to(DEVICE)
                labels = batch.y.float().unsqueeze(1)
                if getattr(model, "supports_batch", False):
                    logits = model(batch)
                else:
                    logits = torch.cat([model(d) for d in batch.to_data_list()], dim=0)
                vc += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
                vs += labels.size(0)
                vloss += crit(logits, labels).item() * labels.size(0)
        vacc = vc / max(vs, 1)
        if vacc > best_acc:
            best_acc = vacc
            best_epoch, best_vloss = ep, vloss / max(vs, 1)
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in tel:
            batch = batch.to(DEVICE)
            ys.extend(batch.y.cpu().numpy().tolist())
            if getattr(model, "supports_batch", False):
                logits = model(batch)
            else:
                logits = torch.cat([model(d) for d in batch.to_data_list()], dim=0)
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
    ys, probs = np.array(ys), np.array(probs)
    preds = (probs > 0.5).astype(float)
    metrics = {
        "accuracy": float(accuracy_score(ys, preds)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.0,
    }
    return metrics, model, best_epoch, best_vloss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="politifact", choices=["politifact", "gossipcop"])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--models", default=None)
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default epoch budget (useful for smoke tests).",
    )
    ap.add_argument(
        "--features",
        default="bert",
        help="Node feature view(s), joined with '+'. CrossModalTree needs bert+profile.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="If set, save each model/seed's best-val-accuracy checkpoint here.",
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    model_names = list(DEFAULT_MODELS) if args.models is None else args.models.split(",")
    lookup = {canonical_model_name(name): name for name in MODELS}
    try:
        model_names = [lookup[canonical_model_name(name)] for name in model_names]
    except (KeyError, ValueError) as exc:
        ap.error(f"Unknown model in --models: {exc}")
    if "CrossModalTree" in model_names and "profile" not in args.features.split("+"):
        ap.error("CrossModalTree requires --features bert+profile")

    train_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset, feature_type=args.features, split="train")
    val_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset, feature_type=args.features, split="val")
    test_ds = UPFDGraphDataset(root="./data/upfd/raw", subset=args.subset, feature_type=args.features, split="test")

    input_dim = train_ds[0].x.shape[1]

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    for name in model_names:
        existing_seeds = {m.get("seed") for m in results.get(name, {}).get("per_seed", [])}
        if all(s in existing_seeds for s in seeds):
            print(f"Skipping {name} (already complete)", flush=True)
            continue
        metrics = list(results.get(name, {}).get("per_seed", []))
        for seed in seeds:
            if any(m.get("seed") == seed for m in metrics):
                continue
            t0 = time.time()
            m, model, best_epoch, best_vloss = train_and_eval(
                MODELS[name], train_ds, val_ds, test_ds, args.subset, seed,
                args.epochs or EPOCHS[args.subset], model_config=MODEL_CONFIGS[name],
            )
            m["seed"] = seed
            metrics.append(m)
            results[name] = {"per_seed": metrics}
            print(f"[{args.subset}] {name} seed={seed}: Acc={m['accuracy']:.4f} AUCROC={m['auc_roc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
            if args.checkpoint_dir:
                registry_name = canonical_model_name(name)
                ckpt_path = os.path.join(
                    args.checkpoint_dir, f"{registry_name}_{args.subset}_seed{seed}.pt"
                )
                save_checkpoint(
                    ckpt_path,
                    model,
                    registry_name,
                    {
                        "input_dim": input_dim,
                        "name": registry_name,
                        **MODEL_CONFIGS[name],
                    },
                    {
                        "subset": args.subset,
                        "feature_type": args.features,
                        "epochs": args.epochs or EPOCHS[args.subset],
                        "batch_size": BS[args.subset],
                        "optimizer": "AdamW",
                        "lr": 1e-3,
                        "weight_decay": 1e-5,
                        "protocol": "multiseed",
                        "selection_metric": "val_accuracy",
                    },
                    epoch=best_epoch,
                    best_val_loss=best_vloss,
                    seed=seed,
                )
                print(f"  -> checkpoint saved: {ckpt_path}", flush=True)
            else:
                print("  -> checkpoint not saved (pass --checkpoint-dir to save)", flush=True)
        mets = [m for m in metrics if "seed" in m]
        results[name] = {
            "acc_mean": float(np.mean([m["accuracy"] for m in mets])),
            "acc_std": float(np.std([m["accuracy"] for m in mets])),
            "f1_mean": float(np.mean([m["f1"] for m in mets])),
            "auc_mean": float(np.mean([m["auc_roc"] for m in mets])),
            "auc_std": float(np.std([m["auc_roc"] for m in mets])),
            "aucpr_mean": float(np.mean([m["auc_pr"] for m in mets])),
            "per_seed": mets,
        }
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> saved {args.out}", flush=True)

    print("\n=== FINAL ===")
    for name in model_names:
        r = results[name]
        print(f"{name:<14} Acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  AUCROC={r['auc_mean']:.4f}+/-{r['auc_std']:.4f}")


if __name__ == "__main__":
    main()
