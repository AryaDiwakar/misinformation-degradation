import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from data.upfd.loader import UPFDGraphDataset, collate_fn
from features.drift import compute_drift
from pipeline.checkpoints import save_checkpoint
from pipeline.config import load_config
from pipeline.data_validation import validate_upfd_dataset
from pipeline.models import build_model, canonical_model_name, uses_drift_auxiliary_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Override node feature view(s), e.g. bert+profile for CrossModalTree",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--drift-weight", type=float, default=None,
                        help="Weight for drift MSE loss in multi-task training")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-data-validation", action="store_true")
    return parser.parse_args()


def choose_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_metrics(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        result["auc_roc"] = float(roc_auc_score(labels, probs))
        result["auc_pr"] = float(average_precision_score(labels, probs))
    return result


def train_epoch(model, loader, optimizer, criterion, device, drift_weight=0.5, use_drift=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    drift_criterion = nn.MSELoss()
    pbar = tqdm(loader, desc="Train")

    for batch in pbar:
        batch = batch.to(device)
        labels = batch.y.float().unsqueeze(1)
        optimizer.zero_grad()

        logits_list = []
        drift_loss = torch.tensor(0.0, device=device)

        if getattr(model, "supports_batch", False) and not use_drift:
            logits_list.append(model(batch))
        else:
            for data in batch.to_data_list():
                if use_drift:
                    logits, drift_pred = model(data, return_drift=True)
                    logits_list.append(logits)
                    parent = getattr(data, "parent", None)
                    n_children = getattr(data, "n_children", None)
                    if parent is None:
                        parent = torch.full((data.x.size(0),), -1, dtype=torch.long, device=data.x.device)
                    if n_children is None:
                        n_children = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
                    target_drift = compute_drift(data.x, parent, data.hop_depth, n_children)
                    drift_loss = drift_loss + drift_criterion(drift_pred, target_drift.detach())
                else:
                    logits_list.append(model(data))

        logits = torch.cat(logits_list, dim=0)
        loss = criterion(logits, labels)

        if use_drift:
            loss = loss + drift_weight * drift_loss

        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=total_correct / max(total_samples, 1))

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_drift=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.float().unsqueeze(1)
        logits_list = []
        if getattr(model, "supports_batch", False) and not use_drift:
            logits_list.append(model(batch))
        else:
            for data in batch.to_data_list():
                if use_drift:
                    logits, _ = model(data, return_drift=True)
                else:
                    logits = model(data)
                logits_list.append(logits)
        logits = torch.cat(logits_list, dim=0)
        loss = criterion(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
        np.array(all_preds),
        np.array(all_labels),
    )


def main():
    args = parse_args()

    config = load_config(args.config)

    if args.features:
        config["data"]["feature_type"] = args.features

    model_name = canonical_model_name(args.model or config["model"]["name"])
    training_config = config.get("training", {})
    batch_size = args.batch_size or config["data"].get("batch_size", 32)
    epochs = args.epochs or int(training_config.get("epochs", 100))
    learning_rate = args.lr or float(training_config.get("lr", 1e-3))
    drift_weight = (
        args.drift_weight
        if args.drift_weight is not None
        else float(training_config.get("drift_weight", 0.5))
    )
    device = choose_device(args.device)
    seed_everything(args.seed)
    print(f"Device: {device} | Model: {model_name} | Seed: {args.seed}")

    if not args.skip_data_validation:
        summary = validate_upfd_dataset(
            config["data"]["root"],
            config["data"]["subset"],
            config["data"].get("feature_type", "bert"),
        )
        print(
            f"Dataset: {summary['num_graphs']} graphs, "
            f"{summary['num_nodes']} nodes, {summary['feature_dim']}-dim features"
        )

    train_ds = UPFDGraphDataset(
        root=config["data"]["root"],
        subset=config["data"]["subset"],
        feature_type=config["data"]["feature_type"],
        split="train",
    )
    val_ds = UPFDGraphDataset(
        root=config["data"]["root"],
        subset=config["data"]["subset"],
        feature_type=config["data"]["feature_type"],
        split="val",
    )
    test_ds = UPFDGraphDataset(
        root=config["data"]["root"],
        subset=config["data"]["subset"],
        feature_type=config["data"]["feature_type"],
        split="test",
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_dim = train_ds[0].x.shape[1]
    model_config = dict(config.get("model", {}))
    model_config["input_dim"] = input_dim
    model_config["name"] = model_name
    model = build_model(model_name, input_dim, model_config).to(device)
    # The drift auxiliary loss is a multi-task variant. It is *not* the
    # protocol behind the paper's reported numbers (those come from
    # multiseed.py, which uses plain BCE). Set training.drift_weight to 0 or
    # use_drift_aux_loss: false to train without it.
    use_drift = uses_drift_auxiliary_loss(model_name) and bool(
        training_config.get("use_drift_aux_loss", True)
    ) and drift_weight > 0

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training_config.get("weight_decay", 1e-5)),
    )
    criterion = nn.BCEWithLogitsLoss()

    patience = int(training_config.get("early_stop_patience", 10))
    epochs_no_improve = 0

    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=config["logging"]["log_dir"])
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            drift_weight=drift_weight, use_drift=use_drift,
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, use_drift=use_drift,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f"  -> best validation loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, use_drift=use_drift,
    )
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    probs = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if getattr(model, "supports_batch", False):
                probs.extend(torch.sigmoid(model(batch)).flatten().cpu().numpy())
            else:
                for data in batch.to_data_list():
                    output = model(data)
                    probs.append(torch.sigmoid(output).item())

    checkpoint_dir = config["logging"].get("checkpoint_dir", "checkpoints")
    result_dir = config["logging"].get("result_dir", "results")
    # The multiseed runner writes "<model>_<subset>_seed<N>.pt" for the paper's
    # canonical protocol (plain BCE, best-validation-accuracy selection). This
    # trainer uses a different protocol (drift auxiliary loss, best-validation
    # -loss selection), so it must use a distinct filename — otherwise running
    # it silently replaces the canonical paper checkpoint with a weaker one.
    tag = f"{model_name}_{config['data']['subset']}_seed{args.seed}_trainaug"
    checkpoint_path = os.path.join(checkpoint_dir, f"{tag}.pt")
    save_checkpoint(
        checkpoint_path,
        model,
        model_name,
        model_config,
        config,
        epoch=best_epoch,
        best_val_loss=best_val_loss,
        seed=args.seed,
    )
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{tag}.json")
    with open(result_path, "w") as handle:
        json.dump(
            {
                "model": model_name,
                "dataset": config["data"]["subset"],
                "seed": args.seed,
                "device": str(device),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_metrics": model_metrics(labels, probs),
                "checkpoint": checkpoint_path,
                "protocol": "train_config",
                "selection_metric": "val_loss",
                "uses_drift_auxiliary_loss": use_drift,
                "duration_s": round(time.time() - started, 2),
                "history": history,
            },
            handle,
            indent=2,
        )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {result_path}")
    writer.close()


if __name__ == "__main__":
    main()
