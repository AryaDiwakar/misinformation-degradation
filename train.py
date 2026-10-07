import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN
from models.trajectory_head import TrajectoryPredictor
from features.drift import compute_drift


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="baseline_lstm",
                        choices=["baseline_lstm", "degradation_gnn"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--drift_weight", type=float, default=0.5,
                        help="Weight for drift MSE loss in multi-task training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, drift_weight=0.5, is_gnn=False):
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

        for data in batch.to_data_list():
            if is_gnn:
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

        if is_gnn:
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
def evaluate(model, loader, criterion, device, is_gnn=False):
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
        for data in batch.to_data_list():
            if is_gnn:
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Device: {device}")

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
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_dim = train_ds[0].x.shape[1]
    is_gnn = args.model == "degradation_gnn"

    if args.model == "baseline_lstm":
        model = BaselineLSTM(
            input_dim=input_dim,
            hidden_dim=config["model"].get("hidden_dim", 128),
            num_layers=config["model"].get("num_layers", 2),
            dropout=config["model"].get("dropout", 0.3),
            bidirectional=config["model"].get("bidirectional", True),
        ).to(device)
    elif is_gnn:
        model = DegradationGNN(
            input_dim=input_dim,
            hidden_dim=config["model"].get("hidden_dim", 128),
            num_layers=config["model"].get("num_layers", 2),
            dropout=config["model"].get("dropout", 0.3),
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    patience = config.get("training", {}).get("early_stop_patience", 10)
    epochs_no_improve = 0

    os.makedirs(config["logging"]["log_dir"], exist_ok=True)
    os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=config["logging"]["log_dir"])
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            drift_weight=args.drift_weight, is_gnn=is_gnn,
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, is_gnn=is_gnn,
        )

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
            epochs_no_improve = 0
            ckpt_path = os.path.join(config["logging"]["checkpoint_dir"], "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(
        torch.load(os.path.join(config["logging"]["checkpoint_dir"], "best_model.pt"))
    )
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, is_gnn=is_gnn,
    )
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
