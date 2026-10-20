"""
Train the TrajectoryPredictor on top of a trained DegradationGNN.

Extracts per-graph drift sequences from the GNN's drift head, then
trains the Transformer predictor to forecast future drift.
"""
import os, sys, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, os.path.dirname(__file__))

from data.upfd.loader import UPFDGraphDataset, collate_fn
from models.degradation_gnn import DegradationGNN
from models.trajectory_head import TrajectoryPredictor, predict_degradation_curve
from pipeline.checkpoints import load_checkpoint

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


def extract_drift_sequences(model, loader, device):
    """Extract per-graph drift sequences from GNN drift head.

    Returns list of (drift_seq, label) where drift_seq is [seq_len, drift_dim].
    """
    model.eval()
    sequences = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            for data in batch.to_data_list():
                _, drift_pred = model(data, return_drift=True)
                parent = getattr(data, "parent", None)
                if parent is None:
                    parent = torch.full((data.x.size(0),), -1, dtype=torch.long, device=device)
                hop_order = data.hop_depth.argsort()
                sorted_drift = drift_pred[hop_order]
                sorted_parent = parent[hop_order]
                root_mask = sorted_parent == -1
                non_root = ~root_mask
                if non_root.sum() < 2:
                    continue
                drift_seq = sorted_drift[non_root]
                sequences.append(drift_seq.cpu())
                labels_list.append(data.y.item())

    return sequences, labels_list


class DriftSequenceDataset(Dataset):
    """Dataset of drift sequences for trajectory prediction.

    Each item: (observed_drift, target_drift, observed_len)
    where observed_drift is the first k hops (to predict from),
    target_drift is the next hop (to predict).
    """
    def __init__(self, sequences, seq_len=5):
        self.samples = []
        for seq in sequences:
            if seq.size(0) < seq_len + 1:
                continue
            for i in range(seq.size(0) - seq_len):
                obs = seq[i:i + seq_len]
                tgt = seq[i + seq_len]
                self.samples.append((obs, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, tgt = self.samples[idx]
        return obs.unsqueeze(0), tgt.unsqueeze(0)


def collate_drift(batch):
    obs_list = [b[0] for b in batch]
    tgt_list = [b[1] for b in batch]
    obs_batch = torch.stack(obs_list)
    tgt_batch = torch.stack(tgt_list)
    return obs_batch, tgt_batch


def train_trajectory_predictor(
    gnn_checkpoint: str,
    subset: str = "politifact",
    epochs: int = 50,
    seq_len: int = 5,
    hidden_dim: int = 64,
):
    """Load a trained GNN, extract drift sequences, train trajectory predictor."""
    print(f"Loading GNN from {gnn_checkpoint}")
    # Accepts the structured checkpoints written by the canonical pipeline as
    # well as a legacy raw state dict.
    gnn, _metadata = load_checkpoint(
        gnn_checkpoint, DEVICE, model_name="degradation_gnn"
    )
    gnn.eval()

    ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="train")
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    sequences, labels = extract_drift_sequences(gnn, loader, DEVICE)
    print(f"Extracted {len(sequences)} drift sequences, {sum(1 for l in labels if l==1)} fake")

    drift_dim = sequences[0].size(-1) if sequences else 5
    train_ds = DriftSequenceDataset(sequences, seq_len=seq_len)
    print(f"Training samples: {len(train_ds)}")

    if len(train_ds) == 0:
        print("Not enough samples. Try shorter seq_len.")
        return

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_drift
    )

    model = TrajectoryPredictor(
        drift_dim=drift_dim,
        d_model=hidden_dim,
        nhead=4,
        num_layers=2,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"\nTraining TrajectoryPredictor for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for obs_batch, tgt_batch in train_loader:
            obs_batch = obs_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)
            optimizer.zero_grad()

            output = model(obs_batch)
            loss = criterion(output[:, -1:, :], tgt_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss / len(train_loader):.6f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/trajectory_predictor_{subset}.pt"
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")

    # Quick test: predict 3 steps ahead for a sample
    print("\nSample trajectory prediction:")
    model.eval()
    with torch.no_grad():
        sample_obs = sequences[0][:seq_len].unsqueeze(0).to(DEVICE)
        curve = predict_degradation_curve(model, sample_obs, num_future_steps=3)
        print(f"  Observed ({seq_len} hops): {sample_obs.cpu().numpy()[0, :, :2]}")
        print(f"  Predicted next 3 hops (cosine_dist, l2_dist): {curve.cpu().numpy()[0, :, :2]}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained DegradationGNN checkpoint")
    parser.add_argument("--subset", type=str, default="politifact")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=5)
    args = parser.parse_args()

    train_trajectory_predictor(
        gnn_checkpoint=args.checkpoint,
        subset=args.subset,
        epochs=args.epochs,
        seq_len=args.seq_len,
    )
