import os
import sys
import argparse
import yaml
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

sys.path.insert(0, os.path.dirname(__file__))

from data.upfd.loader import UPFDGraphDataset, get_upfd_loaders
from models.baseline_lstm import BaselineLSTM
from models.degradation_gnn import DegradationGNN
from models.early_detector import EarlyDetector


METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc_roc": lambda y, p: roc_auc_score(y, p),
    "auc_pr": lambda y, p: average_precision_score(y, p),
}


def compute_metrics(y_true, y_pred, y_prob):
    results = {}
    for name, fn in METRICS.items():
        try:
            if name in ["auc_roc", "auc_pr"]:
                results[name] = fn(y_true, y_prob)
            else:
                results[name] = fn(y_true, y_pred)
        except Exception:
            results[name] = 0.0
    return results


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= max(len(y_true), 1)
    return float(ece)


def evaluate_baselines(train_loader, test_loader):
    """Evaluate simple baselines, training on train set and evaluating on test set."""
    print("\n--- Baselines ---")

    train_features, train_labels = [], []
    for batch in train_loader:
        train_labels.extend(batch.y.numpy())
        for data in batch.to_data_list():
            feat = data.x.mean(dim=0).numpy()
            train_features.append(feat)

    test_features, test_labels = [], []
    for batch in test_loader:
        test_labels.extend(batch.y.numpy())
        for data in batch.to_data_list():
            feat = data.x.mean(dim=0).numpy()
            test_features.append(feat)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("SVM (RBF)", SVC(probability=True)),
    ]:
        clf.fit(train_features, train_labels)
        preds = clf.predict(test_features)
        probs = clf.predict_proba(test_features)[:, 1]
        metrics = compute_metrics(test_labels, preds, probs)
        print(f"  {name}: {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="baseline_lstm")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)

    train_loader, _, test_loader = get_upfd_loaders(
        root=config["data"]["root"],
        subset=config["data"]["subset"],
        feature_type=config["data"]["feature_type"],
        batch_size=config["data"]["batch_size"],
    )

    sample = next(iter(test_loader)).to(device)
    input_dim = sample.x.shape[1]
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

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Full cascade evaluation
    all_preds, all_probs, all_labels = [], [], []
    for batch in test_loader:
        batch = batch.to(device)
        labels = batch.y.float()
        with torch.no_grad():
            logits_list = []
            for data in batch.to_data_list():
                if is_gnn:
                    logits, _ = model(data, return_drift=True)
                else:
                    logits = model(data)
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)
        probs = torch.sigmoid(logits).flatten()
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    print("\n--- Full Cascade Evaluation ---")
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    ece = compute_ece(all_labels, all_probs)
    print(f"  ece: {ece:.4f}")

    # Early detection evaluation
    print("\n--- Early Detection ---")
    all_graphs = []
    all_labels_list = []
    for batch in test_loader:
        for data in batch.to_data_list():
            all_graphs.append(data)
            all_labels_list.append(data.y.item())

    detector = EarlyDetector(model)
    truncations = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = detector.evaluate(all_graphs, all_labels_list, truncations=truncations)

    for frac, m in results.items():
        print(f"  Truncation {frac:.0%}: F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")

    evaluate_baselines(train_loader, test_loader)


def evaluate_drift(model, test_loader, device):
    """Evaluate drift prediction quality for GNN models."""
    if not hasattr(model, "drift_head"):
        return None

    import torch.nn as nn
    from features.drift import compute_drift

    model.eval()
    drift_mse = nn.MSELoss()
    all_mse = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            for data in batch.to_data_list():
                logits, drift_pred = model(data, return_drift=True)
                parent = getattr(data, "parent", None)
                n_children = getattr(data, "n_children", None)
                if parent is None:
                    parent = torch.full((data.x.size(0),), -1, dtype=torch.long, device=data.x.device)
                if n_children is None:
                    n_children = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
                target_drift = compute_drift(
                    data.x, parent, data.hop_depth, n_children
                )
                mse = drift_mse(drift_pred, target_drift).item()
                all_mse.append(mse)

    return float(np.mean(all_mse))


if __name__ == "__main__":
    main()
