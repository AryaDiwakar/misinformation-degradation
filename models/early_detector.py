import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from torch_geometric.data import Data


class EarlyDetector:
    """
    Wraps a trained model to measure performance at partial cascade sizes.

    Usage:
        detector = EarlyDetector(model)
        curve = detector.evaluate(test_data, test_labels, truncations=[0.1, 0.25, 0.5, 0.75, 1.0])
    """
    def __init__(self, model: nn.Module, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    @torch.no_grad()
    def evaluate(
        self,
        graphs: List[Data],
        labels: List[int],
        truncations: List[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    ) -> Dict[float, Dict[str, float]]:
        device = next(self.model.parameters()).device
        results = {}
        for frac in truncations:
            probs = []
            for graph in graphs:
                truncated = self._truncate_cascade(graph.clone(), frac)
                logits = self.model(truncated.to(device))
                prob = torch.sigmoid(logits).item()
                probs.append(prob)

            probs = np.array(probs)
            preds = (probs > self.threshold).astype(float)
            labels_arr = np.array(labels)
            acc = (preds == labels_arr).mean()
            tp = (preds[labels_arr == 1] == 1).sum()
            fp = (preds[labels_arr == 0] == 1).sum()
            fn = (preds[labels_arr == 1] == 0).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            results[frac] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            if len(np.unique(labels_arr)) > 1:
                try:
                    results[frac]["auc_roc"] = float(
                        __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels_arr, probs)
                    )
                except Exception:
                    pass

        return results

    def _truncate_cascade(self, graph: Data, fraction: float) -> Data:
        order = graph.hop_depth.argsort()
        n = graph.x.size(0)
        keep = max(1, int(n * fraction))
        keep_nodes = order[:keep]
        node_map = {old.item(): i for i, old in enumerate(keep_nodes)}
        mask = torch.zeros(n, dtype=torch.bool)
        mask[keep_nodes] = True
        edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        new_edge_index = graph.edge_index[:, edge_mask].clone()
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = node_map[new_edge_index[0, i].item()]
            new_edge_index[1, i] = node_map[new_edge_index[1, i].item()]
        return Data(
            x=graph.x[keep_nodes],
            edge_index=new_edge_index,
            y=graph.y,
            hop_depth=graph.hop_depth[keep_nodes],
        )
