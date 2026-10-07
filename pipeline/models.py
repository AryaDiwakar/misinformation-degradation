"""Model registry used by training, evaluation, and inference."""

from typing import Any, Dict

from models.baseline_lstm import BaselineLSTM
from models.baselines import BiGCNModel, GATModel, GCNModel
from models.degradation_gnn import DegradationGNN
from models.drift_pool import DriftPoolHybrid, DriftPoolOnly
from models.hybrid import HybridDegradLSTM


MODEL_ALIASES = {
    "baseline_lstm": "baseline_lstm",
    "lstm": "baseline_lstm",
    "degradation_gnn": "degradation_gnn",
    "degradgnn": "degradation_gnn",
    "hybrid": "hybrid",
    "hybrid_degrad_lstm": "hybrid",
    "drift_pool": "drift_pool",
    "driftpool": "drift_pool",
    "drift_pool_only": "drift_pool_only",
    "driftpoolonly": "drift_pool_only",
    "gcn": "gcn",
    "gat": "gat",
    "bigcn": "bigcn",
}

MODEL_CLASSES = {
    "baseline_lstm": BaselineLSTM,
    "degradation_gnn": DegradationGNN,
    "hybrid": HybridDegradLSTM,
    "drift_pool": DriftPoolHybrid,
    "drift_pool_only": DriftPoolOnly,
    "gcn": GCNModel,
    "gat": GATModel,
    "bigcn": BiGCNModel,
}


def canonical_model_name(name: str) -> str:
    """Return the stable registry name for a model alias."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CLASSES))
        raise ValueError(f"Unknown model '{name}'. Choose from: {available}") from exc


def build_model(name: str, input_dim: int, config: Dict[str, Any] | None = None):
    """Build a registered model from the model section of a config."""
    model_name = canonical_model_name(name)
    config = config or {}
    kwargs = {
        "input_dim": input_dim,
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "num_layers": int(config.get("num_layers", 2)),
        "dropout": float(config.get("dropout", 0.3)),
    }

    if model_name == "baseline_lstm":
        kwargs["bidirectional"] = bool(config.get("bidirectional", True))
    elif model_name == "drift_pool":
        kwargs["pool_dim"] = int(config.get("pool_dim", 64))
        kwargs["use_pool"] = bool(config.get("use_pool", True))
        kwargs["use_tree"] = bool(config.get("use_tree", True))
    elif model_name == "drift_pool_only":
        kwargs.pop("num_layers")
    elif model_name in {"gcn", "gat", "bigcn"}:
        kwargs.pop("bidirectional", None)

    return MODEL_CLASSES[model_name](**kwargs)


def uses_drift_auxiliary_loss(name: str) -> bool:
    """Whether a model exposes a learned per-node drift prediction head."""
    return canonical_model_name(name) in {"degradation_gnn", "hybrid"}
