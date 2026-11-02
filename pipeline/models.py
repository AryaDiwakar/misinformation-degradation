"""Model registry used by training, evaluation, and inference."""

import re
from typing import Any, Dict

from models.baseline_lstm import BaselineLSTM
from models.baselines import BiGCNModel, GATModel, GCNModel
from models.degradation_gnn import DegradationGNN
from models.degradation_trajectory import DegradationTrajectoryNetwork
from models.drift_pool import DriftPoolHybrid, DriftPoolOnly
from models.hybrid import HybridDegradLSTM
from models.cross_modal_tree import CrossModalTreeNet


MODEL_ALIASES = {
    "baseline_lstm": "baseline_lstm",
    "baselinelstm": "baseline_lstm",
    "lstm": "baseline_lstm",
    "degradation_gnn": "degradation_gnn",
    "degradgnn": "degradation_gnn",
    "degrad_gnn": "degradation_gnn",
    "hybrid": "hybrid",
    "hybrid_degrad_lstm": "hybrid",
    "hybriddegradlstm": "hybrid",
    "drift_pool": "drift_pool",
    "driftpool": "drift_pool",
    "drift_pool_only": "drift_pool_only",
    "driftpoolonly": "drift_pool_only",
    "cross_modal_tree": "cross_modal_tree",
    "crossmodaltree": "cross_modal_tree",
    "cross_modal_tree_net": "cross_modal_tree",
    "gcn": "gcn",
    "gat": "gat",
    "bigcn": "bigcn",
    "bi_gcn": "bigcn",
    "dtn": "dtn",
    "degradation_trajectory": "dtn",
}

MODEL_CLASSES = {
    "baseline_lstm": BaselineLSTM,
    "degradation_gnn": DegradationGNN,
    "hybrid": HybridDegradLSTM,
    "drift_pool": DriftPoolHybrid,
    "drift_pool_only": DriftPoolOnly,
    "cross_modal_tree": CrossModalTreeNet,
    "gcn": GCNModel,
    "gat": GATModel,
    "bigcn": BiGCNModel,
    "dtn": DegradationTrajectoryNetwork,
}


def canonical_model_name(name: str) -> str:
    """Return the stable registry name for a model alias.

    Handles snake_case, kebab-case, spaces and CamelCase (``BaselineLSTM``,
    ``HybridDegradLSTM``) so that every registered spelling resolves.
    """
    raw = name.strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    # CamelCase -> snake_case, keeping acronyms together:
    # "BaselineLSTM" -> "baseline_lstm", "DegradGNN" -> "degrad_gnn".
    camel = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", raw
    ).lower().replace("-", "_").replace(" ", "_")
    if camel in MODEL_ALIASES:
        return MODEL_ALIASES[camel]
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
        # DriftPoolOnly has no message-passing layers.
        kwargs.pop("num_layers")
    elif model_name == "dtn":
        kwargs.pop("num_layers")
    elif model_name == "cross_modal_tree":
        kwargs["profile_dim"] = int(config.get("profile_dim", 10))
        kwargs["hop_dim"] = int(config.get("hop_dim", 16))
        kwargs["modality_dropout"] = float(config.get("modality_dropout", 0.1))
    elif model_name in {"gcn", "gat", "bigcn"}:
        kwargs.pop("bidirectional", None)

    return MODEL_CLASSES[model_name](**kwargs)


def uses_drift_auxiliary_loss(name: str) -> bool:
    """Whether a model exposes a learned per-node drift prediction head."""
    return canonical_model_name(name) in {"degradation_gnn", "hybrid"}
