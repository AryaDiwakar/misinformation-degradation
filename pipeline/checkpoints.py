"""Portable checkpoint helpers with architecture and experiment metadata."""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .models import build_model, canonical_model_name


def _torch_load(path: str | Path, device: torch.device):
    """Load checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    model_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    *,
    epoch: int,
    best_val_loss: float,
    seed: int,
) -> None:
    """Save weights together with everything needed to reconstruct the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_name": canonical_model_name(model_name),
        "model_config": dict(model_config),
        "experiment_config": experiment_config,
        "input_dim": int(model_config["input_dim"]),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "seed": int(seed),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    *,
    model_name: str | None = None,
    model_config: Dict[str, Any] | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a structured checkpoint, or an explicitly identified legacy state dict."""
    payload = _torch_load(path, device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = payload
        resolved_name = payload.get("model_name") or model_name
        resolved_config = dict(payload.get("model_config") or model_config or {})
        input_dim = int(payload.get("input_dim", resolved_config.get("input_dim", 768)))
    else:
        if not model_name:
            raise ValueError(
                "Legacy state-dict checkpoints require --model and model configuration."
            )
        state_dict = payload
        metadata = {"format_version": 0, "model_name": model_name}
        resolved_name = model_name
        resolved_config = dict(model_config or {})
        input_dim = int(resolved_config.pop("input_dim", 768))

    if not resolved_name:
        raise ValueError(f"Checkpoint does not specify a model name: {path}")

    resolved_config["input_dim"] = input_dim
    model = build_model(resolved_name, input_dim, resolved_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata
