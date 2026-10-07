"""Predict fake/real for one UPFD-format cascade."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.checkpoints import load_checkpoint
from pipeline.config import load_config
from pipeline.inference import load_cascade_npz, predict_cascade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npz cascade input")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Required for legacy checkpoints")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, metadata = load_checkpoint(
        args.checkpoint,
        device,
        model_name=args.model,
        model_config=config.get("model"),
    )
    graph = load_cascade_npz(args.input)
    expected_dim = int(metadata.get("input_dim", graph.x.size(1)))
    if graph.x.size(1) != expected_dim:
        raise SystemExit(
            f"Input feature dimension {graph.x.size(1)} does not match checkpoint {expected_dim}"
        )
    output = predict_cascade(model, graph, device)
    output.update({
        "model": metadata.get("model_name", args.model),
        "checkpoint": args.checkpoint,
        "device": str(device),
    })
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
