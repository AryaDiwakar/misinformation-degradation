"""Create a tiny synthetic cascade for local inference and demo smoke tests."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/sample_cascade.npz")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=768)
    args = parser.parse_args()
    if args.nodes < 2:
        raise SystemExit("--nodes must be at least 2")

    rng = np.random.default_rng(7)
    x = rng.normal(size=(args.nodes, args.feature_dim)).astype("float32")
    edges = [(0, 0)]
    for node in range(1, args.nodes):
        parent = max(0, (node - 1) // 2)
        edges.extend([(node, node), (parent, node)])
    edge_index = np.asarray(edges, dtype="int64").T
    hop_depth = np.zeros(args.nodes, dtype="int64")
    for node in range(1, args.nodes):
        hop_depth[node] = hop_depth[max(0, (node - 1) // 2)] + 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, x=x, edge_index=edge_index, hop_depth=hop_depth)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
