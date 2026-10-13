"""Precompute a compact cache for the Streamlit drift-explorer demo.

Loading the full UPFD split takes tens of seconds and ~1 GB of RAM, which is a
bad experience for a live demo. This script extracts only what the explorer
needs and writes it to ``demo/assets/``:

  - drift_summary.npz : real/fake drift-magnitude samples, per-hop means, counts
  - examples/*.npz    : a few individual cascades (real + fake) for inspection

Re-run this whenever the dataset or the drift definition changes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift

MAX_SAMPLES = 20000  # nodes per class kept for the histogram


def drift_magnitude(graph):
    drift = compute_drift(graph.x, graph.parent, graph.hop_depth, graph.n_children)
    return torch.linalg.vector_norm(drift[:, :2], dim=1).numpy()


def collect_examples(dataset, rng, min_nodes=40, max_nodes=90, per_class=3):
    """Pick a handful of mid-sized cascades, balanced across classes."""
    by_class = {0: [], 1: []}
    for idx in range(len(dataset)):
        graph = dataset[idx]
        size = graph.x.size(0)
        if min_nodes <= size <= max_nodes:
            by_class[int(graph.y.item())].append(idx)
    examples = []
    for label in (0, 1):
        candidates = by_class[label]
        rng.shuffle(candidates)
        for idx in candidates[:per_class]:
            graph = dataset[idx]
            examples.append((label, idx, graph))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="gossipcop")
    parser.add_argument("--root", default="data/upfd/raw")
    parser.add_argument("--feature-type", default="bert")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="demo/assets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.subset}/{args.split} ({args.feature_type}) ...", flush=True)
    dataset = UPFDGraphDataset(
        root=args.root,
        subset=args.subset,
        feature_type=args.feature_type,
        split=args.split,
    )
    print(f"  {len(dataset)} graphs", flush=True)

    rng = np.random.default_rng(args.seed)
    real_mag, fake_mag = [], []
    real_hop, fake_hop = {}, {}

    for idx in range(len(dataset)):
        graph = dataset[idx]
        mag = drift_magnitude(graph)
        hops = graph.hop_depth.numpy()
        if int(graph.y.item()) == 1:
            fake_mag.append(mag)
            target = fake_hop
        else:
            real_mag.append(mag)
            target = real_hop
        for hop, value in zip(hops.tolist(), mag.tolist()):
            target.setdefault(int(hop), []).append(value)
        if (idx + 1) % 500 == 0:
            print(f"  processed {idx + 1}/{len(dataset)}", flush=True)

    real_mag = np.concatenate(real_mag)
    fake_mag = np.concatenate(fake_mag)

    def sample(values):
        if values.size <= MAX_SAMPLES:
            return values
        choice = rng.choice(values.size, MAX_SAMPLES, replace=False)
        return values[np.sort(choice)]

    hops = sorted(set(real_hop) | set(fake_hop))
    hop_real_mean = np.array([np.mean(real_hop.get(h, [np.nan])) for h in hops])
    hop_fake_mean = np.array([np.mean(fake_hop.get(h, [np.nan])) for h in hops])
    hop_real_n = np.array([len(real_hop.get(h, [])) for h in hops])
    hop_fake_n = np.array([len(fake_hop.get(h, [])) for h in hops])

    summary_path = out_dir / "drift_summary.npz"
    np.savez_compressed(
        summary_path,
        subset=args.subset,
        real_mag=sample(real_mag),
        fake_mag=sample(fake_mag),
        real_mag_full_mean=np.array([real_mag.mean()]),
        fake_mag_full_mean=np.array([fake_mag.mean()]),
        real_nodes=np.array([real_mag.size]),
        fake_nodes=np.array([fake_mag.size]),
        hops=np.array(hops),
        hop_real_mean=hop_real_mean,
        hop_fake_mean=hop_fake_mean,
        hop_real_n=hop_real_n,
        hop_fake_n=hop_fake_n,
    )
    ratio = float(fake_mag.mean() / real_mag.mean())
    print(
        f"Saved {summary_path} | real {real_mag.size:,} nodes "
        f"fake {fake_mag.size:,} nodes | ratio (fake/real) = {ratio:.4f}",
        flush=True,
    )

    print("Collecting example cascades ...", flush=True)
    examples = collect_examples(dataset, rng)
    index = []
    for label, idx, graph in examples:
        path = out_dir / "examples" / f"graph_{idx:05d}.npz"
        np.savez_compressed(
            path,
            x=graph.x.numpy().astype("float32"),
            edge_index=graph.edge_index.numpy().astype("int64"),
            hop_depth=graph.hop_depth.numpy().astype("int64"),
            parent=graph.parent.numpy().astype("int64"),
            n_children=graph.n_children.numpy().astype("int64"),
            y=np.array([label], dtype="int64"),
        )
        index.append(
            {
                "file": path.name,
                "graph_index": int(idx),
                "label": int(label),
                "n_nodes": int(graph.x.size(0)),
                "max_depth": int(graph.hop_depth.max().item()),
                "drift_magnitude": float(drift_magnitude(graph).mean()),
            }
        )
        print(f"  wrote {path.name} (label={label}, nodes={graph.x.size(0)})", flush=True)

    with (out_dir / "examples.json").open("w") as handle:
        json.dump({"subset": args.subset, "examples": index}, handle, indent=2)
    print(f"Saved {out_dir / 'examples.json'}", flush=True)


if __name__ == "__main__":
    main()
