"""Streamlit demo: the drift finding + single-cascade exploration.

Tab 1 (The Finding) reads a precomputed cache from ``scripts/build_demo_cache.py``
and visualises the headline result: fake news drifts less than real news.

Tab 2 (Cascade Explorer) draws individual cascades coloured by per-node drift.

Tab 3 (Live Predictor) is optional: if a canonical HybridDegradLSTM checkpoint is
present, it classifies an uploaded cascade and explains the drift.

Build the cache first:

    python3 scripts/build_demo_cache.py
    streamlit run demo/drift_explorer.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.drift import compute_drift  # noqa: E402
from pipeline.inference import cascade_from_arrays, predict_cascade  # noqa: E402

ASSETS = ROOT / "demo" / "assets"
REAL = "#1f77b4"
FAKE = "#d62728"

st.set_page_config(page_title="Misinformation Drift Explorer", page_icon="〰️", layout="wide")
st.title("Misinformation Drift Explorer")
st.caption(
    "Research prototype — feature drift along misinformation cascades (UPFD GossipCop)."
)


@st.cache_data(show_spinner=False)
def load_summary():
    path = ASSETS / "drift_summary.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@st.cache_data(show_spinner=False)
def load_examples():
    index_path = ASSETS / "examples.json"
    if not index_path.is_file():
        return None
    with index_path.open() as handle:
        index = json.load(handle)
    for entry in index["examples"]:
        with np.load(ASSETS / "examples" / entry["file"], allow_pickle=False) as data:
            entry["arrays"] = {key: data[key] for key in data.files}
    return index


def drift_of(graph_arrays):
    x = torch.as_tensor(graph_arrays["x"], dtype=torch.float32)
    parent = torch.as_tensor(graph_arrays["parent"], dtype=torch.long)
    hop = torch.as_tensor(graph_arrays["hop_depth"], dtype=torch.long)
    children = torch.as_tensor(graph_arrays["n_children"], dtype=torch.long)
    return compute_drift(x, parent, hop, children).numpy()


def draw_tree(ax, graph_arrays, drift, title):
    hop = graph_arrays["hop_depth"]
    mag = np.linalg.norm(drift[:, :2], axis=1)
    edge_index = graph_arrays["edge_index"]

    levels = {}
    for node, depth in enumerate(hop.tolist()):
        levels.setdefault(int(depth), []).append(node)
    pos = {}
    for depth, nodes in levels.items():
        nodes = sorted(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (i - (len(nodes) - 1) / 2, -depth)

    seen = set()
    for src, dst in edge_index.T:
        src, dst = int(src), int(dst)
        if src == dst or (src, dst) in seen or (dst, src) in seen:
            continue
        if src in pos and dst in pos:
            seen.add((src, dst))
            ax.plot(
                [pos[src][0], pos[dst][0]],
                [pos[src][1], pos[dst][1]],
                color="#c8c8c8",
                lw=0.8,
                zorder=1,
            )

    scatter = ax.scatter(
        [pos[v][0] for v in pos],
        [pos[v][1] for v in pos],
        c=mag[list(pos)],
        cmap="viridis",
        s=42,
        zorder=2,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_title(title)
    ax.axis("off")
    return scatter


def page_finding(summary):
    st.header("The finding")
    st.markdown(
        "Across **{} test cascades**, fake-news nodes show less content drift from "
        "their parent than real-news nodes. The effect emerges at propagation depth "
        "≥ 2 and holds on a second dataset (PolitiFact).".format(
            f"{summary['real_nodes'][0] + summary['fake_nodes'][0]:,}"
        )
    )

    real = summary["real_mag"]
    fake = summary["fake_mag"]
    real_mean = float(real.mean())
    fake_mean = float(fake.mean())
    ratio = fake_mean / real_mean

    col1, col2, col3 = st.columns(3)
    col1.metric("Real mean drift", f"{real_mean:.2f}")
    col2.metric("Fake mean drift", f"{fake_mean:.2f}")
    col3.metric("Fake / real", f"{ratio:.3f}", delta=f"{(ratio - 1) * 100:+.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 18, 60)
    axes[0].hist(real, bins=bins, alpha=0.55, label=f"Real (n={summary['real_nodes'][0]:,})", color=REAL)
    axes[0].hist(fake, bins=bins, alpha=0.55, label=f"Fake (n={summary['fake_nodes'][0]:,})", color=FAKE)
    axes[0].axvline(real_mean, color=REAL, ls="--", lw=1.5)
    axes[0].axvline(fake_mean, color=FAKE, ls="--", lw=1.5)
    axes[0].set_xlim(0, 12)
    axes[0].set_xlabel("Per-node drift magnitude")
    axes[0].set_ylabel("Node count")
    axes[0].set_title("Drift distribution")
    axes[0].legend()

    hops = summary["hops"].tolist()[:5]
    r = summary["hop_real_mean"][: len(hops)]
    f = summary["hop_fake_mean"][: len(hops)]
    x = np.arange(len(hops))
    axes[1].bar(x - 0.2, r, 0.4, label="Real", color=REAL)
    axes[1].bar(x + 0.2, f, 0.4, label="Fake", color=FAKE)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"hop {h}" for h in hops])
    axes[1].set_ylabel("Mean drift magnitude")
    axes[1].set_title("Drift by propagation depth")
    axes[1].legend()
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.info(
        "A drift-only random forest (12 aggregate statistics, no text and no graph "
        "learning) reaches ~0.87 AUC on this dataset and still ~0.79 AUC when only "
        "the first 10% of each cascade is visible."
    )


def page_cascades(examples):
    st.header("Cascade explorer")
    labels = {
        0: "Real news",
        1: "Fake news",
    }
    names = [
        f"#{e['graph_index']} — {labels[e['label']]} ({e['n_nodes']} nodes, depth {e['max_depth']})"
        for e in examples["examples"]
    ]
    choice = st.selectbox("Pick a cascade", range(len(names)), format_func=lambda i: names[i])
    entry = examples["examples"][choice]
    drift = drift_of(entry["arrays"])
    magnitude = np.linalg.norm(drift[:, :2], axis=1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Label", labels[entry["label"]])
    col2.metric("Mean drift", f"{magnitude.mean():.2f}")
    col3.metric("Max drift", f"{magnitude.max():.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = draw_tree(ax, entry["arrays"], drift, f"{labels[entry['label']]} cascade #{entry['graph_index']}")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("drift magnitude (L2)")
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    st.caption("Node colour is the per-node drift from its parent; root drift is zero by definition.")


def resolve_checkpoint():
    """Return (path, model_name, description) for the best available checkpoint."""
    canonical = ROOT / "checkpoints" / "hybrid_gossipcop_seed1.pt"
    if canonical.is_file():
        return canonical, "hybrid", "canonical HybridDegradLSTM checkpoint"
    legacy = ROOT / "checkpoints" / "best_model.pt"
    if legacy.is_file():
        return (
            legacy,
            "baseline_lstm",
            "legacy BaselineLSTM checkpoint (fallback — the canonical hybrid is still training)",
        )
    return None, None, None


def page_predictor():
    st.header("Live predictor")
    checkpoint, model_name, description = resolve_checkpoint()
    if checkpoint is None:
        st.warning(
            "No checkpoint available. Train one with:\n\n"
            "```\npython3 train.py --config configs/default.yaml --model hybrid --seed 1\n```"
        )
        return

    st.caption(f"Using `{checkpoint.relative_to(ROOT)}` — {description}.")

    examples = load_examples()
    source = st.radio("Input", ["Example cascade", "Upload .npz"])
    arrays = None
    if source == "Example cascade" and examples:
        entry = examples["examples"][0]
        arrays = entry["arrays"]
        st.caption(f"Using example #{entry['graph_index']} (true label: {entry['label']})")
    elif source == "Upload .npz":
        uploaded = st.file_uploader("Cascade (.npz with x and edge_index)", type=["npz"])
        if uploaded is not None:
            import io

            with np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}

    if arrays is None:
        return

    from pipeline.checkpoints import load_checkpoint

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    graph = cascade_from_arrays(arrays)
    try:
        model, metadata = load_checkpoint(str(checkpoint), device, model_name=model_name)
    except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
        st.error(f"Could not load checkpoint: {exc}")
        return
    result = predict_cascade(model, graph, device)

    col1, col2 = st.columns(2)
    col1.metric("Prediction", result["label"].upper())
    col2.metric("Fake probability", f"{result['probability_fake']:.1%}")
    st.json(
        {
            key: result[key]
            for key in ("cascade_size", "feature_dim", "max_depth", "mean_content_drift")
        }
    )
    import pandas as pd

    by_hop = result["content_drift_by_hop"]
    chart = pd.DataFrame(
        {"hop": [int(h) for h in by_hop], "mean_drift": list(by_hop.values())}
    ).set_index("hop")
    st.bar_chart(chart)


def main():
    summary = load_summary()
    examples = load_examples()

    if summary is None:
        st.error(
            "No demo cache found. Build it first:\n\n"
            "```\npython3 scripts/build_demo_cache.py\n```"
        )
        st.stop()

    tab_finding, tab_cascades, tab_predict = st.tabs(
        ["The Finding", "Cascade Explorer", "Live Predictor"]
    )
    with tab_finding:
        page_finding(summary)
    with tab_cascades:
        if examples is None:
            st.warning("No example cascades in the cache.")
        else:
            page_cascades(examples)
    with tab_predict:
        page_predictor()


if __name__ == "__main__":
    main()
