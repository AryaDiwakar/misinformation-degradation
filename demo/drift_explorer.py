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


def show_figure(fig):
    """Render a matplotlib figure, using full width where Streamlit supports it.

    ``st.pyplot(width="stretch")`` requires Streamlit >= 1.49; older versions
    only accept a fixed ``use_container_width`` flag.
    """
    try:
        st.pyplot(fig, width="stretch")
    except TypeError:
        st.pyplot(fig, use_container_width=True)

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
    n_cascades = int(summary.get("n_cascades", [0])[0])
    n_nodes = int(
        summary.get(
            "real_nodes_total",
            summary["real_nodes"] + summary["fake_nodes"],
        )[0]
        + summary.get("fake_nodes_total", [0])[0]
    )
    st.markdown(
        "Across **{:,} test cascades** ({:,} nodes), fake-news nodes show slightly "
        "less content drift from their parent than real-news nodes. The effect is "
        "modest after accounting for propagation depth. The corresponding "
        "cascade-level comparison on PolitiFact points in the same direction but "
        "is not statistically significant (ratio 0.969, $p=0.522).".format(n_cascades, n_nodes)
    )

    real = summary["real_mag"]
    fake = summary["fake_mag"]
    # Use the full-split means for the headline numbers. The histogram arrays
    # are subsampled to keep the cache small, so their means carry a small
    # sampling error; the stored full means are exact.
    real_mean = float(summary.get("real_mag_full_mean", [real.mean()])[0])
    fake_mean = float(summary.get("fake_mag_full_mean", [fake.mean()])[0])
    ratio = fake_mean / real_mean
    standardized = float(summary.get("hop_standardized_ratio", [float("nan")])[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Real mean drift", f"{real_mean:.2f}")
    col2.metric("Fake mean drift", f"{fake_mean:.2f}")
    col3.metric("Fake / real", f"{ratio:.3f}", delta=f"{(ratio - 1) * 100:+.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 18, 60)
    axes[0].hist(real, bins=bins, alpha=0.55, label=f"Real non-root (n={summary['real_nodes'][0]:,})", color=REAL)
    axes[0].hist(fake, bins=bins, alpha=0.55, label=f"Fake non-root (n={summary['fake_nodes'][0]:,})", color=FAKE)
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
    show_figure(fig)
    plt.close(fig)

    if standardized == standardized:  # not NaN
        st.caption(
            f"Pooled ratio (fake/real) = {ratio:.3f}. After standardising over "
            f"propagation depth — which removes the difference in how deep the "
            f"two classes' cascades go — the ratio is {standardized:.3f}. The "
            "pooled figure mixes the content effect with a structural one."
        )

    st.info(
        "A random forest over 12 aggregate cascade statistics reaches ~0.88 AUC "
        "on this dataset. Note that only 4 of those 12 statistics are content "
        "drift; 6 are structural (depth, children) and 2 are pure cascade size. "
        "Content drift on its own reaches ~0.85 AUC, and cascade size alone "
        "~0.79 AUC."
    )

    st.caption(
        "The feature-group AUCs use exploratory cross-validation inside the "
        "official GossipCop test partition; they are not independent held-out results."
    )


def page_dataset(summary, examples):
    st.header("Dataset and experiment setup")
    st.markdown(
        "This project uses **UPFD** (User Preference-aware Fake News Detection), "
        "a benchmark of fact-checking-labelled propagation cascades. Each row "
        "in the experiment is one complete cascade; posts are nodes and "
        "repost/reply links are edges. The demo uses cached test-split graph "
        "examples, not live social-media data."
    )

    st.subheader("Official cascade split sizes")
    import pandas as pd

    splits = pd.DataFrame(
        {
            "Dataset": ["GossipCop", "PolitiFact"],
            "Train cascades": [1092, 62],
            "Validation cascades": [546, 31],
            "Test cascades": [3826, 221],
            "Use in this study": [
                "Primary five-seed model comparison + analyses",
                "Drift comparison; small split, no stable model ranking",
            ],
        }
    )
    st.dataframe(splits, hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    left.metric("GossipCop test cascades", f"{int(summary.get('n_cascades', [3826])[0]):,}")
    total_nodes = int(
        summary.get("real_nodes_total", summary["real_nodes"])[0]
        + summary.get("fake_nodes_total", summary["fake_nodes"])[0]
    )
    left.metric("GossipCop test nodes", f"{total_nodes:,}")
    right.metric("Real / fake test cascades", f"{int(summary['hop_real_n'][0]):,} / {int(summary['hop_fake_n'][0]):,}")
    right.metric("Frozen text-vector size", "768 dimensions per post")

    st.subheader("What is inside one cascade?")
    schema = pd.DataFrame(
        [
            ("Label", "One label per cascade: 0 = real, 1 = fake in this codebase."),
            ("Node", "A post represented by a precomputed, frozen 768-dimensional BERT vector."),
            ("Edge", "A parent-to-child repost/reply relation from the UPFD propagation tree."),
            ("Depth", "Number of edges from the root; this is topological depth, not elapsed time."),
            ("Content drift", "Cosine distance and L2 distance between a post vector and its parent's vector."),
            ("Root", "Has no parent, so both parent-relative distances are set to zero by definition."),
        ],
        columns=["Field", "Meaning"],
    )
    st.dataframe(schema, hide_index=True, use_container_width=True)

    st.subheader("What the stored UPFD example actually looks like")
    st.write(
        "Choose one cached test cascade. The table below is the numeric graph "
        "record used by the model: one row per post/node. UPFD's supplied text "
        "view here is already a BERT vector, so this preview does not contain "
        "readable post text."
    )
    if examples and examples.get("examples"):
        labels = {0: "Real", 1: "Fake"}
        choices = examples["examples"]
        choice_names = [
            f"#{item['graph_index']} — {labels[item['label']]} — "
            f"{item['n_nodes']} nodes, depth {item['max_depth']}"
            for item in choices
        ]
        chosen = st.selectbox(
            "Cached GossipCop test cascade",
            range(len(choices)),
            format_func=lambda i: choice_names[i],
            key="dataset_raw_cascade",
        )
        item = choices[chosen]
        arrays = item["arrays"]
        x = np.asarray(arrays["x"])
        parent = np.asarray(arrays["parent"], dtype=int)
        depth = np.asarray(arrays["hop_depth"], dtype=int)
        children = np.asarray(arrays["n_children"], dtype=int)
        drift = drift_of(arrays)

        st.caption(
            f"Cascade #{item['graph_index']} · label {item['label']} ({labels[item['label']]}) · "
            f"{x.shape[0]} nodes · {int(np.asarray(arrays['edge_index']).shape[1])} edges · "
            f"x shape {tuple(x.shape)}"
        )

        raw_schema = pd.DataFrame(
            [
                ("x", str(tuple(x.shape)), "768-dimensional BERT vector for each post"),
                ("edge_index", str(tuple(np.asarray(arrays['edge_index']).shape)), "Propagation links between node indices"),
                ("parent", str(tuple(parent.shape)), "Parent node index; -1 marks the root"),
                ("hop_depth", str(tuple(depth.shape)), "Tree depth of each node"),
                ("n_children", str(tuple(children.shape)), "Number of direct child posts"),
                ("y", str(tuple(np.asarray(arrays['y']).shape)), "Cascade label: 0 real, 1 fake"),
            ],
            columns=["Stored array", "Shape", "What it represents"],
        )
        st.dataframe(raw_schema, hide_index=True, use_container_width=True)

        node_table = pd.DataFrame(
            {
                "node_id": np.arange(x.shape[0]),
                "parent_id": parent,
                "depth": depth,
                "children": children,
                "cosine_drift": np.round(drift[:, 0], 4),
                "L2_drift": np.round(drift[:, 1], 4),
                "depth_norm": np.round(drift[:, 2], 4),
                "children_norm": np.round(drift[:, 3], 4),
                "is_root": drift[:, 4].astype(int),
                "x[0]": np.round(x[:, 0], 4),
                "x[1]": np.round(x[:, 1], 4),
                "x[2]": np.round(x[:, 2], 4),
                "x[3]": np.round(x[:, 3], 4),
            }
        )
        st.caption("First 20 node rows; x[0]–x[3] are just the first four of 768 embedding coordinates.")
        st.dataframe(node_table.head(20), hide_index=True, use_container_width=True)

        node_id = st.selectbox(
            "Inspect one node's embedding coordinates",
            range(x.shape[0]),
            index=min(1, x.shape[0] - 1),
            key="dataset_raw_node",
        )
        vector_preview = pd.DataFrame(
            {
                "dimension": np.arange(12),
                "value": np.round(x[node_id, :12], 6),
            }
        )
        st.caption(f"Node {node_id}: showing 12 of its 768 numeric embedding values.")
        st.dataframe(vector_preview, hide_index=True, use_container_width=True)

        edge_rows = [
            {"parent_node": int(parent[node]), "child_node": node}
            for node in range(len(parent))
            if parent[node] >= 0
        ]
        st.caption("Parent → child links (first 20); this is the propagation-tree structure.")
        st.dataframe(pd.DataFrame(edge_rows).head(20), hide_index=True, use_container_width=True)
    else:
        st.warning("No cached example cascades found. Rebuild them with scripts/build_demo_cache.py.")

    st.subheader("How the experiment is run")
    st.markdown(
        "1. Keep UPFD's supplied train/validation/test cascade split.\n"
        "2. Use the supplied frozen BERT vectors and tree structure; we do not re-encode raw post text.\n"
        "3. Compute parent-relative drift and structural features for each node.\n"
        "4. Compare drift across labels with depth controls; train primary GossipCop models and select checkpoints on validation accuracy.\n"
        "5. Report test results on the fixed split. The feature-group cross-validation shown in the paper is an explicitly exploratory analysis performed within the test partition."
    )
    st.warning(
        "This is an offline research prototype. It does not ingest a live platform "
        "stream, verify claims, or determine whether any individual post is true."
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
    show_figure(fig)
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
            "legacy BaselineLSTM checkpoint (fallback — no canonical hybrid checkpoint found)",
        )
    return None, None, None


def page_predictor():
    st.header("Live predictor")
    checkpoint, model_name, description = resolve_checkpoint()
    if checkpoint is None:
        st.warning(
            "No checkpoint available. Train one with:\n\n"
            "```\n"
            "python3 multiseed.py --subset gossipcop --models Hybrid --seeds 1 "
            "--out results/_canonical_hybrid_seed1.json --checkpoint-dir checkpoints\n"
            "```"
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

    tab_dataset, tab_finding, tab_cascades, tab_predict = st.tabs(
        ["Dataset & Raw Data", "The Finding", "Cascade Explorer", "Optional Predictor"]
    )
    with tab_dataset:
        page_dataset(summary, examples)
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
