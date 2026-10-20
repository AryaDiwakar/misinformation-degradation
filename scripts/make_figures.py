"""Regenerate all paper figures with the final, reproducible data."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.upfd.loader import UPFDGraphDataset
from features.drift import compute_drift

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REAL = "#1f77b4"
FAKE = "#d62728"
MODEL_COLORS = {
    "GCN": "#7f7f7f", "GAT": "#8c564b", "BiGCN": "#bcbd22",
    "DegradGNN": "#ff7f0e", "BaselineLSTM": "#2ca02c", "Hybrid": "#9467bd",
    "DriftPool": "#e377c2",
}

# Optional supplementary result files merged into the canonical ones.
EXTRA_RESULT_FILES = ["multiseed_driftpool.json"]
EARLY_EXTRA_FILES = ["early_detection_driftpool.json"]


def load_result_entry(model_key):
    """Load a model entry, checking canonical + extra result files."""
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    for fname in ["multiseed_all.json"] + EXTRA_RESULT_FILES:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        rec = d.get("gossipcop", d).get(model_key)
        if rec is not None:
            return fname, rec
    return None, None


def load_drift(subset):
    ds = UPFDGraphDataset(root="./data/upfd/raw", subset=subset, feature_type="bert", split="test")
    real_mag, fake_mag = [], []
    real_drifts, fake_drifts = [], []
    real_hop, fake_hop = {}, {}
    for i in range(len(ds)):
        d = ds[i]
        drift = compute_drift(d.x, d.parent, d.hop_depth, d.n_children).numpy()
        mag = np.linalg.norm(drift[:, :2], axis=1)
        hd = d.hop_depth.numpy()
        if d.y.item() == 1:
            fake_mag.append(mag); fake_drifts.append(drift)
            for h, m in zip(hd.tolist(), mag.tolist()):
                fake_hop.setdefault(h, []).append(m)
        else:
            real_mag.append(mag); real_drifts.append(drift)
            for h, m in zip(hd.tolist(), mag.tolist()):
                real_hop.setdefault(h, []).append(m)
    return {
        "real_mag": np.concatenate(real_mag), "fake_mag": np.concatenate(fake_mag),
        "real_drifts": np.concatenate(real_drifts), "fake_drifts": np.concatenate(fake_drifts),
        "real_hop": real_hop, "fake_hop": fake_hop,
    }


# ---------------------------------------------------------------- Fig 1: drift analysis
print("Fig 1: drift_analysis.png", flush=True)
data = {s: load_drift(s) for s in ["gossipcop", "politifact"]}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for row, subset in enumerate(["gossipcop", "politifact"]):
    d = data[subset]
    rm, fm = d["real_mag"], d["fake_mag"]

    ax = axes[row][0]
    bins = np.linspace(0, 18, 60)
    ax.hist(rm, bins=bins, alpha=0.55, label=f"Real (n={len(rm):,})", color=REAL)
    ax.hist(fm, bins=bins, alpha=0.55, label=f"Fake (n={len(fm):,})", color=FAKE)
    ax.axvline(rm.mean(), color=REAL, ls="--", lw=1.5)
    ax.axvline(fm.mean(), color=FAKE, ls="--", lw=1.5)
    ax.set_title(f"{subset.title()} — drift magnitude" + ("" if row else ""))
    if row == 0:
        ax.legend(); ax.set_ylabel("Node count")
    ax.set_xlim(0, 12)

    ax = axes[row][1]
    hops = sorted(set(d["real_hop"]) | set(d["fake_hop"]))[:5]
    r_means = [np.mean(d["real_hop"][h]) for h in hops]
    f_means = [np.mean(d["fake_hop"][h]) for h in hops]
    x = np.arange(len(hops)); w = 0.35
    ax.bar(x - w / 2, r_means, w, label="Real news", color=REAL)
    ax.bar(x + w / 2, f_means, w, label="Fake news", color=FAKE)
    ax.set_xticks(x); ax.set_xticklabels([f"hop {h}" for h in hops])
    ax.set_title(f"{subset.title()} — drift by depth")
    ax.set_ylabel("Mean drift magnitude")

    ax = axes[row][2]
    all_drifts = np.concatenate([d["real_drifts"], d["fake_drifts"]])
    # Node-level labels: real_drifts and fake_drifts are per-node drift
    # vectors, so the label array must be built per node (not per graph).
    labels = np.concatenate([
        np.zeros(len(d["real_drifts"]), dtype=np.int64),
        np.ones(len(d["fake_drifts"]), dtype=np.int64),
    ])
    rng = np.random.default_rng(0)
    if len(all_drifts) > 20000:
        idx = rng.integers(0, len(all_drifts), 20000)
    else:
        idx = np.arange(len(all_drifts))
    pca = PCA(n_components=2)
    emb = pca.fit_transform(all_drifts[idx])
    ax.scatter(emb[labels[idx] == 0, 0], emb[labels[idx] == 0, 1], s=1, alpha=0.4, label="Real", color=REAL)
    ax.scatter(emb[labels[idx] == 1, 0], emb[labels[idx] == 1, 1], s=1, alpha=0.4, label="Fake", color=FAKE)
    ax.set_title(f"{subset.title()} — PCA of drift vectors")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(markerscale=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "drift_analysis.png"))
plt.close()
print("Saved figures/drift_analysis.png", flush=True)

# ---------------------------------------------------------------- Fig 2: main results
print("Fig 2: main_results.png", flush=True)
order = ["GCN", "GAT", "BiGCN", "DegradGNN", "BaselineLSTM", "Hybrid", "DriftPool"]
gc = {}
for model_key in order:
    fname, rec = load_result_entry(model_key)
    if rec is not None:
        gc[model_key] = rec
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for col, metric in enumerate(["acc_mean", "auc_mean"]):
    ax = axes[col]
    names = [n for n in order if n in gc]
    means = [gc[n][metric] * 100 for n in names]
    stds = [gc[n][metric.replace("mean", "std")] * 100 for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    bars = ax.bar(np.arange(len(names)), means, yerr=stds, capsize=4, color=colors, alpha=0.9, error_kw={"lw": 1.5})
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Accuracy (%)" if col == 0 else "AUC-ROC (%)")
    ax.set_title("GossipCop test " + ("accuracy" if col == 0 else "AUC-ROC") + " (3 seeds)")
    ax.set_ylim((60 if col == 0 else 82), 100)
    ax.grid(axis="y", alpha=0.3)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.5, f"{m:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "main_results.png"))
plt.close()
print("Saved figures/main_results.png", flush=True)

# ---------------------------------------------------------------- Fig 3: ablation
print("Fig 3: ablation.png", flush=True)
abl10 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "ablations_10pct.json")))
abl100 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "ablations.json")))
variants = ["Full", "w/o drift", "w/o hop"]
prior_names = {"Full": "Full (drift+hop+tree)", "w/o drift": "w/o drift", "w/o hop": "w/o hop"}
# Both files are produced by the same protocol (AblationGNN, seed 1, best
# validation accuracy), so the two 100% bars agree; the extra variant from
# the 100%-only script is shown as context.
full10 = [abl10[v]["full"]["accuracy"] * 100 for v in variants]
trunc10 = [abl10[v]["truncated_10pct"]["accuracy"] * 100 for v in variants]
full100 = [abl100[prior_names[v]]["accuracy"] * 100 for v in variants]
extra_variants = [v for v in ("w/o drift+hop", "w/o tree (MLP)", "BERT-MLP") if v in abl100]

x = np.arange(len(variants)); w = 0.35
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.bar(x - w / 2, full100, w, label="100% cascade", color=REAL, alpha=0.85)
ax.bar(x + w / 2, trunc10, w, label="10% cascade", color=FAKE, alpha=0.85)
for bars, values in (
    (ax.containers[0], full100),
    (ax.containers[1], trunc10),
):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
        )
ax.axhline(0.5 * 100, color="k", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(variants)
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Ablation: drift/hop features hurt at 100% and 10% (GossipCop)")
ax.legend(loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 100)

# The remaining 100%-only variants, shown as a separate hatched group so the
# chart stays on one protocol for the compared variants.
if extra_variants:
    extra_acc = [abl100[v]["accuracy"] * 100 for v in extra_variants]
    offset = len(variants) + 0.4
    ax.bar(
        np.arange(len(extra_variants)) + offset,
        extra_acc,
        w * 1.6,
        color="#aec7e8",
        alpha=0.9,
        hatch="//",
        edgecolor="white",
        label="100% only",
    )
    ax.set_xticks(
        list(x)
        + list(np.arange(len(extra_variants)) + offset)
    )
    ax.set_xticklabels(
        variants + [v.replace(" (MLP)", "\n(MLP)").replace("+hop", "+hop") for v in extra_variants],
        fontsize=9,
    )
    ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ablation.png"))
plt.close()
print("Saved figures/ablation.png", flush=True)

# ---------------------------------------------------------------- Fig 4: drift-only discriminator
print("Fig 4: discriminator.png", flush=True)
disc = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "drift_discriminator_robustness.json")))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(2); w = 0.35
for i, (tag, label) in enumerate([("full", "Full cascade"), ("truncated_10pct", "First 10%")]):
    v = disc["gossipcop"][tag]
    ax.bar(x + (i - 0.5) * w, [v["auc_mean"], disc["politifact"][tag]["auc_mean"]], w,
           label=label, color=[REAL, FAKE][i], alpha=0.85,
           yerr=[v["auc_std"], disc["politifact"][tag]["auc_std"]], capsize=4)
ax.axhline(0.5, color="k", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(["GossipCop", "PolitiFact"])
ax.set_ylabel("AUC-ROC")
ax.set_ylim(0.4, 1.0)
ax.set_title("Drift-only discriminator (random forest, 12 features)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "discriminator.png"))
plt.close()
print("Saved figures/discriminator.png", flush=True)

# ---------------------------------------------------------------- Fig 5: architecture
print("Fig 5: architecture.png", flush=True)
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")


def box(x, y, w, h, text, fc="#e8e8e8", ec="#555555", fs=10, tc="#111111"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                 fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.8))


box(0.2, 1.9, 1.5, 0.7, "Node features\nx_i (BERT 768-d)", fc="#dbeafe")
box(2.0, 1.9, 1.2, 0.7, "Drift features\nd_i (5-d)", fc="#fde68a")
box(2.0, 1.1, 1.2, 0.7, "Hop embed\nh_i (16-d)", fc="#fde68a")

# LSTM branch
box(3.6, 2.6, 1.7, 0.8, "Bi-LSTM\n(over hop-sorted\nnodes)", fc="#dcfce7")
# GNN branch
box(3.6, 0.7, 1.7, 1.6, "DegradGNN\n(drift + hop\nmessage passing)", fc="#fae8ff")

box(5.8, 1.9, 1.4, 0.8, "Concat\n256-d + 128-d", fc="#ffe4e6")
box(7.6, 1.9, 1.4, 0.8, "MLP\nclassifier", fc="#dcfce7")
box(9.2, 1.9, 0.6, 0.8, "y", fc="#fff", ec="#000")

arrow(1.7, 2.25, 2.0, 2.25)
arrow(2.0, 2.25, 3.6, 2.7)
arrow(2.0, 1.45, 3.6, 1.5)
arrow(1.7, 2.25, 3.6, 2.25)
arrow(3.6, 3.0, 5.8, 2.5)
arrow(3.6, 2.3, 5.8, 2.2)
arrow(5.8, 2.3, 7.6, 2.3)
arrow(7.6, 2.3, 9.2, 2.3)
ax.set_title("HybridDegradLSTM: bidirectional LSTM over cascade order fused with\nDegradGNN structural encoding (drift + hop-depth auxiliary signals)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "architecture.png"))
plt.close()
print("Saved figures/architecture.png", flush=True)

# ---------------------------------------------------------------- Fig 6: example trees
print("Fig 6: example_tree.png", flush=True)
ds = UPFDGraphDataset(root="./data/upfd/raw", subset="gossipcop", feature_type="bert", split="test")
real_ex = next((d for d in ds if d.y.item() == 0), None)
fake_ex = next((d for d in ds if d.y.item() == 1), None)
# pick moderately sized trees
for d in ds:
    if d.y.item() == 0 and 40 <= d.x.size(0) <= 90:
        real_ex = d; break
for d in ds:
    if d.y.item() == 1 and 40 <= d.x.size(0) <= 90:
        fake_ex = d; break

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, d, title in [(axes[0], real_ex, "Real news tree"), (axes[1], fake_ex, "Fake news tree")]:
    drift = compute_drift(d.x, d.parent, d.hop_depth, d.n_children).numpy()
    mag = np.linalg.norm(drift[:, :2], axis=1)
    ei = d.edge_index.numpy()
    hd = d.hop_depth.numpy()
    n = d.x.size(0)
    levels = {}
    for v in range(n):
        levels.setdefault(int(hd[v]), []).append(v)
    pos = {}
    for h, nodes in levels.items():
        nodes = sorted(nodes)
        for i, v in enumerate(nodes):
            pos[v] = (i - (len(nodes) - 1) / 2, -h)
    seen_edges = set()
    for src, dst in ei.T:
        if src == dst or (src, dst) in seen_edges or (dst, src) in seen_edges:
            continue
        if src in pos and dst in pos:
            seen_edges.add((src, dst))
            ax.plot([pos[src][0], pos[dst][0]], [pos[src][1], pos[dst][1]],
                    color="#bbbbbb", lw=0.8, zorder=1)
    sc = ax.scatter([pos[v][0] for v in pos], [pos[v][1] for v in pos],
                    c=mag[list(pos)], cmap="viridis", s=28, zorder=2, edgecolors="black", linewidths=0.3)
    ax.set_title(f"{title} ({n} nodes)")
    ax.axis("off")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("drift magnitude (L2)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "example_tree.png"))
plt.close()
print("Saved figures/example_tree.png", flush=True)

# ---------------------------------------------------------------- Fig 7: early detection (multi-seed)
print("Fig 7: early_detection.png", flush=True)
try:
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    ed = {}
    for fname in ["early_detection_multiseed.json"] + EARLY_EXTRA_FILES:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            ed.update(json.load(open(path)))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    pcts = [10, 25, 50, 100]
    for model, marker, color in [("BaselineLSTM", "o", "#2ca02c"), ("DegradGNN", "s", "#ff7f0e"), ("Hybrid", "^", "#9467bd"), ("DriftPool", "D", "#e377c2")]:
        if model not in ed:
            continue
        if not all(f"acc_{p}pct_mean" in ed[model] for p in pcts):
            print(f"  skipping {model}: incomplete early-detection record", flush=True)
            continue
        means = [ed[model][f"acc_{p}pct_mean"] * 100 for p in pcts]
        stds = [ed[model].get(f"acc_{p}pct_std", 0.0) * 100 for p in pcts]
        ax.errorbar(pcts, means, yerr=stds, marker=marker, color=color, label=model, lw=2, markersize=7, capsize=4)
    ax.axhline(0.5 * 100, color="k", ls=":", lw=1, label="random")
    ax.set_xscale("log")
    ax.set_xticks(pcts)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Percentage of propagation tree used")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Partial-cascade performance (GossipCop, 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "early_detection.png"))
    plt.close()
    print("Saved figures/early_detection.png", flush=True)
except FileNotFoundError:
    print("early_detection_multiseed.json not ready; skipping early_detection.png", flush=True)

print("All figures done.")
