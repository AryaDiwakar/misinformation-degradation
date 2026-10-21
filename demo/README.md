# Demo

Two Streamlit demos.

## 1. Drift Explorer (recommended — shows the actual finding)

Interactive view of the paper's core claim: **fake news drifts less than real news**.

```bash
python3 scripts/build_demo_cache.py          # one-time: precomputes the test-split summary
streamlit run demo/drift_explorer.py
```

Tabs:
- **The Finding** — real-vs-fake drift histogram + per-hop bars over all 3,826
  GossipCop test cascades, showing both the pooled fake/real ratio (0.890) and
  the hop-standardised ratio (0.977), which removes the difference in how deep
  the two classes' cascades go. It also summarises the drift-only
  discriminator: content drift alone reaches 0.85 AUC, while cascade size
  alone reaches 0.79 AUC.
- **Cascade Explorer** — pick a real or fake test cascade, see its propagation tree
  colored by per-node drift.
- **Live Predictor** — upload an `.npz` cascade (or reuse a cached example) and get a
  HybridDegradLSTM prediction with per-hop drift. Needs a trained checkpoint:

```bash
python3 multiseed.py --subset gossipcop --models Hybrid --seeds 1 \
  --out results/_canonical_hybrid_seed1.json --checkpoint-dir checkpoints
```

## 2. Single-cascade predictor (the original app)

```bash
pip install -r requirements-demo.txt
python3 scripts/create_sample_cascade.py
streamlit run demo/app.py
```

The demo expects a metadata-rich checkpoint created by `train.py` and an `.npz`
cascade containing `x` and `edge_index`. `hop_depth`, `parent`, and `n_children`
are optional and are reconstructed when absent.