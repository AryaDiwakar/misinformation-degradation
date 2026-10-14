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
  GossipCop test cascades (the 0.89 fake/real ratio) and the drift-only discriminator
  story (0.87 AUC full, 0.79 AUC at 10% truncation).
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