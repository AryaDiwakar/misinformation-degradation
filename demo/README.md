# Research Demo

The recommended faculty demo is the Streamlit **Misinformation Drift Explorer**.
It shows the dataset setup, the reported finding, example propagation trees, and
an optional model predictor. It is an offline research prototype, not a live
fact-checker.

## Launch it

From the repository root:

```bash
python3 -m pip install -r requirements-demo.txt
streamlit run demo/drift_explorer.py
```

The cached GossipCop examples are included under `demo/assets/`. To rebuild the
cache from the local UPFD files (for example, after changing the drift
definition), run:

```bash
python3 scripts/build_demo_cache.py
```

## Tabs

- **Dataset & Raw Data** — UPFD split counts plus an actual cached test-cascade
  preview: stored array names/shapes, node rows, parent IDs, depths, drift
  values, edge list, and the first coordinates of a 768-dimensional BERT vector.
  The released preview contains vectors and graph structure, not readable post
  text.
- **The Finding** — real/fake drift distributions, drift by propagation depth,
  pooled versus depth-standardized comparison, and the exploratory
  feature-group AUC comparison. The paper's key values are a GossipCop
  depth-standardized ratio of 0.977 and cascade-level ratio of 0.975
  (nominal `p=0.048`); PolitiFact's cascade-level ratio is 0.969 and is not
  significant (`p=0.522`).
- **Cascade Explorer** — select a cached real or fake test cascade and inspect
  its tree, with each node colored by its parent-relative drift magnitude.
- **Optional Predictor** — uses a trained checkpoint if one is available. It
  is not required for the dataset and results demonstration; predictions are
  research outputs, not verified truth judgments.

The GossipCop primary models use the supplied fixed split and validation-based
checkpoint selection. The feature-group AUC analysis uses cross-validation
inside the official test partition and is explicitly exploratory, not an
independent test estimate. The PolitiFact set is much smaller and is used for
the drift comparison, not stable model ranking.

## Original single-cascade app

`demo/app.py` is also available for a standalone cascade prediction. It expects
a metadata-rich checkpoint and an `.npz` cascade with `x` and `edge_index`;
`hop_depth`, `parent`, and `n_children` can be reconstructed when absent.
