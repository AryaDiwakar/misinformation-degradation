# Feature Degradation in Misinformation Propagation

Research code and paper draft for studying how content representations change as misinformation propagates through social-network cascades.

## Overview

This project introduces **feature drift**, the change between a cascade node's representation and its parent's representation. The framework evaluates whether drift is a measurable misinformation signal and how it can be used in graph and sequence models.

The work combines:

- Cross-dataset statistical analysis of drift on UPFD GossipCop and PolitiFact cascades, with the propagation-depth composition controlled for.
- A content-drift-only random-forest discriminator using cascade-level statistics, without a text encoder or learned graph embeddings.
- DegradGNN, a drift-aware propagation model.
- HybridDegradLSTM, which fuses a bidirectional LSTM over cascade order with a propagation GNN.
- Partial-cascade evaluation for early misinformation detection.

## Headline Results

Results below are reported from the current saved experiments. GossipCop is the primary benchmark; PolitiFact is substantially smaller and is reported with a stability caveat.

| Finding | Result |
| --- | --- |
| Drift difference, GossipCop | Fake-news nodes drift less than real-news nodes. Pooled node-level ratio 0.901; **hop-standardised ratio 0.977** (-2.3% at matched depth); cascade-level ratio 0.975, p = 0.048 |
| Drift difference, PolitiFact | Pooled ratio 0.883; hop-standardised ratio 0.911; cascade-level ratio 0.969, **p = 0.52** (not significant) |
| Content-drift-only discriminator | **0.846 AUC** on GossipCop using 4 content-drift statistics (no text encoder, no graph learning) |
| Full 12-statistic discriminator | 0.801 accuracy / 0.879 AUC on GossipCop; cascade *size* alone reaches 0.792 AUC |
| Drift-only at 10% cascade size | 0.733 accuracy / 0.797 AUC on GossipCop; **0.707 AUC** once the 25.7% of root-only cascades are excluded |
| HybridDegradLSTM | 0.9585 accuracy / 0.9876 AUC on GossipCop, 3-seed mean |

> **Note on the drift effect size.** An earlier version of this work reported an "~11% less drift, p < 1e-70" headline. That statistic pools every test node, which mixes the *content* effect with the classes' different cascade shapes (real cascades average 76 nodes vs 39 for fake ones, and are hop-1-heavy). Controlling for propagation depth the effect is ~2.3% on GossipCop and not significant at cascade level on PolitiFact. Both the confounded and the controlled numbers are reported; see `paper/report.md` §5.2 and §6.2, and `results/drift_standardized.json`.

The HybridDegradLSTM margin over the strong LSTM baseline is not statistically significant at three seeds, and it is below published UPFD results on the same split — it is the best model in our own six-model comparison, not a state-of-the-art claim. Per-node drift features also hurt classifier performance in the ablation, while aggregate drift remains discriminative. These limitations are documented deliberately in the paper rather than hidden behind a single best number.

## Repository Structure

```text
data/       Dataset loaders and crawling utilities; raw UPFD data is not tracked
features/   Drift and representation feature construction
models/     GNN, LSTM, hybrid, and drift-pooling models
pipeline/   Shared model registry, validation, checkpoints, and truncation utilities
demo/       Optional Streamlit demonstration
scripts/    Analysis, evaluation, ablation, and figure-generation scripts
figures/    Figures used in the paper draft
results/    Selected verified JSON result artifacts
paper/      Markdown and LaTeX paper drafts, plus compiled PDF
tests/      Unit tests for the canonical pipeline
```

## Setup

Python 3.10+ and a PyTorch installation compatible with the target device are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments use the UPFD GossipCop and PolitiFact datasets with precomputed 768-dimensional BERT features. The dataset is intentionally excluded from Git because it is approximately 1.2 GB and contains data subject to upstream platform and licensing restrictions. Download it from the [UPFD project](https://github.com/safe-graph/GNN-FakeNews) and place the extracted files under `data/upfd/raw/`.

## Reproduce Key Analyses

Run commands from the repository root after installing the dataset:

```bash
# Validate the local UPFD data before training
python3 scripts/validate_data.py

# Train the paper's canonical HybridDegradLSTM (best-validation-accuracy protocol)
python3 multiseed.py --subset gossipcop --models Hybrid --seeds 1 \
  --out results/_canonical_hybrid_seed1.json --checkpoint-dir checkpoints

# Config-driven trainer (multi-task variant with the drift auxiliary loss)
python3 train.py --config configs/default.yaml --model hybrid --seed 1

# Evaluate a structured checkpoint
python3 evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/hybrid_gossipcop_seed1.pt \
  --skip-baselines

# Run the automated checks
python3 -m pytest -q

# Drift-only discrimination on full and truncated cascades
python3 scripts/run_discriminator_robustness.py

# Decompose the discriminator into content / structural / size feature groups
python3 scripts/analysis_discriminator_decomposition.py

# Re-derive the drift phenomenon with the propagation-depth confound removed
python3 scripts/analysis_drift_standardized.py

# Three-seed early-detection evaluation
python3 scripts/run_early_detection.py --seeds 1,2,3

# Multi-seed benchmark evaluation for the primary dataset
python3 multiseed.py --subset gossipcop --seeds 1,2,3 --out results/gossipcop_reproduction.json

# Validate the tracked paper artifacts
python3 scripts/validate_results.py

# Check that the manuscript's numbers still match the artifacts
python3 scripts/check_paper_numbers.py

# Create an inference example and run the prediction CLI
python3 scripts/create_sample_cascade.py
python3 predict.py \
  --checkpoint checkpoints/hybrid_gossipcop_seed1.pt \
  --input examples/sample_cascade.npz

# Start the optional Streamlit demo
pip install -r requirements-demo.txt
python3 scripts/build_demo_cache.py   # one-time: precompute the test-split drift cache
streamlit run demo/drift_explorer.py  # science demo (finding + cascade explorer + predictor)

# Run the complete compute-intensive reproduction plan
python3 scripts/run_reproduction.py --seeds 1,2,3
```

Most training runs are compute-intensive. Apple Silicon uses MPS when available; otherwise the scripts fall back to CPU.

Both training paths write a metadata-rich checkpoint containing the model architecture, input dimension, dataset, seed, validation score, and full experiment configuration; `multiseed.py --checkpoint-dir` and `train.py` share the same on-disk format. The two paths use **different protocols** and therefore **different filenames**: `multiseed.py` writes `<model>_<subset>_seed<N>.pt` (plain BCE, best-validation-accuracy — the paper's canonical protocol) while `train.py` writes `<model>_<subset>_seed<N>_trainaug.pt` (drift auxiliary loss, best-validation-loss). This prevents the config-driven trainer from silently replacing the canonical paper checkpoint.

The existing research runners remain available for reproducing historical comparison tables.

## Paper

The current manuscript is available in [`paper/report.md`](paper/report.md), [`paper/paper.tex`](paper/paper.tex), and [`paper/paper.pdf`](paper/paper.pdf). Figures are in [`figures/`](figures/).

## Status

Active research repository. The canonical training, evaluation, inference, and demo paths are complete; the manuscript and saved results represent the current reproducible protocol. Older or explicitly invalidated result files are excluded from the tracked artifact set.

## Citation

This repository is currently a private research workspace. A formal citation entry will be added when the manuscript and author metadata are finalized.
