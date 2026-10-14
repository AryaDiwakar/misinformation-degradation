# Changelog

## 1.1.1 - 2026-10-14

- `multiseed.py` now actually saves a structured checkpoint for each model/seed via the
  new `--checkpoint-dir` flag. Previously it logged "checkpoint saved" without writing
  one; checkpoints use the same `pipeline.checkpoints` format as `train.py`.
- Regenerated `checkpoints/hybrid_gossipcop_seed1.pt` from the paper's canonical protocol
  (plain BCE, best-validation-accuracy) — test acc 0.9574, AUC 0.9875. The demo's Live
  Predictor now uses the model behind the reported GossipCop Hybrid result rather than
  the drift-auxiliary-loss variant (`train.py --model hybrid`, which scores 0.9213).
- Updated `README.md` and the demo docs to point at the canonical training command.

## 1.1.0 - 2026-10-13

- Added a drift-explorer Streamlit demo (`demo/drift_explorer.py`) with three tabs:
  the cross-dataset drift finding, per-cascade tree inspection, and a live predictor.
- Added `scripts/build_demo_cache.py` to precompute a lightweight test-split cache so
  the demo boots instantly instead of loading the full UPFD dataset.
- Added `paper/claims_and_baselines.md` — the locked claim/baseline positioning doc,
  including externally verified published UPFD results (Dou et al., SIGIR'21).
- Ran the 5-seed analysis (`results/significance_5seed.json`): Hybrid vs LSTM is not
  significant (acc +0.20pt, p=0.118; AUC p=0.214); DriftPool vs DegradGNN is
  significant (p=0.0009, d=18.9, n=3).
- Flagged that DriftPool has no seeds 4–5 yet (n=5 rows are NaN in the analysis).
- Documented the "state-of-the-art" wording risk in `paper/report.md` abstract.

## 1.0.0 - 2026-10-07

- Added the canonical configuration-driven training pipeline.
- Added UPFD dataset validation and structured checkpoints.
- Added shared cascade truncation and early-detection handling.
- Added single-cascade `.npz` inference and an optional Streamlit demo.
- Added reproduction orchestration, result validation, tests, and CI.
- Documented the research limitations and non-reproducible legacy result files.
