# Changelog

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
