# Changelog

## 1.2.0 - 2026-10-21

### Headline claims corrected downward after a confound analysis

The two headline numbers did not survive a confound check and are now reported at
their defensible size. Both corrections are reproducible from new scripts.

- **Drift effect: ~11% → ~2.3%.** The old headline pooled every test node, which
  mixed the *content* effect with the classes' different cascade shapes (real
  GossipCop cascades average 76 nodes vs 39 for fake ones and are hop-1-heavy;
  hop-1 nodes carry ~2.4x the drift of deep nodes). Controlled statistics:
  hop-standardised ratio 0.977 (GossipCop) / 0.911 (PolitiFact); cascade-level
  ratio 0.975 (p = 0.048) on GossipCop and 0.969 (**p = 0.52**, not significant)
  on PolitiFact; node-level regression with hop fixed effects and
  cascade-clustered SEs gives -2.4% on GossipCop. The pooled and controlled
  numbers are both reported.
- **Discriminator: "no text or graph features" → "content drift without a text
  encoder".** The 12 aggregate statistics are 4 content + 6 structural + 2
  cascade-size; cascade size alone reaches 0.792 AUC on GossipCop. Content drift
  alone reaches 0.846 AUC.
- **Truncation claim weakened.** At 10% truncation 25.7% of GossipCop cascades
  retain only the root node, whose drift is zero *by construction*. Excluding
  them, the 12-feature AUC drops from 0.797 to 0.707 (content-only: 0.803 →
  0.718).
- The paper no longer makes a "state-of-the-art" claim for HybridDegradLSTM; it
  is the best model in our own six-model comparison and is competitive with
  published UPFD results.

New artifacts: `results/drift_standardized.json`,
`results/drift_discriminator_decomposition.json`.
New scripts: `scripts/analysis_drift_standardized.py`,
`scripts/analysis_discriminator_decomposition.py`, `pipeline/drift_stats.py`.

### Bug fixes

- `multiseed.py` recorded `hidden_dim=128` for every model, but `DriftPoolOnly`
  defaults to 64 — every DriftPool/DriftPoolOnly checkpoint was unloadable
  (shape mismatch on reload). Per-model configs are now recorded.
- `train.py` and `multiseed.py` wrote the *same* checkpoint filename, so running
  the config-driven trainer silently replaced the canonical paper checkpoint
  with the weaker drift-auxiliary-loss variant. `train.py` now writes
  `<model>_<subset>_seed<N>_trainaug.pt`.
- `pipeline/models.py` could not resolve CamelCase model names
  (`BaselineLSTM`, `HybridDegradLSTM`), so saving a checkpoint for those models
  raised `KeyError`. Alias resolution now handles CamelCase with acronyms.
- `scripts/train_dtn.py` supervised with `labels[data.y.item()]` — indexing the
  batch label vector by the label *value* rather than the graph's position.
  Wrong supervision, and an `IndexError` for a batch of size one.
- `scripts/train_dtn.py` never saved a checkpoint, so `scripts/evaluate_dtn.py`
  (which requires one) could never run. It now writes structured checkpoints and
  `evaluate_dtn.py` loads them through `pipeline.checkpoints`.
- Cascade truncation was implemented four times with three different semantics.
  All of them now use `pipeline.truncation.truncate_cascade`; the DTN variants
  previously copied parent indices without remapping them, producing corrupt
  drift targets.
- `scripts/run_early_detection.py` discarded previously stored aggregate keys
  when resuming, and printed `0.0000` for skipped models.
- `data/upfd/loader.py::build_parent_index` assumed a directed edge list; with a
  symmetrised list the root adopted a child as its parent. Parent assignment is
  now disambiguated by hop depth (no change to the real UPFD data: verified 0
  differing assignments over 28,282 nodes).
- `pipeline/checkpoints.py` silently ignored the caller's `model_config` when
  the checkpoint carried one, and defaulted `input_dim` to 768. A shape
  mismatch now raises a message naming the stored architecture.
- `scripts/compare_gnn.py` executed its experiment at import time; it is now
  guarded by `main()`.
- `scripts/degradation_trace.py` referenced an undefined global and printed a
  hard-coded conclusion before computing it.
- `scripts/validate_10pct.py` had a no-op list comprehension.
- `scripts/make_figures.py`: the PCA panel built its label array at the wrong
  granularity, and the ablation figure mixed two different protocols in one
  chart.
- `demo/drift_explorer.py` labelled a node count as "test cascades" and
  computed the headline ratio from the subsampled histogram arrays instead of
  the stored full-split means. It also used `st.pyplot(width="stretch")`, which
  requires Streamlit >= 1.49 while `requirements-demo.txt` pinned >= 1.30.

### Reproducibility

- `results/ablations.json` was cited by the paper but had **no producing
  script**. Added `scripts/run_ablation_100pct.py`; the regenerated table keeps
  the same ranking and reproduces the 10%-run's "Full" variant exactly.
- `results/multiseed_all.json`, `results/multiseed_driftpool.json`,
  `results/ablations.json` and `results/early_detection_driftpool.json` were
  gitignored although the paper's main table and figures depend on them. They
  are now tracked.
- `scripts/validate_results.py` now checks all eight tracked artifacts and
  cross-checks the numbers that appear in more than one.
- The DOCX and PPTX generators hard-coded every metric and cited result files
  that do not exist. They now read from `scripts/paper_numbers.py` and print
  `n/a` for a missing artifact instead of inventing a number.
- Removed plaintext administrator credentials from `review_two/review_criteria.md`.

### Tests and CI

- New `tests/test_drift_stats.py` (depth-standardisation, cascade-level
  statistics, regression recovery, truncation sharing) and
  `tests/test_paper_numbers.py` (artifact validation, graceful degradation).
- `tests/test_models_and_checkpoints.py` gained regression tests for the
  DriftPoolOnly checkpoint, the per-model config/shape agreement, and the
  checkpoint mismatch message.
- CI now compiles `demo/`, `multiseed.py`, `predict.py` and the `run_*.py`
  scripts, and installs `statsmodels`.

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
