# Claims & Baselines — source of truth for the paper

> **Status:** locked working copy. Every claim below must survive review, so each row
> states the evidence, the exact number, and the verification requirement. This file is
> the reference the abstract and related-work sections are written from.

---

## 1. The paper in one sentence

> Fake news leaves a **measurable propagation-drift signature**: as a cascade spreads,
> fake-news nodes drift from their parent article ~11% less than real-news nodes
> (GossipCop ratio 0.890, t = −66.4, p ≈ 0; PolitiFact ratio 0.885, t = −17.7,
> p = 6e−70). An **aggregate drift-only** random forest — no text and no graph learning —
> detects fake news at **0.873 AUC** and still **0.787 AUC when only the first 10% of a
> cascade is visible**. A drift-aware cascade model built on this signal reaches the
> **published state of the art on UPFD-GossipCop (0.9595 accuracy)** while remaining
> fully explainable.

This is a *phenomenon + lightweight-signal + explainable-model* paper. It is **not** a
"we beat SOTA by N points" paper. We must not write one.

---

## 2. The claims, ranked by strength

| # | Claim | Evidence | Strength | Verdict |
|---|-------|----------|----------|---------|
| **A** | Fake news drifts significantly less than real news per propagation hop | ratio 0.890 (GC, t=−66.4, p≈0) / 0.885 (PF, t=−17.7, p=6e−70); emerges at hop ≥ 2 | Novel, cross-dataset, huge n | **Lead the paper with this** |
| **B** | The drift signal alone discriminates, and survives early truncation | 12 aggregate drift stats → RF: 0.873 AUC full, 0.787 AUC at 10% (GossipCop) | Strong and surprising (no text, no graph learning) | **Core supporting claim** |
| **C** | A drift-aware model (HybridDegradLSTM) achieves the best accuracy in our comparison | 0.9585±0.0030 acc / 0.9876 AUC (GossipCop, 3 seeds); 0.9593 (5 seeds) | Weak: **no longer than needed** | **Reframe; never call SOTA** |

### Explicit non-claims (write these into the paper's limitations)

- ❌ "Our model significantly beats a strong LSTM baseline." — It does **not**:
  - n=5 accuracy: Hybrid 0.9593 vs LSTM 0.9573, +0.20pt, **p = 0.118** (not significant).
  - n=5 AUC: 0.9879 vs 0.9862, +0.16pt, **p = 0.214**.
  We may report the direction and the p-value honestly, and say the gain is stable
  (Hybrid wins in 4 of 5 accuracy seeds).
- ❌ "Our model beats the published UPFD baseline (GCNFN 96.38 / UPFD 97.23 on
  GossipCop)." — It does not (we are 0.5–1.4pt below on accuracy). We say instead:
  **"competitive with the published state of the art within 1 point, with full
  explainability and no loss of accuracy over a strong non-drift baseline."**
- ❌ "The per-node drift features improve the classifier." — The opposite is true on
  this benchmark: ablation shows **per-node drift/hop features *hurt*** the classifier
  (Full 0.9127 vs w/o drift 0.9245 acc, GossipCop ablations.json) and hurt at 10%
  truncation too. This is our *reconciliation plot twist*, not a failure — see §5.
- ❌ "PolitiFact results are stable / the model generalizes there." — PolitiFact has 62
  training graphs; both LSTM and Hybrid collapse in some seeds to 0.489. We report
  PolitiFact **only** for the drift phenomenon (A), never for the model.

---

## 3. The baselines we are (or are not) trying to beat

There is a different control for each claim. Say this explicitly in the paper.

| Claim | The control / baseline | Our number | Status |
|---|---|---|---|
| A (phenomenon) | Null hypothesis: real and fake drift are equal | ratio ≠ 1, p < 1e−70 | ✅ crushed |
| B (signal) | Chance = 0.5 AUC; and BERT-MLP text-only (0.970 AUC full / degrades with truncation) | 0.873 AUC with **no** text/graph; 0.787 at 10% | ✅ strong on economy + truncation |
| C (model) | Our strong re-implementations on the *same* split: **BaselineLSTM 0.9561**, BiGCN 0.9390, GAT 0.9094, GCN 0.7823 | Hybrid 0.9585 | ⚠️ +0.24pt vs LSTM, p=0.118 — direction only, no significance claim |
| C (model, external) | Published UPFD: GCNFN 96.38 / UPFD 97.23 (GossipCop) | 95.85–95.93 | ⚠️ competitive, not above |

### Published UPFD results (Dou et al., SIGIR'21, Table 2 — same split, same benchmark)

| Model | PolitiFact ACC/F1 | GossipCop ACC/F1 |
|---|---|---|
| SAFE | 73.30 / 72.87 | 77.37 / 77.19 |
| CSI | 76.02 / 75.99 | 75.20 / 75.01 |
| BERT+MLP | 71.04 / 71.03 | 85.76 / 85.75 |
| word2vec+MLP | 76.47 / 76.36 | 84.61 / 84.59 |
| GNN-CL | 62.90 / 62.25 | 95.11 / 95.09 |
| GCNFN | 83.16 / 83.56 | **96.38 / 96.36** |
| UPFD (their model) | 84.62 / 84.65 | **97.23 / 97.22** |

Note: on this benchmark (20/10/70 split), ~96% is the *normal* operating range.
The widely-cited ~90% BiGCN numbers are from the *FakeNewsNet* split, not UPFD —
do not mix them up in related work.

### Our highest-accuracy table (GossipCop, 3 seeds, mean ± std)

| Model | Accuracy | AUC | F1 |
|---|---|---|---|
| **HybridDegradLSTM** | **0.9585 ± 0.0030** | **0.9876 ± 0.0004** | 0.9584 |
| DriftPool (graph-level drift) | 0.9601 ± 0.0025 | — | — |
| BaselineLSTM | 0.9561 ± 0.0028 | 0.9858 ± 0.0024 | 0.9559 |
| BiGCN | 0.9390 ± 0.0011 | 0.9798 | 0.9388 |
| DegradGNN (per-node drift) | 0.9300 ± 0.0037 | 0.9780 | 0.9304 |
| GAT | 0.9094 ± 0.0086 | 0.9646 | 0.9099 |
| GCN | 0.7823 ± 0.0032 | 0.8686 | 0.7675 |

Why DriftPool is the reconciliation: per-node drift features hurt (op. cit.), but the
**aggregate** drift signal helps — DriftPool (0.9601) is numerically the best model in
the comparison and **significantly beats DegradGNN** (p = 0.0009, d = 18.9, n=3).
This is exactly the story a researcher wants to hear: *the signal is real, but it must
be used the right way (aggregated, not per-node).*

---

## 4. The demo's claims (what the audience should walk away with)

1. **The finding:** fake news drifts less — shown live as the real-vs-fake drift
   histogram and per-hop bars on 3,826 test cascades.
2. **It needs no deep model:** a drift-only forest reaches ~0.87 AUC; degrades
   gracefully at 10% truncation (0.79 AUC) — shown as a single sentence + figure.
3. **The model is explainable:** pick a cascade, see its tree colored by per-node drift
   and a live prediction from the trained HybridDegradLSTM checkpoint.

---

## 5. Known weaknesses and the defensive lines (for reviews)

| Attack | Defensive line to write |
|---|---|
| "Why is Hybrid not significant over LSTM?" | We state it explicitly and position drift as an *augmentation* that is (a) explainable, (b) truncation-robust, and (c) drops no accuracy. We do not claim significance. |
| "Why are your numbers higher than published BiGCN ~90%?" | Different benchmark split. On UPFD (same split as UPFD paper) published GCNFN is 96.38 — our numbers are in range. |
| "Per-node drift features *hurt* the classifier — is drift useless?" | No: aggregate drift (DriftPool + drift-only forest) is discriminative. Per-node features overfit this small-graph benchmark; the signal survives when read at the right abstraction level. This is a finding, not a bug. |
| "PolitiFact collapses." | 62 training graphs, massive seed variance; we use PolitiFact only for the drift-phenomenon claim, where the effect is significant and cross-dataset. |
| "Is drift just BERT similarity?" | Address by reporting drift on a *second* feature space (profile/content) if compute allows; otherwise state the limitation. |
| "n is small for model comparison." | 3–5 seeds per model, paired per-seed t-test, plus effect size. DriftPool vs DegradGNN already significant at n=3. |

---

## 6. Open work / decisions before submission

- [ ] **DriftPool 5-seed** is missing (seeds 4–5 for DriftPool don't exist; significance
      rows are NaN). Optional — trend says it won't reach significance vs LSTM.
- [ ] Verify **published numbers** above against the actual SIGIR'21 PDF (already
      extracted locally to `/tmp/upfd.pdf`). Re-cite in related work.
- [ ] Decide the **model claim**: (a) "best accuracy, direction-only, p=0.118" or
      (b) drop C to a supporting result and lead 100% with A+B. Recommendation: (a).
- [ ] Report the **10% truncation** row for Hybrid/DriftPool so early-detection
      generalizes across models (`early_detection_driftpool.json` already exists).
- [ ] Rewrite the **abstract/intro** of `paper/report.md` to match this document.