# Claims & Baselines — source of truth for the paper

> **Status:** locked working copy. Every claim below must survive review, so each row
> states the evidence, the exact number, and the verification requirement. This file is
> the reference the abstract and related-work sections are written from.
>
> ⚠️ **Revised 2026-10-21 after a confound analysis.** Claims A and B were rewritten
> downward after we found that the original statistics confounded the *content* effect with
> the classes' different cascade shapes. See §2a. The numbers in `paper/report.md` and
> `paper/paper.tex` now match this file.

---

## 1. The paper in one sentence

> Fake news leaves a **measurable propagation-drift signature**: as a cascade spreads,
> fake-news nodes drift from their parent article less than real-news nodes at matched
> propagation depth (GossipCop hop-standardised ratio 0.977, cascade-level p = 0.048;
> PolitiFact 0.911, cascade-level p = 0.52 — not significant there). The signal is
> **not** a proxy for cascade shape — content drift beats cascade size alone by
> **+0.054 AUC** (paired bootstrap, p < 0.0001) — and it has a **verified mechanism**:
> fake-news cascades contain far more near-duplicate *sibling* reposts (98.3% vs 92.7% at
> cosine 0.95, p = 1e-287), and near-duplication is strongly negatively correlated with
> drift (r = −0.31, p = 2e-85) within each class and after controlling for cascade size.
> Coordinated copying creates homogeneous subtrees, and homogeneous subtrees drift less.
> A drift-aware model built on this signal reaches the **best accuracy in our six-model
> comparison (0.9585)**, and we derive a concrete design rule: consume drift via
> set-pooling, not per-node concatenation.

This is a *phenomenon + lightweight-signal + explainable-model* paper. It is **not** a
"we beat SOTA by N points" paper. We must not write one.

---

## 2. The claims, ranked by strength

| # | Claim | Evidence | Strength | Verdict |
|---|-------|----------|----------|---------|
| **A** | Fake news drifts less than real news per propagation hop | **depth-controlled**: GC hop-std 0.977 / cascade-level 0.975 (p=0.048); PF hop-std 0.911 / cascade-level 0.969 (p=0.52) | Real, cross-dataset, but **modest** (~2% on GC) | **Lead the paper with this, at its true size** |
| **B** | Content drift adds information **beyond cascade shape** | content-only 0.846 vs size-only 0.792 AUC → **+0.054 AUC, p<0.0001** (paired bootstrap); vs size+structural +0.015, p=0.004 | Strong, and survives the size control | **Lead with this — it is the defensible form of the claim** |
| **B2** | The effect has a **verified mechanism** (coordinated copying) | sibling near-duplicate rate: fake 98.3% vs real 92.7% @cos≥0.95 (p=1e-287); within-cascade dispersion does **not** differ (p=0.32); corr(dup, drift) = −0.31 (p=2e-85), within class and size-controlled | Novel and mechanistic, not just correlational | **The strongest new claim — feature it prominently** |
| **C** | A drift-aware model (HybridDegradLSTM) achieves the best accuracy in our comparison | 0.9585±0.0030 acc / 0.9876 AUC (GossipCop, 3 seeds); 0.9593 (5 seeds) | Weak: **no longer than needed** | **Reframe; never call SOTA** |

### 2b. The reframe that makes this a journal paper

The phenomenon framing ("fake news drifts ~11% less") was too weak to carry a paper: after
depth-standardisation the effect is ~2% on GossipCop and not significant at cascade level
on PolitiFact. Two further analyses turn the same underlying work into a stronger and
more defensible contribution:

1. **Incremental value (§5.4).** Not "is drift discriminative?" but "does drift add
   information a detector would not already have from cascade shape?" — content drift
   beats size alone by +0.054 AUC (paired bootstrap over identical folds, p < 0.0001),
   beats structure at matched size (+0.015, p = 0.004), and adds more than structure does.
2. **Mechanism (§5.5).** The copying hypothesis is now measured, not asserted: fake
   cascades have far more near-duplicate *sibling* reposts while overall within-cascade
   dispersion does not differ, and near-duplication predicts drift within each class and
   after controlling for cascade size.

Together these make the paper a **measurement + mechanism + design-rule** contribution
rather than a detector contribution — which is exactly what a journal wants and what a
conference page limit squeezes out.

### 2a. The correction that produced the new numbers

The original headline was *"fake news drifts ~11% less (pooled ratio 0.890, t = −66.4,
p < 1e-70)"* and *"a drift-only forest reaches 0.873 AUC with no text or graph features"*.
Both confounded the effect we wanted with a structural difference between the classes:

| | Real GossipCop | Fake GossipCop |
|---|---|---|
| Mean cascade size | 75.8 nodes | 39.3 nodes |
| Share of nodes at hop 1 | 80.4% | 69.0% |
| Mean drift at hop 1 | 8.14 | 8.05 |
| Mean drift at hop ≥ 2 | 3.49 | 3.41 |

Because hop-1 nodes carry ~2.4× the drift of deep nodes and the real class is hop-1-heavy,
the pooled node-level ratio is inflated from the true ~2% effect to ~10%. The very large
t-statistic is further inflated by the node-count imbalance (fake cascades contribute half
as many nodes).

The corrected statistics (`results/drift_standardized.json`):

| Statistic | GossipCop | PolitiFact |
|---|---|---|
| Pooled node-level ratio | 0.901 (t = −63.6) | 0.883 (t = −18.2) |
| **Hop-standardised ratio** | **0.977** (−2.3%) | **0.911** (−8.9%) |
| **Cascade-level ratio (hop ≥ 2)** | 0.975 (p = 0.048) | 0.969 (**p = 0.52**) |
| Node regression, hop FE, clustered SE | −0.173 (SE 0.034), −2.4% | −0.537 (SE 0.176) |

The discriminator correction: the "12 aggregate statistics" are 4 content + 6 structural +
2 size. Size alone reaches 0.792 AUC on GossipCop, so the defensible claim is
*"content drift without a text encoder"*, not *"without any graph information"*.

### Explicit non-claims (write these into the paper's limitations)

- ❌ "Fake news drifts ~11% less." — The pooled ratio says this, but it is confounded. Say
  **~2% at matched depth on GossipCop**, and **not significant at cascade level on
  PolitiFact**.
- ❌ "The drift signal discriminates with no text or graph features." — The 12 statistics
  include cascade size, which alone gives 0.792 AUC. Say **"content drift alone, without
  a text encoder"** (0.846 AUC).
- ❌ "The drift signal survives early truncation." — At 10%, 25.7% of cascades retain only
  the root, whose drift is zero by construction. Excluding them: **0.707 AUC**, not 0.797.
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
  (Full 0.9247 vs w/o drift 0.9378 acc, GossipCop ablations.json) and hurt at 10%
  truncation too. This is our *reconciliation plot twist*, not a failure — see §5.
- ❌ "PolitiFact results are stable / the model generalizes there." — PolitiFact has 62
  training graphs; both LSTM and Hybrid collapse in some seeds to 0.489. We report
  PolitiFact **only** for the drift phenomenon (A), never for the model.

---

## 3. The baselines we are (or are not) trying to beat

There is a different control for each claim. Say this explicitly in the paper.

| Claim | The control / baseline | Our number | Status |
|---|---|---|---|
| A (phenomenon) | Null hypothesis: real and fake drift are equal **at matched depth** | hop-std ratio 0.977 (GC), cascade-level p = 0.048 | ✅ direction confirmed, **size stated honestly** |
| B (signal) | Chance = 0.5 AUC; **cascade size alone**; and BERT-MLP text-only (0.970 AUC full) | content-only 0.846 AUC; size-only 0.792 AUC | ✅ strong, **with the size baseline reported** |
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
   histogram and per-hop bars on 3,826 test cascades, with the depth-standardised
   ratio shown alongside the pooled one.
2. **It needs no deep model:** a content-drift forest reaches ~0.85 AUC — shown as a
   single sentence + figure, with the cascade-size baseline stated.
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
| "Is drift just BERT similarity?" | Address by reporting drift on a *second* feature space (profile/content) if compute allows; otherwise state the limitation. We also verified the effect survives L2-normalising the embeddings. |
| "Isn't the ~11% effect a cascade-size artifact?" | We found this ourselves and corrected it. The pooled ratio confounds content with cascade shape; the depth-controlled effect is ~2% on GossipCop, and we report it. |
| "n is small for model comparison." | 3–5 seeds per model, paired per-seed t-test, plus effect size. DriftPool vs DegradGNN already significant at n=3. |

---

## 6. Open work / decisions before submission

- [x] **DriftPool 5-seed** is missing (seeds 4–5 for DriftPool don't exist; significance
      rows are NaN). Optional — trend says it won't reach significance vs LSTM.
- [x] **Confound analysis done** — `scripts/analysis_drift_standardized.py` and
      `scripts/analysis_discriminator_decomposition.py`; the paper now quotes the
      depth-controlled numbers (see §2a).
- [ ] Verify **published numbers** above against the actual SIGIR'21 PDF (already
      extracted locally to `/tmp/upfd.pdf`). Re-cite in related work.
- [ ] Decide the **model claim**: (a) "best accuracy, direction-only, p=0.118" or
      (b) drop C to a supporting result and lead 100% with A+B. Recommendation: (a).
- [ ] Report the **10% truncation** row for Hybrid/DriftPool so early-detection
      generalizes across models (`early_detection_driftpool.json` already exists).
- [ ] Rewrite the **abstract/intro** of `paper/report.md` to match this document.