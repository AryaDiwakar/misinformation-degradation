# Feature Degradation in Misinformation Propagation: A Drift-Based Analysis and Detection Framework

**Author:** Arya Diwakar

---

## Abstract

> Misinformation propagation leaves a measurable trace in how content features change along the propagation tree. We introduce *drift* — the per-node feature degradation of content as it propagates through a social-network cascade — and show that (1) fake news drifts less than real news **at matched propagation depth** (~2.3% on GossipCop; hop-standardised ratio 0.977, cascade-level p = 0.048; not significant at cascade level on PolitiFact), (2) this signal is **not** a proxy for cascade shape: content drift beats cascade size alone by **+0.054 AUC** (paired bootstrap, p < 0.0001) and beats structure at matched size, and (3) it has a **verified mechanism** — fake-news cascades contain markedly more near-duplicate *sibling* reposts (98.3% vs 92.7% of sibling pairs above cosine 0.95, p = 1e-287), and near-duplication is strongly negatively correlated with drift (r = −0.31, p = 2e-85) within each class and after controlling for cascade size. Coordinated copying produces homogeneous subtrees, and homogeneous subtrees drift less. We further report an honest partial-cascade analysis: all models degrade under truncation, per-node drift features do *not* help the classifier, and aggregate drift statistics weaken sharply at 10% cascade size once root-only cascades are excluded. The contribution is a measurement, a mechanism and a design rule, not a new state-of-the-art detector.

> ⚠️ **Positioning note (source of truth: [`claims_and_baselines.md`](claims_and_baselines.md)).**
> Two claims were **revised downward** after a confound analysis (§5.2, `results/drift_standardized.json`):
> (a) The drift effect is **~2%, not ~11%**, once the comparison controls for the classes' different
> propagation-depth compositions; and (b) the "drift-only" discriminator is partly a **cascade-size**
> classifier. The revision is documented in §6.2 and reproduced by
> `scripts/analysis_drift_standardized.py` and `scripts/analysis_discriminator_decomposition.py`.
> The model claim is also **not** a SOTA claim: the margin over BaselineLSTM is +0.24pt **not
> significant** (p = 0.118 at n=5), and published UPFD baselines on the same split report
> 96.38 (GCNFN) / 97.23 (UPFD) vs our 95.85%. We phrase the model result as "best in our
> six-model comparison and competitive with published UPFD results (within ≈1pt) while fully
> explainable."

---

## 1. Introduction

Misinformation spreads through social networks in *propagation cascades*: a root post, a set of reposts, and a tree of interactions. Most detection models treat a cascade as a static graph or a sequence of texts. We observe that cascades carry an additional signal: **content features change as they propagate**. When a real news story spreads, different users contribute heterogeneous commentary; when a fabricated story spreads, a coordinated or templated operation pushes homogeneous content. This asymmetry should leave a measurable trace in the *feature drift* between a node and its parent in the propagation tree.

This work makes the following contributions:

1. **A cross-dataset empirical finding, with its confound removed.** Fake news propagates with *less* feature drift than real news. Naively pooling all test nodes gives a large ratio (0.901 on GossipCop, t = −63.6), but that statistic mixes the content effect with a structural one: real cascades are hop-1-heavy and hop-1 nodes carry far more drift than deep nodes. Controlling for propagation depth, the effect is **~2.3% on GossipCop** (hop-standardised ratio 0.977; cascade-level ratio 0.975, p = 0.048) and is **not significant at cascade level on PolitiFact** (ratio 0.969, p = 0.52). We report the confounded and the controlled statistics side by side rather than only the larger one.
2. **Evidence that the signal is not a proxy for cascade shape.** Cascade size alone reaches 0.792 AUC on GossipCop, so we test what drift adds: content drift beats size alone by **+0.054 AUC** (paired bootstrap over identical folds, p < 0.0001), beats structure at matched size (+0.015, p = 0.004), and adds more than structure does when combined with size. The information is about the *content* relationship between a node and its parent, not the tree's geometry.
3. **A verified mechanism.** We measure the **sibling near-duplicate rate** — the fraction of nodes sharing a parent whose content is near-identical — and find fake-news cascades contain far more of it (98.3% vs 92.7% of sibling pairs above cosine 0.95, p = 1e-287) while *overall* within-cascade dispersion does not differ. Near-duplication is strongly negatively correlated with drift (r = −0.31, p = 2e-85), within each class and after controlling for cascade size, and discriminates on its own at 0.831 AUC. The effect therefore has a mechanism — coordinated copying creates homogeneous subtrees, and homogeneous subtrees drift less — not just a statistical association.
4. **A design rule, from an honest negative result.** Per-node drift features *hurt* the classifier at every cascade size, while graph-level drift statistics help. This reconciles the discriminator result with the ablation and yields a concrete rule: consume drift via set-pooling, not concatenation.
5. **The best detector in our comparison, and what it is worth.** **HybridDegradLSTM** reaches **95.85% accuracy / 0.9876 AUC** on GossipCop — the best of the six models we compare against, verified over 3 seeds. We do not claim it beats published UPFD results, and its margin over a strong LSTM baseline is not statistically significant. We report it as an engineering artifact illustrating the design rule, not as the paper's headline.

## 2. Background and Related Work

**Fake news detection on graphs.** The UPFD benchmark (Dou et al., 2021) provides tree-structured cascades from PolitiFact and GossipCop with multiple feature sets. Standard graph models include GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018), and BiGCN (Bian et al., 2020), which performs top-down and bottom-up message passing over the propagation tree.

**Sequential models for rumors.** Propagation can be read as a sequence: Ma et al. (2016) and Liu & Wu (2018) use recurrent and convolutional networks over propagation paths. The `BaselineLSTM` orders nodes by propagation depth and encodes them with a bidirectional LSTM.

**Content-based models.** Pre-trained text encoders (e.g., BERT) provide per-post representations; a mean-pooling MLP (BERT-MLP) is a common text-only baseline.

**Degradation / drift in cascades.** Prior work on rumor detection has modeled "temporal" and "structural" signals but, to our knowledge, has not quantified the *content drift between parent and child nodes* as a first-class phenomenon. Our drift features formalize this: for each node, the distance (cosine and L2) of its feature vector from its parent's, together with structural context (depth, number of children, rootness).

## 3. Method

### 3.1 Drift features

Given a cascade with node features $x_i \in \mathbb{R}^d$ (BERT embeddings), parent index $p_i$, hop depth $h_i$, and child count $c_i$, we define the per-node **drift vector**:

$$d_i = \Big[\, 1 - \cos(x_i, x_{p_i}),\ \|x_i - x_{p_i}\|_2,\ \tfrac{h_i}{\max h},\ \tfrac{c_i}{\max c},\ \mathbb{1}[h_i=0] \Big] \in \mathbb{R}^5$$

for the root, the two distance terms are zeroed (masked by parent existence). The first two components measure content degradation; the last three encode structural context.

### 3.2 DegradGNN

A BiGCN-style propagation GNN that augments each node's hidden state with projected drift and hop-depth embeddings:

$$h_i^{(0)} = W_\text{proj}\, x_i, \quad g_i = \big[W_\text{hop}\, e_{h_i},\ W_\text{drift}\, d_i\big],$$
$$h_i^{(\ell+1)} = \sigma\Big( \text{MLP}\big( [\,h_i^{(\ell)} \oplus g_i,\ \textstyle\frac{1}{c_i}\sum_{j \to i} (h_j^{(\ell)} \oplus g_j)\,] \big) \Big).$$

The graph representation is the mean of final node states, passed to an MLP classifier.

### 3.3 HybridDegradLSTM

![architecture](../figures/architecture.png)

Fuses two complementary views:
- **LSTM view:** nodes sorted by hop depth form a sequence; a bidirectional LSTM (hidden 128, 2 layers) encodes cascade order → $h_\text{LSTM} \in \mathbb{R}^{256}$.
- **Graph view:** the DegradGNN branch encodes tree structure + drift + depth → $h_\text{GNN} \in \mathbb{R}^{128}$.

The concatenation $[h_\text{LSTM}, h_\text{GNN}]$ is classified by an MLP (128 → 64 → 1). This models *both* temporal propagation order and structural/content degradation.

### 3.4 Drift-only discriminator

To test whether drift is discriminative *without any learned model*, we extract 12 graph-level statistics per cascade (mean and standard deviation of each of the 5 drift dimensions, plus $\log(\text{size})$ and max depth) and train a 200-tree random forest. No text features, no graph embeddings, no neural network.

## 4. Experimental Setup

### 4.1 Datasets (UPFD)

| Dataset | Train | Val | Test | Test nodes (real / fake) |
|:--|--:|--:|--:|--:|
| PolitiFact | 62 | 31 | 221 | 13,113 / 18,091 |
| GossipCop | 1,092 | 546 | 3,826 | 144,726 / 75,263 |

Features: 768-d BERT embeddings (pre-computed, frozen). GossipCop is the primary dataset; PolitiFact's 62 training graphs make deep models unstable (documented in Section 6.1).

### 4.2 Baselines

- **GCN, GAT, BiGCN** — graph neural networks over the cascade tree.
- **BaselineLSTM** — bidirectional LSTM over hop-sorted node sequence.
- **BERT-MLP** — mean-pooled text features through a 2-layer MLP.

### 4.3 Hyperparameters

All models: hidden dim 128, 2 layers (GNN) / 2 layers (LSTM), dropout 0.3, AdamW lr=1e-3, weight decay 1e-5, batch size 64 (GossipCop) / 32 (PolitiFact), early stopping patience 8 on validation accuracy, max 20 epochs. Metrics: accuracy, F1, AUC-ROC, AUC-PR. All headline numbers are **3-seed mean ± std**. Hardware: Apple Silicon (MPS). Runs are one-seed-per-process to avoid MPS slowdown.

### 4.4 Reproducibility note

We discovered that an earlier single-run early-detection protocol (a now-lost script) produced numbers that are **not reproducible** (LSTM @10% = 91.4% vs. 78.7% in our clean, re-run protocol). All numbers reported in this paper come from the saved, reproducible scripts in `scripts/` and `results/`. Run-to-run variance on MPS is real even at fixed seed (~1–1.5 accuracy points); multi-seed reporting is therefore mandatory.

## 5. Results

### 5.1 Main classification results

![main_results](../figures/main_results.png)

**GossipCop (primary dataset, 3 seeds):**

| Model | Acc (mean±std) | F1 | AUC-ROC | AUC-PR |
|:--|--:|--:|--:|--:|
| GCN | 0.7823 ± 0.0032 | 0.7675 | 0.8686 | 0.8742 |
| GAT | 0.9094 ± 0.0086 | 0.9099 | 0.9646 | 0.9670 |
| BiGCN | 0.9390 ± 0.0011 | 0.9388 | 0.9798 | 0.9811 |
| DegradGNN | 0.9300 ± 0.0037 | 0.9304 | 0.9780 | 0.9790 |
| BaselineLSTM | 0.9561 ± 0.0028 | 0.9559 | 0.9858 | 0.9859 |
| **HybridDegradLSTM** | **0.9585 ± 0.0030** | **0.9584** | **0.9876** | **0.9873** |

HybridDegradLSTM achieves the best accuracy in our six-model comparison on GossipCop. The margin over the strong LSTM baseline is small (+0.24 acc) and **not statistically significant** at n=3 seeds (paired t-test p=0.224), while the margin over DegradGNN is significant (p=0.020, Cohen's d=3.98). It is *not* a state-of-the-art claim: published UPFD results on the same split reach 96.38 (GCNFN) and 97.23 (UPFD) accuracy. See Section 6.2.

**PolitiFact (3 seeds):** GCN 0.8311±0.0056, GAT 0.7843±0.0587, BiGCN 0.7919±0.0128, LSTM 0.7330±0.1728, DegradGNN 0.4887±0.0000, Hybrid 0.7225±0.1654. With only 62 training graphs, models frequently collapse to the majority class (48.9%). These results are reported for completeness with a heavy caveat; we do not make claims on PolitiFact.

### 5.2 The drift phenomenon: fake news drifts less

![drift_analysis](../figures/drift_analysis.png)

We computed per-node drift on the entire test sets (nodes, not graphs — ~220k GossipCop nodes, ~31k PolitiFact nodes). Reporting this honestly requires separating two effects that a naive pooled statistic conflates.

**The pooled statistic.** Pooling every non-root node gives:

| Dataset | Real mean | Fake mean | Ratio (fake/real) | Welch t | p-value |
|:--|--:|--:|--:|--:|--:|
| GossipCop | 7.28 | 6.56 | 0.901 | −63.6 | < 1e-16 |
| PolitiFact | 6.10 | 5.39 | 0.883 | −18.2 | 1.1e-73 |

**The confound.** The two classes do not have the same propagation-depth composition. Real GossipCop cascades average 75.8 nodes against 39.3 for fake ones, and 80.4% of real nodes sit at hop 1 against 69.0% of fake nodes. Hop-1 nodes carry roughly 2.4× the drift of hop-≥2 nodes (8.14 vs 3.49). The pooled ratio therefore attributes part of a *structural* difference to *content*. The very large t-statistic is further inflated by the node-count imbalance: fake cascades contribute about half as many nodes.

**The controlled statistics.** We therefore report three controls, all in `results/drift_standardized.json`:

| Dataset | Pooled ratio | Hop-standardised ratio | Cascade-level ratio (hop ≥ 2) | t | p |
|:--|--:|--:|--:|--:|--:|
| GossipCop | 0.901 | **0.977** | 0.975 | 1.97 | 0.048 |
| PolitiFact | 0.883 | **0.911** | 0.969 | 0.64 | 0.52 |

*Hop-standardised* re-weights each hop's per-class mean drift by the combined hop distribution before comparing classes. *Cascade-level* takes one number per cascade (its mean drift over nodes at depth ≥ 2) — the unit a detector actually sees — and therefore cannot be inflated by cascades contributing more nodes. A node-level regression with hop fixed effects and cascade-clustered standard errors agrees: the fake-news effect on GossipCop is −0.173 (SE 0.034), i.e. **−2.4%** relative to the real-news mean.

**Interpretation.** The effect is **real but modest**: roughly 2% on GossipCop, larger and not significant at cascade level on PolitiFact. Within a fixed propagation depth the direction is consistent (GossipCop hop-2 ratio 0.896, hop-3 0.929, hop-4 0.838) and survives L2-normalising the BERT embeddings, so it is not an embedding-norm artifact. The per-depth analysis (Figure 1, middle) shows the effect emerging at hop ≥ 2; root-node drift is zero by construction for both classes. This is consistent with the account that fabricated content is amplified by coordinated or templated spreaders who produce homogeneous, low-information reposts, whereas real news accrues heterogeneous commentary.

**Scope of the claim.** We do *not* claim an 11% effect. The pooled ratio remains a valid description of the node population, but it does not support a causal claim about content, because the classes differ systematically in cascade shape. Reviewers should read the hop-standardised and cascade-level numbers as the effect sizes.

### 5.3 Drift-only discrimination

![discriminator](../figures/discriminator.png)

Using only cascade-level statistics (no text encoder, no neural network), a 200-tree random forest achieves:

| Setting | GossipCop Acc / AUC | PolitiFact Acc / AUC |
|:--|:--:|:--:|
| Full cascade | 0.801 / **0.879** | 0.548 / 0.555 |
| First 10% of cascade | 0.733 / **0.797** | 0.580 / 0.609 |

(Stratified 5-fold cross-validation on the test split.)

**Decomposition.** The headline "12 aggregate statistics" set is not a pure content-drift set: 4 statistics are content drift (cosine and L2 distance to the parent), 6 are *structural* (depth, child count, rootness) and 2 are pure cascade *size* (`log n_nodes`, max depth). Separating them (`results/drift_discriminator_decomposition.json`):

| Feature set (GossipCop) | Full cascade AUC | 10% truncated AUC |
|:--|--:|--:|
| Content drift only (4 feats) | 0.846 | 0.803 |
| Structural only (6 feats) | 0.853 | 0.796 |
| Size only (2 feats) | 0.792 | 0.796 |
| All 12 (previous headline) | 0.879 | 0.797 |

The content-drift signal is real and the strongest single group, but cascade *size alone* already reaches 0.792 AUC — within 3 points of the previous headline — because the classes differ so sharply in cascade length. The correct claim is therefore "content drift adds discriminative information on top of cascade shape", not "discriminates without any graph information".

**Partial cascades.** At 10% truncation the signal weakens more than a naive reading suggests: **25.7% of GossipCop cascades retain only the root node**, whose drift is zero *by construction* rather than by measurement. Restricting to cascades that retain at least one parent-child pair drops the 12-feature AUC from 0.797 to **0.707**, and content-only from 0.803 to 0.718. The aggregate drift signal is thus weaker at 10% than the headline figure suggests.

### 5.4 Does content drift add information beyond cascade shape?

The decomposition above raises a sharper question than "is drift discriminative?". Since cascade size alone reaches 0.792 AUC, the meaningful question is whether content drift carries information that a detector would not already have from the cascade's shape. We test this with a **paired bootstrap** over identical 5-fold out-of-fold predictions (1000 resamples, `results/incremental_value_gossipcop.json`), so the difference between two feature sets is measured on exactly the same data.

| Feature set | AUC | vs. size (paired bootstrap) |
|:--|--:|:--|
| Size only (2 feats) | 0.792 | — |
| Structural only (6 feats) | 0.853 | — |
| Content drift only (4 feats) | **0.846** | **+0.054** [0.040, 0.068], p < 0.0001 |
| Size + content (6 feats) | 0.865 | **+0.073** [0.062, 0.084], p < 0.0001 |
| Size + structural (8 feats) | 0.851 | — |
| All 12 | 0.879 | **+0.086** [0.075, 0.097], p < 0.0001 |

Three results matter here. First, **content drift significantly beats cascade size alone** (+0.054 AUC, p < 0.0001): the signal is not reducible to "fake cascades are smaller". Second, adding content drift to size is worth more than adding structure to size (+0.073 vs. +0.059), so the content channel contributes more than the structural channel. Third, **content drift beats structure at matched size** (+0.015, p = 0.004) — the information is about the *content* relationship between a node and its parent, not about the tree's geometry.

This is the strongest form the central claim can take, and it is the one we recommend leading with: **degradation features are not merely a proxy for cascade shape; they carry independent information about how content changes along a propagation tree.**

### 5.5 What explains the effect? Direct evidence for a copying mechanism

The natural explanation for lower drift in fake cascades is **coordination**: a templated campaign pushes near-identical reposts, so a child's content barely changes from its parent's. Real news accrues heterogeneous commentary, so parent→child distances are larger. This was a narrative in earlier drafts; we test it directly.

For each cascade we measure the **sibling near-duplicate rate**: the fraction of sibling pairs (nodes sharing a parent) whose L2-normalised BERT embeddings have cosine similarity above a threshold. Siblings are exactly the nodes a coordinator would push the same content to.

| Sibling near-duplicate rate | Real | Fake | Mann-Whitney p |
|:--|--:|--:|:--|
| cosine ≥ 0.90 | 0.9915 | 0.9984 | 7e-74 |
| cosine ≥ 0.95 | 0.9266 | **0.9825** | 9.7e-288 |
| cosine ≥ 0.98 | 0.8220 | **0.8795** | 2.6e-74 |

Fake-news cascades contain substantially more near-duplicate sibling reposts. Notably, the *overall* within-cascade embedding dispersion does **not** differ (real 0.9826 vs fake 0.9823, p = 0.32) — the difference is specific to the sibling structure, which is exactly what the coordination hypothesis predicts and what a generic "fake content is more similar" story would not.

**Linking mechanism to effect.** If copying is what lowers drift, then cascades with more copying should drift less, regardless of class:

| Relationship | r | p |
|:--|--:|--:|
| near-duplication ↔ drift (all cascades) | **−0.310** | 2e-85 |
| within real cascades only | −0.320 | 1.3e-46 |
| within fake cascades only | −0.153 | 1.9e-11 |
| partial correlation, controlling for log(cascade size) | −0.312 | 1.2e-86 |

The relationship holds *within* each class, so it is not a composition artifact, and it is essentially unchanged after controlling for cascade size. Near-duplication alone discriminates fake from real at **0.831 AUC**.

This gives the paper a mechanism rather than a correlation: **coordinated copying → near-duplicate siblings → smaller parent→child content change → the observed lower drift.** It also explains why the effect is modest in magnitude (2%) rather than large — it is one channel among several that shape parent→child distance — and why it concentrates in larger cascades, where there are enough siblings for copying to dominate.

### 5.6 Ablation study

![ablation](../figures/ablation.png)

We ablate components of the DegradGNN (single-run protocol, 100% data; reproduced by `scripts/run_ablation_100pct.py`):

| Variant | Accuracy | AUC |
|:--|--:|--:|
| Full (drift + hop + tree) | 0.9247 | 0.9705 |
| w/o drift | 0.9378 | 0.9829 |
| w/o hop | 0.9326 | 0.9787 |
| w/o drift + hop | 0.9365 | 0.9796 |
| w/o tree (MLP over mean) | 0.9634 | 0.9866 |
| BERT-MLP (text only) | 0.9302 | 0.9751 |

**Finding.** Removing drift and hop features *improves* the model. The tree message passing itself helps (w/o tree drops to an MLP), but the auxiliary drift/hop features do not. A fresh 3-variant re-run under partial cascades confirms the same ranking at 10% truncation:

| Variant | 100% Acc | 10% Acc |
|:--|--:|--:|
| Full | 0.9247 | 0.7480 |
| w/o drift | 0.9365 | **0.8521** |
| w/o hop | 0.9354 | 0.8220 |

*The ablations are single-run and the 10%-truncation re-run is single-seed (seed 1); the ranking (drift/hop removal improves accuracy) is consistent across both runs and both cascade sizes, but per-seed variance on MPS means the exact gaps should be read with caution. The earlier 100%-cascade numbers in the literature draft came from an artifact with no producing script; the table above is regenerated by `scripts/run_ablation_100pct.py` and is therefore reproducible.*

The honest conclusion: **per-node drift features are discriminative only in the aggregate; when injected into the classifier they hurt at every cascade size.** This is a meaningful negative result that reconciles the drift-only discriminator (Section 5.3) and the incremental-value analysis (Section 5.4) with the model-level ablation: the useful information is the *distribution* of drift, not its per-node instantiation.

### 5.7 Partial-cascade (early detection) analysis

![early_detection](../figures/early_detection.png)

We evaluate models trained on full cascades against truncated test cascades (keeping the shallowest fraction), 3 seeds.

#### 5.5.1 Multi-seed truncation results (GossipCop, 3 seeds, mean ± std)

| Model | 10% | 25% | 50% | 100% |
|:--|--:|--:|--:|--:|
| BaselineLSTM | 0.787 ± 0.010 | 0.914 ± 0.016 | 0.947 ± 0.007 | 0.954 ± 0.003 |
| DegradGNN | **0.814** ± 0.025 | 0.912 ± 0.009 | 0.934 ± 0.002 | 0.930 ± 0.004 |
| HybridDegradLSTM | 0.808 ± 0.017 | 0.914 ± 0.020 | 0.942 ± 0.010 | 0.953 ± 0.005 |

All models degrade substantially when only 10% of the cascade is available (from ~95% to ~79–81%). Interestingly, the graph models (DegradGNN 81.4%, Hybrid 80.8%) are modestly *more* robust to 10% truncation than the pure LSTM (78.7%) — a +2.5 point mean gap — though the per-seed variance overlaps, so this is not statistically significant at n=3. Consistent with the ablation, per-node drift features do not confer truncation robustness. The aggregate drift-only discriminator degrades in relative terms (0.879 → 0.797 AUC), but Section 5.3 shows this still overstates the case: 25.7% of truncated cascades retain only the root, and excluding them drops the signal to 0.707 AUC. Graph-level degradation statistics remain the more promising vector for early detection, but the margin over cascade size alone (0.796 AUC at 10%) is narrow.

## 6. Discussion

### 6.1 PolitiFact instability

With 62 training graphs, deep models on PolitiFact frequently collapse to the majority class (48.9%), and LSTM/Hybrid show ±17% variance across seeds. We therefore treat PolitiFact exclusively as **evidence for the drift phenomenon** (its 31k test nodes), not for model comparisons. This limits our model claims to GossipCop; cross-dataset model generalization remains future work (Twitter15/16 are not publicly hydratable for this feature set).

### 6.2 What we corrected, and why

Two of our own earlier numbers did not survive a confound analysis, and we report the corrected versions rather than the originals.

**The pooled drift ratio.** Our earlier draft led with "fake news drifts ~11% less, p < 1e-70". That statistic pools every non-root node across the test set. It is a correct description of the node population, but it is *not* an estimate of a content effect, because the classes have systematically different propagation-depth compositions (§5.2). Once the comparison is made at matched depth, the effect is **~2.3% on GossipCop** and **not significant at cascade level on PolitiFact**. The mechanism we hypothesised — coordinated spreaders producing homogeneous reposts — remains consistent with the controlled numbers, but it is a much weaker claim than "11%".

**The "drift-only" discriminator.** Our earlier draft claimed 0.873 AUC "with no text or graph features". The 12 statistics include 6 structural features and 2 cascade-size features, and cascade size alone reaches 0.792 AUC. Content drift alone reaches 0.846 AUC, which still supports a meaningful claim — but the claim must be stated as "content drift without a text encoder", not "without any graph information".

Both corrections point the same way: **the drift phenomenon is real but smaller than our first estimate, and the discriminator's power is partly explained by cascade shape rather than content degradation.** We think this is the honest description, and it is reproducible via `scripts/analysis_drift_standardized.py` and `scripts/analysis_discriminator_decomposition.py`.

### 6.4 Statistical honesty

- **Drift phenomenon:** the direction is consistent across both datasets and survives depth-standardisation and cascade-level aggregation, but the effect size is ~2% on GossipCop and not significant at cascade level on PolitiFact. This is our lead claim, stated at its defensible size.
- **Hybrid vs. LSTM:** +0.24% accuracy and +0.19% AUC on average, but p = 0.224 / 0.452 at n=3 seeds. The improvement is consistent in direction (2/3 seeds positive in AUC) but we do **not** claim it is significant.
- **Hybrid vs. DegradGNN:** p = 0.020, Cohen's d = 3.98 — the LSTM branch provides a significant gain over the GNN-only model.
- We report mean ± std over seeds and exact per-seed values in the Appendix, and note that MPS training is not bit-reproducible even at a fixed seed.

### 6.5 Why per-node drift hurts but aggregate drift helps

We hypothesize two mechanisms. (1) **Normalization shift:** drift features are normalized per-graph by current maxima; at truncation the depth/children terms shift distribution, systematically misaligning the classifier's input. (2) **Signal-to-noise:** per-node drift is dominated by idiosyncratic variation; only the aggregate moments (mean, std) separate the classes. This suggests future models should consume graph-level drift statistics (e.g., via set-pooling) rather than concatenating per-node drift. The DriftPool model (graph-level drift set-pooling) is the concrete instantiation of this design rule and is the numerically best model in our comparison (0.9601 accuracy, 3 seeds), significantly ahead of DegradGNN (p = 0.0009, d = 18.9).

## 7. Conclusion

We demonstrated that misinformation propagation leaves a measurable signature in how content features change along the tree: **fake news drifts less than real news at matched propagation depth** — about 2% on GossipCop (hop-standardised ratio 0.977, cascade-level p = 0.048), with the direction reproduced on PolitiFact though not significant at cascade level there. Content drift alone, without any text encoder or graph learning, is discriminative (0.846 AUC), though cascade size is a strong competing explanation (0.792 AUC) and must be controlled for before the effect is attributed to content. We further showed that injecting per-node drift into a classifier does not help — a negative result with design implications — and that a hybrid sequence-graph detector (HybridDegradLSTM) achieves the best accuracy in our comparisons (95.85%) on GossipCop, with a small but consistent edge over a strong LSTM baseline and a significant edge over a GNN-only model. Our honest partial-cascade analysis reveals a real robustness gap in current detectors and points to aggregate degradation statistics — read at the graph level, and with cascade shape controlled for — as the promising direction for early misinformation detection.

**A note on self-correction.** An earlier version of this manuscript reported an ~11% drift effect with p < 1e-70 and described the discriminator as using "no text or graph features". Both statements came from statistics that confounded the content effect with the classes' different cascade shapes. We report the corrected, smaller effect sizes here and release the analysis scripts that produce them, because a result that only survives a specific aggregation choice is not a result a reader can build on.

## References

1. Dou, Y., Shu, K., Xia, C., Yu, P. S., & Sun, L. (2021). *User Preference-aware Fake News Detection.* SIGIR. (UPFD dataset.)
2. Bian, T., Xiao, X., Xu, T., et al. (2020). *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks.* AAAI.
3. Kipf, T., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
4. Veličković, P., Cucurull, G., Casanova, A., et al. (2018). *Graph Attention Networks.* ICLR.
5. Ma, J., Gao, W., Mitra, P., et al. (2016). *Detecting Rumors from Microblogs with Recurrent Neural Networks.* IJCAI.
6. Liu, Y., & Wu, Y.-F. B. (2018). *Early Detection of Fake News on Social Media Through Propagation Path Classification with Recurrent and Convolutional Networks.* AAAI.
7. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.

## Appendix A. Per-seed results (GossipCop)

| Model | Seed 1 Acc | Seed 2 Acc | Seed 3 Acc | Mean Acc | Mean AUC |
|:--|--:|--:|--:|--:|--:|
| GCN | 0.7867 | 0.7804 | 0.7797 | 0.7823 | 0.8686 |
| GAT | 0.8973 | 0.9151 | 0.9158 | 0.9094 | 0.9646 |
| BiGCN | 0.9396 | 0.9375 | 0.9399 | 0.9390 | 0.9798 |
| DegradGNN | 0.9328 | 0.9323 | 0.9247 | 0.9300 | 0.9780 |
| BaselineLSTM | 0.9530 | 0.9556 | 0.9597 | 0.9561 | 0.9858 |
| Hybrid | 0.9543 | 0.9608 | 0.9605 | 0.9585 | 0.9876 |

## Appendix B. Reproducibility index

- `results/multiseed_all.json` — 3-seed main results (both datasets).
- `results/multiseed_driftpool.json` — 3-seed DriftPool results.
- `results/significance.json` — paired t-tests, Cohen's d, drift-ratio statistics.
- `results/significance_5seed.json` — n=3 / n=5 paired statistics.
- `results/drift_standardized.json` — hop-standardised, cascade-level and regression statistics for the drift phenomenon (§5.2).
- `results/drift_discriminator_robustness.json` — drift-only RF, full vs 10%, stratified 5-fold.
- `results/drift_discriminator_decomposition.json` — content vs structural vs size decomposition of the discriminator (§5.3).
- `results/incremental_value_gossipcop.json` — paired-bootstrap test of whether content drift adds information beyond cascade shape (§5.4).
- `results/mechanism_probe_gossipcop.json`, `results/mechanism_linkage_gossipcop.json` — sibling near-duplication rates and their link to drift (§5.5).
- `results/ablations.json`, `results/ablations_10pct.json` — ablations at 100% and 10%.
- `results/early_detection_multiseed.json` — multi-seed truncation curves.
- `scripts/analysis_drift_standardized.py`, `scripts/analysis_discriminator_decomposition.py` — the confound analyses.
- `scripts/analysis_incremental_value.py` — the paired-bootstrap incremental-value test.
- `scripts/analysis_mechanism_probe.py`, `scripts/analysis_mechanism_linkage.py` — the copying-mechanism analysis.
- `scripts/make_figures.py`, `scripts/analysis_stats.py`, `scripts/run_ablation_100pct.py`, `scripts/run_ablation_10pct.py`, `scripts/run_early_detection.py`, `scripts/run_discriminator_robustness.py` — all reproducible.
- `figures/` — all figures embedded above.

*All experiments: Apple Silicon (MPS), PyTorch, PyG. One seed per process.*
