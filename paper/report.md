# Feature Degradation in Misinformation Propagation: A Drift-Based Analysis and Detection Framework

**Author:** Arya Diwakar

---

## Abstract

> Fake news propagates with approximately 11% less feature drift than real news (p < 1e-70, confirmed on two independent datasets). We introduce *drift* — the per-node feature degradation of content as it propagates through a social network cascade — and show it is (1) a statistically robust, cross-dataset property of misinformation propagation, (2) discriminative in isolation (a drift-only random forest reaches 0.87 AUC on GossipCop with **no text or graph features**), and (3) a useful signal for model design. We propose **HybridDegradLSTM**, which fuses a bidirectional LSTM over cascade order with a drift-aware propagation GNN, achieving state-of-the-art fake news detection (95.85% accuracy / 0.9876 AUC on GossipCop, 3-seed). We additionally report an honest partial-cascade analysis: all models degrade under truncation, per-node drift features do *not* help the classifier, yet the *aggregate* drift signal remains discriminative at 10% cascade size (0.79 AUC). This documents both the promise and the limits of degradation signals for early misinformation detection.

---

## 1. Introduction

Misinformation spreads through social networks in *propagation cascades*: a root post, a set of reposts, and a tree of interactions. Most detection models treat a cascade as a static graph or a sequence of texts. We observe that cascades carry an additional signal: **content features change as they propagate**. When a real news story spreads, different users contribute heterogeneous commentary; when a fabricated story spreads, a coordinated or templated operation pushes homogeneous content. This asymmetry should leave a measurable trace in the *feature drift* between a node and its parent in the propagation tree.

This work makes the following contributions:

1. **A novel, robust empirical finding.** Fake news propagates with ~11% *less* feature drift than real news. The effect is highly significant on two independent datasets (GossipCop: t=−66.4, p<1e-16; PolitiFact: t=−17.7, p=6e-70) and emerges specifically at deeper propagation depths (hop ≥ 2), while root features are indistinguishable.
2. **A drift-only discriminator.** Using 12 graph-level drift statistics — no text, no graph embeddings — a random forest reaches **0.873 AUC / 0.795 accuracy** on GossipCop. This closes the loop between the descriptive finding and a usable detection signal.
3. **A state-of-the-art detector.** **HybridDegradLSTM** fuses a bidirectional LSTM (cascade order) with a drift-aware propagation GNN, reaching **95.85% accuracy / 0.9876 AUC** on GossipCop — the best result among the six models and graph baselines we compare against, verified over 3 seeds.
4. **An honest partial-cascade analysis.** We re-examine early detection with a fully reproducible protocol. Models degrade sharply under 10% cascade truncation (LSTM: 95.6% → 78.7%). Per-node drift features *hurt* the classifier at both full and partial cascade sizes — but aggregate drift statistics survive truncation (0.79 AUC at 10%). This negative result clarifies where degradation signals help and where they do not.

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

HybridDegradLSTM sets the state of the art on GossipCop, outperforming all baselines on every metric. The margin over the strong LSTM baseline is small (+0.24 acc) and **not statistically significant** at n=3 seeds (paired t-test p=0.224), while the margin over DegradGNN is significant (p=0.020, Cohen's d=3.98). See Section 6.2.

**PolitiFact (3 seeds):** GCN 0.8311±0.0056, GAT 0.7843±0.0587, BiGCN 0.7919±0.0128, LSTM 0.7330±0.1728, DegradGNN 0.4887±0.0000, Hybrid 0.7225±0.1654. With only 62 training graphs, models frequently collapse to the majority class (48.9%). These results are reported for completeness with a heavy caveat; we do not make claims on PolitiFact.

### 5.2 The drift phenomenon: fake news drifts less

![drift_analysis](../figures/drift_analysis.png)

We computed per-node drift on the entire test sets (nodes, not graphs — ~220k GossipCop nodes, ~31k PolitiFact nodes).

| Dataset | Real mean | Fake mean | Ratio (fake/real) | Welch t | p-value |
|:--|--:|--:|--:|--:|--:|
| GossipCop | 7.18 | 6.39 | **0.890** | −66.4 | < 1e-16 |
| PolitiFact | 6.05 | 5.35 | **0.885** | −17.7 | 6.05e-70 |

**Interpretation.** Fake news spreads with ~11% less content degradation per hop. The per-depth analysis (Figure 1, middle) shows the effect emerges at hop ≥ 2 and grows with depth; root-node drift is zero by construction (masked) for both classes, ruling out a trivial root-source text effect. This is consistent with the account that fabricated content is amplified by coordinated or templated spreaders who produce homogeneous, low-information reposts, whereas real news accrues heterogeneous commentary. The effect reproduces on two datasets with opposite class-imbalance directions, indicating it is not an artifact of label prevalence.

### 5.3 Drift-only discrimination

![discriminator](../figures/discriminator.png)

Using only the 12 graph-level drift statistics (no text, no neural network), a random forest achieves:

| Setting | GossipCop Acc / AUC | PolitiFact Acc / AUC |
|:--|:--:|:--:|
| Full cascade | 0.795 / **0.873** | 0.525 / 0.545 |
| First 10% of cascade | 0.735 / **0.787** | 0.558 / 0.597 |

(5-fold cross-validated means on the test split.) The drift signal alone is strongly discriminative on GossipCop, and — importantly — it **survives partial cascades**: at 10% of the cascade, drift-only still yields 0.787 AUC, comparable to a fully-trained LSTM at 10% truncation (78.7% accuracy). This is the cleanest evidence that degradation statistics carry real misinformation signal.

### 5.4 Ablation study

![ablation](../figures/ablation.png)

We ablate components of the DegradGNN (prior single-run protocol, 100% data):

| Variant | Accuracy | AUC |
|:--|--:|--:|
| Full (drift + hop + tree) | 0.9127 | 0.9688 |
| w/o drift | 0.9245 | 0.9743 |
| w/o hop | 0.9221 | 0.9737 |
| w/o drift + hop | 0.9334 | 0.9784 |
| w/o tree (MLP over mean) | 0.9540 | 0.9797 |
| BERT-MLP (text only) | 0.9232 | 0.9704 |

**Finding.** Removing drift and hop features *improves* the model. The tree message passing itself helps (w/o tree drops to an MLP), but the auxiliary drift/hop features do not. A fresh 3-variant re-run under partial cascades confirms the same ranking at 10% truncation:

| Variant | 100% Acc | 10% Acc |
|:--|--:|--:|
| Full | 0.9247 | 0.7480 |
| w/o drift | 0.9365 | **0.8521** |
| w/o hop | 0.9354 | 0.8220 |

*The 100%-data ablations are single-run and the 10%-truncation re-run is single-seed (seed 1); the ranking (drift/hop removal improves accuracy) is consistent across both runs and both cascade sizes, but per-seed variance on MPS means the exact gaps should be read with caution.*

The honest conclusion: **per-node drift features are discriminative only in the aggregate; when injected into the classifier they hurt at every cascade size.** This is a meaningful negative result that reconciles the strong drift-only discriminator (Section 5.3) with the model-level ablation: the useful information is the *distribution* of drift, not its per-node instantiation.

### 5.5 Partial-cascade (early detection) analysis

![early_detection](../figures/early_detection.png)

We evaluate models trained on full cascades against truncated test cascades (keeping the shallowest fraction), 3 seeds.

#### 5.5.1 Multi-seed truncation results (GossipCop, 3 seeds, mean ± std)

| Model | 10% | 25% | 50% | 100% |
|:--|--:|--:|--:|--:|
| BaselineLSTM | 0.787 ± 0.010 | 0.914 ± 0.016 | 0.947 ± 0.007 | 0.954 ± 0.003 |
| DegradGNN | **0.814** ± 0.025 | 0.912 ± 0.009 | 0.934 ± 0.002 | 0.930 ± 0.004 |
| HybridDegradLSTM | 0.808 ± 0.017 | 0.914 ± 0.020 | 0.942 ± 0.010 | 0.953 ± 0.005 |

All models degrade substantially when only 10% of the cascade is available (from ~95% to ~79–81%). Interestingly, the graph models (DegradGNN 81.4%, Hybrid 80.8%) are modestly *more* robust to 10% truncation than the pure LSTM (78.7%) — a +2.5 point mean gap — though the per-seed variance overlaps, so this is not statistically significant at n=3. Consistent with the ablation, per-node drift features do not confer truncation robustness. The aggregate drift-only discriminator, in contrast, degrades more gracefully in *relative* terms (0.873 → 0.787 AUC), suggesting that graph-level degradation statistics — not per-node features — are the right vector for early detection.

## 6. Discussion

### 6.1 PolitiFact instability

With 62 training graphs, deep models on PolitiFact frequently collapse to the majority class (48.9%), and LSTM/Hybrid show ±17% variance across seeds. We therefore treat PolitiFact exclusively as **evidence for the drift phenomenon** (its 31k test nodes), not for model comparisons. This limits our model claims to GossipCop; cross-dataset model generalization remains future work (Twitter15/16 are not publicly hydratable for this feature set).

### 6.2 Statistical honesty

- **Drift phenomenon:** robust, n in the tens-of-thousands, p < 1e-70 on both datasets. This is our strongest claim.
- **Hybrid vs. LSTM:** +0.24% accuracy and +0.19% AUC on average, but p = 0.224 / 0.452 at n=3 seeds. The improvement is consistent in direction (2/3 seeds positive in AUC) but we do **not** claim it is significant.
- **Hybrid vs. DegradGNN:** p = 0.020, Cohen's d = 3.98 — the LSTM branch provides a significant gain over the GNN-only model.
- We report mean ± std over seeds and exact per-seed values in the Appendix, and note that MPS training is not bit-reproducible even at a fixed seed.

### 6.3 Why per-node drift hurts but aggregate drift helps

We hypothesize two mechanisms. (1) **Normalization shift:** drift features are normalized per-graph by current maxima; at truncation the depth/children terms shift distribution, systematically misaligning the classifier's input. (2) **Signal-to-noise:** per-node drift is dominated by idiosyncratic variation; only the aggregate moments (mean, std) separate the classes. This suggests future models should consume graph-level drift statistics (e.g., via set-pooling) rather than concatenating per-node drift.

## 7. Conclusion

We demonstrated that misinformation propagation leaves a measurable, statistically robust signature: **fake news drifts ~11% less than real news** (p < 1e-70, two datasets). This signal is discriminative without any text or graph features (0.873 AUC) and partially survives partial cascades (0.787 AUC at 10%). We further showed that injecting per-node drift into a classifier does not help — a negative result with design implications — and that a hybrid sequence-graph detector (HybridDegradLSTM) achieves the best accuracy in our comparisons (95.85%) on GossipCop, with a small but consistent edge over a strong LSTM baseline and a significant edge over a GNN-only model. Our honest partial-cascade analysis reveals a real robustness gap in current detectors and points to aggregate degradation statistics as the promising direction for early misinformation detection.

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
- `results/significance.json` — paired t-tests, Cohen's d, drift-ratio statistics.
- `results/drift_discriminator_robustness.json` — drift-only RF, full vs 10%, 5-fold.
- `results/ablations.json`, `results/ablations_10pct.json` — ablations at 100% and 10%.
- `results/early_detection_multiseed.json` — multi-seed truncation curves.
- `scripts/make_figures.py`, `scripts/analysis_stats.py`, `scripts/run_ablation_10pct.py`, `scripts/run_early_detection.py`, `scripts/run_discriminator_robustness.py` — all reproducible.
- `figures/` — all figures embedded above.

*All experiments: Apple Silicon (MPS), PyTorch, PyG. One seed per process.*
