# RESULTS (Draft)

> **Project:** Cross‑lingual “truth‑direction” transfer via linear alignment of hidden states  
> **Model:** `mlx-community/Mistral-7B-v0.3-4bit` (dense)  
> **Dataset:** `data/wikidata_enfr_facts_v1.jsonl` (N=200: 4 relations × 50 examples; each example has EN/FR true + EN/FR false)  
> **Core idea:** Learn an **English→French rotation** \(W\) that aligns hidden‑state representations across languages, then rotate the **English truth direction** \(w_{\text{eng}}\) into French as \( \hat{w}_{\text{fr}} = W^\top w_{\text{eng}} \). Test whether this improves truth separation in French.

This file summarizes the main findings from the figures currently in the repo. It’s written to be readable for a mentor review and to serve as the “results skeleton” for the paper.

---

## 1) What was measured (high-level)

For each transformer layer \(k\), we extract a sentence representation from the model’s hidden states (end‑of‑block). We then compute:

1. **EN→FR alignment gain**  
   - Metric: mean cosine similarity between paired EN and FR vectors **after** vs **before** Procrustes alignment  
   - Reported as:  
     \[
       \Delta_{\text{align}}(k) = \mathbb{E}[\cos(EW, F)] - \mathbb{E}[\cos(E, F)]
     \]

2. **Truth-direction cosine gain**  
   - Define the English truth direction as a contrast mean:  
     \[
     w_{\text{eng}} \propto \mathbb{E}[E_{\text{true}} - E_{\text{false}}]
     \]
   - Define the “French oracle” direction similarly:  
     \[
     w^*_{\text{fr}} \propto \mathbb{E}[F_{\text{true}} - F_{\text{false}}]
     \]
   - Compare direction similarity **before** and **after** rotating the English direction:  
     \[
     \Delta_{\cos}(k) = \cos(W^\top w_{\text{eng}}, w^*_{\text{fr}}) - \cos(w_{\text{eng}}, w^*_{\text{fr}})
     \]

3. **French truth-separation margin** (continuous, preferred metric)  
   - For any direction \(w\), define a pairwise margin on French test pairs:  
     \[
       m_{\text{fr}}(w) = \mathbb{E}\big[(F_{\text{true}} - F_{\text{false}})\cdot w\big]
     \]
   - We plot margins for:
     - \(w_{\text{eng}}\) (unaligned transfer baseline)
     - \(W^\top w_{\text{eng}}\) (aligned transfer)
     - \(w^*_{\text{fr}}\) (French upper bound)

4. **French accuracy gain** (thresholded metric; noisier)  
   - Pairwise accuracy: fraction of test pairs where score(true) > score(false).
   - We plot aligned − unaligned.

All plots include uncertainty bands across multiple random seeds/splits (CI shown in filenames).

---

## 2) Main results on the full dataset (base setting)

### 2.1 Cross-lingual alignment becomes easier in later layers
**Observation:** EN→FR Procrustes alignment gains are near zero in early layers and grow steadily toward late layers, peaking around the final blocks (≈ layers 29–30).

**Interpretation:** The later layers produce representations that are more “coordinate‑system compatible” across languages under a linear orthogonal mapping.

**Figure:**  
- `outputs/figures/fig_align_delta.png`  
- (also appears in `outputs/figures_byrel_ci95/fig_align_delta.png`)

### 2.2 The truth-direction becomes increasingly transferable in later layers
**Observation:** The cosine similarity between the *rotated* English truth direction and the French “oracle” direction improves with depth (Δ cosine grows toward late layers).

**Interpretation:** The truth-separating feature is not purely language‑specific; at least part of it behaves like the *same geometric object* expressed in two different coordinate systems (EN vs FR), especially late in the network.

**Figure:**  
- `outputs/figures/fig_cos_gain.png`  
- (also in `outputs/figures_byrel_ci95/fig_cos_gain.png`)

### 2.3 Truth-separation in French improves (margin gain is the clean headline)
**Observation:** French truth-separation margin gain (aligned − unaligned) increases with depth, reaching its largest values in the final layers (roughly ~0.15–0.17 by the last layer).

**Interpretation:** Rotating the English truth direction using the learned alignment \(W\) improves French truth separation in a smooth, measurable way.

**Figure:**  
- `outputs/figures/fig_margin_gain.png`  
- (also in `outputs/figures_byrel_ci95/fig_margin_gain.png`)

### 2.4 Accuracy gain is small/noisy (expected with small test sets)
**Observation:** French accuracy gain is close to zero with wide uncertainty. It does not mirror the clean improvements seen in margins and direction cosines.

**Interpretation:** Accuracy is a **thresholded** metric. With ~40 held‑out test pairs per split (for N=200, test_frac=0.2), accuracy changes in coarse steps (≈ 0.025). Margin improvements can be real without flipping many comparisons.

**Figure:**  
- `outputs/figures/fig_acc_gain.png`  
- (also in `outputs/figures_byrel_ci95/fig_acc_gain.png`)

### 2.5 The three‑curve “margins” plot shows the end‑to‑end story
**Observation:** In French:
- \(m_{\text{fr}}(W^\top w_{\text{eng}})\) (aligned) grows above \(m_{\text{fr}}(w_{\text{eng}})\) (unaligned), especially in late layers.
- \(m_{\text{fr}}(w^*_{\text{fr}})\) stays higher than both (as expected: it is a French‑trained upper bound).

**Figure:**  
- `outputs/figures/fig_fr_margins.png`  
- (also `outputs/figures_byrel_ci95/fig_fr_margins_base.png`)

---

## 3) Per‑relation breakdown (heterogeneous effects)

We split results by Wikidata relation type. The overall positive effect is not uniform.

**Figures (margin gain by relation):**
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_birthplace.png`
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_capital.png`
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_atomic_number.png`
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_director.png`

### 3.1 Birthplace drives large improvements
**Observation:** Birthplace shows a strong positive margin-gain curve that grows sharply in later layers (largest of all relations).

**Interpretation:** For this relation, the “truth geometry” appears especially transferable after alignment.

### 3.2 Capital shows moderate late‑layer gains
**Observation:** Capital exhibits smaller gains than birthplace but becomes clearly positive in late layers.

### 3.3 Atomic number and director show weak/negative late‑layer behavior
**Observation:** Atomic_number rises slightly then turns negative very late; director is near‑zero early and trends negative/noisy late.

**Interpretation:** Transferability depends on the “type” of fact. Numeric relations and some name‑heavy relations may behave differently under the same alignment procedure.

A useful sanity check: the overall dataset margin gain near the final layer is consistent with the approximate average of the four per‑relation curves (birthplace and capital positive, atomic_number and director slightly negative).

---

## 4) Negative control: shuffled pairing (base vs shuffled)

To verify that improvements require *correct EN↔FR pairing* (not just “any rotation”), we re‑learn \(W\) after shuffling the EN/FR pairings in training.

**Key outcome:** the shuffled condition does **not** reproduce the base improvements; it often becomes **strongly negative**, indicating the mapping is meaningful and pairing‑dependent.

**Figures (base vs shuffled):**
- `outputs/figures_compare_ci95/fig_align_delta.png`
- `outputs/figures_compare_ci95/fig_cos_gain.png`
- `outputs/figures_compare_ci95/fig_margin_gain.png`
- `outputs/figures_compare_ci95/fig_acc_gain.png`
- `outputs/figures_compare_ci95/fig_fr_margins_base.png`
- `outputs/figures_compare_ci95/fig_fr_margins_shuffled.png`

### 4.1 Shuffled alignment gain is strongly negative
**Observation:** While base alignment gains are positive (especially late), shuffled alignment gains are large negative values.

**Interpretation:** The method is not “free lunch.” Learning \(W\) from incorrect correspondences makes the real paired vectors less aligned.

### 4.2 Shuffled truth-direction transfer collapses (cosine + margin)
**Observation:** In the shuffled condition, truth-direction cosine “gain” becomes negative, and margin gain becomes negative across much of the network (especially late layers).

**Interpretation:** The transfer effect depends on learning a *real correspondence* between EN and FR representations. This is a strong causal control in favor of the base interpretation.

---

## 5) Relation holdout (generalization test)

We also ran a “holdout relation” setting: learn on a subset of relations and evaluate on a held‑out relation type.

**Figures (holdout):**
- `outputs/figures_holdout_ci95/fig_align_delta.png`
- `outputs/figures_holdout_ci95/fig_cos_gain.png`
- `outputs/figures_holdout_ci95/fig_margin_gain.png`
- `outputs/figures_holdout_ci95/fig_acc_gain.png`
- `outputs/figures_holdout_ci95/fig_fr_margins_holdout.png`

### 5.1 Holdout breaks margin transfer (near-zero gains)
**Observation:** Holdout margin gain is approximately zero (small fluctuations), and the holdout margins plot shows the aligned and unaligned curves largely overlap.

**Interpretation:** The truth-direction transfer effect is **not fully relation‑agnostic** in this setting. The learned \(W\) and/or learned truth direction may be partially relation‑dependent.

### 5.2 Holdout alignment gains can become negative
**Observation:** Unlike the base setting, the holdout EN→FR alignment gain trends negative across much of the depth.

**Interpretation:** This suggests that “one global rotation” may not generalize across semantic domains, or that the held‑out relation occupies a different subspace / region of representation space.

### 5.3 Cosine gain remains positive (but doesn’t translate to margin gains)
**Observation:** Truth-direction cosine gain can still increase with depth in holdout, even though margin gain is near zero.

**Interpretation:** Direction similarity alone may not guarantee improved separation on a new relation; the relevant geometry for classification may depend on relation-specific structure beyond a single direction.

---

## 6) Notes / caveats for interpretation (important)

- **Layer 0 spikes:** Several plots show a layer‑0 spike followed by near‑zero at layer 1. This likely reflects a representation indexing boundary (embedding / pre‑block vs post‑block). For claims, focus on layers 1–31 or define the exact hidden-state indexing explicitly in Methods.
- **Accuracy is low-resolution at N=200:** With test sets of ~40 pairs per split, accuracy is coarse and noisy. Margin is the preferred metric here.
- **Not “universal truth geometry” yet:** The holdout results suggest limits. A more cautious claim is:  
  **Within the same relation distribution, the truth direction is partially transferable across languages via a linear alignment—especially in late layers.**

---

## 7) Suggested “mentor questions” (what feedback we want)

1. Are the chosen metrics and controls sufficient to support the base claim?
2. How should we interpret the relation heterogeneity (e.g., numeric vs named entities)?
3. Is the holdout setting best framed as a limitation, or as an additional result about domain-specific geometry?
4. What is the cleanest figure set for the paper (likely: alignment gain, cosine gain, margin gain, compare base vs shuffled, and one per‑relation panel)?

---

## 8) Reproduction (high-level)

Exact commands depend on your script names/paths, but the intended flow is:

1. Build dataset: `scripts/make-dataset.py` → `data/wikidata_enfr_facts_v1.jsonl`
2. Extract vectors: `scripts/extract-vectors-wikidata.py` → `outputs/*_vecs.npz`
3. Run transfers across seeds/layers: `scripts/procrustes-truth-transfer.py` → `outputs/runs_*.jsonl`
4. Aggregate + plot: `scripts/aggregate_plot.py` → `outputs/figures*/` and `outputs/agg_*.csv`

(If you want, I can tailor this section to match your exact CLI flags and filenames so it’s copy‑paste runnable.)

---

### Figure index (quick list)

**Core (base):**
- `outputs/figures/fig_align_delta.png`
- `outputs/figures/fig_cos_gain.png`
- `outputs/figures/fig_fr_margins.png`
- `outputs/figures/fig_margin_gain.png`
- `outputs/figures/fig_acc_gain.png`

**Per relation:**
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_birthplace.png`
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_capital.png`
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_atomic_number.png`
- `outputs/figures_byrel_ci95/by_relation/fig_margin_gain_director.png`

**Compare (base vs shuffled):**
- `outputs/figures_compare_ci95/fig_align_delta.png`
- `outputs/figures_compare_ci95/fig_cos_gain.png`
- `outputs/figures_compare_ci95/fig_margin_gain.png`
- `outputs/figures_compare_ci95/fig_acc_gain.png`
- `outputs/figures_compare_ci95/fig_fr_margins_base.png`
- `outputs/figures_compare_ci95/fig_fr_margins_shuffled.png`

**Holdout:**
- `outputs/figures_holdout_ci95/fig_align_delta.png`
- `outputs/figures_holdout_ci95/fig_cos_gain.png`
- `outputs/figures_holdout_ci95/fig_margin_gain.png`
- `outputs/figures_holdout_ci95/fig_acc_gain.png`
- `outputs/figures_holdout_ci95/fig_fr_margins_holdout.png`
