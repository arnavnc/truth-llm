# TRUTH-LLM: Cross-Lingual Truth Geometry via Linear Alignment (MLX)

This repo is a small, reproducible research project studying whether a **“truth direction”** learned from English internal representations in a dense LLM can be **mapped to French** using a simple geometric transformation.

We treat hidden states as vectors. For each factual statement, we compare a **true** vs **false** version and measure whether there exists a direction in activation space that separates them. Then we ask:

> If English and French are the “same meaning” expressed in different token systems, can we align their internal vector spaces with a rotation and transfer the truth direction?

This project is designed to run on **Apple Silicon** (e.g., M1 Pro) using the **MLX** stack.

---

## Core idea (high level)

1. **Representations as vectors:**  
   For a sentence `x`, the model produces a hidden state at each layer (a high-dimensional vector).

2. **True/false contrast:**  
   We build minimal pairs `(true_sentence, false_sentence)` and look at the difference in hidden states:
   \\[
   \\Delta(x) = h(\\text{true}) - h(\\text{false})
   \\]
   Averaging these differences yields a “truth direction”:
   \\[
   w \\propto \\mathbb{E}[\\Delta(x)]
   \\]

3. **Cross-lingual alignment:**  
   We learn an **orthogonal Procrustes** map (a rotation) \\(W\\) that aligns English and French representations:
   \\[
   E W \\approx F
   \\]
   If this is true, the corresponding weight/direction transfer is:
   \\[
   \\hat w_{fr} = W^T w_{eng}
   \\]

4. **Evaluate transfer:**  
   We test whether \\(\\hat w_{fr}\\) separates French true vs false examples better than using \\(w_{eng}\\) directly.

---

## Model

We use a dense open-weights model compatible with MLX:

- `mlx-community/Mistral-7B-v0.3-4bit`

(Using a dense model is intentional: it provides a stable baseline before attempting MoE routing complications.)

---

## Dataset

The default dataset is built from Wikidata triples and converted into bilingual minimal pairs:

- 4 relations (balanced):
  - `capital`
  - `birthplace`
  - `director`
  - `atomic_number`
- 200 total examples (50 per relation)
- Each example produces:
  - English true + English false
  - French true + French false

Files:
- `data/wikidata_enfr_facts_v1.jsonl` (main dataset)
- `data/capitals_pairs.jsonl` (older/simpler dataset)

---

## Repo structure

```
TRUTH-LLM/
  data/                  # datasets (JSONL)
  scripts/               # runnable pipeline scripts
  outputs/               # generated artifacts (runs, vecs, figures)
    figures/             # plots
  paper/                 # LaTeX draft + bib
```

Key artifacts produced:
- `outputs/wikidata_mistral7b_v0.3_4bit_vecs.npz`  
  Cached hidden-state vectors (E_true/E_false/F_true/F_false) for all layers.
- `outputs/runs_*.jsonl`  
  One row per (seed, layer) with alignment + transfer metrics.
- `outputs/agg_*.csv`  
  Aggregated results across seeds.
- `outputs/figures*/`  
  Plots (alignment gain, cosine gain, margin gain, accuracy gain).

---

## Setup

### Requirements
- Apple Silicon Mac recommended
- Python 3.12 recommended (MLX lens tooling may require it)

### Create an environment
Using conda (example):
```bash
conda create -n truthgeom312 python=3.12 -y
conda activate truthgeom312
```

Install dependencies:
```bash
python -m pip install -U numpy pandas matplotlib
python -m pip install -U mlx mlx-lm mlx-lm-lens
```

---

## Pipeline: reproduce the main figures

### 1) Sanity check: lens + hidden states
```bash
python scripts/lens-sanity.py
```

### 2) Extract vectors for the Wikidata dataset (cached)
This produces the `.npz` used by downstream analysis.
```bash
python scripts/extract-vectors-wikidata.py
```

### 3) Run Procrustes truth transfer across seeds
Baseline:
```bash
OUT_BASE=outputs/runs_mistral7b_wikidata_r256.jsonl
rm -f $OUT_BASE

for s in $(seq 0 19); do
  python scripts/procrustes-truth-transfer.py \
    --layers all --r 256 --seed $s --tag base \
    --stratify_relations \
    --out_jsonl $OUT_BASE
done
```

### 4) Aggregate + plot (95% CI bands)
```bash
python scripts/aggregate_plot.py \
  --runs base:outputs/runs_mistral7b_wikidata_r256.jsonl \
  --out_csv outputs/agg_mistral7b_wikidata_r256_ci95.csv \
  --fig_dir outputs/figures_ci95 \
  --band ci95 \
  --relation __all__
```

---

## Controls (recommended)

### Shuffled pairing control (should destroy the effect)
Fit the rotation \\(W\\) using randomly mismatched EN↔FR pairs.

```bash
OUT_SHUF=outputs/runs_mistral7b_wikidata_r256_shuffled.jsonl
rm -f $OUT_SHUF

for s in $(seq 0 19); do
  python scripts/procrustes-truth-transfer.py \
    --layers all --r 256 --seed $s --tag shuffled \
    --shuffle_train_pairs \
    --stratify_relations \
    --out_jsonl $OUT_SHUF
done
```

Compare base vs shuffled:
```bash
python scripts/aggregate_plot.py \
  --runs base:outputs/runs_mistral7b_wikidata_r256.jsonl \
  --runs shuffled:outputs/runs_mistral7b_wikidata_r256_shuffled.jsonl \
  --out_csv outputs/agg_compare_ci95.csv \
  --fig_dir outputs/figures_compare_ci95 \
  --band ci95 \
  --relation __all__
```

Expected: the shuffled condition collapses toward ~0 gain in alignment/cosine/margin.

---

## Per-relation analysis (optional)

Generate per-relation rows:
```bash
OUT_REL=outputs/runs_mistral7b_wikidata_r256_byrel.jsonl
rm -f $OUT_REL

for s in $(seq 0 19); do
  python scripts/procrustes-truth-transfer.py \
    --layers all --r 256 --seed $s --tag base \
    --emit_by_relation \
    --stratify_relations \
    --out_jsonl $OUT_REL
done
```

Plot per relation:
```bash
python scripts/aggregate_plot.py \
  --runs base:outputs/runs_mistral7b_wikidata_r256_byrel.jsonl \
  --out_csv outputs/agg_byrel_ci95.csv \
  --fig_dir outputs/figures_byrel_ci95 \
  --band ci95 \
  --relation __all__ \
  --per_relation
```

---

## Notes on metrics

We report several metrics layer-by-layer:

- **Alignment gain**: mean cosine similarity improvement after Procrustes:
  \\[
  \\Delta = \\cos(EW, F) - \\cos(E, F)
  \\]

- **Truth-direction cosine gain**: whether the transferred direction better matches the French direction:
  \\[
  \\Delta = \\cos(W^T w_{eng}, w^*_{fr}) - \\cos(w_{eng}, w^*_{fr})
  \\]

- **Margin gain (preferred)**: continuous separation strength in French:
  \\[
  \\Delta = \\mathbb{E}[(f_{true}-f_{false})^\\top (W^T w_{eng})] - \\mathbb{E}[(f_{true}-f_{false})^\\top w_{eng}]
  \\]

- **Accuracy gain (noisy)**: discrete pairwise accuracy difference; can be high variance with small test sets.

---

## Git hygiene (recommended)

See `.gitignore` in the repo root. The default ignores large generated artifacts (runs, cached vectors),
while keeping figures and the paper draft trackable.

---

## Paper draft

The manuscript lives in:
- `paper/draft.tex`
- `paper/ref.bib`

We typically write results after the base + shuffled control is stable.
