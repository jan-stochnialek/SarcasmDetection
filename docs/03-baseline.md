# 03 — TF-IDF baseline (`baseline.py`)

The baseline ([`../project/baseline.py`](../project/baseline.py)) is a classical
text classifier: turn each comment into a **TF-IDF** vector, then fit a linear
classifier. It is fast, interpretable, and runs on a CPU / M1 Air in minutes.

## Why have a baseline at all?

- It is a **lower bound**: a transformer is only worth its cost if it clearly
  beats this.
- It is **cheap and reproducible**, so you can iterate on the data pipeline
  quickly before touching a GPU.
- It has its **own context switch**, so you can check whether thread context helps
  even a simple bag-of-words model — a useful comparison point for RQ2.

## What TF-IDF does

`TfidfVectorizer` converts text to a sparse vector where each dimension is a word
n-gram, weighted by:

- **TF** (term frequency) — how often the term appears in the comment, and
- **IDF** (inverse document frequency) — down-weighting terms common across all
  comments.

Configured (see `BaselineConfig`) with:

| Setting | Value | Meaning |
|---------|-------|---------|
| `ngram_range` | (1, 2) | word uni- and bi-grams (captures short phrases) |
| `min_df` | 2 | ignore terms in fewer than 2 documents (drops noise) |
| `max_features` | 100k | cap vocabulary size for speed/memory |
| `sublinear_tf` | True | use `1 + log(tf)` so very frequent terms don't dominate |

## The classifier

Two choices via `BaselineConfig.classifier`:

- `logreg` (**default**) — `LogisticRegression`. Fast and gives probabilities
  (needed for ROC-AUC).
- `svm` — `LinearSVC`, often strong on text. It has no native probabilities, so we
  wrap it in `CalibratedClassifierCV` to recover them.

The vectorizer and classifier are bundled in a scikit-learn `Pipeline`, which
ensures the vocabulary learned on **train** is the one applied to **test** — no
leakage.

## Context variant

- no context → vectorize `comment`.
- with context → vectorize `"parent [SEP] comment"` (built by
  `build_text_single`).

## Run it

```bash
cd project
python run.py baseline                 # comment only
python run.py baseline --context       # parent + comment
python run.py baseline --classifier svm
```

Outputs (in `../results/`):
- `baseline_logreg_noctx_metrics.json` — accuracy / precision / recall / F1 / AUC
- `baseline_logreg_noctx_cm.png` — confusion matrix

Compare the `noctx` vs `ctx` JSONs to see whether context moves the numbers for
the classical model.

## What to expect

On balanced SARC, bag-of-words baselines land well below typical text-classification
scores — sarcasm is subtle and the labels are self-annotated (noisy). Treat the
baseline number as the bar the transformers must clear, not as a strong result.
