# Sarcasm Detection in Social Media using Transformer-based Language Models

> University course project plan
> Author: Jan Stochniałek
> Last updated: 2026-06-21

---

## 1. Overview

Sarcasm is pervasive in social media and is a well-known failure mode for sentiment
analysis, opinion mining, and content-moderation systems: the literal meaning of a
sarcastic comment is the opposite of its intended meaning. This project builds and
evaluates automatic sarcasm classifiers for Reddit comments, and tests a single focused
research question: **does adding conversation-thread context improve sarcasm detection
compared to using the target comment alone?**

We compare three families of models on the same data and metrics:

1. **TF-IDF + linear classifier** — a strong, cheap, classical baseline.
2. **Transformer (BERT / RoBERTa), comment-only** — fine-tuned on the target comment.
3. **Transformer (BERT / RoBERTa), comment + context** — fine-tuned on the parent
   comment(s) concatenated with the target comment.

The contribution is not a new architecture but a **controlled empirical comparison**:
classical vs. transformer, and context vs. no-context, holding data and evaluation fixed.

---

## 2. Research questions and hypotheses

- **RQ1 — Transformers vs. classical baseline.** Do fine-tuned transformer models
  outperform a TF-IDF baseline on balanced SARC?
  *Hypothesis H1:* yes, transformers improve over TF-IDF, but the margin on balanced
  SARC is moderate (sarcasm is subtle and self-annotated, so absolute scores stay well
  below typical text-classification numbers).

- **RQ2 — Does context help? (main question)** Does conditioning on the parent comment
  improve classification of the target comment?
  *Hypothesis H2:* yes — context resolves cases where sarcasm only emerges relative to
  what was said before (e.g. a literal-sounding reply that contradicts the parent).
  The SARC paper itself notes that human annotators are more accurate when given context,
  motivating this hypothesis.

- **RQ3 — BERT vs. RoBERTa.** Does RoBERTa's improved pretraining (more data, dynamic
  masking, no NSP objective) translate into better sarcasm detection?
  *Hypothesis H3:* RoBERTa ≥ BERT on the same setup, consistent with the RoBERTa paper's
  reported gains on GLUE/SQuAD/RACE.

---

## 3. Background and related work

The three reference papers map onto the three pillars of the project.

### 3.1 SARC — the dataset (Khodak, Saunshi & Vodrahalli, 2018)
- **Self-Annotated Reddit Corpus.** Sarcasm labels come from the Reddit convention of
  appending the marker **`/s`** to a comment to flag it as sarcastic. This yields a very
  large corpus with *author-provided* (not third-party) labels.
- Order of **hundreds of millions** of comments overall, with on the order of a **million
  sarcastic** comments; the authors release **balanced** and **unbalanced** variants plus
  a **politics (`pol`)** subset.
- Each example carries **conversation context**: the chain of ancestor comments leading to
  the target, which is exactly what RQ2 needs.
- Reported baselines include **Bag-of-Words** and **Bag-of-Bigrams**; **human accuracy**
  is higher *with* context than without, and machine baselines trail humans — establishing
  headroom and motivating both transformer models and the use of context.
- **Caveats to discuss in the report:** label noise (not every sarcastic comment is tagged
  `/s`; `/s` is more common in some subreddits/communities), self-selection bias, and the
  fact that `/s` is stripped from the text before labeling so the model cannot cheat off
  the marker.

### 3.2 BERT — the model (Devlin et al., 2019)
- Bidirectional Transformer encoder pretrained with **Masked Language Modeling** and
  **Next Sentence Prediction**.
- Fine-tuning for classification: feed the sequence, take the **`[CLS]`** representation,
  add a linear + softmax head.
- Native **sentence-pair** input format — `[CLS] A [SEP] B [SEP]` with segment embeddings —
  which maps naturally onto our context setup: **A = parent/context, B = target comment**.

### 3.3 RoBERTa — the improved model (Liu et al., 2019)
- Same architecture as BERT, **better pretraining recipe**: more data, longer training,
  larger batches, **dynamic masking**, **removal of the NSP objective**, and byte-level BPE.
- Generally outperforms BERT on downstream benchmarks.
- *Practical note:* RoBERTa has **no NSP** and no segment-embedding distinction between
  sentences; pairs are joined with the separator (`</s>`). Context still works — it is just
  encoded as one sequence with a separator rather than via segment IDs.

### 3.4 How this project differs
We move SARC from classical bag-of-words baselines to modern fine-tuned transformers, and
we isolate the **context effect** with an otherwise-identical experimental setup.

---

## 4. Dataset

- **Source:** SARC via Kaggle — `danofer/sarcasm`
  (https://www.kaggle.com/datasets/danofer/sarcasm). Primary file:
  **`train-balanced-sarcasm.csv`** (~1.01M rows, label-balanced 50/50).
- **Key columns:** `label` (1 = sarcastic, 0 = not), `comment` (target),
  `parent_comment` (immediate thread context), plus metadata
  (`author`, `subreddit`, `score`, `ups`, `downs`, `date`, `created_utc`).
- **Context available here:** the Kaggle balanced file provides the **immediate parent
  comment**. (The full SARC release on the authors' site provides the *full ancestor
  chain*; if multi-turn context becomes a research goal, switch to that release. For this
  project, parent-comment context is the primary setting and is sufficient to answer RQ2.)

### 4.1 Splits
- Use a **stratified** split preserving the 50/50 label balance:
  **train 80% / validation 10% / test 10%**, fixed `random_state` for reproducibility.
- **Group-aware splitting (recommended):** avoid leakage by ensuring the same
  `author`/thread does not appear in both train and test where feasible; at minimum,
  deduplicate identical comments.
- Report final split sizes in a table in the report.

### 4.2 Subsampling for compute
Fine-tuning a transformer on ~1M rows on free GPUs (Colab/Kaggle) is slow. Plan:
- **Development subset:** ~100k–200k rows (stratified) for the main experiments.
- **Scaling check (optional):** if time permits, re-run the best config on the full set to
  confirm the trend holds.
- Keep the **validation and test sets fixed** across all models so every comparison is
  apples-to-apples.

### 4.3 Preprocessing
- Drop rows with empty/NaN `comment` or `parent_comment`.
- Light normalization only (transformers handle raw text well): collapse whitespace,
  optionally normalize URLs → `[URL]` and user mentions → `[USER]`.
- **Do not** remove stopwords or lowercase for the transformer models (use the model's own
  tokenizer / cased-or-uncased as appropriate). For the TF-IDF baseline, standard
  lowercasing + stopword handling is fine.
- Verify the `/s` marker is **not** present in the comment text (SARC strips it); if any
  leaks exist, remove them so the model cannot cheat.

---

## 5. Methodology

Three model conditions, evaluated identically.

### 5.1 Baseline — TF-IDF + Logistic Regression / Linear SVM
- `TfidfVectorizer` with word **uni- + bi-grams**, `min_df` tuning, optional
  `sublinear_tf=True`.
- Classifier: **Logistic Regression** (primary) and/or **Linear SVM**; tune `C` on the
  validation set.
- Two input variants to mirror the transformer conditions:
  - **comment-only:** vectorize `comment`.
  - **comment + context:** vectorize `parent_comment + " " + comment` (or concatenated
    TF-IDF feature blocks).
- Purpose: cheap, interpretable lower bound; also tells us whether context helps even a
  bag-of-words model.

### 5.2 Transformer, comment-only
- Models: **`bert-base-uncased`** and **`roberta-base`** (Hugging Face).
- Input: `[CLS] comment [SEP]` (BERT) / `<s> comment </s>` (RoBERTa).
- `AutoModelForSequenceClassification` with a 2-class head; fine-tune end-to-end.

### 5.3 Transformer, comment + context (main contribution)
- Input as a **sentence pair**: `parent_comment` as sequence A, `comment` as sequence B.
  - BERT: `[CLS] parent [SEP] comment [SEP]` (uses segment embeddings).
  - RoBERTa: `<s> parent </s></s> comment </s>` (single stream + separators).
  - The tokenizer handles this automatically when you pass **two text fields**
    (`tokenizer(parent, comment, ...)`) with `truncation=True`.
- **Truncation strategy matters:** the *target comment* is what we classify, so prefer
  `truncation="only_first"` (truncate the context, keep the comment intact) or cap context
  length explicitly. Document the choice.
- Everything else (head, optimizer, schedule, seeds) identical to 5.2 so the **only**
  difference is the presence of context → clean answer to RQ2.

### 5.4 Controlled-comparison discipline
- Same train/val/test rows, same seeds, same max length budget, same epochs/LR per model
  family. Only the **input construction** changes between 5.2 and 5.3.
- Run each transformer condition with **≥3 seeds** and report **mean ± std** so that the
  context effect is not confused with seed noise.

---

## 6. Experimental setup

### 6.1 Hyperparameters (starting points, from the BERT/RoBERTa fine-tuning recipes)
| Hyperparameter | Value(s) to try |
|---|---|
| Max sequence length | 128 (comment-only), 192–256 (with context) |
| Batch size | 16 or 32 (use grad accumulation if GPU memory limited) |
| Learning rate | 2e-5, 3e-5 (AdamW) |
| Epochs | 2–4 (early stop on val F1) |
| Weight decay | 0.01 |
| Warmup | 6–10% of steps, linear decay |
| Precision | fp16/bf16 mixed precision for speed |
| Seeds | 3 (e.g. 13, 42, 123) |

### 6.2 Compute
- **Google Colab** or **Kaggle Notebooks** free GPU (T4). Kaggle is convenient since the
  dataset already lives there.
- Save checkpoints/metrics to Drive or as Kaggle outputs so runs are recoverable.
- Track experiments with a simple results CSV (or Weights & Biases if desired).

### 6.3 Evaluation metrics
Because the dataset is balanced, **accuracy** is meaningful, but report the full set:
- **Accuracy**, **Precision**, **Recall**, **F1** (binary, sarcastic = positive class),
  and **Macro-F1**.
- **ROC-AUC** / **PR-AUC** from predicted probabilities.
- **Confusion matrix** for error analysis.
- **Statistical significance:** McNemar's test (or bootstrap CIs over the test set) when
  comparing context vs. no-context for the same model, so the RQ2 claim is defensible.

### 6.4 Primary comparison table (to fill in)
| Model | Context | Acc | P | R | F1 | AUC |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | no | | | | | |
| TF-IDF + LogReg | yes | | | | | |
| BERT-base | no | | | | | |
| BERT-base | yes | | | | | |
| RoBERTa-base | no | | | | | |
| RoBERTa-base | yes | | | | | |

The headline result is the **within-model context delta** (yes − no) for BERT and RoBERTa.

---

## 7. Error analysis & interpretation

- **Where context helps:** sample comments the comment-only model gets wrong but the
  context model gets right; categorize them (e.g. literal-looking replies, callbacks,
  agreement-flips). This is the qualitative evidence for RQ2.
- **Where it hurts/doesn't help:** cases where long/irrelevant context dilutes the signal
  or causes truncation of the comment.
- **Subreddit / length effects:** does context help more in conversational subreddits?
  Does performance drop on very short comments?
- **Interpretability (optional):** attention or integrated-gradients visualization on a
  few examples to show the context model attending to the parent.

---

## 8. Project structure

The code is organised for a beginner: one small script per action (no command-line
flags), one `settings.py` for the knobs, and the shared logic in `engine/`.

```
EmiliaProject/
├── plan.md                       # this file (research plan)
├── data/raw/                     # train-balanced-sarcasm.csv from Kaggle
├── results/                      # scores saved here after each run
├── docs/                         # README, how-it-works, kaggle guide
└── project/
    ├── README.md                 # how to install and run
    ├── requirements.txt
    ├── settings.py               # the only knobs (sample size, epochs, batch)
    ├── check_data.py             # 1. load + check the data
    ├── train_baseline.py             /  train_baseline_context.py    # TF-IDF
    ├── train_bert.py                 /  train_bert_context.py        # BERT
    ├── train_roberta.py              /  train_roberta_context.py     # RoBERTa
    ├── show_results.py           # print the comparison table
    ├── run_everything.py         # run all of the above in order
    ├── notebooks/                # ready-made Colab + Kaggle notebooks
    └── engine/                   # shared code the scripts call (not run directly)
        ├── data.py               # load, clean, split
        ├── baseline.py           # TF-IDF + Logistic Regression
        ├── transformer.py        # fine-tune BERT / RoBERTa
        └── scoring.py            # metrics + significance + results table
```

### 8.1 Core dependencies
```
python==3.12          # 3.13+/3.14 don't have PyTorch wheels yet
torch
transformers>=4.46    # needs the processing_class API
datasets
scikit-learn
pandas, numpy, scipy
matplotlib
accelerate            # backs the Hugging Face Trainer
```

---

## 9. Timeline / milestones

Adjust to the course deadline; this assumes ~6 working weeks.

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Setup + data acquisition + EDA | Repo skeleton, cleaned data, EDA notebook, split stats |
| 2 | TF-IDF baselines (both input variants) | Baseline metrics in results table |
| 3 | Transformer fine-tuning pipeline (comment-only) | Working `train.py`, BERT & RoBERTa no-context results |
| 4 | Context condition + multi-seed runs | Full comparison table with mean±std |
| 5 | Error analysis, significance tests, plots | Analysis notebook, figures, RQ answers |
| 6 | Write-up + slides + buffer | Final report + presentation |

**Critical path:** getting the fine-tuning pipeline working on free GPU (Week 3). De-risk
early by running a tiny subset end-to-end in Week 1.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Full dataset too large for free GPU | Use stratified ~100–200k subset; fix val/test |
| Context truncates the target comment | `truncation="only_first"`; cap context tokens; log truncation rate |
| Label noise (`/s` self-annotation) | Acknowledge in report; rely on balanced split; error analysis |
| Data leakage (same author/thread in train & test) | Group-aware split; dedupe comments |
| Seed variance hides the context effect | ≥3 seeds, report mean±std, significance test |
| Colab timeouts / lost runs | Checkpoint to Drive/Kaggle outputs; resumable Trainer |
| Overfitting on subtle signal | Early stopping on val F1, weight decay, 2–4 epochs only |

---

## 11. Deliverables

1. **`plan.md`** (this document).
2. **Reproducible code** (`project/` + notebooks) with a `README` describing how to run each
   stage.
3. **Results table** comparing TF-IDF / BERT / RoBERTa × {no-context, context} on a fixed
   test set, with metrics and significance tests.
4. **Error analysis** with concrete examples illustrating where context changes the
   prediction.
5. **Final report** answering RQ1–RQ3, plus a short **presentation**.

---

## 12. References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for
  Sarcasm.* LREC 2018. (Paper: L18-1102)
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019. (Paper: N19-1423)
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M.,
  Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining
  Approach.* arXiv:1907.11692.
- Dataset: SARC on Kaggle — https://www.kaggle.com/datasets/danofer/sarcasm
- Library: Hugging Face Transformers — https://github.com/huggingface/transformers
