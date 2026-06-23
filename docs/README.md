# Documentation — Sarcasm Detection on SARC

This folder documents the implementation that lives in [`../project`](../project).
The high-level research plan (motivation, hypotheses, related work, timeline) is
in [`../plan.md`](../plan.md). These docs explain **how the code works and how to
run it**.

## Read in this order

| Doc | What it covers |
|-----|----------------|
| [01-setup.md](01-setup.md) | Install dependencies, download the dataset, hardware (M1 / GPU) notes |
| [02-data.md](02-data.md) | The data pipeline: cleaning, splitting, tokenization, how context is built |
| [03-baseline.md](03-baseline.md) | The TF-IDF + linear classifier baseline |
| [04-transformers.md](04-transformers.md) | Fine-tuning BERT / RoBERTa, with and without context |
| [05-evaluation.md](05-evaluation.md) | Metrics and the McNemar significance test |
| [06-usage.md](06-usage.md) | End-to-end commands, the experiment matrix, and the Colab notebook |
| [kaggle.md](kaggle.md) | Complete step-by-step guide to running on Kaggle's free GPU |

## The three model conditions (recap)

1. **TF-IDF + Logistic Regression / SVM** — classical baseline.
2. **BERT / RoBERTa, comment only** — transformer without context.
3. **BERT / RoBERTa, parent + comment** — transformer *with* thread context.

The project's headline question (RQ2) is whether condition 3 beats condition 2 —
i.e. **does conversation context help?**

## Where things are written

| Path | Contents |
|------|----------|
| `../data/raw/` | the downloaded SARC CSV (you provide this) |
| `../results/`  | metrics JSONs, confusion-matrix PNGs, per-example predictions |
| `../models/`   | fine-tuned transformer checkpoints |
