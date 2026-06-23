# 04 — Transformer fine-tuning (`train.py`)

[`../project/train.py`](../project/train.py) fine-tunes a pretrained transformer
(BERT or RoBERTa) for binary sarcasm classification, with or without thread
context. It uses the Hugging Face `Trainer`, so the training loop, evaluation,
mixed precision and checkpointing are handled for us.

## The key idea: change ONLY the input

Both transformer conditions share the same model, optimizer, schedule and seeds.
The **only** difference is how the input is built (`use_context`):

- **comment only** → `[CLS] comment [SEP]`
- **with context** → `[CLS] parent [SEP] comment [SEP]`

Because everything else is held constant, any performance gap is attributable to
context — which is exactly how we answer **RQ2**.

## BERT vs RoBERTa (and why the input format differs slightly)

| | BERT (`bert-base-uncased`) | RoBERTa (`roberta-base`) |
|---|---|---|
| Pretraining | MLM + Next-Sentence-Prediction | MLM only, more data, dynamic masking |
| Sentence pairs | uses segment ids (`token_type_ids`) | one stream with `</s>` separators |
| In our code | tokenizer returns `token_type_ids` | no `token_type_ids` — expected |

You select the model with `--model`; the tokenizer handles each format
automatically, so the same code path serves both.

## Walkthrough of `train_one()`

1. **Seed** everything (`set_seed`) for reproducibility.
2. **Detect device** (CUDA → MPS → CPU).
3. **Build data** with the shared pipeline (`load_and_clean`, `make_splits`).
4. **Tokenize** via `build_tokenized_datasets` (context handled here).
5. **Load model**: `AutoModelForSequenceClassification` adds a fresh 2-class head
   on top of the pretrained encoder.
6. **TrainingArguments**: see hyperparameters below. `load_best_model_at_end=True`
   with `metric_for_best_model="f1"` keeps the best epoch by validation F1 (a
   simple form of early stopping).
7. **Train**, then **predict on the held-out test split**.
8. **Save** test metrics (`*_metrics.json`) and per-example predictions
   (`*_preds.npz`, needed for the McNemar test).

## Hyperparameters (`TransformerConfig`)

| Hyperparameter | Default | Notes |
|---|---|---|
| `max_length` | 128 (no ctx) / 256 (ctx) | `run.py` raises it automatically when `--context` is set |
| `batch_size` | 16 | lower to 8 on an 8 GB M1 |
| `learning_rate` | 2e-5 | 2e-5 / 3e-5 are standard for BERT fine-tuning |
| `epochs` | 3 | 2–4 with best-epoch selection on val F1 |
| `weight_decay` | 0.01 | mild regularization |
| `warmup_ratio` | 0.1 | linear warmup then decay |
| `seed` | 42 | vary (e.g. 13, 42, 123) to measure variance |

`fp16` mixed precision is turned on **only** on CUDA (it is unstable on MPS), so
on a Mac the model trains in fp32 automatically.

## Run it

```bash
cd project
python run.py train --model bert-base-uncased            # BERT, no context
python run.py train --model bert-base-uncased --context  # BERT, with context
python run.py train --model roberta-base                 # RoBERTa, no context
python run.py train --model roberta-base --context       # RoBERTa, with context
```

Vary the seed to get error bars:

```bash
python run.py train --model roberta-base --context --seed 13
python run.py train --model roberta-base --context --seed 123
```

## Why multiple seeds matter

The context effect is expected to be **small**. Fine-tuning is stochastic, so a
single run's difference could be noise. Running ≥3 seeds and reporting
**mean ± std** (plus the McNemar test in [05-evaluation.md](05-evaluation.md))
separates a real effect from luck.
