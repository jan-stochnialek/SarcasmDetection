# 06 — End-to-end usage

This is the recipe for reproducing the full comparison from `plan.md`. Run all
commands from inside the `project/` folder with your virtual environment active.

## 0. One-time setup

See [01-setup.md](01-setup.md): install deps and place
`data/raw/train-balanced-sarcasm.csv`.

```bash
cd project
python run.py prepare        # sanity-check splits and class balance
```

## 1. Baseline (runs on CPU / M1)

```bash
python run.py baseline                 # TF-IDF, comment only
python run.py baseline --context       # TF-IDF, parent + comment
```

## 2. Transformers

Run each model in both conditions. For solid results, repeat with several seeds.

```bash
# BERT
python run.py train --model bert-base-uncased
python run.py train --model bert-base-uncased --context

# RoBERTa
python run.py train --model roberta-base
python run.py train --model roberta-base --context

# Extra seeds for error bars (example for the main config)
python run.py train --model roberta-base --context --seed 13
python run.py train --model roberta-base --context --seed 123
```

## 3. Significance tests

```bash
python run.py compare \
    --a ../results/bert-base-uncased_noctx_seed42_preds.npz \
    --b ../results/bert-base-uncased_ctx_seed42_preds.npz

python run.py compare \
    --a ../results/roberta-base_noctx_seed42_preds.npz \
    --b ../results/roberta-base_ctx_seed42_preds.npz
```

## 4. Build the comparison table

Once you have run some of the configurations above, aggregate every
`results/*_metrics.json` into one table (seeds of the same model+context are
averaged as mean ± std):

```bash
python run.py report
```

This writes `../results/summary.md` (paste straight into plan.md Section 6.4) and
`../results/summary.csv` (one row per run, for your own plots).

## The full experiment matrix

| Model | Context | Command |
|-------|---------|---------|
| TF-IDF | no | `python run.py baseline` |
| TF-IDF | yes | `python run.py baseline --context` |
| BERT | no | `python run.py train --model bert-base-uncased` |
| BERT | yes | `python run.py train --model bert-base-uncased --context` |
| RoBERTa | no | `python run.py train --model roberta-base` |
| RoBERTa | yes | `python run.py train --model roberta-base --context` |

Fill the results table in `plan.md` Section 6.4 from the `*_metrics.json` files in
`../results/`. The headline number is the **within-model context delta** (yes − no)
for BERT and RoBERTa, plus its McNemar p-value.

## Recommended compute split (M1 Air)

| Stage | Where | Why |
|-------|-------|-----|
| `prepare`, `baseline` | **M1 Air** | CPU-bound, minutes |
| Debug a fine-tune (tiny subset, 1 epoch) | **M1 Air (MPS)** | confirm code runs |
| Full transformer matrix (2 models × 2 contexts × 3 seeds) | **Colab / Kaggle GPU** | ~12 runs is too slow on a fanless laptop |

To shrink a run for local debugging, set `DataConfig.subset_size` small (e.g. 3000)
in `config.py`, or pass `--epochs 1 --batch-size 8`. On MPS, prefix with
`PYTORCH_ENABLE_MPS_FALLBACK=1` if you hit an unsupported-operator error.

## Running on Colab (notebook)

A ready-made notebook is provided at
[`../project/notebooks/sarcasm_colab.ipynb`](../project/notebooks/sarcasm_colab.ipynb).
Open it in Colab, select a **T4 GPU** runtime, and run the cells top to bottom: it
gets the code, installs dependencies, downloads SARC via the Kaggle API, then runs
the same `prepare → baseline → train → compare → report` steps and renders the
final comparison table inline.

## Running on Kaggle

Kaggle is the easiest place to run this — the SARC dataset is already hosted there,
and the GPU is free. There is a dedicated, complete walkthrough in
**[kaggle.md](kaggle.md)** (account setup, attaching the dataset, getting the code
in, running every stage, saving outputs, and troubleshooting), plus a ready-made
notebook at
[`../project/notebooks/sarcasm_kaggle.ipynb`](../project/notebooks/sarcasm_kaggle.ipynb).

The one Kaggle-specific detail: point the pipeline at the read-only dataset mount
with the `SARC_CSV` env var (the notebook does this automatically):

```python
import os, glob
os.environ["SARC_CSV"] = glob.glob("/kaggle/input/**/train-balanced-sarcasm.csv", recursive=True)[0]
```

## Adapting to your own notebook (optional)

If your course expects a notebook you wrote, you can import these modules instead
of re-implementing them, e.g. in a cell:

```python
from config import DataConfig, TransformerConfig
from train import train_one
train_one(DataConfig(), TransformerConfig(model_name="roberta-base", use_context=True))
```

That keeps the heavily-commented logic in one place while giving you the
notebook's interactive output.
