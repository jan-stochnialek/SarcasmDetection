# 01 — Setup

## 1. Python environment

Use Python 3.10 or newer. Create an isolated virtual environment so the project's
packages don't clash with anything else on your machine:

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download the dataset

The data comes from the SARC corpus, packaged on Kaggle as
[`danofer/sarcasm`](https://www.kaggle.com/datasets/danofer/sarcasm).

1. Download the dataset (web download, or `kaggle datasets download -d danofer/sarcasm`).
2. Unzip it and place the balanced file here:

```
data/raw/train-balanced-sarcasm.csv
```

That single file (~1M rows, 50/50 balanced) is all the code needs. Its relevant
columns are `label`, `comment`, and `parent_comment` (the immediate context).

## 3. Verify everything works

```bash
cd project
python run.py prepare
```

You should see the cleaned row count and three lines reporting the size and
sarcastic fraction (~0.5) of the train / val / test splits. If that runs, the
data pipeline is healthy.

## 4. Hardware notes

The code auto-detects the best device in `config.get_device()`:

| Device | When | Notes |
|--------|------|-------|
| **CUDA** | NVIDIA GPU (Colab, Kaggle) | Fastest; mixed-precision (fp16) is enabled automatically. |
| **MPS** | Apple Silicon (M1/M2/M3) | Works for fine-tuning, but slower and trains in fp32. |
| **CPU** | fallback | Fine for the TF-IDF baseline; very slow for transformers. |

### Running on an M1 Air (recommended hybrid workflow)

- The **baseline** (`python run.py baseline`) and **data prep** run comfortably on
  the M1 — minutes, no GPU needed.
- **Transformer fine-tuning** runs on the M1's MPS backend, but it is slow and
  memory-bound. Use it to *debug* on a tiny subset, then run the real experiments
  on a free cloud GPU.

To debug a fine-tune quickly on the M1, shrink the data and epochs. Edit
`config.DataConfig.subset_size` to e.g. `3000`, then:

```bash
python run.py train --model bert-base-uncased --epochs 1 --batch-size 8
```

If you hit an "operator not implemented for MPS" error, allow a CPU fallback for
that op:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python run.py train --model bert-base-uncased
```

On an **8 GB** M1, keep `--batch-size` small (8) and `max_length` at 128. For the
full experiment matrix (both models × context × 3 seeds), prefer Colab/Kaggle —
see [06-usage.md](06-usage.md).

### Running on Colab / Kaggle

Upload the `project/` folder (or clone it), `pip install -r requirements.txt`, put
the CSV where `config.RAW_CSV` expects it, and run the same `python run.py ...`
commands. Kaggle is convenient because the dataset is already hosted there.
