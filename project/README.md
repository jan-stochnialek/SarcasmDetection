# Sarcasm Detection — how to run it

This project tests whether a computer can tell if a Reddit comment is **sarcastic**,
and whether showing it the **previous comment** (the "context") helps.

You run it by typing one command at a time. Each command trains one model and
prints its score. There are **no options to remember** — each script does one thing.

---

## First-time setup (do this once)

1. Install **Python 3.12** (newer versions don't work with the AI libraries yet).
2. Open a terminal **in this `project` folder** and run:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the data from Kaggle (https://www.kaggle.com/datasets/danofer/sarcasm)
   and put the file **`train-balanced-sarcasm.csv`** into the `data/raw` folder.

---

## Running it

**Every time** you open a new terminal, first turn on the environment:

```bash
source .venv/bin/activate
```

Then run these one at a time (top to bottom):

```bash
python check_data.py                 # 1. check the data loaded correctly
python train_baseline.py             # 2. simple model, comment only
python train_baseline_context.py     # 3. simple model, with context
python train_bert.py                 # 4. BERT, comment only
python train_bert_context.py         # 5. BERT, with context
python train_roberta.py              # 6. RoBERTa, comment only
python train_roberta_context.py      # 7. RoBERTa, with context
python show_results.py               # 8. print the comparison table
```

Or do all of it with a single command:

```bash
python run_everything.py
```

**Prefer a notebook?** Open `notebooks/local_walkthrough.ipynb` in Jupyter or VS
Code. It walks through the whole project on a small sample, step by step, showing
the data and the results inline — the friendliest way to see how it all works.

> The `train_bert*` and `train_roberta*` scripts are slow without a GPU. If you
> don't have one, run them on Kaggle's free GPU — see `../docs/kaggle.md`.

---

## Changing settings

Open **`settings.py`**. You can change how many comments to use, how long to
train, and the batch size. Every setting is explained in that file. **You don't
need to edit anything else.**

---

## What each file is

| You run these | What it does |
|---|---|
| `check_data.py` | loads and checks the data |
| `train_baseline.py` / `..._context.py` | the simple TF-IDF model |
| `train_bert.py` / `..._context.py` | BERT |
| `train_roberta.py` / `..._context.py` | RoBERTa |
| `show_results.py` | prints the results table |
| `run_everything.py` | runs all of the above in order |
| `settings.py` | the settings you can change |
| `engine/` | the shared code the scripts use (you never run these directly) |

Scores are saved in the `results` folder. For an explanation of what TF-IDF, BERT,
and RoBERTa actually are, see `../docs/how-it-works.md`.
