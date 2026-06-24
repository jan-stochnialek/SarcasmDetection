# Running this project on Kaggle (free GPU)

Kaggle is the easiest place to run this: it gives you a free GPU, and the SARC
dataset is already hosted there (nothing to download).

There's a ready-made notebook at
[`../project/notebooks/sarcasm_kaggle.ipynb`](../project/notebooks/sarcasm_kaggle.ipynb)
— import it and run it. The steps below explain what it does.

---

## 0. It's free

Kaggle is free (you only need to verify your account with a phone number to unlock
the GPU). You get ~30 hours of GPU per week, sessions up to 12 hours, and ~20 GB of
working space. That's plenty for this project.

---

## 1. Make a GPU notebook

1. <https://www.kaggle.com/code> → **New Notebook**.
2. Right sidebar → **Settings → Accelerator → GPU T4 x2**.

   > ⚠️ **Pick T4, not P100.** Kaggle's PyTorch no longer supports the older P100
   > (`sm_60`); a P100 crashes with *"CUDA error: no kernel image is available"*.
   > The T4 works fine.

---

## 2. Add the dataset

**+ Add Data** → search **"Sarcasm"** → add **danofer/sarcasm**. It appears
read-only at `/kaggle/input/sarcasm/train-balanced-sarcasm.csv`.

---

## 3. Get the code onto Kaggle

The code must live in the **writable** `/kaggle/working/` folder. Pick one:

- **GitHub:** turn on *Settings → Internet*, then
  `!git clone https://github.com/<you>/EmiliaProject.git /kaggle/working/EmiliaProject`
- **Upload as a dataset:** zip the repo, *+ Add Data → Upload*, then
  `!cp -r /kaggle/input/<your-slug>/EmiliaProject /kaggle/working/`

---

## 4. Tell the code where the data is

The project reads the data path from the `SARC_CSV` environment variable. Set it
once (the notebook does this for you):

```python
import os, glob
os.environ["SARC_CSV"] = glob.glob("/kaggle/input/**/train-balanced-sarcasm.csv", recursive=True)[0]
```

---

## 5. Check the libraries

```python
!pip install -q "transformers>=4.46"        # do NOT add -U, it can break the GPU
import torch
print("CUDA:", torch.cuda.is_available())    # should be True
```

---

## 6. Run it

The simplest way — run the whole project with one command:

```python
!python /kaggle/working/EmiliaProject/project/run_everything.py
```

This trains the baseline, BERT, and RoBERTa (each with and without context) and
prints the comparison table at the end.

Prefer one step at a time?

```python
CODE = "/kaggle/working/EmiliaProject/project"
!python {CODE}/check_data.py
!python {CODE}/train_baseline.py
!python {CODE}/train_baseline_context.py
!python {CODE}/train_bert.py
!python {CODE}/train_bert_context.py
!python {CODE}/train_roberta.py
!python {CODE}/train_roberta_context.py
!python {CODE}/show_results.py
```

### Want it faster, or to use more data?

Open **`project/settings.py`** and change:
- `SAMPLE_SIZE` — smaller = faster (e.g. `20000`); set to `None` to use all ~1M
  comments (best results, slower).
- `EPOCHS` — how long each transformer trains (2–3 is normal).

On a T4, expect roughly 15–25 minutes per transformer run with `SAMPLE_SIZE = 50000`.

---

## 7. Save your results

Outputs are written to `/kaggle/working/EmiliaProject/results/`. Use **Save
Version** to keep them — or, for a long run, *Save & Run All (Commit)* runs the
whole notebook in the background so it won't stop at the idle timeout.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA error: no kernel image is available` / `P100 not compatible` | You got a P100 — switch the accelerator to **GPU T4 x2** and re-run. Don't use `pip install -U` on torch. |
| `assert ... CODE_DIR` fails | The clone/copy in step 3 didn't run or used the wrong slug. |
| `Add the danofer/sarcasm dataset...` | You didn't attach the dataset — do step 2. |
| `CUDA: False` | Accelerator isn't set to GPU — fix in step 1, then restart. |
| GPU/Internet greyed out | Verify your Kaggle account with a phone number. |
| Out of memory | Lower `BATCH_SIZE` in `settings.py` (e.g. to 8). |
