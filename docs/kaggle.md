# Running this project on Kaggle

A complete, step-by-step guide to running the whole pipeline (baseline +
BERT/RoBERTa, with and without context) on Kaggle's **free GPU**.

Kaggle is the easiest place to run this because the SARC dataset is already hosted
there — nothing to download.

There are two ways to do it:

- **The easy path** — import the ready-made notebook
  [`../project/notebooks/sarcasm_kaggle.ipynb`](../project/notebooks/sarcasm_kaggle.ipynb)
  and run it top to bottom (see [§7](#7-the-easy-path-use-the-provided-notebook)).
- **The manual path** — do the steps yourself (§1–§6 below). Read these even if you
  use the notebook, so you understand what each cell does.

---

## 0. Cost and limits (it's free)

Kaggle is free, no credit card. You only need to **verify your account with a phone
number** (Settings → Phone Verification) — this is required to unlock the GPU and
Internet.

| Resource | Free tier |
|---|---|
| GPU | ~**30 hours/week** of NVIDIA T4×2 or P100 (quota resets weekly) |
| Session length | up to **12 h** per run; **~20 min idle** timeout |
| Disk (`/kaggle/working`) | ~20 GB, writable and saved with the notebook version |
| Dataset | `danofer/sarcasm` is public and free to attach |

The full matrix below fits comfortably inside one weekly quota.

---

## 1. Create a GPU notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. In the right-hand sidebar open **Settings → Accelerator** and choose
   **GPU T4 x2** (or P100). The code uses a single GPU; T4×2 is fine.
3. Leave **Settings → Internet** **ON** if you plan to `git clone` your code or
   `pip install` anything (see §3, Method A). It can stay off if you upload your
   code as a dataset (Method B).

> On a CUDA GPU the project automatically enables fp16 mixed precision, so training
> is much faster than on a laptop.

---

## 2. Attach the SARC dataset

1. In the notebook, click **+ Add Data** (right sidebar).
2. Search **"Sarcasm"** and add **danofer/sarcasm**.
3. It mounts read-only under `/kaggle/input/`. The file we need ends up at:

   ```
   /kaggle/input/sarcasm/train-balanced-sarcasm.csv
   ```

`/kaggle/input/` is **read-only** — that's why the code must live (and write its
outputs) under `/kaggle/working/` instead (next step).

---

## 3. Get the project code onto Kaggle

The code must live under the **writable** `/kaggle/working/` directory so it can
save metrics and model checkpoints. Pick one method.

### Method A — clone from GitHub (needs Internet ON)

Push this repository to GitHub, then in a notebook cell:

```python
!git clone https://github.com/<your-username>/EmiliaProject.git /kaggle/working/EmiliaProject
```

### Method B — upload the repo as a Kaggle Dataset (no Internet needed)

1. On your machine, zip the project (the `EmiliaProject` folder, including
   `project/`). You can exclude `project/.venv`, `data/`, `models/`, `results/`.
2. On Kaggle: **+ Add Data → Upload → New Dataset**, upload the zip, give it a
   slug (say `emilia-code`).
3. Attach it to the notebook, then copy it into the writable area:

   ```python
   !cp -r /kaggle/input/emilia-code/EmiliaProject /kaggle/working/
   ```

### Confirm the code is in place

```python
import os
CODE_DIR = "/kaggle/working/EmiliaProject/project"
assert os.path.isdir(CODE_DIR), "Code not found — re-check Method A or B."
print("code at", CODE_DIR)
```

---

## 4. Point the pipeline at the mounted dataset

The project reads the CSV path from the **`SARC_CSV`** environment variable (falling
back to `data/raw/` locally). Set it to the Kaggle mount path — globbing makes this
robust to the exact folder name:

```python
import os, glob
hits = glob.glob("/kaggle/input/**/train-balanced-sarcasm.csv", recursive=True)
assert hits, "Add the danofer/sarcasm dataset via + Add Data first."
os.environ["SARC_CSV"] = hits[0]
print("SARC_CSV =", hits[0])
```

`!` shell commands in the notebook inherit this environment variable, so every
`python run.py ...` call below will use the Kaggle dataset automatically.

---

## 5. Check dependencies

Kaggle ships a recent Python ML stack. The project needs **transformers ≥ 4.46**
(for the `processing_class` API). Upgrade only if needed:

```python
!pip install -q -U "transformers>=4.46" datasets
import torch
print("CUDA available:", torch.cuda.is_available())   # should print True
```

---

## 6. Run the pipeline

All commands are the normal `run.py` subcommands; only the paths are Kaggle-specific.
Outputs are written under `/kaggle/working/EmiliaProject/results/`.

```python
PROJ = "/kaggle/working/EmiliaProject/project"
RES  = "/kaggle/working/EmiliaProject/results"
```

**Sanity-check the data:**

```python
!python {PROJ}/run.py prepare
```

**Baseline (seconds):**

```python
!python {PROJ}/run.py baseline
!python {PROJ}/run.py baseline --context
```

**Fine-tune the transformers** (`--subset` keeps it quick; see §6.1 for sizing):

```python
!python {PROJ}/run.py train --model bert-base-uncased           --subset 50000 --epochs 3
!python {PROJ}/run.py train --model bert-base-uncased --context --subset 50000 --epochs 3
!python {PROJ}/run.py train --model roberta-base               --subset 50000 --epochs 3
!python {PROJ}/run.py train --model roberta-base --context     --subset 50000 --epochs 3
```

**Significance test (does context help?):**

```python
!python {PROJ}/run.py compare \
    --a {RES}/bert-base-uncased_noctx_seed42_preds.npz \
    --b {RES}/bert-base-uncased_ctx_seed42_preds.npz
```

**Build the comparison table:**

```python
!python {PROJ}/run.py report

from IPython.display import Markdown, display
display(Markdown(open(f"{RES}/summary.md").read()))
```

### 6.1 Recommended settings on a Kaggle GPU

| Setting | Suggestion | Why |
|---|---|---|
| `--subset` | `50000` to start; drop the flag for the **full ~966k rows** | T4 fp16 is fast; the full set gives the best numbers |
| `--epochs` | `3` | standard for BERT fine-tuning |
| `--batch-size` | default `16` (raise to `32` on T4 if memory allows) | no MPS limits here, unlike the laptop |
| seeds | run each config with `--seed 13 42 123` | report mean ± std (the report aggregates them) |

Approximate runtime per fine-tune at `--subset 50000`, 3 epochs on a T4: ~15 min
(no context) / ~25 min (context). The full 4-config × 3-seed matrix is a few hours
— within the weekly quota.

For a multi-seed sweep, add seeds like:

```python
for s in (13, 42, 123):
    !python {PROJ}/run.py train --model roberta-base --context --subset 50000 --epochs 3 --seed {s}
```

---

## 7. The easy path: use the provided notebook

Instead of writing the cells yourself:

1. **File → Import Notebook** and upload
   [`../project/notebooks/sarcasm_kaggle.ipynb`](../project/notebooks/sarcasm_kaggle.ipynb).
2. Enable the GPU (§1) and add the dataset (§2).
3. Edit **one line** in the first code cell — the clone/copy line — to point at your
   code (Method A or B from §3).
4. Run all cells top to bottom.

---

## 8. Saving and downloading your results

Everything is written to `/kaggle/working/EmiliaProject/`:

- `results/summary.md` and `results/summary.csv` — the comparison table
- `results/*_metrics.json` — per-run metrics
- `results/*_cm.png` — confusion matrices
- `models/<run_name>/` — fine-tuned checkpoints (large)

To keep them:

- **Save Version** (top-right) snapshots the notebook and its `/kaggle/working`
  outputs. Use **Save & Run All (Commit)** to run the whole notebook **headless**
  (this avoids the 20-minute idle timeout for long sweeps).
- Or open the **Output** tab / **Data** panel and download individual files.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Add the danofer/sarcasm dataset…` assertion | You didn't attach the dataset — do §2. |
| `Code not found` assertion | The clone/copy in §3 didn't run or used the wrong slug. |
| GPU/Internet options greyed out | Verify your account with a phone number (Settings → Phone Verification). |
| `CUDA available: False` | Accelerator isn't set to GPU — fix in §1, then restart the session. |
| `Trainer.__init__() got an unexpected keyword 'processing_class'` | transformers too old — `pip install -U "transformers>=4.46"`. |
| CUDA out of memory | Lower `--batch-size` (e.g. 16 → 8), or reduce `--subset`. |
| Session died mid-sweep | Idle timeout — use **Save & Run All (Commit)** for headless runs (§8). |
| Quota exhausted | The 30 h/week GPU quota resets weekly; spread big sweeps out. |

---

See [06-usage.md](06-usage.md) for the command reference shared across all
environments, and [04-transformers.md](04-transformers.md) for what the training
actually does.
