"""
config.py
=========
Central place for *all* configuration used across the project.

Why a single config file?
  * Reproducibility — every hyperparameter lives in one documented location.
  * Clarity — the experiment "knobs" are separated from the logic that uses them.
  * Safety — paths are computed relative to this file, so the code works no
    matter which directory you launch it from.

This module defines:
  1. Filesystem paths (where data lives, where outputs go).
  2. `get_device()` — picks the best available hardware (GPU/Apple-Silicon/CPU).
  3. Three dataclasses of settings: DataConfig, BaselineConfig, TransformerConfig.
"""

from __future__ import annotations  # lets us write `int | None` on older Pythons

import os
from dataclasses import dataclass
from pathlib import Path

# NOTE: torch is imported lazily inside get_device() (not at module top level) so
# that the lightweight commands (prepare / baseline / report) can import this
# config without requiring PyTorch to be installed.


# ---------------------------------------------------------------------------
# 1. Filesystem layout
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the `project/` folder (where this file lives).
PROJECT_ROOT = Path(__file__).resolve().parent

# REPO_ROOT is the EmiliaProject folder, one level up. It also contains docs/.
REPO_ROOT = PROJECT_ROOT.parent

# Data and output directories. These sit next to project/ and docs/.
DATA_DIR = REPO_ROOT / "data"            # all dataset files
RAW_DIR = DATA_DIR / "raw"               # the unmodified Kaggle download goes here
PROCESSED_DIR = DATA_DIR / "processed"   # cleaned/split data (optional cache)
RESULTS_DIR = REPO_ROOT / "results"      # metrics JSONs, confusion-matrix PNGs, preds
MODELS_DIR = REPO_ROOT / "models"        # fine-tuned model checkpoints

# Path to the balanced SARC CSV. Defaults to data/raw/, but can be overridden with
# the SARC_CSV environment variable — handy on Kaggle/Colab where the dataset is
# mounted read-only at a fixed path (e.g. /kaggle/input/sarcasm/...).
RAW_CSV = Path(os.environ.get("SARC_CSV", RAW_DIR / "train-balanced-sarcasm.csv"))


# ---------------------------------------------------------------------------
# 2. Hardware / device detection
# ---------------------------------------------------------------------------
def get_device():
    """Return the best available torch device.

    Preference order:
      1. CUDA  — NVIDIA GPU (Colab, Kaggle, most cloud machines). Fastest.
      2. MPS   — Apple-Silicon GPU (M1/M2/M3) via the Metal backend. Works on a
                 MacBook Air, but slower than CUDA and without solid fp16 support.
      3. CPU   — always available fallback (fine for the TF-IDF baseline).
    """
    import torch  # local import: only the transformer path needs PyTorch

    if torch.cuda.is_available():
        return torch.device("cuda")
    # `torch.backends.mps` only exists on builds with MPS; guard with getattr.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 3. Experiment settings (grouped by stage)
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """How to load, clean and split the raw SARC CSV."""

    raw_csv: Path = RAW_CSV
    # Column names in the Kaggle CSV.
    text_col: str = "comment"            # the target comment we classify
    context_col: str = "parent_comment"  # the immediate thread context
    label_col: str = "label"             # 1 = sarcastic, 0 = not sarcastic

    # Stratified subsample size for development on limited hardware.
    # Set to None to use the entire ~1M-row dataset (only do this on a real GPU).
    subset_size: int | None = 150_000

    # Split fractions. train = 1 - val_size - test_size.
    val_size: float = 0.10
    test_size: float = 0.10

    # Master seed for splitting/subsampling — keeps val/test FIXED across models.
    seed: int = 42

    # Light, transformer-friendly text normalization toggles.
    normalize_urls: bool = True      # replace links with the token [URL]
    normalize_mentions: bool = True  # replace u/name mentions with [USER]


@dataclass
class BaselineConfig:
    """TF-IDF + linear classifier baseline (plan Section 5.1)."""

    use_context: bool = False     # if True, prepend the parent comment
    ngram_max: int = 2            # word uni- + bi-grams (1, 2)
    min_df: int = 2               # ignore terms appearing in < 2 documents
    max_features: int | None = 100_000  # cap vocabulary size for speed/memory
    C: float = 1.0                # inverse regularization strength
    classifier: str = "logreg"    # "logreg" (LogisticRegression) or "svm" (LinearSVC)


@dataclass
class TransformerConfig:
    """Fine-tuning settings for BERT / RoBERTa (plan Sections 5.2 / 5.3 & 6.1)."""

    model_name: str = "bert-base-uncased"  # or "roberta-base"
    use_context: bool = False              # if True, feed (parent, comment) as a pair

    # Token budget. Use 128 for comment-only; bump to ~256 when context is on.
    max_length: int = 128

    batch_size: int = 16
    learning_rate: float = 2e-5            # AdamW; 2e-5 / 3e-5 are typical
    epochs: int = 3                        # 2-4 with early stopping on val F1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1              # linear warmup then linear decay

    seed: int = 42                         # vary this to measure seed variance
