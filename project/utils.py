"""
utils.py
========
Small, reusable helpers that several modules need:

  * set_seed()   — make a run reproducible by seeding all RNGs.
  * ensure_dir() — create an output directory if it does not exist.
  * save_json()  — write a metrics dict to disk as readable JSON.
  * get_logger() — a consistently-formatted logger.

Keeping these here avoids copy-pasting boilerplate into every script.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path


def set_seed(seed: int) -> None:
    """Seed every random number generator we rely on.

    Reproducibility matters here because Hypothesis H2 (does context help?) is a
    SMALL effect — we must be sure a difference comes from the input, not luck.

    numpy/torch are imported locally so the lightweight commands that call other
    helpers in this module don't have to import PyTorch.
    """
    import numpy as np
    import torch

    random.seed(seed)            # Python's built-in RNG
    np.random.seed(seed)         # NumPy (used by scikit-learn)
    torch.manual_seed(seed)      # PyTorch CPU + MPS share this seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # all CUDA GPUs


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if missing, and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str | Path) -> None:
    """Write a dict to `path` as indented JSON, creating the folder if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_logger(name: str = "sarcasm") -> logging.Logger:
    """Return a logger that prints `time | LEVEL | message`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
