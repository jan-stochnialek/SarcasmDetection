"""
smoke_test.py
=============
Fast, self-contained sanity check. It generates a tiny SYNTHETIC SARC-format
dataset and runs the real data pipeline + TF-IDF baseline + report on it.

Needs only the baseline dependencies (pandas, scikit-learn) — NO torch, NO real
dataset, NO GPU — and finishes in a few seconds. Use it to confirm the code
executes end-to-end before investing in the real dataset or GPU time.

    python tests/smoke_test.py

The metrics are MEANINGLESS (the data is synthetic). The point is only that every
stage runs without error.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make both the project modules and this tests/ folder importable from anywhere.
THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(THIS_DIR))

from config import BaselineConfig, DataConfig          # noqa: E402
from data import load_and_clean, make_splits           # noqa: E402
from baseline import run_baseline                       # noqa: E402
from report import build_report                         # noqa: E402
from make_synthetic_data import generate                # noqa: E402


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="sarc_smoke_"))
    csv_path = work / "synthetic.csv"
    generate(n=3000, seed=0).to_csv(csv_path, index=False)
    print(f"[1/4] wrote synthetic data -> {csv_path}")

    # Point a normal DataConfig at the synthetic file; everything else is default.
    cfg = DataConfig(raw_csv=csv_path)

    # --- like `run.py prepare`: clean + split + class balance ---
    df = load_and_clean(cfg)
    train, val, test = make_splits(df, cfg)
    print(
        f"[2/4] splits: train={len(train)} val={len(val)} test={len(test)} | "
        f"test sarcastic fraction={test['label'].mean():.3f}"
    )

    # --- like `run.py baseline` (and `--context`) ---
    results_dir = work / "results"
    results_dir.mkdir()
    for use_context in (False, True):
        metrics, _y_true, _y_pred = run_baseline(cfg, BaselineConfig(use_context=use_context))
        tag = "ctx" if use_context else "noctx"
        (results_dir / f"baseline_logreg_{tag}_metrics.json").write_text(json.dumps(metrics))
        pretty = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"[3/4] baseline {tag:>5}: {pretty}")

    # --- like `run.py report`: aggregate into a table ---
    print("[4/4] report:")
    build_report(results_dir)
    print("\nSMOKE TEST OK (synthetic data — numbers are not meaningful)")


if __name__ == "__main__":
    main()
