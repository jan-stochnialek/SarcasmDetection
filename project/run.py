"""
run.py
======
Single command-line entry point for the whole project. Run it from inside the
`project/` folder.

Commands
--------
  # 0. Sanity-check the data pipeline (prints split sizes + class balance)
  python run.py prepare

  # 1. TF-IDF baseline (add --context to prepend the parent comment)
  python run.py baseline
  python run.py baseline --context

  # 2. Fine-tune a transformer
  python run.py train --model bert-base-uncased
  python run.py train --model bert-base-uncased --context
  python run.py train --model roberta-base --context --seed 13

  # 3. Significance test (McNemar) between two saved prediction files
  python run.py compare \
      --a ../results/bert-base-uncased_noctx_seed42_preds.npz \
      --b ../results/bert-base-uncased_ctx_seed42_preds.npz
"""

from __future__ import annotations

import argparse

from config import BaselineConfig, DataConfig, RESULTS_DIR, TransformerConfig
from utils import ensure_dir, get_logger, save_json

log = get_logger()


# ---------------------------------------------------------------------------
# Command handlers — one function per subcommand.
# ---------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> None:
    """Load+clean+split and report sizes/balance, so you can verify the data
    before spending GPU time on training."""
    from data import load_and_clean, make_splits

    cfg = DataConfig()
    df = load_and_clean(cfg)
    log.info("Cleaned dataset: %d rows", len(df))
    train, val, test = make_splits(df, cfg)
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        sarc_fraction = frame[cfg.label_col].mean()
        log.info("%-5s : %7d rows | sarcastic fraction = %.3f", name, len(frame), sarc_fraction)


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run the TF-IDF baseline and save metrics + a confusion matrix."""
    from baseline import run_baseline
    from evaluate import save_confusion_matrix

    # Honor --subset so the baseline shares the same split as the transformer runs.
    data_cfg = DataConfig(subset_size=args.subset) if args.subset is not None else DataConfig()
    base_cfg = BaselineConfig(use_context=args.context, classifier=args.classifier)

    metrics, y_true, y_pred = run_baseline(data_cfg, base_cfg)
    tag = "ctx" if args.context else "noctx"
    log.info("Baseline (%s, %s): %s", base_cfg.classifier, tag, metrics)

    results_dir = ensure_dir(RESULTS_DIR)
    save_json(metrics, results_dir / f"baseline_{base_cfg.classifier}_{tag}_metrics.json")
    save_confusion_matrix(
        y_true, y_pred,
        results_dir / f"baseline_{base_cfg.classifier}_{tag}_cm.png",
        title=f"TF-IDF ({tag})",
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune one transformer configuration."""
    from train import train_one

    # Optionally override the training subset size (handy for quick local runs).
    data_cfg = DataConfig(subset_size=args.subset) if args.subset is not None else DataConfig()
    # Give the model a bigger token budget when it also has to read the context.
    max_length = 256 if args.context else 128
    model_cfg = TransformerConfig(
        model_name=args.model,
        use_context=args.context,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=max_length,
    )
    train_one(data_cfg, model_cfg)


def cmd_compare(args: argparse.Namespace) -> None:
    """McNemar test between two *_preds.npz files (same test set required)."""
    import numpy as np

    from evaluate import mcnemar_test

    a = np.load(args.a)
    b = np.load(args.b)
    if not np.array_equal(a["y_true"], b["y_true"]):
        raise ValueError(
            "The two prediction files use different test sets — McNemar requires "
            "the SAME examples. Re-run both with the same DataConfig seed."
        )
    result = mcnemar_test(a["y_true"], a["y_pred"], b["y_pred"])
    log.info("McNemar A=%s vs B=%s -> %s", args.a, args.b, result)


def cmd_report(args: argparse.Namespace) -> None:
    """Aggregate all results/*_metrics.json into the comparison table."""
    from report import build_report

    build_report()


# ---------------------------------------------------------------------------
# Argument parser.
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SARC sarcasm-detection experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="clean + split + report class balance")

    p_base = sub.add_parser("baseline", help="TF-IDF + linear classifier")
    p_base.add_argument("--context", action="store_true", help="prepend parent comment")
    p_base.add_argument("--classifier", choices=["logreg", "svm"], default="logreg")
    p_base.add_argument("--subset", type=int, default=None,
                        help="override DataConfig.subset_size (rows) to match a training run")

    p_train = sub.add_parser("train", help="fine-tune a transformer")
    p_train.add_argument("--model", default="bert-base-uncased",
                         help="HF model id, e.g. bert-base-uncased or roberta-base")
    p_train.add_argument("--context", action="store_true", help="feed parent+comment as a pair")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--subset", type=int, default=None,
                         help="override DataConfig.subset_size (rows) for faster local runs")

    p_cmp = sub.add_parser("compare", help="McNemar test between two prediction files")
    p_cmp.add_argument("--a", required=True, help="path to first *_preds.npz")
    p_cmp.add_argument("--b", required=True, help="path to second *_preds.npz")

    sub.add_parser("report", help="aggregate all metrics into summary.md / summary.csv")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "prepare": cmd_prepare,
        "baseline": cmd_baseline,
        "train": cmd_train,
        "compare": cmd_compare,
        "report": cmd_report,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
