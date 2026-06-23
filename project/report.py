"""
report.py
=========
Collect every per-run metrics file in results/ into ONE comparison table that
matches plan.md Section 6.4.

Each training/baseline run writes `<run_name>_metrics.json`. This module:
  1. parses each filename back into (model, context, seed);
  2. groups runs that differ only by seed and aggregates them as mean ± std;
  3. writes a machine-readable `summary.csv` (one row per run) and a
     human-readable `summary.md` (the aggregated table you paste into the report).

Run via:  python run.py report
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from config import RESULTS_DIR
from utils import ensure_dir, get_logger

log = get_logger()

# The metric columns we report, in display order (see evaluate.py).
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
_SUFFIX = "_metrics.json"


def _parse_filename(path: Path):
    """Turn a `*_metrics.json` filename back into (model_label, context, seed).

    Two filename shapes are produced by the project:
      * baseline_<classifier>_<tag>_metrics.json      e.g. baseline_logreg_ctx
      * <model>_<tag>_seed<NN>_metrics.json           e.g. roberta-base_ctx_seed42
    where <tag> is 'ctx' or 'noctx'.
    """
    stem = path.name[: -len(_SUFFIX)]
    if stem.startswith("baseline_"):
        # baseline_<classifier>_<tag>
        _, classifier, tag = stem.split("_", 2)
        model = f"TF-IDF ({classifier})"
        seed = None
    else:
        # <model>_<tag>_seed<NN>  — rsplit from the right so model names with
        # underscores still parse correctly.
        model, tag, seed_token = stem.rsplit("_", 2)
        seed = int(seed_token.replace("seed", ""))
    context = "yes" if tag == "ctx" else "no"
    return model, context, seed


def collect(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Read every metrics JSON into a flat list of per-run row dicts."""
    rows = []
    for path in sorted(Path(results_dir).glob(f"*{_SUFFIX}")):
        model, context, seed = _parse_filename(path)
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"model": model, "context": context, "seed": seed}
        row.update({k: metrics.get(k) for k in METRIC_KEYS})
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Group rows by (model, context) and compute mean (± sample std) per metric."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["context"])].append(r)

    aggregated = []
    for (model, context), items in groups.items():
        entry = {"model": model, "context": context, "n_runs": len(items)}
        for key in METRIC_KEYS:
            # Drop missing values and NaNs (v == v is False only for NaN).
            values = [it[key] for it in items if it.get(key) is not None and it[key] == it[key]]
            if not values:
                entry[key] = None
                continue
            mean = sum(values) / len(values)
            if len(values) > 1:
                # Sample standard deviation (ddof = 1).
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                entry[key] = (mean, variance ** 0.5)
            else:
                entry[key] = (mean, None)
        aggregated.append(entry)

    # Stable, readable order: baselines first (capital T sorts first), then by
    # model, then no-context before with-context.
    aggregated.sort(key=lambda e: (e["model"], e["context"]))
    return aggregated


def _fmt(cell) -> str:
    """Format a (mean, std) cell as 'mean±std', or '—' if missing."""
    if cell is None:
        return "—"
    mean, std = cell
    return f"{mean:.3f}±{std:.3f}" if std is not None else f"{mean:.3f}"


def to_markdown(aggregated: list[dict]) -> str:
    """Render the aggregated rows as a GitHub-flavoured markdown table."""
    lines = [
        "| Model | Context | Acc | P | R | F1 | Macro-F1 | AUC | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in aggregated:
        lines.append(
            "| {model} | {ctx} | {acc} | {p} | {r} | {f1} | {mf1} | {auc} | {n} |".format(
                model=e["model"],
                ctx=e["context"],
                acc=_fmt(e["accuracy"]),
                p=_fmt(e["precision"]),
                r=_fmt(e["recall"]),
                f1=_fmt(e["f1"]),
                mf1=_fmt(e["macro_f1"]),
                auc=_fmt(e["roc_auc"]),
                n=e["n_runs"],
            )
        )
    return "\n".join(lines)


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write one row per run (raw, un-aggregated) for further analysis."""
    fields = ["model", "context", "seed"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})


def build_report(results_dir: Path = RESULTS_DIR) -> None:
    """Top-level entry: collect, aggregate, write summary.csv + summary.md, print."""
    results_dir = Path(results_dir)
    rows = collect(results_dir)
    if not rows:
        log.warning("No *%s files in %s — run `baseline`/`train` first.", _SUFFIX, results_dir)
        return

    aggregated = aggregate(rows)
    table = to_markdown(aggregated)

    ensure_dir(results_dir)
    (results_dir / "summary.md").write_text(
        "# Results summary\n\n"
        "Aggregated over seeds as mean±std. Compare each model's `no` vs `yes` "
        "context rows to answer RQ2.\n\n" + table + "\n",
        encoding="utf-8",
    )
    _write_csv(rows, results_dir / "summary.csv")
    log.info("Wrote summary.md and summary.csv to %s", results_dir)
    print("\n" + table + "\n")
