"""
engine/scoring.py
=================
Works out how good a model is, saves each result to a small file, and prints the
final comparison table at the end.

You do not need to edit this file.
"""

import json
from pathlib import Path

import numpy as np
# These are ready-made scoring functions from scikit-learn.
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Folder where results are saved (created automatically if it does not exist).
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = PROJECT_FOLDER.parent / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def measure(true_labels, predicted_labels, predicted_scores):
    """Compare the model's predictions to the truth and return a few scores.

    true_labels      the correct answers (0 or 1) for the test comments
    predicted_labels what the model guessed (0 or 1)
    predicted_scores the model's confidence that each comment is sarcastic (0..1)
    """
    accuracy = accuracy_score(true_labels, predicted_labels)
    # precision/recall/f1 for the "sarcastic" class (label 1).
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary", pos_label=1, zero_division=0
    )
    auc = roc_auc_score(true_labels, predicted_scores)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def _file_name(name):
    """Turn a label like 'BERT (with context)' into a safe file name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def save_result(name, scores, true_labels=None, predicted_labels=None):
    """Save one model's scores (and its predictions) and print them.

    name is a short label, e.g. 'BERT (with context)'.
    """
    safe = _file_name(name)

    # Save the scores as a small text file in JSON format (name: value pairs).
    (RESULTS_FOLDER / f"{safe}.json").write_text(
        json.dumps({"name": name, **scores}, indent=2)
    )

    # Save the predictions too. We need these later to check whether context
    # REALLY made a difference (see show_table below).
    if predicted_labels is not None:
        np.savez(
            RESULTS_FOLDER / f"{safe}.npz",
            true=np.array(true_labels),
            predicted=np.array(predicted_labels),
        )
        # Also save a confusion-matrix picture you can drop straight into a report.
        _save_confusion_matrix(name, true_labels, predicted_labels)

    # Print the scores straight away so you see them.
    print(f"\nResult for {name}:")
    for score_name, value in scores.items():
        print(f"   {score_name:10s}: {value:.3f}")


def _save_confusion_matrix(name, true_labels, predicted_labels):
    """Save a confusion-matrix picture (PNG) for one model's predictions.

    The grid shows how many comments fell into each true-vs-predicted box, so you
    can see at a glance whether the model confuses sarcastic and non-sarcastic.
    matplotlib is imported here (not at the top) so the quick scripts that never
    plot don't pay to load it.
    """
    import matplotlib
    matplotlib.use("Agg")            # write a file; no on-screen window needed
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        true_labels, predicted_labels,
        display_labels=["not sarcastic", "sarcastic"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(RESULTS_FOLDER / f"{_file_name(name)}_confusion.png", dpi=120)
    plt.close(fig)


def _is_context_a_real_improvement(no_context_file, with_context_file):
    """Statistics check: did adding context make a REAL difference, or just luck?

    We look only at the comments where the two models DISAGREE and ask whether the
    context model is reliably the one that's right. This is "McNemar's test"; it
    returns a p-value. A small p-value (below 0.05) means the difference is real.
    """
    from scipy.stats import binomtest

    a = np.load(no_context_file)
    b = np.load(with_context_file)
    a_correct = a["predicted"] == a["true"]
    b_correct = b["predicted"] == b["true"]

    only_context_right = int(((~a_correct) & b_correct).sum())  # context fixed it
    only_context_wrong = int((a_correct & (~b_correct)).sum())  # context broke it
    disagreements = only_context_right + only_context_wrong
    if disagreements == 0:
        return None
    smaller = min(only_context_right, only_context_wrong)
    return binomtest(smaller, disagreements, 0.5).pvalue


def show_table():
    """Print a table comparing every model you have run so far."""
    json_files = sorted(RESULTS_FOLDER.glob("*.json"))
    if not json_files:
        print("No results yet. Run a train_*.py script first.")
        return

    results = [json.loads(f.read_text()) for f in json_files]

    print("\n" + "=" * 60)
    print(f"{'Model':28s}{'accuracy':>10s}{'f1':>8s}{'auc':>8s}")
    print("-" * 60)
    for r in sorted(results, key=lambda row: row["name"]):
        print(f"{r['name']:28s}{r['accuracy']:10.3f}{r['f1']:8.3f}{r['auc']:8.3f}")
    print("=" * 60)

    # For each model, if we have BOTH the no-context and with-context runs,
    # check whether the context version is genuinely better.
    print("\nDoes adding context help? (p below 0.05 = a real difference)")
    any_pairs = False
    for base in ["Baseline", "BERT", "RoBERTa"]:
        no_context = RESULTS_FOLDER / f"{base}.npz"
        with_context = RESULTS_FOLDER / f"{base}_with_context.npz"
        if no_context.exists() and with_context.exists():
            any_pairs = True
            p_value = _is_context_a_real_improvement(no_context, with_context)
            if p_value is None:
                # The two models never disagreed, so there is nothing to test.
                print(f"   {base:10s}: the two versions made identical predictions")
                continue
            verdict = "REAL difference" if p_value < 0.05 else "could be luck"
            print(f"   {base:10s}: p = {p_value:.3f}  ({verdict})")
    if not any_pairs:
        print("   (run a model both with and without context to see this)")
