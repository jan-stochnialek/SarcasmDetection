"""
evaluate.py
===========
Shared evaluation utilities so every model is scored identically (plan 6.3):

  * compute_classification_metrics() — accuracy, precision, recall, F1, macro-F1,
    and ROC-AUC.
  * save_confusion_matrix()          — save a 2x2 confusion-matrix figure.
  * mcnemar_test()                   — paired significance test used to decide
    whether the context model is *significantly* different from the no-context
    model on the SAME test set (the core of answering RQ2).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return the dict of metrics named in the plan.

    Parameters
    ----------
    y_true : true labels (0/1)
    y_pred : predicted labels (0/1)
    y_prob : optional predicted probability of the positive (sarcastic) class,
             needed only for ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)

    # Binary metrics with the SARCASTIC class (label 1) as "positive".
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    # Macro-F1 averages the F1 of both classes equally (robust summary).
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # Happens if only one class is present in y_true; report NaN.
            metrics["roc_auc"] = float("nan")

    return metrics


def save_confusion_matrix(y_true, y_pred, path, title: str = "Confusion matrix") -> None:
    """Render and save a 2x2 confusion matrix as a PNG."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend — safe on Colab/servers/M1
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["not sarc", "sarc"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Write the count inside each cell for readability.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's paired test: do two classifiers differ on the SAME test set?

    Why this test? Comparing two models on the identical examples means their
    errors are PAIRED, not independent, so a normal proportion test is invalid.
    McNemar looks only at the examples where the two models DISAGREE:

        n01 = A wrong, B right
        n10 = A right, B wrong

    Under the null hypothesis "the two models are equally good", each disagreement
    is a 50/50 coin flip, so we use an exact two-sided binomial test on the
    discordant pairs. A small p-value (e.g. < 0.05) means the difference between,
    say, the context and no-context model is unlikely to be chance.
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true

    n01 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    n10 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong

    n = n01 + n10
    if n == 0:
        p_value = 1.0  # models made identical predictions -> no evidence of a difference
    else:
        # Exact binomial test on the smaller discordant count vs p=0.5.
        p_value = binomtest(min(n01, n10), n, 0.5).pvalue

    return {
        "n_A_wrong_B_right": n01,
        "n_A_right_B_wrong": n10,
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }
