"""
baseline.py
===========
The classical TF-IDF + linear-classifier baseline (plan Section 5.1).

Why a baseline?
  * It is cheap and runs on a CPU / M1 Air in minutes.
  * It is interpretable.
  * It sets a lower bound: a transformer is only worth the cost if it clearly
    beats this.
It also has its own context switch, so we can see whether thread context helps
even a simple bag-of-words model.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import BaselineConfig, DataConfig
from data import build_text_single, load_and_clean, make_splits
from evaluate import compute_classification_metrics


def _make_classifier(cfg: BaselineConfig):
    """Pick the final estimator based on the config."""
    if cfg.classifier == "svm":
        # LinearSVC is strong for text but has no predict_proba; wrap it in a
        # calibrator so we can still produce probabilities for ROC-AUC.
        return CalibratedClassifierCV(LinearSVC(C=cfg.C))
    # LogisticRegression is the default: fast, probabilistic, well-understood.
    return LogisticRegression(C=cfg.C, max_iter=1000)


def run_baseline(data_cfg: DataConfig, base_cfg: BaselineConfig):
    """Train + evaluate the TF-IDF baseline.

    Returns (metrics_dict, y_true, y_pred) so the caller can also save a
    confusion matrix.
    """
    # 1. Same data pipeline as the transformers -> fair comparison.
    df = load_and_clean(data_cfg)
    train, _val, test = make_splits(df, data_cfg)  # baseline does not need val

    # 2. Build input strings according to the context flag.
    X_train = build_text_single(train, data_cfg, base_cfg.use_context)
    X_test = build_text_single(test, data_cfg, base_cfg.use_context)
    y_train = train[data_cfg.label_col].values
    y_test = test[data_cfg.label_col].values

    # 3. Pipeline = TF-IDF vectorizer -> linear classifier.
    #    Bundling them ensures the SAME vocabulary learned on train is applied
    #    to test (no leakage).
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, base_cfg.ngram_max),  # uni- + bi-grams
                    min_df=base_cfg.min_df,               # drop ultra-rare terms
                    max_features=base_cfg.max_features,   # cap vocab size
                    sublinear_tf=True,                    # dampen frequent-term weight
                ),
            ),
            ("clf", _make_classifier(base_cfg)),
        ]
    )

    # 4. Fit on train, predict on test.
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # Probability of the positive class, if the final estimator supports it.
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    return metrics, y_test, y_pred
