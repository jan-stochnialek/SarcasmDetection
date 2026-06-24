"""
train_baseline_context.py
=========================
Trains the SIMPLE model (TF-IDF + logistic regression) using the comment AND the
parent comment (with context). Fast — runs in seconds, no GPU needed.

Compare its score with train_baseline.py to see if context helped the simple model.

    python train_baseline_context.py
"""

from engine.baseline import train_and_score_baseline

train_and_score_baseline(use_context=True)
