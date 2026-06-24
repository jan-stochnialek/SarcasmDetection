"""
train_baseline.py
=================
Trains the SIMPLE model (TF-IDF + logistic regression) using ONLY the comment
(no context). Fast — runs in seconds, no GPU needed.

    python train_baseline.py
"""

from engine.baseline import train_and_score_baseline

train_and_score_baseline(use_context=False)
