"""
Trains the SIMPLE model (TF-IDF + logistic regression) using the comment AND the
parent comment (with context). Fast — runs in seconds, no GPU needed.
"""

from engine.baseline import train_and_score_baseline

train_and_score_baseline(use_context=True)
