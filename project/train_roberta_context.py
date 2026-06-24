"""
train_roberta_context.py
========================
Fine-tunes RoBERTa using the comment AND the parent comment (with context).

Compare its score with train_roberta.py to see if context helped RoBERTa.
Needs a GPU to be fast.

    python train_roberta_context.py
"""

from engine.transformer import train_and_score

train_and_score(model_name="roberta-base", use_context=True)
