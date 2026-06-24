"""
train_roberta.py
================
Fine-tunes RoBERTa using ONLY the comment (no context).

RoBERTa is a better-trained version of BERT, so it usually scores a bit higher.
Needs a GPU to be fast.

    python train_roberta.py
"""

from engine.transformer import train_and_score

train_and_score(model_name="roberta-base", use_context=False)
