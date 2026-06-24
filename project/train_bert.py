"""
train_bert.py
=============
Fine-tunes BERT using ONLY the comment (no context).

Needs a GPU to be fast (it works on a normal computer too, just slowly).

    python train_bert.py
"""

from engine.transformer import train_and_score

train_and_score(model_name="bert-base-uncased", use_context=False)
