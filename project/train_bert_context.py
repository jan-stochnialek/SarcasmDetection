"""
train_bert_context.py
=====================
Fine-tunes BERT using the comment AND the parent comment (with context).

Compare its score with train_bert.py to see if context helped BERT.

    python train_bert_context.py
"""

from engine.transformer import train_and_score

train_and_score(model_name="bert-base-uncased", use_context=True)
