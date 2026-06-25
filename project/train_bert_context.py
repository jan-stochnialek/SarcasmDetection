"""
Fine-tunes BERT using the comment AND the parent comment (with context).
"""

from engine.transformer import train_and_score

train_and_score(model_name="bert-base-uncased", use_context=True)
