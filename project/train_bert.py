"""
Fine-tunes BERT using ONLY the comment (no context).
"""

from engine.transformer import train_and_score

train_and_score(model_name="bert-base-uncased", use_context=False)
