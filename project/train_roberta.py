"""
Fine-tunes RoBERTa using ONLY the comment (no context).
"""

from engine.transformer import train_and_score

train_and_score(model_name="roberta-base", use_context=False)
