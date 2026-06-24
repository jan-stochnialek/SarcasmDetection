"""
run_everything.py
=================
Runs the WHOLE project in order: the simple baseline, then BERT, then RoBERTa —
each one both without and with context — and finally prints the comparison table.

This is the same as running all the train_*.py scripts one by one, then
show_results.py. It can take a while (the transformers train on the GPU).

    python run_everything.py
"""

from engine.baseline import train_and_score_baseline
from engine.transformer import train_and_score
from engine.scoring import show_table

print("STEP 1/7: simple baseline, no context")
train_and_score_baseline(use_context=False)

print("\nSTEP 2/7: simple baseline, with context")
train_and_score_baseline(use_context=True)

print("\nSTEP 3/7: BERT, no context")
train_and_score(model_name="bert-base-uncased", use_context=False)

print("\nSTEP 4/7: BERT, with context")
train_and_score(model_name="bert-base-uncased", use_context=True)

print("\nSTEP 5/7: RoBERTa, no context")
train_and_score(model_name="roberta-base", use_context=False)

print("\nSTEP 6/7: RoBERTa, with context")
train_and_score(model_name="roberta-base", use_context=True)

print("\nSTEP 7/7: final results")
show_table()
