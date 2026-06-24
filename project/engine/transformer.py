"""
engine/transformer.py
=====================
Fine-tunes a transformer model (BERT or RoBERTa) to detect sarcasm.

"Fine-tuning" means: take a model that ALREADY understands English (it was trained
on a huge amount of text by Google/Facebook) and train it a little more on our
sarcasm examples. That is much cheaper than training from scratch.

This is fast on a GPU. It also runs on a normal computer, just slowly.

You do not need to edit this file.
"""

from pathlib import Path

import numpy as np

import settings
from engine.data import get_train_and_test
from engine.scoring import measure, save_result

# Folder where the trainer keeps temporary files while it works.
MODELS_FOLDER = Path(__file__).resolve().parent.parent.parent / "models"


def _to_probabilities(raw_numbers):
    """Turn the model's raw output numbers into probabilities that add up to 1."""
    shifted = raw_numbers - raw_numbers.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_and_score(model_name, use_context):
    """Fine-tune one transformer and grade it on the test set.

    model_name   "bert-base-uncased" or "roberta-base"
    use_context  True  -> the model reads the parent comment AND the comment
                 False -> the model reads only the comment
    """
    # These libraries are big, so we import them inside the function. That way the
    # quick scripts (like the baseline) don't have to load them.
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    train_data, test_data = get_train_and_test()

    # A friendly name for this run, used in the results table.
    nice_name = "BERT" if "bert-base" in model_name else "RoBERTa"
    if use_context:
        nice_name += " (with context)"
    print(f"\nTraining: {nice_name}   (model = {model_name})")

    # 1. The tokenizer chops text into pieces and turns them into numbers.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. A small helper that tokenizes a batch of rows. With context we hand the
    #    tokenizer TWO texts (parent, comment) so the model reads them as a pair.
    #    Both modes use the SAME length limit (settings.MAX_TOKENS), so there is no
    #    length confound in the comparison. With context, "longest_first" trims
    #    whichever of the two is longer until the pair fits — in practice comments
    #    are short, so the comment is kept whole and only the parent is shortened.
    def to_tokens(rows):
        if use_context:
            return tokenizer(rows["parent_comment"], rows["comment"],
                             truncation="longest_first", max_length=settings.MAX_TOKENS)
        return tokenizer(rows["comment"], truncation=True, max_length=settings.MAX_TOKENS)

    # 3. Convert our pandas tables into the format the trainer wants, then tokenize.
    #    The model expects the answer column to be called "labels".
    columns = ["comment", "parent_comment", "label"]
    train_dataset = Dataset.from_pandas(train_data[columns]).map(to_tokens, batched=True)
    test_dataset = Dataset.from_pandas(test_data[columns]).map(to_tokens, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # 4. Load the pretrained model and add a fresh output layer with 2 choices
    #    (sarcastic / not sarcastic). Seed first so the new layer's random starting
    #    weights — and the training-time shuffling — are the same every run, which
    #    makes the results as reproducible as a GPU allows.
    set_seed(settings.RANDOM_SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        attn_implementation="sdpa",    # PyTorch's faster scaled-dot-product attention
    )

    # 5. Use the GPU if there is one — training is much faster on a GPU. On modern
    #    GPUs (A100/H100) bf16 is faster and more numerically stable than fp16;
    #    older GPUs (e.g. a T4) don't support bf16, so we fall back to fp16 there.
    using_gpu = torch.cuda.is_available()
    use_bf16 = using_gpu and torch.cuda.is_bf16_supported()

    # 6. The settings for the training process.
    options = TrainingArguments(
        output_dir=str(MODELS_FOLDER / _file_safe(nice_name)),
        num_train_epochs=settings.EPOCHS,
        per_device_train_batch_size=settings.BATCH_SIZE,
        per_device_eval_batch_size=settings.BATCH_SIZE,
        learning_rate=settings.LEARNING_RATE,
        seed=settings.RANDOM_SEED,
        bf16=use_bf16,                       # fast path on A100/H100
        fp16=using_gpu and not use_bf16,     # fallback for older GPUs (e.g. T4)
        save_strategy="no",            # don't save checkpoints (keeps the disk tidy)
        report_to="none",
        logging_steps=50,
        dataloader_num_workers=2,      # prepare batches in the background so the GPU waits less
    )

    # 7. The Trainer runs the whole training loop for us.
    trainer = Trainer(
        model=model,
        args=options,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    # 8. Ask the trained model to predict on the test comments.
    output = trainer.predict(test_dataset)
    raw_numbers = output.predictions
    predicted_labels = raw_numbers.argmax(axis=1)               # 0 or 1 per comment
    predicted_scores = _to_probabilities(raw_numbers)[:, 1]     # chance of sarcastic

    # 9. Score it and save the result.
    scores = measure(output.label_ids, predicted_labels, predicted_scores)
    save_result(nice_name, scores, output.label_ids, predicted_labels)
    return scores


def _file_safe(name):
    """Make a folder-safe version of a run name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "")
