"""
train.py
========
Fine-tune a pretrained transformer (BERT or RoBERTa) for sarcasm classification
(plan Sections 5.2 / 5.3).

The Hugging Face `Trainer` handles the training loop, mixed precision, evaluation
and checkpointing for us. Crucially, the ONLY difference between the two main
experimental conditions is how the input text is built (see
`data.build_tokenized_datasets`): everything else — model, optimizer, schedule,
seeds — is identical, so any performance gap is attributable to context (RQ2).

Outputs written to results/:
  * <run_name>_metrics.json  — test-set metrics.
  * <run_name>_preds.npz     — per-example test predictions (used by McNemar).
"""

from __future__ import annotations

import numpy as np

from config import (
    DataConfig,
    MODELS_DIR,
    RESULTS_DIR,
    TransformerConfig,
    get_device,
)
from data import build_tokenized_datasets, load_and_clean, make_splits
from evaluate import compute_classification_metrics
from utils import ensure_dir, get_logger, save_json, set_seed

log = get_logger()


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis (logits -> probabilities)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _compute_metrics(eval_pred):
    """Adapter the HF Trainer calls after each evaluation.

    `eval_pred` is (logits, labels). We convert logits to (a) class predictions
    via argmax and (b) positive-class probabilities via softmax, then reuse our
    shared metric function so val and test are scored the same way.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    probs = _softmax(logits)[:, 1]
    return compute_classification_metrics(labels, preds, probs)


def train_one(data_cfg: DataConfig, model_cfg: TransformerConfig):
    """Fine-tune and evaluate ONE (model, context, seed) configuration.

    Returns (test_metrics, run_name).
    """
    # Heavy imports kept local so the lightweight commands (e.g. baseline) start
    # fast and don't require transformers to be installed.
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    set_seed(model_cfg.seed)
    device = get_device()
    log.info(
        "Device=%s | model=%s | context=%s | seed=%d",
        device, model_cfg.model_name, model_cfg.use_context, model_cfg.seed,
    )

    # --- 1. Data (identical pipeline to the baseline) ----------------------
    df = load_and_clean(data_cfg)
    train_df, val_df, test_df = make_splits(df, data_cfg)
    splits = {"train": train_df, "validation": val_df, "test": test_df}

    # --- 2. Tokenizer + tokenized datasets (context handled inside) --------
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    datasets = build_tokenized_datasets(
        tokenizer, splits, data_cfg, model_cfg.use_context, model_cfg.max_length
    )
    # Dynamic padding: pad each batch to its own longest sequence (efficient).
    collator = DataCollatorWithPadding(tokenizer)

    # --- 3. Model: pretrained encoder + a fresh 2-class head ---------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.model_name, num_labels=2
    )

    # --- 4. Training arguments --------------------------------------------
    # fp16 mixed precision only helps (and is only stable) on CUDA. On MPS/CPU
    # we train in fp32, so we leave it off.
    use_fp16 = device.type == "cuda"
    ctx_tag = "ctx" if model_cfg.use_context else "noctx"
    short_name = model_cfg.model_name.split("/")[-1]
    run_name = f"{short_name}_{ctx_tag}_seed{model_cfg.seed}"
    out_dir = ensure_dir(MODELS_DIR / run_name)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=model_cfg.batch_size,
        per_device_eval_batch_size=model_cfg.batch_size,
        learning_rate=model_cfg.learning_rate,
        num_train_epochs=model_cfg.epochs,
        weight_decay=model_cfg.weight_decay,
        warmup_ratio=model_cfg.warmup_ratio,
        eval_strategy="epoch",       # evaluate on val once per epoch
        save_strategy="epoch",       # checkpoint once per epoch (matches eval)
        load_best_model_at_end=True, # keep the epoch with the best val F1
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=use_fp16,
        seed=model_cfg.seed,
        report_to="none",            # no W&B/TensorBoard unless you want it
        logging_steps=50,
    )

    # --- 5. Trainer: ties model, data, collator and metrics together -------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,  # transformers >=5 renamed this from `tokenizer=`
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    # --- 6. Train, then evaluate on the held-out TEST split ----------------
    trainer.train()
    test_out = trainer.predict(datasets["test"])
    test_preds = test_out.predictions.argmax(axis=-1)
    test_probs = _softmax(test_out.predictions)[:, 1]
    test_metrics = compute_classification_metrics(
        test_out.label_ids, test_preds, test_probs
    )
    log.info("TEST metrics for %s: %s", run_name, test_metrics)

    # --- 7. Persist metrics + per-example predictions ----------------------
    # The .npz keeps y_true and y_pred so `run.py compare` can run McNemar
    # between, e.g., the context and no-context runs.
    save_json(test_metrics, RESULTS_DIR / f"{run_name}_metrics.json")
    np.savez(
        ensure_dir(RESULTS_DIR) / f"{run_name}_preds.npz",
        y_true=test_out.label_ids,
        y_pred=test_preds,
    )
    return test_metrics, run_name
