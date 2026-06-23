"""
data.py
=======
Turns the raw SARC CSV into model-ready inputs. This is the single source of
truth for the data pipeline, so the baseline and the transformers see EXACTLY
the same rows, splits and cleaning — a prerequisite for a fair comparison.

Pipeline stages:
  1. load_and_clean()         — read CSV, normalize text, drop bad rows, dedupe.
  2. make_splits()            — stratified subsample + train/val/test split.
  3. build_text_single()      — one string per row, for the TF-IDF baseline.
  4. build_text_pair()        — (context, comment) pair, for the transformers.
  5. build_tokenized_datasets() — tokenize into Hugging Face datasets.

Key design choice — CONTEXT:
  * "no context"  -> the model sees only the target comment.
  * "with context"-> the model sees the parent comment AND the target comment.
  This single switch is what lets us answer the project's main question (RQ2).
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DataConfig

# ---------------------------------------------------------------------------
# Precompiled regexes for cheap text normalization (compiled once, reused).
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")        # web links
_MENTION_RE = re.compile(r"/?u/\w+")                  # Reddit users: u/name or /u/name
_SARCASM_TAG_RE = re.compile(r"\s*/s\b", re.IGNORECASE)  # leftover '/s' marker
_WS_RE = re.compile(r"\s+")                           # runs of whitespace


def _normalize_text(text: object, cfg: DataConfig) -> str:
    """Apply light, transformer-friendly normalization to a single value.

    We deliberately keep this gentle: pretrained transformers handle raw,
    cased text well, so we only (a) strip any stray '/s' so the model can't
    cheat off the self-annotation marker, (b) optionally mask URLs/mentions,
    and (c) collapse whitespace.
    """
    if not isinstance(text, str):
        return ""  # NaN / missing -> empty string (dropped later)
    text = _SARCASM_TAG_RE.sub(" ", text)              # remove '/s'
    if cfg.normalize_urls:
        text = _URL_RE.sub("[URL]", text)
    if cfg.normalize_mentions:
        text = _MENTION_RE.sub("[USER]", text)
    return _WS_RE.sub(" ", text).strip()               # tidy whitespace


def load_and_clean(cfg: DataConfig) -> pd.DataFrame:
    """Load the SARC CSV and return a cleaned DataFrame.

    Returns columns: [label, comment, parent_comment, subreddit].
    """
    # Only read the columns we need (the full file has many) to save memory.
    wanted = {cfg.label_col, cfg.text_col, cfg.context_col, "subreddit"}
    df = pd.read_csv(cfg.raw_csv, usecols=lambda c: c in wanted)

    # We need BOTH the comment and its parent (the context model requires a parent).
    df = df.dropna(subset=[cfg.text_col, cfg.context_col])

    # Normalize the two text columns.
    df[cfg.text_col] = df[cfg.text_col].map(lambda t: _normalize_text(t, cfg))
    df[cfg.context_col] = df[cfg.context_col].map(lambda t: _normalize_text(t, cfg))

    # Drop rows that became empty after cleaning.
    df = df[(df[cfg.text_col].str.len() > 0) & (df[cfg.context_col].str.len() > 0)]

    # Remove exact duplicate (comment, label) pairs to reduce over-counting/leakage.
    df = df.drop_duplicates(subset=[cfg.text_col, cfg.label_col]).reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, cfg: DataConfig):
    """Stratified (optional) subsample, then a stratified train/val/test split.

    'Stratified' = each split keeps the same 50/50 sarcastic ratio as the whole,
    so accuracy stays interpretable and no split is accidentally imbalanced.

    Returns three DataFrames: (train, val, test).
    """
    # Optional: shrink the data for development on a laptop / free GPU.
    if cfg.subset_size is not None and cfg.subset_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=cfg.subset_size,
            stratify=df[cfg.label_col],
            random_state=cfg.seed,
        )

    # Step 1: peel off the TEST set.
    train_val, test = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=df[cfg.label_col],
        random_state=cfg.seed,
    )

    # Step 2: split the remainder into TRAIN and VALIDATION.
    # The val fraction is expressed relative to the remaining (train+val) portion.
    val_relative = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val[cfg.label_col],
        random_state=cfg.seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_text_single(df: pd.DataFrame, cfg: DataConfig, use_context: bool) -> list[str]:
    """Build ONE string per row, for the TF-IDF baseline.

    * no context : just the comment.
    * with context: "parent [SEP] comment" so bag-of-words sees both.
    """
    if use_context:
        return (df[cfg.context_col] + " [SEP] " + df[cfg.text_col]).tolist()
    return df[cfg.text_col].tolist()


def build_text_pair(df: pd.DataFrame, cfg: DataConfig, use_context: bool):
    """Build inputs for transformer tokenization.

    Returns (texts_a, texts_b):
      * no context : (comment, None)               -> single-sequence input.
      * with context: (parent, comment)            -> a SENTENCE PAIR, tokenized
        as `[CLS] parent [SEP] comment [SEP]` (BERT) so the model can compare the
        reply against what it is replying to.
    """
    if use_context:
        return df[cfg.context_col].tolist(), df[cfg.text_col].tolist()
    return df[cfg.text_col].tolist(), None


def build_tokenized_datasets(tokenizer, splits: dict, cfg: DataConfig,
                             use_context: bool, max_length: int):
    """Tokenize each split into a Hugging Face DatasetDict.

    `splits` is a dict like {"train": df, "validation": df, "test": df}.

    Truncation note (important): when context is on we use
    `truncation="longest_first"`, which trims the LONGER sequence first. In the
    common case (short comment, longer parent) this preferentially trims the
    context and keeps the target comment intact, while remaining robust when a
    comment is itself longer than max_length (which "only_first" would crash on).
    """
    from datasets import Dataset, DatasetDict  # imported lazily (heavy dependency)

    tokenized = {}
    for name, frame in splits.items():
        a, b = build_text_pair(frame, cfg, use_context)

        if b is None:
            # Single-sequence: just the comment.
            enc = tokenizer(a, truncation=True, max_length=max_length)
        else:
            # Sentence pair. Truncate the LONGER sequence first: in the common case
            # (short comment, longer parent) this trims the context and leaves the
            # comment intact, but — unlike "only_first" — it cannot crash when a
            # comment is itself longer than max_length.
            enc = tokenizer(a, b, truncation="longest_first", max_length=max_length)

        # We do NOT pad here — the DataCollatorWithPadding pads each batch
        # dynamically at train time, which is faster and uses less memory.
        data = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": frame[cfg.label_col].tolist(),
        }
        # BERT returns token_type_ids (segment embeddings); RoBERTa does not.
        if "token_type_ids" in enc:
            data["token_type_ids"] = enc["token_type_ids"]

        tokenized[name] = Dataset.from_dict(data)

    return DatasetDict(tokenized)
