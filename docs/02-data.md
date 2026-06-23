# 02 — The data pipeline (`data.py`)

Everything that converts the raw CSV into model inputs lives in
[`../project/data.py`](../project/data.py). The baseline and the transformers
both call these functions, which guarantees they train and test on **exactly the
same rows** — essential for a fair comparison.

## Stage 1 — `load_and_clean(cfg)`

1. Reads only the columns we need (`label`, `comment`, `parent_comment`,
   `subreddit`) to save memory.
2. Drops rows missing a comment or a parent (the context model needs a parent).
3. Normalizes text lightly with `_normalize_text()`:
   - removes any stray `/s` sarcasm marker so the model can't cheat off it;
   - optionally replaces URLs with `[URL]` and `u/name` mentions with `[USER]`;
   - collapses repeated whitespace.
4. Drops rows that became empty after cleaning.
5. De-duplicates identical `(comment, label)` pairs to reduce leakage/over-counting.

> We keep normalization gentle on purpose: pretrained transformers are trained on
> raw, cased text, so aggressive cleaning (stopword removal, stemming) would hurt
> them. The TF-IDF vectorizer does its own lowercasing internally.

## Stage 2 — `make_splits(df, cfg)`

Produces **train / validation / test** DataFrames:

- Optional **stratified subsample** (`DataConfig.subset_size`) to keep runs fast on
  limited hardware. "Stratified" means the 50/50 sarcastic balance is preserved.
- Splits off the **test** set first, then carves **validation** out of the rest.
- A fixed `seed` makes the val/test sets **identical across every model**, so the
  comparison (and the McNemar test later) is valid.

Default proportions: 80% train / 10% val / 10% test.

## Stage 3 — building model inputs

There are two builders because the baseline and the transformers consume text
differently.

### `build_text_single(df, cfg, use_context)` — for TF-IDF

Returns one string per row:
- no context → `comment`
- with context → `"parent [SEP] comment"`

### `build_text_pair(df, cfg, use_context)` — for transformers

Returns a pair `(texts_a, texts_b)`:
- no context → `(comment, None)` → single-sequence input.
- with context → `(parent, comment)` → a **sentence pair**.

## Stage 4 — `build_tokenized_datasets(...)`

Tokenizes each split into a Hugging Face `DatasetDict`. Two important details:

1. **How context is encoded.** Passing two texts to the tokenizer produces the
   model's native pair format:
   - BERT: `[CLS] parent [SEP] comment [SEP]` (plus segment/`token_type_ids`).
   - RoBERTa: `<s> parent </s></s> comment </s>` (no segment ids; that's expected).
   This lets the model *compare* the reply to what it is replying to — the whole
   point of using context.

2. **Truncation favours the comment.** With context we use
   `truncation="longest_first"`, which trims whichever sequence is currently
   longer. In the common case (short comment, longer parent) this trims the
   **context** and leaves the target comment intact — but, unlike `"only_first"`,
   it stays robust when a comment is itself longer than `max_length` (that case
   makes `"only_first"` raise a truncation error).

We deliberately **do not pad** during tokenization — `DataCollatorWithPadding`
(in `train.py`) pads each batch to its own longest example, which is faster and
uses less memory.

## What "context" means here

The Kaggle balanced file gives the **immediate parent comment** (one level up).
That is enough to answer the project question. The original SARC release contains
the *full ancestor chain* if you later want multi-turn context — you would extend
`build_text_pair` to concatenate several ancestors.
