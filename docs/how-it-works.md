# How it works (the ideas behind the project)

This explains, in plain language, what the project does and what the models are.
For *how to run it*, see the project's [README](../project/README.md). For Kaggle,
see [kaggle.md](kaggle.md).

---

## The problem

**Sarcasm** is saying the opposite of what you mean ("Oh *great*, another Monday").
It's hard for a computer because the literal words point one way and the real
meaning points the other. The project asks one main question:

> Does showing the model the **previous comment** (the conversation *context*) help
> it spot sarcasm, compared to showing only the comment by itself?

---

## The data: SARC

We use **SARC** (Self-Annotated Reddit Corpus). Redditers mark their own sarcastic
comments by adding **`/s`** at the end, which gives a huge, author-labelled dataset.
We use the balanced version (half sarcastic, half not). Each row has:

- `comment` — the comment we classify,
- `parent_comment` — the comment it replied to (**the context**),
- `label` — 1 = sarcastic, 0 = not.

---

## The three models

We build three kinds of model and compare them on the same comments.

### 1. The baseline — TF-IDF + Logistic Regression
A model needs **numbers**, not words. **TF-IDF** turns each comment into a list of
numbers — one per word — where common words ("the", "and") count for little and
rare, distinctive words count for more. **Logistic Regression** then learns to
separate sarcastic from non-sarcastic comments using those numbers.

It's fast and runs on any computer, but it only knows *which words appear*, not
their **order** or **meaning** ("dog bites man" = "man bites dog" to it). That's why
it's our **baseline** — the score to beat.

### 2 & 3. The transformers — BERT and RoBERTa
**BERT** is a neural network that actually understands language in context. Two
ideas make it work:
- **Attention:** it builds each word's meaning by looking at the words around it, so
  "bank" means different things in "river bank" and "bank account".
- **Pre-training:** before we touch it, Google trained it on a huge amount of text by
  hiding random words and making it guess them. To do that well it had to learn
  grammar and meaning — so it arrives already understanding English.

We **fine-tune** it: take the pre-trained model and train it a little more on our
sarcasm examples. We run it two ways — **comment only**, and **comment + context**
(where it reads the parent comment and the comment together).

**RoBERTa** is the *same design as BERT, trained better* (more data, smarter
training). It usually scores a little higher.

| | TF-IDF | BERT | RoBERTa |
|---|---|---|---|
| Understands meaning / word order? | no | yes | yes |
| Pre-trained on huge text? | no | yes | yes (more) |
| Needs a GPU? | no | yes (to be fast) | yes (to be fast) |
| Role here | baseline | main model | improved main model |

---

## How "context" is added

For the baseline, context just means gluing the parent comment in front of the
comment. For the transformers, we feed the two as a **pair** (`parent` + `comment`)
so the model can *relate* the reply to what it's replying to — which is often what
reveals the sarcasm.

---

## How we measure

Because the data is 50/50, **accuracy** is meaningful, but we also report
**precision**, **recall**, **F1**, and **AUC** (see `engine/scoring.py`).

We also run a **significance test** (McNemar's test). The context effect is small,
and training has randomness, so a tiny difference could be luck. The test looks only
at the comments where the two models *disagree* and asks whether the context model
is reliably the right one. It gives a **p-value**: below 0.05 means the difference is
probably real, not chance.

---

## What we expect (and found)

- **Transformers beat the baseline** — they understand language; TF-IDF just counts
  words.
- **Context helps the transformer** by a small but real amount, because it can use
  the parent comment. In our runs, context improved BERT and the McNemar test said it
  was statistically significant.
- **Context does *not* help the bag-of-words baseline** (it can even hurt it) —
  extra words just add noise to a model that can't relate them.
- **RoBERTa ≥ BERT.**

Remember the scores stay modest (mid-0.70s to low-0.80s). Sarcasm is genuinely hard,
and the labels are noisy (self-annotated), so **even humans only reach ~0.81–0.83**.
Modest numbers are the honest, correct outcome — not a mistake.
