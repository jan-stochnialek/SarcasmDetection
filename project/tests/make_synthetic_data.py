"""
make_synthetic_data.py
======================
Generate a SMALL, SYNTHETIC dataset in the SARC CSV format, purely to smoke-test
the pipeline without the real ~1 GB Kaggle download.

The text is meaningless — it only carries a mild, ARTIFICIAL signal so the
metrics come out above chance, and so that 'context' (the parent comment) is
genuinely informative, which exercises that code path.

    DO NOT interpret any numbers produced from this data. They say nothing about
    real sarcasm detection. Replace with the real train-balanced-sarcasm.csv for
    actual experiments.

Usage:
    python tests/make_synthetic_data.py [output.csv] [n_rows]
"""

from __future__ import annotations

import random
import sys

import pandas as pd

# Tiny vocabularies. The 'setup' words let the PARENT comment carry signal, so a
# context-aware model can do better than a comment-only one (mirrors RQ2).
NEUTRAL = "the a it is and to of for on with about today game team movie food work day".split()
SARC_MARKERS = [
    "oh great", "yeah right", "totally", "wow amazing", "so thrilled",
    "love that", "brilliant idea", "sure buddy", "fantastic news", "best ever",
]
NEG_SETUP = ["broken", "late", "failed", "worst", "again", "delayed", "rude", "expensive"]
POS_SETUP = ["nice", "helpful", "clear", "fast", "kind", "useful", "fair", "smooth"]


def _sentence(words: list[str], k_lo: int, k_hi: int, rng: random.Random) -> str:
    """A random sentence of k_lo..k_hi words drawn from `words`."""
    k = rng.randint(k_lo, k_hi)
    return " ".join(rng.choice(words) for _ in range(k))


def generate(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Return a balanced synthetic DataFrame with SARC columns.

    Columns: label, comment, parent_comment, subreddit.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % 2  # exactly balanced (alternating), shuffled below
        comment = _sentence(NEUTRAL, 4, 9, rng)

        if label == 1:
            # Sarcastic: usually (not always) contains a marker -> noisy signal.
            if rng.random() < 0.8:
                comment += " " + rng.choice(SARC_MARKERS)
            parent = _sentence(NEUTRAL, 3, 7, rng) + " " + rng.choice(NEG_SETUP)
        else:
            # Non-sarcastic: occasionally a misleading marker so it isn't separable.
            if rng.random() < 0.1:
                comment += " " + rng.choice(SARC_MARKERS)
            parent = _sentence(NEUTRAL, 3, 7, rng) + " " + rng.choice(POS_SETUP)

        # Append a unique token so rows aren't collapsed by de-duplication.
        # (min_df>=2 in the vectorizer drops these singletons, so they don't leak.)
        comment += f" u{i}"

        rows.append(
            {
                "label": label,
                "comment": comment,
                "parent_comment": parent,
                "subreddit": rng.choice(["aww", "news", "movies"]),
            }
        )

    # Shuffle so the alternating labels aren't ordered.
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_sarc.csv"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    generate(n_rows).to_csv(out, index=False)
    print(f"wrote {n_rows} synthetic rows to {out}")
