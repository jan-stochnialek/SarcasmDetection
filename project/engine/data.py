"""
Loads the Reddit comments from the CSV file, cleans the text, and splits the
comments into a TRAINING set (used to teach the models) and a TEST set (used to
grade them on comments they have never seen).
"""
import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import settings


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER.parent / "data" / "raw"
DEFAULT_FILE = DATA_FOLDER / "train-balanced-sarcasm.csv"
DATA_FILE = os.environ.get("SARC_CSV", str(DEFAULT_FILE))


def _clean_text(text):
    """Tidy up one comment.
      - remove the Reddit "/s" sarcasm tag (otherwise the model could just look
        for that tag instead of actually learning),
      - replace web links and usernames with simple placeholders,
      - squeeze repeated spaces into one.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s*/s\b", " ", text, flags=re.IGNORECASE)   # drop the "/s" tag
    text = re.sub(r"http\S+|www\.\S+", "[link]", text)          # web links
    text = re.sub(r"/?u/\w+", "[user]", text)                   # usernames like u/bob
    text = re.sub(r"\s+", " ", text).strip()                    # tidy the spaces
    return text


def load_comments():
    """Read the CSV file, clean it, and return a table of comments.

    The returned table has three columns:
      label          1 = sarcastic, 0 = not sarcastic
      comment        the comment we want to classify
      parent_comment the earlier comment it was replying to (the "context")
    """
    # Read only the three columns we use (the file has more) to save time/memory.
    data = pd.read_csv(DATA_FILE, usecols=["label", "comment", "parent_comment"])

    # Remove rows where the comment OR its parent is missing.
    data = data.dropna(subset=["comment", "parent_comment"])

    # Clean both text columns. `.apply(f)` runs the function f on every value.
    data["comment"] = data["comment"].apply(_clean_text)
    data["parent_comment"] = data["parent_comment"].apply(_clean_text)

    # Drop any rows that became empty after cleaning.
    data = data[(data["comment"] != "") & (data["parent_comment"] != "")]

    # If SAMPLE_SIZE is set, randomly keep only that many comments (much faster).
    # `random_state` makes the random choice the same every run, so results match.
    if settings.SAMPLE_SIZE is not None and settings.SAMPLE_SIZE < len(data):
        data = data.sample(n=settings.SAMPLE_SIZE, random_state=settings.RANDOM_SEED)

    # `reset_index` just re-numbers the rows 0, 1, 2, ... after all that filtering.
    return data.reset_index(drop=True)


def split_train_test(data):
    """Split the comments: 85% for training, 15% for testing.

    We split on the PARENT comment (the thread), not on individual comments, so
    that every reply under a given parent lands entirely in training OR entirely
    in testing — never both. This matters because balanced SARC pairs a sarcastic
    and a non-sarcastic reply under the SAME parent; a naive comment-level split
    would let that parent's text show up in both training and testing and quietly
    leak thread context into the with-context evaluation — the very thing we are
    trying to measure. The fixed `random_state` keeps the split identical every
    run, so results (and the with/without-context comparison) stay reproducible.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=settings.RANDOM_SEED
    )
    train_idx, test_idx = next(splitter.split(data, groups=data["parent_comment"]))
    train_data = data.iloc[train_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)
    return train_data, test_data


def get_train_and_test():
    """Do everything in one call: load + clean + split."""
    return split_train_test(load_comments())
