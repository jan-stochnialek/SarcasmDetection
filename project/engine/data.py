"""
Loads the Reddit comments from the CSV file, cleans the text, and splits the
comments into a TRAINING set (used to teach the models) and a TEST set (used to
grade them on comments they have never seen).
"""

import os
import re
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import settings


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER.parent / "data" / "raw"
DEFAULT_FILE = DATA_FOLDER / "train-balanced-sarcasm.csv"
DATA_FILE = os.environ.get("SARC_CSV", str(DEFAULT_FILE))

# SARC's OFFICIAL held-out test set (used only when settings.USE_OFFICIAL_TEST_SPLIT
# is True). On Kaggle this ships in the raw SARC format, whose comment text lives in
# a separate comments.json — see load_official_test() below. Both locations can be
# overridden with environment variables (handy on Kaggle).
DEFAULT_TEST_FILE = DATA_FOLDER / "test-balanced.csv"
TEST_FILE = os.environ.get("SARC_TEST_CSV", str(DEFAULT_TEST_FILE))
COMMENTS_JSON = os.environ.get("SARC_COMMENTS_JSON", str(DATA_FOLDER / "comments.json"))


def _clean_text(text):
    """Tidy up one comment.

    Pretrained models prefer normal, raw text, so we only do a little cleaning:
      - remove the Reddit "/s" sarcasm tag (otherwise the model could just look
        for that tag instead of actually learning),
      - replace web links and usernames with simple placeholders,
      - squeeze repeated spaces into one.
    """
    # Some rows are empty/missing; pandas stores those as a non-text value.
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
    """Produce a (training, testing) pair of tables.

    Controlled by settings.USE_OFFICIAL_TEST_SPLIT:
      False (default) - carve a held-out test set out of the training file with a
                        parent-grouped split (see split_train_test).
      True            - train on the whole training file and grade on SARC's
                        OFFICIAL held-out test set (see load_official_test), for
                        numbers comparable to the published SARC results.
    """
    if settings.USE_OFFICIAL_TEST_SPLIT:
        train_data = load_comments()
        test_data = load_official_test()
        return train_data, test_data
    return split_train_test(load_comments())


def load_official_test():
    """Load SARC's official held-out test set as a label/comment/parent table.

    The Kaggle dataset ships the test split in the ORIGINAL ("raw") SARC format,
    which is different from the friendly train-balanced-sarcasm.csv: every line is

        post_id ancestor_id ... | response_id ... | label ...

    (the three sections separated by "|", ids space-separated) and the actual
    comment text lives in a separate comments.json lookup. We detect the format
    automatically: if the file already has the friendly columns we read them
    directly; otherwise we rebuild the table from the raw ids + comments.json.
    """
    path = Path(TEST_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Official test file not found: {path}\n"
            "Either set USE_OFFICIAL_TEST_SPLIT = False in settings.py, or put "
            "test-balanced.csv from the SARC/Kaggle dataset into data/raw/ "
            "(or point the SARC_TEST_CSV environment variable at it)."
        )

    # Peek at the header only (no comment rows) to tell the two formats apart.
    header = list(pd.read_csv(path, nrows=0).columns)
    if {"label", "comment", "parent_comment"}.issubset(header):
        data = pd.read_csv(path, usecols=["label", "comment", "parent_comment"])
    else:
        data = _load_raw_sarc_test(path)

    # Clean exactly like the training data so the two are comparable.
    data = data.dropna(subset=["comment", "parent_comment"])
    data["comment"] = data["comment"].apply(_clean_text)
    data["parent_comment"] = data["parent_comment"].apply(_clean_text)
    data = data[(data["comment"] != "") & (data["parent_comment"] != "")]
    return data.reset_index(drop=True)


def _load_raw_sarc_test(path):
    """Rebuild a label/comment/parent table from a raw-format SARC file.

    Needs comments.json (the id -> comment-text lookup) alongside it.
    """
    comments_path = Path(COMMENTS_JSON)
    if not comments_path.exists():
        raise FileNotFoundError(
            f"'{path.name}' is in the raw SARC format, which needs the comment-text "
            f"lookup, but comments.json was not found at {comments_path}.\n"
            "Download comments.json from the SARC/Kaggle dataset into data/raw/ "
            "(or point the SARC_COMMENTS_JSON environment variable at it)."
        )

    with open(comments_path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    def text_of(comment_id):
        entry = comments.get(comment_id)
        return entry.get("text", "") if entry else None

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            context_ids = parts[0].split()
            response_ids = parts[1].split()
            labels = parts[2].split()
            if not context_ids or len(response_ids) != len(labels):
                continue
            parent_text = text_of(context_ids[-1])   # immediate parent = last context id
            if not parent_text:
                continue
            for response_id, label in zip(response_ids, labels):
                comment_text = text_of(response_id)
                if comment_text:
                    rows.append((int(label), comment_text, parent_text))

    return pd.DataFrame(rows, columns=["label", "comment", "parent_comment"])
