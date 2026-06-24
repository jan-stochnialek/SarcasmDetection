"""
The SIMPLE model: TF-IDF + logistic regression.

  - TF-IDF turns each comment into a list of numbers, one per word, where rare,
    distinctive words count for more than common words like "the".
  - Logistic regression then learns a line that separates sarcastic comments from
    non-sarcastic ones in that number-space.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from engine.data import get_train_and_test
from engine.scoring import measure, save_result


def train_and_score_baseline(use_context):
    train_data, test_data = get_train_and_test()

    # Join parent + comment if the run is with context, otherwise just use the comment.
    if use_context:
        train_text = train_data["parent_comment"] + " " + train_data["comment"]
        test_text = test_data["parent_comment"] + " " + test_data["comment"]
        name = "Baseline (with context)"
    else:
        train_text = train_data["comment"]
        test_text = test_data["comment"]
        name = "Baseline"

    # A "pipeline" chains steps together: first TF-IDF turns text into numbers,
    # then logistic regression classifies it. ngram_range=(1, 2) means we count
    # single words AND word-pairs (so "yeah right" is also captured).
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000),
    )

    # Train model
    model.fit(train_text, train_data["label"])

    # Predict on test data
    predicted_labels = model.predict(test_text)
    
    # "[:, 1]" takes the probability of the SARCASTIC class for each comment.
    predicted_scores = model.predict_proba(test_text)[:, 1]

    # Score it and save the result.
    scores = measure(test_data["label"], predicted_labels, predicted_scores)
    save_result(name, scores, test_data["label"], predicted_labels)
    return scores
