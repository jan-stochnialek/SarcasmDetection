"""
check_data.py
=============
RUN FIRST.

Loads the Reddit data, cleans it, and prints how many comments there are and
how they split into a training set and a test set. If this runs without errors,
your data is set up correctly and you can move on to the train_*.py scripts.
"""

from engine.data import load_comments, split_train_test

# Load and clean all the comments.
comments = load_comments()
print(f"Loaded {len(comments):,} comments after cleaning.")

# Split them into a training part and a test part.
train, test = split_train_test(comments)
print(f"Training comments: {len(train):,}")
print(f"Test comments:     {len(test):,}")

# The data should be about half sarcastic, half not — check that here.
print(f"Sarcastic fraction in the test set: {test['label'].mean():.2f}  (expect ~0.50)")
print("\nData looks good. You can now run the train_*.py scripts.")
