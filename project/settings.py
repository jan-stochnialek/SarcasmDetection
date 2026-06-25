# ---------------------------------------------------------------------------
# How many comments to use.
# The full data file has about 1,000,000 comments, which is slow to train on.
# Start small to check everything works, then make it bigger for better results.
# Set this to None to use ALL the comments (only do that on a fast GPU).
# ---------------------------------------------------------------------------
SAMPLE_SIZE = None     # full dataset (intended for a cloud A100/H100 run)

# ---------------------------------------------------------------------------
# How many times the transformer reads through the training data while learning.
# 2 or 3 is normal. More can be a little better but takes longer.
# (This setting does not affect the simple baseline model.)
# ---------------------------------------------------------------------------
EPOCHS = 2

# ---------------------------------------------------------------------------
# How many comments the model works on at once.
# 64 suits a big cloud GPU (A100/H100 80GB). On a smaller GPU lower it to 16, and
# if you still get an "out of memory" error make it smaller again (try 8).
# ---------------------------------------------------------------------------
BATCH_SIZE = 96

# ---------------------------------------------------------------------------
# A fixed number that makes the random parts (shuffling, sampling) behave the
# same way every time, so your results are repeatable. Leave it as is.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# how big each learning step is
LEARNING_RATE = 2e-5

# ---------------------------------------------------------------------------
# The most word-pieces the transformer reads from one input. Anything longer is
# trimmed. 256 is plenty for a Reddit comment, and using the SAME limit whether
# or not context is included keeps the comparison fair (see engine/transformer.py).
# (This setting does not affect the simple baseline model.)
# ---------------------------------------------------------------------------
MAX_TOKENS = 256
