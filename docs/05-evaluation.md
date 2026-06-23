# 05 — Evaluation (`evaluate.py`)

[`../project/evaluate.py`](../project/evaluate.py) holds the scoring code shared by
the baseline and the transformers, so every model is measured the same way.

## Metrics — `compute_classification_metrics(y_true, y_pred, y_prob)`

Because the dataset is balanced, **accuracy** is meaningful, but we report a full
set (plan Section 6.3):

| Metric | What it tells you |
|--------|-------------------|
| **accuracy** | fraction of comments classified correctly |
| **precision** | of those predicted sarcastic, how many really are |
| **recall** | of the truly sarcastic, how many we caught |
| **F1** | harmonic mean of precision & recall (sarcastic = positive class) |
| **macro_f1** | F1 averaged equally over both classes |
| **roc_auc** | ranking quality across all thresholds (needs probabilities) |

The sarcastic class (label `1`) is the positive class throughout.

## Confusion matrix — `save_confusion_matrix(...)`

Saves a 2×2 PNG (true vs predicted, `not sarc` / `sarc`) so you can see *where*
errors fall — e.g. whether the model over-predicts sarcasm. Useful for the error
analysis in `plan.md` Section 7.

## Significance — `mcnemar_test(y_true, pred_a, pred_b)`

This is how you decide whether **context vs no-context** is a *real* difference,
not chance.

When two models are evaluated on the **same** test examples, their results are
**paired**, so an ordinary comparison of accuracies is statistically invalid.
McNemar's test looks only at the examples where the two models **disagree**:

- `n01` = A wrong, B right
- `n10` = A right, B wrong

Under the null hypothesis "both models are equally good", each disagreement is a
fair coin flip. We run an **exact two-sided binomial test** on those discordant
counts. The function returns:

```json
{
  "n_A_wrong_B_right": ...,
  "n_A_right_B_wrong": ...,
  "p_value": ...,
  "significant_at_0.05": true/false
}
```

A `p_value < 0.05` means the difference between the two models is unlikely to be
random.

> The test requires both prediction files to come from the **same test set**.
> `run.py compare` checks this by comparing the stored `y_true` arrays and errors
> out if they differ — which is why the fixed `DataConfig.seed` matters.

## Putting it together

```bash
# After training BERT both ways with the same seed:
python run.py compare \
    --a ../results/bert-base-uncased_noctx_seed42_preds.npz \
    --b ../results/bert-base-uncased_ctx_seed42_preds.npz
```

Report, for each model family, the **context delta** (metric with context minus
without) together with the McNemar p-value. That pair of numbers — effect size and
significance — is the direct answer to the project's main question.
