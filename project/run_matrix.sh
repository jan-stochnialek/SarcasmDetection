#!/usr/bin/env bash
# Run the full LOCAL experiment matrix in one shot:
#   baseline (TF-IDF) + BERT/RoBERTa x {no-context, context}
# all at the SAME --subset so every model shares one train/test split (required
# for a fair comparison and for the McNemar test). Then build the report.
#
# Each run logs to logs/<name>.log and continues even if one run fails.
#
# Usage (from the project/ folder):
#   SUBSET=20000 EPOCHS=2 BATCH=32 bash run_matrix.sh
set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
SUBSET="${SUBSET:-20000}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-32}"
mkdir -p logs

echo "== local matrix: subset=$SUBSET epochs=$EPOCHS batch=$BATCH =="
date

# --- Baselines (seconds) — same subset as the transformers -> shared test split.
$PY run.py baseline --subset "$SUBSET"           > logs/baseline_noctx.log 2>&1 && echo "[ok] baseline noctx"
$PY run.py baseline --subset "$SUBSET" --context > logs/baseline_ctx.log   2>&1 && echo "[ok] baseline ctx"

# --- Transformers.
for MODEL in bert-base-uncased roberta-base; do
  $PY run.py train --model "$MODEL" --subset "$SUBSET" --epochs "$EPOCHS" --batch-size "$BATCH" \
      > "logs/${MODEL}_noctx.log" 2>&1 && echo "[ok] ${MODEL} noctx"
  $PY run.py train --model "$MODEL" --context --subset "$SUBSET" --epochs "$EPOCHS" --batch-size "$BATCH" \
      > "logs/${MODEL}_ctx.log" 2>&1 && echo "[ok] ${MODEL} ctx"
done

# --- Aggregate into results/summary.md + summary.csv.
$PY run.py report > logs/report.log 2>&1
date
echo "MATRIX COMPLETE"
