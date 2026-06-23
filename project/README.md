# Project code — SARC sarcasm detection

All source code lives in this folder. **The full documentation is in
[`../docs`](../docs)** (start with [`../docs/README.md`](../docs/README.md)).

## Quick start

```bash
pip install -r requirements.txt
# Download train-balanced-sarcasm.csv from Kaggle (danofer/sarcasm)
# and place it in ../data/raw/

python run.py prepare                                   # check the data pipeline
python run.py baseline                                  # TF-IDF, no context
python run.py baseline --context                        # TF-IDF, with context
python run.py train --model bert-base-uncased           # BERT, no context
python run.py train --model bert-base-uncased --context # BERT, with context
python run.py report                                    # build the comparison table
```

To run everything on a free GPU, open
[`notebooks/sarcasm_colab.ipynb`](notebooks/sarcasm_colab.ipynb) in Google Colab.

## File map

| File              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `config.py`       | All paths, hyperparameters, and device (CPU/MPS/CUDA) detection |
| `utils.py`        | Seeding, logging, JSON/dir helpers                             |
| `data.py`         | Load, clean, split, and tokenize the SARC data                 |
| `baseline.py`     | TF-IDF + LogisticRegression/SVM baseline                       |
| `train.py`        | Fine-tune BERT/RoBERTa (with/without context)                  |
| `evaluate.py`     | Metrics, confusion matrix, McNemar significance test           |
| `report.py`       | Aggregate all run metrics into the comparison table            |
| `run.py`          | Command-line entry point tying it all together                 |
| `notebooks/`      | Colab notebook that runs the whole pipeline on a free GPU      |

See [`../docs/06-usage.md`](../docs/06-usage.md) for the full experiment workflow.
