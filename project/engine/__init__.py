"""
The `engine` folder holds the shared code that the scripts use.

You do NOT run anything in here directly. You run the scripts in the folder above
(check_data.py, train_bert.py, ...). They call the functions defined here.

Files:
  data.py        - loads, cleans, and splits the Reddit comments
  baseline.py    - the simple TF-IDF model
  transformer.py - fine-tunes BERT / RoBERTa
  scoring.py     - works out the scores and prints the results table
"""
