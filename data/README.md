# Data policy

The only training/evaluation source is the official [Kaggle Titanic competition `train.csv`](https://www.kaggle.com/competitions/titanic/data).

- `python train.py` downloads **only** `train.csv` via the Kaggle API into this folder.
- `test.csv` and `gender_submission.csv` are not downloaded or used.
- `example_inference.csv` is a **hand-authored illustrative inference file**, not sourced from Kaggle and not used for training or evaluation.
- The real training file is excluded from git to keep the repository lightweight.
