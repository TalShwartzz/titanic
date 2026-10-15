# Data

This project uses only Kaggle's [Titanic `train.csv`](https://www.kaggle.com/competitions/titanic/data), included in the repository root. Kaggle's `test.csv` and `gender_submission.csv` are not used.

**Training:**
- `python train.py` — Use local data.
- `python train.py --download` — Fetch from Kaggle (credentials required).

**Sample data:**

`data/example_inference.csv` contains synthetic passengers for testing Streamlit.

Run `python -m streamlit run ds_app.py` and upload the sample CSV to generate predictions. Evaluation metrics require a `Survived` column.
