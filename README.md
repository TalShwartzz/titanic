# 🚢 Titanic Survival Intelligence — ELTA Data Science Assignment

End-to-end, reproducible **PyTorch binary classification pipeline** using **only the official Kaggle Titanic competition `train.csv`**, plus an exploratory Jupyter notebook and a Streamlit dashboard for held-out evaluation and CSV inference.

**Important:** No Kaggle `test.csv` or `gender_submission.csv` is downloaded, used, evaluated, or submitted. The supplied `example_inference.csv` contains *invented illustrative passengers* and is never used for training or evaluation.

## Project contents

```text
elta_titanic_solution/
├── train.py                       # Standalone training entry point
├── ds_app.py                      # Streamlit evaluation and inference app
├── notebooks/
│   └── titanic_eda.ipynb          # Detailed EDA and data-quality investigation
├── src/
│   ├── data.py                    # Kaggle-only train.csv download and validation
│   ├── features.py                # Feature engineering + training-fitted preprocessing
│   ├── model.py                   # PyTorch MLP (BCEWithLogitsLoss)
│   └── evaluation.py              # Metrics, saved model loading, inference
├── tests/                         # End-to-end and unit tests (synthetic fixtures)
├── data/
│   ├── README.md                  # Kaggle source attribution
│   └── example_inference.csv      # Small, illustrative unlabeled inference file
├── artifacts/                     # Generated when train.py runs, not checked into git
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## 1. Setup

Use Python **3.11 or 3.12** (with PyTorch-compatible binaries). From a terminal in the project root:

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation differs by OS/GPU. The supplied pipeline runs on **CPU by default**; consult [PyTorch's installation selector](https://pytorch.org/get-started/locally/) if you prefer a CUDA build.

### Kaggle credentials

1. Sign in to [Kaggle](https://www.kaggle.com/), open the [Titanic competition](https://www.kaggle.com/competitions/titanic/data), and accept any applicable rules.
2. Configure the official **Kaggle API** using supported Kaggle credentials (for example, `~/.kaggle/kaggle.json` on macOS/Linux or `%USERPROFILE%\.kaggle\kaggle.json` on Windows, as applicable to your Kaggle account/API version). Alternatively configure the environment variables supported by your Kaggle client.
3. Run `python train.py`. It uses the official Kaggle Python API to download **the single file `train.csv`** to `data/train.csv`. A previously downloaded file is reused on later runs.

Never commit API tokens or your `kaggle.json`. `.gitignore` excludes them.

**No credentials?** Download **only** `train.csv` yourself from the [Kaggle Titanic data page](https://www.kaggle.com/competitions/titanic/data), save it as `data/train.csv`, and use the same training command. This manual alternative is for local access; the code's normal first-run route is a direct Kaggle API download.

## 2. Explore data in Jupyter

```bash
jupyter lab notebooks/titanic_eda.ipynb
```

Set `LOCAL_TRAIN_CSV` in the first code cell to the full path of your locally downloaded Kaggle `train.csv` (for example `r"C:\Users\YourUsername\Downloads\train.csv"`). Alternatively leave it blank and place the file at `data/train.csv` inside the project. The notebook does **not** download the data or require Kaggle credentials. Run **Run All** to cover schema/data quality, missingness, categorical survival differences, age/fare and family features, sex × passenger-class interactions, and data leakage safeguards. All plots are generated from actual Kaggle records during execution; no results are invented in the notebook.

## 3. Train and evaluate

```bash
python train.py
```

Or customize a reproducible experiment:

```bash
python train.py --seed 42 --epochs 80 --val-size 0.20 --batch-size 64 --learning-rate 0.001
```

If you've manually placed the official dataset elsewhere:

```bash
python train.py --data-path "C:/mydata/train.csv"
```

`train.py` loads **only** the allowed training file, validates it, makes a **stratified 80/20 split**, fits preprocessing exclusively on the training partition, fits a logistic regression comparison model, and trains a compact MLP in PyTorch. The number of epochs is predetermined; it does not choose a checkpoint by optimizing the reported validation metrics. It computes final held-out metrics at **threshold = 0.50**.

Generated files in `artifacts/`:

| File | Role |
|---|---|
| `model.pt` | PyTorch weights as `state_dict` (not a serialized Python model object) |
| `preprocessor.joblib` | Train-fitted imputers, scaler, categorical encoder |
| `model_config.json` | Model architecture, split info, source hash, training settings |
| `metrics.json` | Accuracy, precision, recall, F1, ROC AUC, AP and confusion matrices for both models |
| `history.csv` | Epoch-by-epoch training and validation loss diagnostics |
| `validation.csv` | **Held-out subset of Kaggle train.csv** with labels, saved after the split |
| `validation_predictions.csv` | Held-out MLP predictions at default threshold |
| `split_manifest.json` | Training/validation passenger IDs to audit disjointness |

The exact numeric results depend on the genuine Kaggle source and the training environment. **No actual Titanic-trained checkpoint or score is bundled with this source repository.** Execute `python train.py` once to create both.

## 4. Run Streamlit

```bash
streamlit run ds_app.py
```

The app provides:

- **Held-out validation** (default): run inference anew on `artifacts/validation.csv`; show accuracy, precision, recall, F1, ROC AUC, confusion matrix, score distributions, ROC/PR curves, calibration diagnostics, and baseline comparison.
- **Upload CSV**: predict from an uploaded `.csv` and evaluate if a complete binary `Survived` column is included.
- **Local CSV path**: load a path such as `data/example_inference.csv` and show predictions even if no labels are present.
- Interactive decision **threshold** for classification and downloadable `titanic_predictions.csv`.

**Inference CSV required columns:**

```text
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
```

`Name` is optional but recommended, because a passenger's title is an engineered feature. `PassengerId` is optional and is shown in downloaded predictions. `Survived` is optional and is **only** used for evaluation if present. Missing cells within expected columns are supported. Unseen categorical values are handled by the trained encoder. A missing required column raises a clear error instead of silently changing the schema.

### Example inference file

```text
PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Fare,Embarked
9001,1,"Example, Mrs. Alice",female,34,1,0,88.0,S
9002,3,"Example, Mr. Bob",male,26,0,0,7.9,Q
```

These are **fabricated sample inputs** only. Choose **Local CSV path** in Streamlit and enter `data/example_inference.csv` to test unlabeled inference after the real model is trained.

## 5. Run automated tests

```bash
python -m pytest -q
```

Tests exercise feature extraction, required-column errors, separation between inputs and labels, saved model round-trip, reproducible disjoint split, labeled/unlabeled inference, and binary metrics. The fixtures use synthesized Titanic-*shaped* records so tests run offline without confusing them with the actual Kaggle competition data.

## Architecture & methodological decisions

```text
Kaggle API: Titanic/train.csv
              │
      schema + label checks
              │
       stratified split
          ┌───┴────────┐
      80% train     20% validation (never used to fit preprocessing)
          │              │
 feature engineering ────┘ (identical deterministic transforms)
          │
 train-only median imputer / standard scaler / one-hot encoder
          │
   ┌──────┴──────────────────────────┐
   │                                 │
 logistic regression            PyTorch MLP
   baseline                       32 → 16 → 1
   │                               │
   └──────── validation metrics ───┘
                                   │
                   save weights + preprocessing + config
                                   │
                 Streamlit: infer / evaluate / export CSV
```

- **Why these features?** Passenger class, sex, age, family members, fare, port of embarkation, and derived `FamilySize`, `IsAlone`, `FarePerPerson`, `Title`. Raw `Ticket` and `Cabin` are omitted because of high cardinality and/or missingness; `PassengerId` is an identifier and `Survived` is the target.
- **Why these imputations?** Median numerical imputation is resistant to extreme fares; categorical blanks are assigned a stable `Unknown` category. Scaling prevents fare magnitudes from dominating MLP optimization; one-hot encoding avoids imposing false order on categories.
- **Why a simple MLP?** With approximately 900 rows, a 2-layer MLP (ReLU, dropout) has enough flexibility without a needlessly large deep-learning model. `BCEWithLogitsLoss` is numerically stable; AdamW handles optimization.
- **Why logistic regression?** Its comparable scores provide a meaningful sanity check on whether extra neural-network complexity is justified.
- **Why accuracy isn't enough?** Titanic has unequal survival classes. F1, recall, precision, ROC AUC, PR curves, error distribution and confusion matrix reveal different kinds of failures.
- **Limitations:** A single 80/20 split gives noisy estimates on a small dataset. Fixed hyperparameters avoid explicitly selecting on validation, but findings should still be viewed as an estimate, not a definitive generalization guarantee. Changing the threshold in the UI is exploratory; results after selecting a threshold by looking at validation labels are no longer an unbiased final estimate. For further work, use stratified cross-validation and separate final holdout/test data **drawn exclusively from `train.csv`**, if the assignment permits it.

### Reproducibility and trust

The script stores its random seed, input CSV SHA-256, split IDs and pipeline definition alongside the checkpoint. It defaults to deterministic, CPU training. GPU/PyTorch versions may lead to minor numeric variation. `torch.load(..., weights_only=True)` loads only tensor weights; **joblib deserialization is still unsafe for untrusted files**, so use only artifacts you generated yourself. Do not share or commit Kaggle API credentials.

## Suggested GitHub submission checklist

1. Push project **source**, notebook, tests, example CSV and README.
2. Run `python train.py`, `python -m pytest -q` and `streamlit run ds_app.py` locally before submitting.
3. Capture screenshots of the notebook's EDA figures and Streamlit validation tab and include them under `assets/` if desired.
4. Do not upload Kaggle credentials, Kaggle `test.csv`, or `gender_submission.csv`.

This repository fulfills the **classification task** described in the provided ELTA assignment. If a separate task was omitted from the uploaded instructions, it is intentionally not invented here.
