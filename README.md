# Titanic Survival Analysis — Exploratory Data Analysis & Feature Engineering

A reproducible exploratory data analysis (EDA) of the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data), focused on **data quality, statistical exploration, feature engineering, and leakage-safe preprocessing** for a passenger-survival classification task.

This repository documents the **EDA notebook**. Its logistic-regression experiments are diagnostic feature-ablation studies, **not** the final PyTorch training application.

## Project goals

- Examine feature types, missing values, duplicates, and data integrity.
- Summarize numerical variables with mean, median, standard deviation, percentiles, IQR, and skewness.
- Study survival rates across passenger sex, class, age, fare, family size, and embarkation port.
- Investigate outliers, feature correlations, and redundancy.
- Engineer additional predictive features and test their value using stratified cross-validation.
- Demonstrate preprocessing that avoids fitting statistics on validation data.

## Dataset and evaluation setup

The analysis uses **only `train.csv`** from Kaggle's Titanic competition (891 labeled passengers). It **does not use** Kaggle's `test.csv` or `gender_submission.csv`.

| Setting | Value |
| --- | --- |
| Prediction target | `Survived` (0 = did not survive; 1 = survived) |
| Original dataset | 891 passengers |
| Development / EDA partition | 712 passengers (80%) |
| Held-out validation partition | 179 passengers (20%) |
| Split | Stratified by `Survived` |
| Random seed | `42` |
| Feature-ablation evaluation | 5-fold stratified cross-validation on development data only |

**Data leakage precaution:** Preprocessing is fitted separately within each training fold during cross-validation. The held-out 20% is not used to score feature-ablation candidates. The EDA results below refer to the 712-passenger development partition unless otherwise stated.

## Key EDA findings

| Observation | Result (development partition) | Implication |
| --- | ---: | --- |
| Overall survival rate | 38.34% | Report F1 and ROC AUC alongside accuracy |
| Missing `Age` | 19.24% | Median imputation; preserve missingness indicator |
| Missing `Cabin` | 77.53% | Exclude raw `Cabin` from the initial feature set |
| Missing `Embarked` | 0.28% | Handle the missing category explicitly |
| Duplicate rows / passenger IDs | 0 / 0 | No duplicate removal required |
| Median fare | 14.45 | Mean alone does not represent a typical fare |
| Raw fare skewness | 4.65 | Investigate a logarithmic fare transformation |
| `LogFare` skewness | 0.31 | Log transform substantially reduces right skew |

Survival rates differ substantially across sex and passenger class, motivating an explicit `SexPclass` interaction feature. An IQR-based check flags some high fares; those observations are **not automatically dropped**, because expensive tickets may be legitimate.

EDA shows **associations**, not causal effects.

## Feature engineering

The notebook evaluates the original passenger attributes alongside these engineered variables:

| Engineered feature | Definition / motivation |
| --- | --- |
| `FamilySize` | `SibSp + Parch + 1`; traveling group size |
| `IsAlone` | Indicator that `FamilySize == 1` |
| `FarePerPerson` | `Fare / FamilySize`; approximate normalized fare |
| `Title` | Extract `Mr`, `Mrs`, `Miss`, `Master`, or `Rare` from passenger names |
| `LogFare` | `log1p(Fare)` to compress the fare distribution |
| `AgeMissing` | Indicator showing whether age is missing |
| `SexPclass` | Combined sex and passenger-class category |

The complete feature representation produces **29 model input columns** after numerical processing and one-hot encoding.

### Preprocessing evaluated in the notebook

- **Numerical:** median imputation followed by `StandardScaler`.
- **Categorical:** missing/unknown-category handling and `OneHotEncoder(handle_unknown="ignore")`.
- **Validation:** the fitted transformations are reused on held-out data; no separate fitting is performed on validation rows.

## Feature-ablation results

Five-fold stratified cross-validation of a **logistic-regression diagnostic baseline** on the 712 development passengers:

| Feature configuration | Mean ROC AUC | Standard deviation |
| --- | ---: | ---: |
| Original seven features | 0.8532 | 0.0242 |
| + Family features | 0.8561 | 0.0241 |
| + Title | 0.8684 | 0.0217 |
| + LogFare | 0.8581 | 0.0278 |
| + Sex × Pclass | 0.8607 | 0.0177 |
| **All engineered features** | **0.8718** | 0.0229 |

The full engineered set gives the highest **mean** ROC AUC in this diagnostic experiment. The differences between configurations are modest relative to fold-to-fold variation; these results do not prove that every feature is individually useful or that the same ranking will hold for a PyTorch model.

## How to run

### Google Colab (simplest)

1. Download/open the repository's Titanic EDA `.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Choose **Runtime → Run all**.
3. When prompted, upload the **Kaggle `train.csv`** file from your computer.

Google Colab runs remotely, so a Windows path such as `C:\Users\...\train.csv` is not accessible to it. The notebook includes an upload fallback.

### Local Jupyter / PyCharm

```bash
git clone https://github.com/TalShwartzz/titanic.git
cd titanic
python -m pip install numpy pandas matplotlib seaborn scikit-learn ipython notebook ipykernel
jupyter notebook
```

Open the Titanic EDA notebook and run all cells. Place `train.csv` next to the notebook or in a local `data/` folder, or update `LOCAL_TRAIN_CSV` in its first code cell. If the repository includes `requirements.txt`, you may instead install with `python -m pip install -r requirements.txt`.

The dataset can be obtained from the [official Kaggle competition page](https://www.kaggle.com/competitions/titanic/data). The notebook does not require Kaggle's competition test set.

## Notebook contents

The notebook covers dataset integrity, descriptive and categorical statistics, missingness, class balance, survival-group comparisons, numerical distributions, outlier checks, feature engineering, redundancy and correlations, preprocessing validation, feature-ablation cross-validation, and evidence-based conclusions.

## Scope and limitations

- This repository documents the exploratory analysis and diagnostic logistic-regression comparisons, **not a deployed inference application**.
- Preprocessing choices are supported by the statistics, but downstream feature/model selection should be confirmed through additional validation.
- Small differences in cross-validation scores may be sampling variation.
- Historical passenger characteristics are correlated with survival; the results should not be interpreted as causal estimates.

---

**Dataset credit:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data).
