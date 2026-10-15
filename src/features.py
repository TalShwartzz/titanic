"""Feature definitions matching the EDA notebook (not fitted on labels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import REQUIRED_FEATURES

NUMERIC = ["Age", "SibSp", "Parch", "Fare", "LogFare", "FamilySize", "IsAlone", "FarePerPerson", "AgeMissing"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "SexPclass"]
COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}
ENGINEERED = ["LogFare", "FamilySize", "IsAlone", "FarePerPerson", "AgeMissing", "Title", "SexPclass"]

# Compare the original features, title-only, and all engineered features *for the MLP*.
FEATURE_SETS = {
    "original": (["Age", "SibSp", "Parch", "Fare"], ["Pclass", "Sex", "Embarked"]),
    "title": (["Age", "SibSp", "Parch", "Fare"], ["Pclass", "Sex", "Embarked", "Title"]),
    "all": (NUMERIC, CATEGORICAL),
}


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive explanatory variables using passenger details only (never Survived)."""
    missing = sorted(set(REQUIRED_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    out = pd.DataFrame(index=data.index)
    for col in ("Age", "SibSp", "Parch", "Fare"):
        out[col] = pd.to_numeric(data[col], errors="coerce")
    out["LogFare"] = np.log1p(out["Fare"].clip(lower=0))
    out["AgeMissing"] = out["Age"].isna().astype(float)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(float)
    out.loc[out["FamilySize"].isna(), "IsAlone"] = np.nan
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].where(out["FamilySize"] > 0)

    pclass = pd.to_numeric(data["Pclass"], errors="coerce")
    out["Pclass"] = pclass.map({1: "1", 2: "2", 3: "3"}).fillna("Unknown")
    out["Sex"] = data["Sex"].astype("string").str.strip().str.lower().fillna("unknown")
    out["Embarked"] = data["Embarked"].astype("string").str.strip().str.upper().fillna("Unknown")
    out["SexPclass"] = out["Sex"].astype(str) + "_" + out["Pclass"].astype(str)

    if "Name" in data:
        title = data["Name"].astype("string").str.extract(
            r",\s*([A-Za-z]+)\.", expand=False
        ).str.title()
        out["Title"] = title.where(title.isin(COMMON_TITLES), "Rare").fillna("Unknown")
    else:
        out["Title"] = "Unknown"
    return out[NUMERIC + CATEGORICAL]


def build_preprocessor(numeric_cols=None, categorical_cols=None) -> ColumnTransformer:
    """Each fit learns medians, means and categories from training rows only."""
    numeric_cols = NUMERIC if numeric_cols is None else list(numeric_cols)
    categorical_cols = CATEGORICAL if categorical_cols is None else list(categorical_cols)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", num_pipe, numeric_cols), ("categorical", cat_pipe, categorical_cols)],
        remainder="drop", sparse_threshold=0.0,
    )


def transformed_array(preprocessor, data: pd.DataFrame) -> np.ndarray:
    arr = np.asarray(preprocessor.transform(make_features(data)), dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("NaN or infinity after preprocessing")
    return arr
