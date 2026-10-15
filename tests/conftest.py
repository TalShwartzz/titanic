"""Synthetic Titanic-shaped records for local tests; not official Kaggle data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(123)
    n = 180
    sex = rng.choice(["male", "female"], n)
    classes = rng.choice([1, 2, 3], n)
    age = rng.normal(32, 13, n).clip(1, 75)
    fare = (5 + (4 - classes) * 15 + rng.exponential(10, n)).round(2)
    siblings = rng.integers(0, 3, n)
    parents = rng.integers(0, 2, n)
    embarked = rng.choice(["S", "C", "Q"], n)
    logits = -1.0 + 1.4 * (sex == "female") + 0.55 * (classes == 1) - 0.015 * age
    probability = 1 / (1 + np.exp(-logits))
    survived = rng.binomial(1, probability)
    df = pd.DataFrame({
        "PassengerId": np.arange(1, n + 1), "Survived": survived,
        "Pclass": classes, "Name": [f"Example, {'Mrs' if x == 'female' else 'Mr'}. Person{i}" for i, x in enumerate(sex)],
        "Sex": sex, "Age": age, "SibSp": siblings, "Parch": parents,
        "Fare": fare, "Embarked": embarked,
    })
    df.loc[[1, 9, 18], "Age"] = np.nan
    df.loc[[5, 20], "Embarked"] = None
    return df
