"""Run: pytest -q. Uses generated synthetic test fixtures, not Kaggle test.csv."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import load_training_data
from src.evaluation import evaluate_binary, load_bundle, predict_probabilities
from src.features import build_preprocessor, make_features, transformed_array
from train import parse_args, train


def test_feature_extraction_and_missing_values(synthetic_data):
    table = make_features(synthetic_data)
    assert {"FamilySize", "Title", "IsAlone", "FarePerPerson"} <= set(table)
    transformer = build_preprocessor().fit(table.iloc[:120])
    output = transformed_array(transformer, synthetic_data.iloc[120:])
    assert output.shape[0] == 60
    assert np.isfinite(output).all()


def test_survival_label_is_not_a_feature(synthetic_data):
    source = synthetic_data.copy()
    changed = source.copy()
    changed["Survived"] = 1 - changed["Survived"]
    changed["PassengerId"] += 10000
    pd.testing.assert_frame_equal(make_features(source), make_features(changed))


def test_missing_required_columns(synthetic_data):
    with pytest.raises(ValueError, match="Missing required"):
        make_features(synthetic_data.drop(columns="Fare"))


def test_one_class_metrics_are_defined_except_auc():
    result = evaluate_binary([1, 1], [0.8, 0.2])
    assert result["support"] == 2
    assert result["roc_auc"] is None
    assert result["confusion_matrix"] == [[0, 0], [1, 1]]


def test_end_to_end_save_load_and_reproducible_split(tmp_path, synthetic_data):
    training_csv = tmp_path / "train.csv"
    synthetic_data.to_csv(training_csv, index=False)
    assert len(load_training_data(training_csv)) == len(synthetic_data)
    outputs = tmp_path / "trained"
    args = parse_args([
        "--data-path", str(training_csv), "--output-dir", str(outputs),
        "--epochs", "3", "--batch-size", "48", "--seed", "42",
    ])
    summary = train(args)
    for name in ["model.pt", "preprocessor.joblib", "model_config.json", "metrics.json", "history.csv", "validation.csv", "validation_predictions.csv", "split_manifest.json"]:
        assert (outputs / name).is_file(), name
    preprocessor, model, config = load_bundle(outputs)
    val_frame = pd.read_csv(outputs / "validation.csv")
    probabilities = predict_probabilities(val_frame, preprocessor, model)
    assert config["n_validation"] == len(val_frame)
    assert np.allclose(probabilities, pd.read_csv(outputs / "validation_predictions.csv")["survival_probability"])
    assert "logistic_regression" in summary
    ids = json.loads((outputs / "split_manifest.json").read_text())
    assert not set(ids["training_passenger_ids"]) & set(ids["validation_passenger_ids"])
    # CSV for unlabeled inference is accepted after training.
    unlabeled = val_frame.drop(columns=["Survived", "Name"])
    assert len(predict_probabilities(unlabeled, preprocessor, model)) == len(val_frame)
