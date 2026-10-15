"""Classification metrics and inference for arbitrary Titanic-style CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from .features import transformed_array
from .model import TitanicMLP


def evaluate_binary(y_true, probabilities, threshold: float = 0.5) -> dict:
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(truth) != len(probabilities) or len(truth) == 0:
        raise ValueError("Labels and probabilities must be aligned and nonempty.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(truth, predictions, labels=[0, 1])
    out = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "support": int(len(truth)),
        "threshold": float(threshold),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(truth)) == 2:
        out["roc_auc"] = float(roc_auc_score(truth, probabilities))
        out["average_precision"] = float(average_precision_score(truth, probabilities))
    return out


def load_bundle(artifacts_dir: str | Path = "artifacts"):
    """Load only artifacts created by this project. Never load untrusted joblib files."""
    folder = Path(artifacts_dir)
    config = json.loads((folder / "model_config.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(folder / "preprocessor.joblib")
    model = TitanicMLP(
        input_dim=int(config["input_dim"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return preprocessor, model, config


def predict_probabilities(data: pd.DataFrame, preprocessor, model: TitanicMLP, batch_size: int = 2048):
    features = transformed_array(preprocessor, data)
    if features.shape[1] != model.network[0].in_features:
        raise ValueError("Preprocessor/model feature dimension mismatch.")
    if len(features) == 0:
        return np.asarray([], dtype=float)
    with torch.inference_mode():
        parts = []
        for start in range(0, len(features), batch_size):
            logits = model(torch.from_numpy(features[start:start + batch_size]))
            parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(parts).astype(float)
