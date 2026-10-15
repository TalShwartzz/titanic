#!/usr/bin/env python3
"""EDA-aligned Titanic PyTorch training, with development-only model selection.

Local run (train.csv beside this file):
    python train.py
    python train.py --cv-epochs 30 40 --cv-folds 5
    python train.py --no-cv --epochs 40     # fast reference run
Optional Kaggle fetch (only competition train.csv):
    python train.py --download

Use only Kaggle train.csv; the reserved 20% is NOT used to select features or epochs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import REQUIRED_FEATURES, download_train_csv, load_training_data, sha256_file
from src.evaluation import evaluate_binary, predict_probabilities
from src.features import (
    CATEGORICAL, ENGINEERED, FEATURE_SETS, NUMERIC,
    build_preprocessor, make_features, transformed_array,
)
from src.model import TitanicMLP

ROOT = Path(__file__).resolve().parent
HIDDEN_DIMS = (32, 16)
DROPOUT = 0.15
THRESHOLD = 0.50


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ELTA Titanic EDA-aligned PyTorch classifier")
    p.add_argument("--data-path", type=Path, default=ROOT / "train.csv")
    p.add_argument("--download", action="store_true", help="Download only Kaggle train.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-epochs", type=int, nargs="+", default=[30, 40])
    p.add_argument("--no-cv", action="store_true", help="Skip CV and train all features with --epochs")
    p.add_argument("--epochs", type=int, default=40, help="Used with --no-cv")
    args = p.parse_args(argv)
    if not 0 < args.val_size < 0.5:
        p.error("--val-size must be between 0 and 0.5")
    if args.batch_size < 1 or args.learning_rate <= 0 or args.epochs < 1:
        p.error("--batch-size, --learning-rate, --epochs must be positive")
    if args.cv_folds < 2 or any(x < 1 for x in args.cv_epochs):
        p.error("--cv-folds must be >= 2; --cv-epochs must be positive")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_mlp(x, y, *, epochs, batch_size, lr, seed, device, record_history=False):
    """Fit on provided rows only. Does not see held-out labels or features."""
    seed_everything(seed)
    model = TitanicMLP(x.shape[1], HIDDEN_DIMS, DROPOUT).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            loss_sum += float(loss.item()) * len(yb)
        if record_history:
            train_loss = loss_sum / len(ds)
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": np.nan})
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d}/{epochs} | train loss={train_loss:.4f}")
    return model.cpu().eval(), history


def probabilities_from_matrix(model, matrix, batch_size=2048):
    results = []
    with torch.inference_mode():
        for offset in range(0, len(matrix), batch_size):
            logits = model(torch.from_numpy(matrix[offset:offset + batch_size]))
            results.append(torch.sigmoid(logits).numpy())
    return np.concatenate(results).astype(float)


def select_with_cv(development: pd.DataFrame, args, device):
    """Compare feature sets and epoch counts using ONLY development rows."""
    folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    y = development["Survived"].to_numpy(dtype=np.int64)
    rows = []
    candidates = [(feature_set, ep)
                  for feature_set in FEATURE_SETS
                  for ep in sorted(set(args.cv_epochs))]
    for feature_set, epochs in candidates:
        numeric, categorical = FEATURE_SETS[feature_set]
        for fold_no, (fit_idx, check_idx) in enumerate(folds.split(development, y), start=1):
            fit_df = development.iloc[fit_idx]
            check_df = development.iloc[check_idx]
            pre = build_preprocessor(numeric, categorical)
            x_fit = np.asarray(pre.fit_transform(make_features(fit_df)), dtype=np.float32)
            x_check = transformed_array(pre, check_df)
            model, _ = fit_mlp(
                x_fit, y[fit_idx], epochs=epochs, batch_size=args.batch_size,
                lr=args.learning_rate, seed=args.seed + fold_no, device=device,
            )
            p = probabilities_from_matrix(model, x_check)
            true = y[check_idx]
            rows.append({
                "feature_set": feature_set, "epochs": epochs, "fold": fold_no,
                "n_features": x_fit.shape[1],
                "roc_auc": roc_auc_score(true, p),
                "f1": f1_score(true, p >= THRESHOLD, zero_division=0),
                "accuracy": accuracy_score(true, p >= THRESHOLD),
            })
        subset = [r for r in rows if r["feature_set"] == feature_set and r["epochs"] == epochs]
        print(f"CV {feature_set:8s} {epochs:2d} epochs | "
              f"AUC {np.mean([r['roc_auc'] for r in subset]):.3f} "
              f"± {np.std([r['roc_auc'] for r in subset], ddof=1):.3f}")

    fold_table = pd.DataFrame(rows)
    summary = (fold_table.groupby(["feature_set", "epochs"], as_index=False)
               .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                    f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                    accuracy_mean=("accuracy", "mean"),
                    accuracy_std=("accuracy", "std")))
    # Choose a single configuration based only on the mean training-only CV ROC AUC.
    # Tie-break deterministically in favor of shorter training.
    best = summary.sort_values(["roc_auc_mean", "epochs", "feature_set"],
                               ascending=[False, True, True]).iloc[0]
    return fold_table, summary, str(best["feature_set"]), int(best["epochs"])


def train(args):
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    path = args.data_path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if args.download:
        path = download_train_csv(path, force=True)
    if not path.is_file():
        raise FileNotFoundError(f"No train.csv at {path}. Copy it next to train.py or use --download")
    print(f"Loading Titanic training data from {path}")
    df = load_training_data(path)
    tr_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=args.val_size,
        stratify=df["Survived"].to_numpy(), random_state=args.seed,
    )
    tr = df.iloc[tr_idx].copy()
    val = df.iloc[val_idx].copy()
    print(f"Train/dev: {len(tr)} | held-out: {len(val)} | device: {device}")

    if args.no_cv:
        feature_set, epochs = "all", args.epochs
        fold_table = summary = None
        print(f"CV skipped: all-engineered features, {epochs} fixed epochs.")
    else:
        fold_table, summary, feature_set, epochs = select_with_cv(tr, args, device)
        print(f"Selected from training-only CV: features={feature_set}, epochs={epochs}")

    numeric, categorical = FEATURE_SETS[feature_set]
    preprocessor = build_preprocessor(numeric, categorical)
    x_tr = np.asarray(preprocessor.fit_transform(make_features(tr)), dtype=np.float32)
    x_val = transformed_array(preprocessor, val)
    y_tr = tr["Survived"].to_numpy(dtype=np.int64)
    y_val = val["Survived"].to_numpy(dtype=np.int64)
    if not np.isfinite(x_tr).all():
        raise ValueError("Training features contain NaN or infinity")

    model, history = fit_mlp(
        x_tr, y_tr, epochs=epochs, batch_size=args.batch_size,
        lr=args.learning_rate, seed=args.seed, device=device, record_history=True,
    )
    p = predict_probabilities(val, preprocessor, model)
    neural_metrics = evaluate_binary(y_val, p, threshold=THRESHOLD)
    # Compare against a simple model with the *same* features, train split and holdout.
    baseline = LogisticRegression(max_iter=2000, random_state=args.seed)
    baseline.fit(x_tr, y_tr)
    baseline_metrics = evaluate_binary(y_val, baseline.predict_proba(x_val)[:, 1], THRESHOLD)
    metrics = {"pytorch_mlp": neural_metrics, "logistic_regression": baseline_metrics}

    # No validation-set metrics were used during training/CV selection.
    # This final validation loss is computed once, after the model is fixed.
    with torch.inference_mode():
        val_logits = model(torch.from_numpy(x_val))
        val_loss = float(nn.BCEWithLogitsLoss()(val_logits, torch.from_numpy(y_val.astype(np.float32))))
    history[-1]["val_loss"] = val_loss

    out = args.output_dir.expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    config = {
        "input_dim": int(x_tr.shape[1]), "hidden_dims": list(HIDDEN_DIMS), "dropout": DROPOUT,
        "input_columns": list(REQUIRED_FEATURES),
        "optional_columns": ["Name", "PassengerId", "Survived"],
        "feature_set": feature_set,
        "numeric_features": list(numeric), "categorical_features": list(categorical),
        "engineered_features_available": ENGINEERED,
        "threshold": THRESHOLD, "seed": args.seed, "n_train": len(tr), "n_validation": len(val),
        "source_sha256": sha256_file(path), "epochs": epochs,
        "validation_policy": "fixed stratified holdout; all selection within training-only CV",
        "cv_folds": 0 if args.no_cv else args.cv_folds,
    }
    (out / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    val.to_csv(out / "validation.csv", index=False)
    predictions = val[[c for c in ("PassengerId", "Survived") if c in val]].copy()
    predictions["survival_probability"] = p
    predictions["prediction"] = (p >= THRESHOLD).astype(int)
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    manifest = {
        "training_passenger_ids": tr["PassengerId"].tolist() if "PassengerId" in tr else tr_idx.tolist(),
        "validation_passenger_ids": val["PassengerId"].tolist() if "PassengerId" in val else val_idx.tolist(),
        "stratified": True, "seed": args.seed, "val_fraction": args.val_size,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if summary is not None:
        summary.to_csv(out / "cv_summary.csv", index=False)
        fold_table.to_csv(out / "cv_per_fold.csv", index=False)
    else:
        (out / "cv_summary.csv").unlink(missing_ok=True)
        (out / "cv_per_fold.csv").unlink(missing_ok=True)

    print("\nHeld-out evaluation (threshold 0.50):")
    for name, result in metrics.items():
        print(f"{name:22s} Accuracy {result['accuracy']:.3f} | "
              f"Precision {result['precision']:.3f} | Recall {result['recall']:.3f} | "
              f"F1 {result['f1']:.3f} | ROC AUC {result['roc_auc']:.3f}")
    print(f"Final model: {feature_set} features / {x_tr.shape[1]} inputs / {epochs} epochs")
    print(f"Saved compatible Streamlit artifacts to {out}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
