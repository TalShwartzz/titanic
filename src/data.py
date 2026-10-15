"""Download ONLY Titanic competition train.csv and validate its schema."""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "titanic"
REQUIRED_FEATURES = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
TARGET = "Survived"


def download_train_csv(destination: str | Path = "data/train.csv", force: bool = False) -> Path:
    """Fetch precisely train.csv from Kaggle (no test.csv or gender_submission.csv).

    Kaggle credentials and acceptance of any competition rules are the user's
    responsibility. Existing training data are reused by default for offline runs.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return destination

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Install project dependencies (pip install -r requirements.txt).") from exc

    # Download into an isolated temporary directory, so an older CSV or ZIP
    # cannot be mistaken for a fresh Kaggle download (including force=True).
    with tempfile.TemporaryDirectory(prefix="kaggle_titanic_", dir=destination.parent) as staging:
        try:
            api = KaggleApi()
            api.authenticate()
            api.competition_download_file(COMPETITION, "train.csv", path=staging, force=True, quiet=True)
        except Exception as exc:
            raise RuntimeError(
                "Kaggle download failed. Configure Kaggle API authentication, ensure the "
                "Titanic competition access is enabled, and retry; see README.md. "
                "Or pass --data-path to a previously downloaded Kaggle train.csv."
            ) from exc

        # Kaggle versions may return CSV bytes or a single-file ZIP archive.
        staging_path = Path(staging)
        candidates = [staging_path / "train.csv.zip", staging_path / "train.csv"]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError("Kaggle returned neither train.csv nor train.csv.zip")
        if zipfile.is_zipfile(found):
            with zipfile.ZipFile(found) as archive:
                members = [name for name in archive.namelist() if Path(name).name == "train.csv"]
                if len(members) != 1:
                    raise ValueError("Expected exactly one train.csv inside Kaggle archive.")
                destination.write_bytes(archive.read(members[0]))
        elif found.suffix == ".csv":
            destination.write_bytes(found.read_bytes())
        else:
            raise ValueError(f"Unexpected Kaggle archive format: {found.name}")
    return destination


def load_training_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    required = set(REQUIRED_FEATURES) | {TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")
    if len(df) < 30:
        raise ValueError("Training dataset is too small (requires at least 30 rows).")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Survived must contain only non-null 0/1 labels.")
    if df[TARGET].nunique() < 2:
        raise ValueError("Survived must contain both classes.")
    if "PassengerId" in df and df["PassengerId"].duplicated().any():
        raise ValueError("Duplicate PassengerId values detected; check data source.")
    return df


def sha256_file(csv_path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(csv_path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
