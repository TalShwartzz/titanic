"""Mocked Kaggle download checks only the permitted competition file is used."""

from __future__ import annotations

import io
import sys
import types
import zipfile

from src.data import download_train_csv


def test_download_single_csv_from_kaggle(monkeypatch, tmp_path):
    calls = []
    class MockApi:
        def authenticate(self):
            calls.append("auth")
        def competition_download_file(self, competition, file_name, path, force, quiet):
            calls.append((competition, file_name, force, quiet))
            from pathlib import Path
            Path(path, "train.csv").write_text("Survived,Pclass\n1,2\n", encoding="utf-8")

    package = types.ModuleType("kaggle")
    api_pkg = types.ModuleType("kaggle.api")
    api_mod = types.ModuleType("kaggle.api.kaggle_api_extended")
    api_mod.KaggleApi = MockApi
    monkeypatch.setitem(sys.modules, "kaggle", package)
    monkeypatch.setitem(sys.modules, "kaggle.api", api_pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", api_mod)
    destination = download_train_csv(tmp_path / "my_train.csv")
    assert destination.read_text() == "Survived,Pclass\n1,2\n"
    assert ("titanic", "train.csv", True, True) in calls
    assert download_train_csv(destination) == destination  # cache hit, no extra request
    assert calls.count("auth") == 1


def test_kaggle_zip_is_read_without_other_files(monkeypatch, tmp_path):
    class MockApi:
        def authenticate(self):
            pass
        def competition_download_file(self, competition, file_name, path, force, quiet):
            from pathlib import Path
            with zipfile.ZipFile(Path(path, "train.csv.zip"), "w") as z:
                z.writestr("train.csv", "Survived,Pclass\n1,2\n")
                z.writestr("test.csv", "should never be extracted")
    package = types.ModuleType("kaggle")
    api_pkg = types.ModuleType("kaggle.api")
    api_mod = types.ModuleType("kaggle.api.kaggle_api_extended")
    api_mod.KaggleApi = MockApi
    monkeypatch.setitem(sys.modules, "kaggle", package)
    monkeypatch.setitem(sys.modules, "kaggle.api", api_pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", api_mod)
    destination = download_train_csv(tmp_path / "train.csv", force=True)
    assert destination.read_text() == "Survived,Pclass\n1,2\n"
    assert not (tmp_path / "test.csv").exists()
