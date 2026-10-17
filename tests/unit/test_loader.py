"""Unit tests for CSV loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.core.exceptions import DataLoadError
from app.data.loader import load_csv, load_raw_dataset


class TestLoadCsv:
    def test_loads_valid_csv(self, tmp_path: Path, valid_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        valid_frame.to_csv(path, index=False)
        loaded = load_csv(path)
        assert loaded.shape == valid_frame.shape
        assert list(loaded.columns) == list(valid_frame.columns)

    def test_values_match_source(self, tmp_path: Path, valid_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        valid_frame.to_csv(path, index=False)
        loaded = load_csv(path)
        pd.testing.assert_frame_equal(loaded, valid_frame)

    def test_accepts_string_path(self, tmp_path: Path, valid_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        valid_frame.to_csv(path, index=False)
        assert not load_csv(str(path)).empty

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="not found"):
            load_csv(tmp_path / "missing.csv")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataLoadError, match="not a file"):
            load_csv(tmp_path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DataLoadError):
            load_csv(path)

    def test_header_only_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "header.csv"
        path.write_text("a,b\n", encoding="utf-8")
        with pytest.raises(DataLoadError, match="no rows"):
            load_csv(path)

    def test_source_file_is_unchanged(self, tmp_path: Path, valid_frame: pd.DataFrame) -> None:
        path = tmp_path / "data.csv"
        valid_frame.to_csv(path, index=False)
        before = path.read_bytes()
        load_csv(path)
        assert path.read_bytes() == before


class TestLoadRawDataset:
    def test_resolves_from_settings(
        self, make_frame: Callable[..., pd.DataFrame], tmp_path: Path
    ) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        make_frame(3).to_csv(raw_dir / "ai4i2020.csv", index=False)
        settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")
        loaded = load_raw_dataset(settings=settings)
        assert len(loaded) == 3

    def test_explicit_path_overrides_settings(
        self, make_frame: Callable[..., pd.DataFrame], tmp_path: Path
    ) -> None:
        path = tmp_path / "custom.csv"
        make_frame(2).to_csv(path, index=False)
        assert len(load_raw_dataset(path)) == 2

    def test_missing_default_raises(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path, artifacts_dir=tmp_path / "artifacts")
        with pytest.raises(DataLoadError):
            load_raw_dataset(settings=settings)
