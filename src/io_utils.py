# src/io_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass
class CSVDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def load_csv_dataset(csv_path: str, label_col: str) -> CSVDataset:
    """
    Loads a CSV file into (X, y).
    Assumptions (simple on purpose):
      - header row exists
      - label_col exists
      - all other columns are numeric features
      - label is 0/1 (or numeric)

    Example CSV:
      f1,f2,f3,label
      1.2,0.3,9.1,1
      ...
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV must have a header row.")

        if label_col not in reader.fieldnames:
            raise ValueError(f"Label column '{label_col}' not found. Columns: {reader.fieldnames}")

        feature_cols = [c for c in reader.fieldnames if c != label_col]

        X_rows: list[list[float]] = []
        y_rows: list[float] = []

        for row in reader:
            X_rows.append([float(row[c]) for c in feature_cols])
            y_rows.append(float(row[label_col]))

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("CSV parse failed: X must be 2D and y must be 1D.")

    if len(X) != len(y):
        raise ValueError("CSV parse failed: X and y have different lengths.")

    return CSVDataset(X=X, y=y, feature_names=feature_cols)