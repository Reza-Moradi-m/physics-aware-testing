# src/adapters/nasa_adapter.py
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


def setup_nasa_dataset(
    out_path: str = "data/nasa_jet_engine.csv",
    failure_window: int = 30,
) -> Path:
    """
    Download and prepare NASA C-MAPSS FD001 for binary classification.

    Label:
      1 = engine is within `failure_window` cycles of failure
      0 = not yet in imminent-failure window

    Output:
      Standardized CSV with numeric feature columns + label
    """
    print("Downloading NASA C-MAPSS FD001 dataset...")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(NASA_CMAPSS_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        target_name = "train_FD001.txt"
        if target_name not in zf.namelist():
            raise FileNotFoundError(f"{target_name} not found inside NASA ZIP archive.")

        with zf.open(target_name) as fh:
            cols = (
                ["engine_id", "cycle", "setting1", "setting2", "setting3"]
                + [f"s{i}" for i in range(1, 22)]
            )
            df = pd.read_csv(
                fh,
                sep=r"\s+",
                header=None,
                names=cols,
                engine="python",
            )

    print("Processing NASA turbofan data...")

    # Remaining Useful Life
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]

    # Binary target
    df["label"] = (df["RUL"] <= failure_window).astype(int)

    # Keep a focused subset for the first product version.
    keep_cols = [
        "engine_id",
        "cycle",
        "s2",
        "s3",
        "s4",
        "s7",
        "s11",
        "s12",
        "s15",
        "label",
    ]
    final_df = df[keep_cols].copy()

    final_df.columns = [
        "engine_id",
        "cycle",
        "temp_inlet",
        "temp_lpc",
        "temp_hpc",
        "fan_speed",
        "pressure_static",
        "pressure_lpc",
        "fuel_flow",
        "label",
    ]

    # Remove obvious missing/infinite values if any appear
    final_df = final_df.replace([float("inf"), float("-inf")], pd.NA).dropna()

    final_df.to_csv(out_file, index=False)

    label_dist = final_df["label"].value_counts(normalize=True).to_dict()
    print(f"Saved NASA dataset to: {out_file}")
    print(f"Rows: {len(final_df):,}")
    print(f"Columns: {list(final_df.columns)}")
    print(f"Label distribution: {label_dist}")

    return out_file


if __name__ == "__main__":
    setup_nasa_dataset()