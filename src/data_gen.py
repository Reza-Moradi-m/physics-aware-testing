# src/data_gen.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_smart_factory_data(
    n_samples: int = 2000,
    out_path: str = "data/factory_sensors.csv",
    seed: int = 42,
) -> None:
    """
    Generates a synthetic dataset for a Smart Factory 'Quality Control' AI.
    Features: Temperature, Pressure, Vibration, Power, Cycle Time, Humidity.
    Label: 1 (Defective/Maintenance Needed), 0 (Healthy).
    """
    rng = np.random.default_rng(seed)
    Path("data").mkdir(exist_ok=True)

    # Physical-ish sensors
    temp = rng.normal(70, 5, n_samples)            # °C
    pressure = rng.normal(100, 15, n_samples)      # PSI
    vibration = rng.uniform(0.1, 5.0, n_samples)   # mm/s
    power = rng.normal(500, 50, n_samples)         # watts
    cycle_time = rng.normal(12, 2, n_samples)      # seconds
    humidity = rng.uniform(30, 60, n_samples)      # %

    # "Defect physics" rule (gives the model something learnable but not trivial)
    defect_score = (
        0.35 * (temp - 75)
        + 0.70 * (vibration - 3.2)
        - 0.25 * (pressure - 92)
        + 0.10 * (cycle_time - 12)
        + 0.05 * (humidity - 45)
    )

    probs = 1 / (1 + np.exp(-defect_score))
    label = (probs > 0.65).astype(int)

    df = pd.DataFrame(
        {
            "temp_c": temp,
            "pressure_psi": pressure,
            "vibration_mm_s": vibration,
            "power_watts": power,
            "cycle_time_s": cycle_time,
            "humidity_pct": humidity,
            "label": label,
        }
    )

    df.to_csv(out_path, index=False)
    print(f"✅ Dataset generated: {out_path} ({n_samples} samples)")


if __name__ == "__main__":
    generate_smart_factory_data()