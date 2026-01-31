# src/plot_results.py

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_noise(csv_path: str, out_path: str) -> None:
    stds: list[float] = []
    accs: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stds.append(float(row["std"]))
            accs.append(float(row["accuracy"]))

    plt.figure()
    plt.plot(stds, accs, marker="o")
    plt.xlabel("Noise std (Gaussian)")
    plt.ylabel("Accuracy")
    plt.title("Week 02 - Accuracy vs Sensor Noise")
    plt.grid(True)

    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


def plot_latency(csv_path: str, out_path: str) -> None:
    delays: list[int] = []
    raw_accs: list[float] = []
    eff_accs: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delays.append(int(row["delay_ms"]))
            raw_accs.append(float(row["raw_accuracy"]))
            eff_accs.append(float(row["effective_accuracy"]))

    plt.figure()
    plt.plot(delays, raw_accs, marker="o", label="Raw accuracy")
    plt.plot(delays, eff_accs, marker="o", label="Effective accuracy")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Week 04 - Accuracy vs Latency (Raw vs Effective)")
    plt.grid(True)
    plt.legend()

    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


def plot_staleness_dynamics(csv_path: str, out_path: str) -> None:
    delays: list[int] = []
    acc_slow: list[float] = []
    acc_fast: list[float] = []

    drift_slow_val: float | None = None
    drift_fast_val: float | None = None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delays.append(int(row["delay_ms"]))
            acc_slow.append(float(row["accuracy_slow"]))
            acc_fast.append(float(row["accuracy_fast"]))

            if drift_slow_val is None:
                drift_slow_val = float(row["drift_slow"])
            if drift_fast_val is None:
                drift_fast_val = float(row["drift_fast"])

    label_slow = "Slow world"
    label_fast = "Fast world"
    if drift_slow_val is not None and drift_fast_val is not None:
        label_slow = f"Slow world (drift_per_ms={drift_slow_val})"
        label_fast = f"Fast world (drift_per_ms={drift_fast_val})"

    plt.figure()
    plt.plot(delays, acc_slow, marker="o", label=label_slow)
    plt.plot(delays, acc_fast, marker="o", label=label_fast)
    plt.xlabel("Staleness delay (ms)")
    plt.ylabel("Accuracy")
    plt.title("Week 04 - Staleness Dynamics (Slow vs Fast World)")
    plt.grid(True)
    plt.legend()

    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


def main():
    results_dir = Path("results")

    # Week 02 noise
    noise_csv = results_dir / "week02_noise_results.csv"
    if noise_csv.exists():
        plot_noise(str(noise_csv), str(results_dir / "week02_noise_plot.png"))

    # Week 04 latency
    latency_csv = results_dir / "week04_latency_results.csv"
    if latency_csv.exists():
        plot_latency(str(latency_csv), str(results_dir / "week04_latency_plot.png"))

    # Week 04 staleness dynamics (NEW)
    staleness_csv = results_dir / "week04_staleness_dynamics.csv"
    if staleness_csv.exists():
        plot_staleness_dynamics(
            str(staleness_csv),
            str(results_dir / "week04_staleness_dynamics_plot.png"),
        )


if __name__ == "__main__":
    main()