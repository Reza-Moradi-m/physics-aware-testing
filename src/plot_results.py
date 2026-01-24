from pathlib import Path
import csv

import matplotlib.pyplot as plt


def main():
    csv_path = Path("results/week02_noise_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}. Run `python -m src.test_runner` first.")

    std_vals = []
    acc_vals = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            std_vals.append(float(row["std"]))
            acc_vals.append(float(row["accuracy"]))

    plt.figure()
    plt.plot(std_vals, acc_vals, marker="o")
    plt.xlabel("Noise standard deviation (std)")
    plt.ylabel("Accuracy")
    plt.title("Physics-Aware Test: Accuracy vs Sensor Noise")
    plt.grid(True)

    out_path = Path("results/week02_noise_plot.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()