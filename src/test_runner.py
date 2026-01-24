# src/test_runner.py

from pathlib import Path
from datetime import datetime

from src.model import train_and_evaluate, evaluate_with_noise


def write_noise_report(results: dict[float, float], baseline: float) -> Path:
    """
    Writes a human-readable report to results/week02_noise_results.txt
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week02_noise_results.txt"

    lines: list[str] = []
    lines.append("Week 02 – Physics-Aware AI Testing (Sensor Noise)\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Baseline (no constraints): {baseline:.4f}")
    lines.append(f"Baseline (std=0.0): {results.get(0.0, baseline):.4f}\n")

    lines.append("Noise sweep (Gaussian noise added to test inputs):")
    for std in sorted(results.keys()):
        lines.append(f"std={std:.1f} -> accuracy={results[std]:.4f}")

    lines.append("\nObservation:")
    lines.append(
        "Accuracy decreases as sensor noise increases, revealing robustness limits "
        "that baseline accuracy testing does not capture."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_noise_csv(results: dict[float, float]) -> Path:
    """
    Writes machine-readable results to results/week02_noise_results.csv
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week02_noise_results.csv"
    lines = ["std,accuracy"]
    for std in sorted(results.keys()):
        lines.append(f"{std},{results[std]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main():
    baseline = train_and_evaluate()
    print(f"Baseline accuracy (no constraints): {baseline:.4f}")

    print("\nPhysics-aware test: Gaussian noise injection")

    std_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    results: dict[float, float] = {}

    for std in std_values:
        acc = evaluate_with_noise(std=std, seed=0)
        results[std] = acc
        print(f"noise std={std:.1f} -> accuracy={acc:.4f}")

    report_file = write_noise_report(results=results, baseline=baseline)
    csv_file = write_noise_csv(results=results)

    print(f"\nSaved report to: {report_file}")
    print(f"Saved CSV to: {csv_file}")


if __name__ == "__main__":
    main()