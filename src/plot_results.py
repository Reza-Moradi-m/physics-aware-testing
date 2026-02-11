# src/plot_results.py

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _ensure_parent(out_path: str) -> None:
    Path(out_path).parent.mkdir(exist_ok=True)


def plot_noise(csv_path: str, out_path: str) -> None:
    stds: list[float] = []
    accs: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stds.append(float(row["std"]))
            accs.append(float(row["accuracy"]))

    # sort by std
    pairs = sorted(zip(stds, accs), key=lambda t: t[0])
    stds = [p[0] for p in pairs]
    accs = [p[1] for p in pairs]

    plt.figure()
    plt.plot(stds, accs, marker="o")
    plt.xlabel("Noise std (Gaussian)")
    plt.ylabel("Accuracy")
    plt.title("Week 02 - Accuracy vs Sensor Noise")
    plt.grid(True)

    _ensure_parent(out_path)
    plt.savefig(out_path)
    plt.close()
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

    # sort by delay
    triples = sorted(zip(delays, raw_accs, eff_accs), key=lambda t: t[0])
    delays = [t[0] for t in triples]
    raw_accs = [t[1] for t in triples]
    eff_accs = [t[2] for t in triples]

    plt.figure()
    plt.plot(delays, raw_accs, marker="o", label="Raw accuracy")
    plt.plot(delays, eff_accs, marker="o", label="Effective accuracy")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Week 04 - Accuracy vs Latency (Raw vs Effective)")
    plt.grid(True)
    plt.legend()

    _ensure_parent(out_path)
    plt.savefig(out_path)
    plt.close()
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

    # sort by delay
    triples = sorted(zip(delays, acc_slow, acc_fast), key=lambda t: t[0])
    delays = [t[0] for t in triples]
    acc_slow = [t[1] for t in triples]
    acc_fast = [t[2] for t in triples]

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

    _ensure_parent(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to: {out_path}")


def plot_world_speed(csv_path: str, out_path: str) -> None:
    slow_points: list[tuple[int, float]] = []
    fast_points: list[tuple[int, float]] = []
    other_points: dict[str, list[tuple[int, float]]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delay = int(row["delay_ms"])
            acc = float(row["accuracy"])
            world = (row.get("world") or "").strip()

            if world == "slow_world":
                slow_points.append((delay, acc))
            elif world == "fast_world":
                fast_points.append((delay, acc))
            else:
                other_points.setdefault(world or "unknown", []).append((delay, acc))

    slow_points.sort(key=lambda t: t[0])
    fast_points.sort(key=lambda t: t[0])

    plt.figure()
    if slow_points:
        plt.plot([d for d, _ in slow_points], [a for _, a in slow_points], marker="o",
                 label="Slow World (stable environment)")
    if fast_points:
        plt.plot([d for d, _ in fast_points], [a for _, a in fast_points], marker="o",
                 label="Fast World (dynamic environment)")

    # plot any unexpected world labels, just in case
    for world, pts in other_points.items():
        pts.sort(key=lambda t: t[0])
        plt.plot([d for d, _ in pts], [a for _, a in pts], marker="o", label=f"World: {world}")

    plt.xlabel("Delay (ms)")
    plt.ylabel("Accuracy")
    plt.title("Week 05 – Staleness: Fast vs Slow World")
    plt.grid(True)
    plt.legend()

    _ensure_parent(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to: {out_path}")


def plot_latency_boundary(csv_path: str, out_path: str) -> None:
    delays: list[int] = []
    accs: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delays.append(int(row["delay_ms"]))
            accs.append(float(row["effective_accuracy"]))

    pairs = sorted(zip(delays, accs), key=lambda t: t[0])
    delays = [p[0] for p in pairs]
    accs = [p[1] for p in pairs]

    plt.figure()
    plt.plot(delays, accs, marker="o")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Effective Accuracy")
    plt.title("Week 05 - AI Failure Boundary vs Latency")
    plt.grid(True)

    _ensure_parent(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to: {out_path}")


def plot_safety_margin_vs_latency(csv_path: str, out_path: str) -> None:
    delays: list[int] = []
    safety_rate: list[float] = []
    eff_safety_rate: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            delays.append(int(row["delay_ms"]))
            safety_rate.append(float(row["safety_rate"]))
            eff_safety_rate.append(float(row["effective_safety_rate"]))

    triples = sorted(zip(delays, safety_rate, eff_safety_rate), key=lambda t: t[0])
    delays = [t[0] for t in triples]
    safety_rate = [t[1] for t in triples]
    eff_safety_rate = [t[2] for t in triples]

    plt.figure()
    plt.plot(delays, safety_rate, marker="o", label="Safety rate (no latency penalties)")
    plt.plot(delays, eff_safety_rate, marker="o", label="Effective safety rate (timeouts/drops fail)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Safety Rate")
    plt.title("Week 05 - Control Safety Margin vs Latency")
    plt.grid(True)
    plt.legend()

    _ensure_parent(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to: {out_path}")


def main() -> None:
    results_dir = Path("results")

    # Week 02 noise
    noise_csv = results_dir / "week02_noise_results.csv"
    if noise_csv.exists():
        plot_noise(str(noise_csv), str(results_dir / "week02_noise_plot.png"))

    # Week 04 latency
    latency_csv = results_dir / "week04_latency_results.csv"
    if latency_csv.exists():
        plot_latency(str(latency_csv), str(results_dir / "week04_latency_plot.png"))

    # Week 04 staleness dynamics
    staleness_csv = results_dir / "week04_staleness_dynamics.csv"
    if staleness_csv.exists():
        plot_staleness_dynamics(
            str(staleness_csv),
            str(results_dir / "week04_staleness_dynamics_plot.png"),
        )

    # Week 05 staleness world speed
    world_speed_csv = results_dir / "week05_staleness_world_speed.csv"
    if world_speed_csv.exists():
        plot_world_speed(
            str(world_speed_csv),
            str(results_dir / "week05_staleness_world_speed_plot.png"),
        )

    # Week 05 latency boundary
    boundary_csv = results_dir / "week05_latency_boundary.csv"
    if boundary_csv.exists():
        plot_latency_boundary(
            str(boundary_csv),
            str(results_dir / "week05_latency_boundary_plot.png"),
        )

    # Week 05 safety margin vs latency (optional)
    safety_csv = results_dir / "week05_safety_margin_vs_latency.csv"
    if safety_csv.exists():
        plot_safety_margin_vs_latency(
            str(safety_csv),
            str(results_dir / "week05_safety_margin_vs_latency_plot.png"),
        )


if __name__ == "__main__":
    main()