# src/run_experiments.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from src.model import (
    train_and_evaluate,
    evaluate_with_noise,
    evaluate_with_latency_effects,
    evaluate_with_noise_and_latency,
    evaluate_with_staleness,
    evaluate_control_safety_margin,  # Week 05 NEW (if you added it in model.py)
)


# ---------------------------
# Week 02 - Noise
# ---------------------------
def write_noise_report(results: dict[float, float], baseline: float) -> Path:
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
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week02_noise_results.csv"
    lines = ["std,accuracy"]
    for std in sorted(results.keys()):
        lines.append(f"{std},{results[std]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------
# Week 04 - Latency
# ---------------------------
def write_latency_report(results: dict[int, dict], baseline: float) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week04_latency_results.txt"

    lines: list[str] = []
    lines.append("Week 04 – Physics-Aware AI Testing (Latency + Timeouts)\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Baseline (no constraints): {baseline:.4f}\n")

    lines.append("Latency sweep (delay with timeout/drop effects):")
    for delay_ms in sorted(results.keys()):
        m = results[delay_ms]
        lines.append(
            f"delay_ms={delay_ms} -> raw={m['raw_accuracy']:.4f} "
            f"effective={m['effective_accuracy']:.4f} "
            f"(failed={m['failed_frac']:.2%}, dropped={m['dropped_frac']:.2%}, timed_out={m['timed_out_frac']:.2%})"
        )

    lines.append("\nObservation:")
    lines.append(
        "Unlike a pure sleep(), this latency model affects correctness by treating "
        "timeouts and dropped samples as failures. This better matches real systems "
        "where missing deadlines is a functional failure."
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_latency_csv(results: dict[int, dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week04_latency_results.csv"
    header = [
        "delay_ms",
        "raw_accuracy",
        "effective_accuracy",
        "failed_frac",
        "dropped_frac",
        "timed_out_frac",
    ]
    lines = [",".join(header)]

    for delay_ms in sorted(results.keys()):
        m = results[delay_ms]
        lines.append(
            f"{delay_ms},{m['raw_accuracy']},{m['effective_accuracy']},"
            f"{m['failed_frac']},{m['dropped_frac']},{m['timed_out_frac']}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_latency_effects_csv(rows: list[dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week04_latency_effects_results.csv"
    header = [
        "delay_ms",
        "timeout_ms",
        "drop_rate",
        "raw_accuracy",
        "effective_accuracy",
        "failed_frac",
        "dropped_frac",
        "timed_out_frac",
    ]

    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_combined_csv(rows: list[dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week04_combined_noise_latency_results.csv"
    header = [
        "noise_std",
        "delay_ms",
        "timeout_ms",
        "drop_rate",
        "raw_accuracy",
        "effective_accuracy",
        "failed_frac",
        "dropped_frac",
        "timed_out_frac",
    ]

    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------
# Week 04 - Staleness Dynamics (Option 3)
# ---------------------------
def write_staleness_dynamics_csv(rows: list[dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week04_staleness_dynamics.csv"
    header = ["delay_ms", "accuracy_slow", "accuracy_fast", "drift_slow", "drift_fast"]

    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------
# Week 05 - Staleness World Speed (NEW)
# ---------------------------
def write_week05_world_speed_csv(rows: list[dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week05_staleness_world_speed.csv"
    header = ["world", "delay_ms", "drift_per_ms", "accuracy"]

    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------
# Week 05 - Latency Boundary (NEW)
# ---------------------------
def write_week05_latency_boundary_csv(rows: list[dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week05_latency_boundary.csv"
    header = ["delay_ms", "effective_accuracy"]

    lines = [",".join(header)]
    for r in rows:
        lines.append(f"{r['delay_ms']},{r['effective_accuracy']}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------
# Week 05 - Control Safety Margin (NEW)
# ---------------------------
def write_week05_safety_margin_csv(rows: list[dict]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    out_path = results_dir / "week05_safety_margin_vs_latency.csv"
    header = [
        "delay_ms",
        "timeout_ms",
        "drop_rate",
        "tolerance",
        "raw_accuracy",
        "effective_accuracy",
        "safety_rate",
        "effective_safety_rate",
        "failed_frac",
        "dropped_frac",
        "timed_out_frac",
    ]

    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in header))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    baseline = train_and_evaluate()
    print(f"Baseline accuracy (no constraints): {baseline:.4f}")

    # ---------------------------
    # Week 02 - Noise
    # ---------------------------
    print("\n[Week 02] Gaussian noise injection")
    std_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    noise_results: dict[float, float] = {}

    for std in std_values:
        acc = evaluate_with_noise(std=std, seed=0)
        noise_results[std] = acc
        print(f"noise std={std:.1f} -> accuracy={acc:.4f}")

    noise_report_file = write_noise_report(results=noise_results, baseline=baseline)
    noise_csv_file = write_noise_csv(results=noise_results)
    print(f"Saved noise report to: {noise_report_file}")
    print(f"Saved noise CSV to: {noise_csv_file}")

    # ---------------------------
    # Week 04 - Latency sweep
    # ---------------------------
    print("\n[Week 04] Latency Simulation (timeout/drop impacts)")

    timeout_ms = 120
    drop_rate = 0.05

    delays = [0, 50, 100, 200]
    latency_results: dict[int, dict] = {}

    for d in delays:
        metrics = evaluate_with_latency_effects(
            delay_ms=d,
            timeout_ms=timeout_ms,
            drop_rate=drop_rate,
            seed=0,
        )
        latency_results[d] = metrics
        print(
            f"delay={d}ms -> raw={metrics['raw_accuracy']:.4f} "
            f"effective={metrics['effective_accuracy']:.4f} "
            f"(failed={metrics['failed_frac']:.2%})"
        )

    latency_report_file = write_latency_report(results=latency_results, baseline=baseline)
    latency_csv_file = write_latency_csv(results=latency_results)
    print(f"Saved latency report to: {latency_report_file}")
    print(f"Saved latency CSV to: {latency_csv_file}")

    # ---------------------------
    # Week 04 - Latency Effects sweep
    # ---------------------------
    print("\n[Week 04] Latency Effects (explicit scenarios)")

    latency_cases = [
        {"delay_ms": 50, "timeout_ms": 120, "drop_rate": 0.00},
        {"delay_ms": 100, "timeout_ms": 120, "drop_rate": 0.05},
        {"delay_ms": 200, "timeout_ms": 120, "drop_rate": 0.05},
    ]

    latency_rows: list[dict] = []
    for c in latency_cases:
        metrics = evaluate_with_latency_effects(
            delay_ms=c["delay_ms"],
            timeout_ms=c["timeout_ms"],
            drop_rate=c["drop_rate"],
            seed=0,
        )
        latency_rows.append({**c, **metrics})

    latency_effects_csv = write_latency_effects_csv(latency_rows)
    print(f"Saved latency effects CSV to: {latency_effects_csv}")

    # ---------------------------
    # Week 04 - Combined (Noise + Latency)
    # ---------------------------
    print("\n[Week 04] Combined (Noise + Latency Effects)")

    combined_cases = [
        {"noise_std": 0.2, "delay_ms": 50, "timeout_ms": 120, "drop_rate": 0.00},
        {"noise_std": 0.5, "delay_ms": 100, "timeout_ms": 120, "drop_rate": 0.05},
        {"noise_std": 0.8, "delay_ms": 200, "timeout_ms": 120, "drop_rate": 0.05},
    ]

    combined_rows: list[dict] = []
    for c in combined_cases:
        metrics = evaluate_with_noise_and_latency(
            noise_std=c["noise_std"],
            delay_ms=c["delay_ms"],
            timeout_ms=c["timeout_ms"],
            drop_rate=c["drop_rate"],
            seed=0,
        )
        combined_rows.append({**c, **metrics})

    combined_csv = write_combined_csv(combined_rows)
    print(f"Saved combined CSV to: {combined_csv}")

    # ---------------------------
    # Week 04 - Staleness Dynamics (slow vs fast world)
    # ---------------------------
    print("\n[Week 04] Staleness Dynamics (slow vs fast world)")

    staleness_delays = [0, 25, 50, 100, 200, 300, 500]
    drift_slow = 0.003
    drift_fast = 0.015

    staleness_rows: list[dict] = []
    for d in staleness_delays:
        a_slow = evaluate_with_staleness(delay_ms=d, drift_per_ms=drift_slow, seed=0)
        a_fast = evaluate_with_staleness(delay_ms=d, drift_per_ms=drift_fast, seed=0)
        staleness_rows.append(
            {
                "delay_ms": d,
                "accuracy_slow": a_slow,
                "accuracy_fast": a_fast,
                "drift_slow": drift_slow,
                "drift_fast": drift_fast,
            }
        )

    staleness_csv = write_staleness_dynamics_csv(staleness_rows)
    print(f"Saved staleness dynamics CSV to: {staleness_csv}")

    # ---------------------------
    # Week 05 - World speed CSV (slow vs fast)
    # ---------------------------
    print("\n[Week 05] Staleness – Slow World vs Fast World CSV")

    delays_w5 = [0, 50, 100, 150, 200]
    worlds = {"slow_world": 0.003, "fast_world": 0.02}

    world_rows: list[dict] = []
    for world_name, drift in worlds.items():
        for d in delays_w5:
            acc = evaluate_with_staleness(delay_ms=d, drift_per_ms=drift, seed=0)
            world_rows.append(
                {"world": world_name, "delay_ms": d, "drift_per_ms": drift, "accuracy": acc}
            )

    world_csv = write_week05_world_speed_csv(world_rows)
    print(f"Saved Week 05 world-speed CSV to: {world_csv}")

    # ---------------------------
    # Week 05 - Latency boundary sweep (for your plot)
    # ---------------------------
    print("\n[Week 05] Latency Failure Boundary Sweep (0..300ms)")

    boundary_rows: list[dict] = []
    for delay in range(0, 301, 10):
        metrics = evaluate_with_latency_effects(
            delay_ms=delay,
            timeout_ms=timeout_ms,
            drop_rate=drop_rate,
            seed=0,
        )
        boundary_rows.append({"delay_ms": delay, "effective_accuracy": metrics["effective_accuracy"]})
        print(f"delay={delay}ms -> effective_accuracy={metrics['effective_accuracy']:.4f}")

    boundary_csv = write_week05_latency_boundary_csv(boundary_rows)
    print(f"Saved Week 05 boundary CSV to: {boundary_csv}")

    # ---------------------------
    # Week 05 - Control safety margin vs latency (optional but useful)
    # ---------------------------
    print("\n[Week 05] Control Safety Margin vs Latency (optional)")

    tolerance = 0.25
    safety_rows: list[dict] = []

    for delay in range(0, 301, 10):
        m = evaluate_control_safety_margin(
            delay_ms=delay,
            timeout_ms=timeout_ms,
            drop_rate=drop_rate,
            tolerance=tolerance,
            noise_std=0.0,
            drift_per_ms=0.0,
            seed=0,
        )
        safety_rows.append(
            {
                "delay_ms": delay,
                "timeout_ms": timeout_ms,
                "drop_rate": drop_rate,
                "tolerance": tolerance,
                **m,
            }
        )

    safety_csv = write_week05_safety_margin_csv(safety_rows)
    print(f"Saved Week 05 safety margin CSV to: {safety_csv}")


if __name__ == "__main__":
    main()