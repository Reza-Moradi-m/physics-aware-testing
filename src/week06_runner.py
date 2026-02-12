# src/week06_runner.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.io_utils import load_csv_dataset
from src.constraints import (
    apply_noise,
    apply_staleness,
    simulate_latency,
    apply_intermittent_dropout,
    apply_stuck_at_value,
    apply_bias_drift,
)
from src.pytorch_model import train_mlp_binary, predict_mlp_binary


def ensure_results_dir() -> Path:
    d = Path("results")
    d.mkdir(exist_ok=True)
    return d


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_report_md(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    for h, body in sections:
        lines.append(f"## {h}\n")
        lines.append(body.strip() + "\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def effective_accuracy_from_latency(preds: np.ndarray, y: np.ndarray, sim: dict) -> float:
    dropped = np.asarray(sim["dropped_mask"], dtype=bool)
    timed = np.asarray(sim["timed_out_mask"], dtype=bool)
    failed = dropped | timed

    correct = (preds == y).copy()
    correct[failed] = False
    return float(correct.mean())


def safety_grade(worst_case: float) -> str:
    if worst_case >= 0.90:
        return "A"
    if worst_case >= 0.80:
        return "B"
    if worst_case >= 0.70:
        return "C"
    if worst_case >= 0.60:
        return "D"
    return "F"


def run(csv_path: str, label_col: str) -> None:
    results_dir = ensure_results_dir()

    ds = load_csv_dataset(csv_path, label_col=label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.30, random_state=42, stratify=ds.y
    )

    # ------------------------
    # Train PyTorch model
    # ------------------------
    res = train_mlp_binary(X_train, y_train, X_test, y_test, epochs=40, lr=1e-3, seed=0)
    baseline_preds = predict_mlp_binary(res.model, X_test)
    baseline_acc = float(accuracy_score(y_test, baseline_preds))
    model_note = f"PyTorch MLP (train_acc={res.train_acc:.4f}, test_acc={res.test_acc:.4f})"

    # ------------------------
    # Main results CSV
    # ------------------------
    rows: list[list[object]] = []
    header = ["test_name", "param", "raw_accuracy", "effective_accuracy", "notes"]

    # Baseline
    rows.append(["baseline", "-", baseline_acc, baseline_acc, model_note])

    # Noise sweep
    for std in [0.0, 0.1, 0.2, 0.5, 0.8]:
        Xn = apply_noise(X_test, std=std, seed=0)
        preds = predict_mlp_binary(res.model, Xn)
        acc = float(accuracy_score(y_test, preds))
        rows.append(["noise", f"std={std}", acc, acc, "Gaussian sensor noise"])

    # Latency sweep (effective accuracy)
    for delay in [0, 50, 100, 150, 200]:
        sim = simulate_latency(
            n_samples=len(y_test),
            delay_ms=delay,
            timeout_ms=120,
            drop_rate=0.05,
            seed=0,
        )
        eff = effective_accuracy_from_latency(baseline_preds, y_test, sim)
        rows.append(["latency_effects", f"delay_ms={delay}", baseline_acc, eff, "drop+timeout counted as failures"])

    # Staleness (slow vs fast)
    for delay in [0, 50, 100, 200, 300]:
        for drift_name, drift in [("slow", 0.003), ("fast", 0.015)]:
            Xs = apply_staleness(X_test, delay_ms=delay, drift_per_ms=drift, seed=0)
            preds = predict_mlp_binary(res.model, Xs)
            acc = float(accuracy_score(y_test, preds))
            rows.append(["staleness", f"{drift_name}: delay={delay}, drift={drift}", acc, acc, "staleness drift"])

    # Advanced failure modes
    X_drop = apply_intermittent_dropout(X_test, drop_prob=0.03, seed=0)
    acc_drop = float(accuracy_score(y_test, predict_mlp_binary(res.model, X_drop)))
    rows.append(["failure_mode", "intermittent_dropout=0.03", acc_drop, acc_drop, "rows zeroed out"])

    X_stuck = apply_stuck_at_value(X_test, stuck_prob=0.03, seed=0)
    acc_stuck = float(accuracy_score(y_test, predict_mlp_binary(res.model, X_stuck)))
    rows.append(["failure_mode", "stuck_at_value=0.03", acc_stuck, acc_stuck, "rows repeat previous"])

    X_bias = apply_bias_drift(X_test, bias_per_ms=0.0008, delay_ms=200)
    acc_bias = float(accuracy_score(y_test, predict_mlp_binary(res.model, X_bias)))
    rows.append(["failure_mode", "bias_drift(bias_per_ms=0.0008,delay=200)", acc_bias, acc_bias, "systematic bias"])

    out_csv = results_dir / "week06_product_demo_results.csv"
    write_csv(out_csv, header, rows)

    # ------------------------
    # NEW: Operating envelope grid (noise vs latency)
    # We use EFFECTIVE accuracy with latency drops/timeouts.
    # ------------------------
    env_header = ["noise_std", "delay_ms", "effective_accuracy", "pass_fail"]
    env_rows: list[list[object]] = []

    # Set threshold for pass/fail boundary (startup-like safety boundary)
    THRESH = 0.80

    noise_vals = [0.0, 0.1, 0.2, 0.3, 0.5]
    delay_vals = [0, 50, 100, 150, 200]

    for std in noise_vals:
        # Apply noise first, then evaluate baseline model predictions under noise
        Xn = apply_noise(X_test, std=std, seed=0)
        preds = predict_mlp_binary(res.model, Xn)

        for delay in delay_vals:
            sim = simulate_latency(
                n_samples=len(y_test),
                delay_ms=delay,
                timeout_ms=120,
                drop_rate=0.05,
                seed=0,
            )
            eff = effective_accuracy_from_latency(preds, y_test, sim)
            pf = "PASS" if eff >= THRESH else "FAIL"
            env_rows.append([std, delay, eff, pf])

    env_csv = results_dir / "week06_operating_envelope.csv"
    write_csv(env_csv, env_header, env_rows)

    # ------------------------
    # Report
    # ------------------------
    worst_eff = min(float(r[3]) for r in rows if isinstance(r[3], (int, float)))
    grade = safety_grade(worst_eff)

    positives = (
        f"- Baseline accuracy on Smart Factory CSV: **{baseline_acc:.4f}**\n"
        f"- Runs physics stress tests: noise, latency(drop/timeout), staleness(slow/fast), and failure modes.\n"
        f"- Produces an Operating Envelope grid (noise vs latency) with PASS/FAIL boundary."
    )

    concerns = (
        f"- Worst effective accuracy observed: **{worst_eff:.4f}** (Grade **{grade}**)\n"
        "- Fast-world staleness and sensor failure modes can drop accuracy significantly.\n"
        "- Effective accuracy is the real deployment metric: late/missing decisions count as failures."
    )

    envelope = (
        "- The Operating Envelope shows safe vs unsafe regions:\n"
        "  - X-axis: latency delay\n"
        "  - Y-axis: noise level\n"
        "  - Cell color (in dashboard) indicates PASS/FAIL based on effective accuracy threshold."
    )

    artifacts = (
        f"- Results CSV: `{out_csv}`\n"
        f"- Envelope CSV: `{env_csv}`\n"
        f"- Dashboard: `results/master_dashboard.png` (generated next)\n"
    )

    report_md = results_dir / "week06_report.md"
    write_report_md(
        report_md,
        "Week 06 – Smart Factory Model Reliability Report (Startup Demo)",
        [
            ("Summary", f"Model: {model_note}\n\nSafety Grade: **{grade}** (worst effective accuracy={worst_eff:.4f})"),
            ("Positives", positives),
            ("Issues / Concerns", concerns),
            ("Operating Envelope", envelope),
            ("Artifacts", artifacts),
        ],
    )

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {env_csv}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV dataset")
    ap.add_argument("--label", required=True, help="Label column name (e.g., 'label')")
    args = ap.parse_args()

    run(csv_path=args.csv, label_col=args.label)