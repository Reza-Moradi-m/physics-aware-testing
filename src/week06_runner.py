# src/week06_runner.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import sys
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from src.io_utils import load_csv_dataset
from src.profiles import load_profile
from src.reliability_regression import (
    save_gold_standard,
    compare_to_gold_standard,
    write_regression_artifacts,
    prescriptive_mitigations,
)
from src.constraints import (
    apply_noise,
    apply_staleness,
    simulate_latency,
    apply_intermittent_dropout,
    apply_stuck_at_value,
    apply_bias_drift,
    apply_sensor_saturation,
    apply_quantization,
    simulate_packet_burst_loss,
)
from src.pytorch_model import train_mlp_binary, predict_mlp_binary


# ------------------------
# IO helpers
# ------------------------
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


# ------------------------
# Metrics
# ------------------------
def effective_accuracy_from_latency(preds: np.ndarray, y: np.ndarray, sim: dict[str, Any]) -> float:
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


# ------------------------
# Safety gate (warning mode)
# ------------------------
def _yellow_box_warning(lines: list[str]) -> str:
    Y = "\033[33m"
    R = "\033[0m"
    width = max(len(x) for x in lines) + 4
    top = "┌" + ("─" * (width - 2)) + "┐"
    mid = ["│ " + x.ljust(width - 4) + " │" for x in lines]
    bot = "└" + ("─" * (width - 2)) + "┘"
    return Y + "\n".join([top] + mid + [bot]) + R


def apply_safety_gate(results_rows: list[list[object]], threshold: float, model_name: str) -> int:
    """
    Returns:
      0 = PASS
      2 = SAFETY VIOLATION (audit still completed)
    """
    eff_vals: list[float] = []
    for r in results_rows:
        eff = r[4]  # effective_accuracy index in our new schema
        if isinstance(eff, (int, float)):
            eff_vals.append(float(eff))

    if not eff_vals:
        return 0

    worst_eff = min(eff_vals)
    if worst_eff < threshold:
        msg = _yellow_box_warning([
            "[WARNING] SAFETY MARGIN VIOLATION",
            f'Model "{model_name}" completed the audit run but failed safety thresholds.',
            f"Worst Effective Accuracy: {worst_eff:.2%}  (threshold={threshold:.2%})",
            "Deployment is NOT recommended for real-time / safety-critical systems.",
        ])
        print("\n" + msg + "\n")
        return 2

    return 0


# ------------------------
# Main runner
# ------------------------
def run(
    csv_path: str,
    label_col: str,
    profile_name: str = "industrial_wifi",
    model_name: str = "MLP_v1",
    save_gold: bool = False,
    gold_path: str = "results/gold_standard.json",
) -> int:
    results_dir = ensure_results_dir()
    profile = load_profile(profile_name)

    ds = load_csv_dataset(csv_path, label_col=label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.30, random_state=42, stratify=ds.y
    )

    # ------------------------
    # Train models
    # ------------------------
    mlp_res = train_mlp_binary(X_train, y_train, X_test, y_test, epochs=40, lr=1e-3, seed=0)
    mlp_preds = predict_mlp_binary(mlp_res.model, X_test)
    mlp_acc = float(accuracy_score(y_test, mlp_preds))
    mlp_note = f"PyTorch MLP (train_acc={mlp_res.train_acc:.4f}, test_acc={mlp_res.test_acc:.4f})"

    logreg = LogisticRegression(max_iter=2000, solver="lbfgs")
    logreg.fit(X_train, y_train)
    lr_preds = logreg.predict(X_test)
    lr_acc = float(accuracy_score(y_test, lr_preds))
    lr_note = f"Logistic Regression (test_acc={lr_acc:.4f})"

    # ------------------------
    # Results table schema
    # ------------------------
    header = ["model", "test_name", "param", "raw_accuracy", "effective_accuracy", "notes"]
    rows: list[list[object]] = []

    def add(model: str, test: str, param: str, raw: float, eff: float, notes: str) -> None:
        rows.append([model, test, param, float(raw), float(eff), notes])

    # Baselines
    add("PyTorch MLP", "baseline", "-", mlp_acc, mlp_acc, mlp_note)
    add("Logistic Regression", "baseline", "-", lr_acc, lr_acc, lr_note)

    # ------------------------
    # Noise sweep (per model)
    # ------------------------
    for std in profile.noise_stds:
        Xn = apply_noise(X_test, std=std, seed=0)

        pm = predict_mlp_binary(mlp_res.model, Xn)
        add("PyTorch MLP", "noise", f"std={std}", float(accuracy_score(y_test, pm)), float(accuracy_score(y_test, pm)), "Gaussian sensor noise")

        pl = logreg.predict(Xn)
        add("Logistic Regression", "noise", f"std={std}", float(accuracy_score(y_test, pl)), float(accuracy_score(y_test, pl)), "Gaussian sensor noise")

    # ------------------------
    # Latency sweep (effective accuracy uses baseline preds)
    # ------------------------
    for delay in profile.latency_delays_ms:
        sim = simulate_latency(
            n_samples=len(y_test),
            delay_ms=int(delay),
            timeout_ms=int(profile.timeout_ms),
            drop_rate=float(profile.drop_rate),
            seed=0,
        )

        eff_mlp = effective_accuracy_from_latency(mlp_preds, y_test, sim)
        add("PyTorch MLP", "latency_effects", f"delay_ms={delay}", mlp_acc, eff_mlp, "drop+timeout counted as failures")

        eff_lr = effective_accuracy_from_latency(lr_preds, y_test, sim)
        add("Logistic Regression", "latency_effects", f"delay_ms={delay}", lr_acc, eff_lr, "drop+timeout counted as failures")

    # ------------------------
    # Staleness (per model)
    # ------------------------
    for delay in [0, 50, 100, 200, 300]:
        for drift_name, drift in [("slow", 0.003), ("fast", 0.015)]:
            Xs = apply_staleness(X_test, delay_ms=delay, drift_per_ms=drift, seed=0)

            pm = predict_mlp_binary(mlp_res.model, Xs)
            accm = float(accuracy_score(y_test, pm))
            add("PyTorch MLP", "staleness", f"{drift_name}: delay={delay}, drift={drift}", accm, accm, "staleness drift")

            pl = logreg.predict(Xs)
            accl = float(accuracy_score(y_test, pl))
            add("Logistic Regression", "staleness", f"{drift_name}: delay={delay}, drift={drift}", accl, accl, "staleness drift")

    # ------------------------
    # Failure modes (per model)
    # ------------------------
    def failure_mode_eval(param: str, Xv: np.ndarray, note: str) -> None:
        pm = predict_mlp_binary(mlp_res.model, Xv)
        accm = float(accuracy_score(y_test, pm))
        add("PyTorch MLP", "failure_mode", param, accm, accm, note)

        pl = logreg.predict(Xv)
        accl = float(accuracy_score(y_test, pl))
        add("Logistic Regression", "failure_mode", param, accl, accl, note)

    # Existing
    failure_mode_eval("intermittent_dropout=0.03", apply_intermittent_dropout(X_test, drop_prob=0.03, seed=0), "rows zeroed out")
    failure_mode_eval("stuck_at_value=0.03", apply_stuck_at_value(X_test, stuck_prob=0.03, seed=0), "rows repeat previous")
    failure_mode_eval("bias_drift(bias_per_ms=0.0008,delay=200)", apply_bias_drift(X_test, bias_per_ms=0.0008, delay_ms=200), "systematic bias")

    # Hardware-realistic
    failure_mode_eval("sensor_saturation(per_feature,p99.5)", apply_sensor_saturation(X_test, per_feature=True, ref_X=X_train, clip_percentile=99.5), "caps sensor outputs (flatline)")
    failure_mode_eval("quantization(decimals=1)", apply_quantization(X_test, decimals=1), "low-resolution sensor rounding")

    # Packet burst loss => effective accuracy (dropped fail)
    burst = simulate_packet_burst_loss(
        n_samples=len(y_test),
        burst_ms=int(profile.burst_ms),
        sample_period_ms=int(profile.sample_period_ms),
        seed=0,
    )
    dropped_mask = np.asarray(burst["dropped_mask"], dtype=bool)

    corr_mlp = (mlp_preds == y_test).copy()
    corr_mlp[dropped_mask] = False
    add("PyTorch MLP", "failure_mode", f"packet_burst_loss({profile.burst_ms}ms)", mlp_acc, float(corr_mlp.mean()), "contiguous blackout window; dropped=fail")

    corr_lr = (lr_preds == y_test).copy()
    corr_lr[dropped_mask] = False
    add("Logistic Regression", "failure_mode", f"packet_burst_loss({profile.burst_ms}ms)", lr_acc, float(corr_lr.mean()), "contiguous blackout window; dropped=fail")
    
        # ------------------------
    # Combined Constraints Evaluation (MLP only)
    # Explicit category for requirement coverage
    # ------------------------
    combined_cases = [
        {"noise_std": 0.2, "delay_ms": 100, "drift_per_ms": 0.003},
        {"noise_std": 0.3, "delay_ms": 100, "drift_per_ms": 0.015},
        {"noise_std": 0.5, "delay_ms": 150, "drift_per_ms": 0.015},
    ]

    for case in combined_cases:
        noise_std = case["noise_std"]
        delay_ms = case["delay_ms"]
        drift_per_ms = case["drift_per_ms"]

        # Apply noise first
        X_combo = apply_noise(X_test, std=noise_std, seed=0)

        # Then apply staleness drift
        X_combo = apply_staleness(
            X_combo,
            delay_ms=delay_ms,
            drift_per_ms=drift_per_ms,
            seed=0,
        )

        # Predict with MLP
        combo_preds = predict_mlp_binary(mlp_res.model, X_combo)
        raw_acc = float(accuracy_score(y_test, combo_preds))

        # Then apply latency/drop effects to convert raw -> effective
        sim = simulate_latency(
            n_samples=len(y_test),
            delay_ms=delay_ms,
            timeout_ms=int(profile.timeout_ms),
            drop_rate=float(profile.drop_rate),
            seed=0,
        )

        combo_eff = effective_accuracy_from_latency(combo_preds, y_test, sim)

        add(
            "PyTorch MLP",
            "combined_constraints",
            f"noise={noise_std}|delay={delay_ms}|drift={drift_per_ms}",
            raw_acc,
            combo_eff,
            "combined noise + latency + staleness",
        )

    # ------------------------
    # Feature sensitivity (MLP only)
    # ------------------------
    feature_names = list(ds.feature_names)
    sens_std = float(profile.sensitivity_noise_std)
    for idx, feat in enumerate(feature_names):
        X_feat = np.asarray(X_test, dtype=float).copy()
        rng = np.random.default_rng(0)
        noise = rng.normal(loc=0.0, scale=sens_std, size=len(X_feat))
        X_feat[:, idx] = X_feat[:, idx] + noise

        preds_feat = predict_mlp_binary(mlp_res.model, X_feat)
        acc_feat = float(accuracy_score(y_test, preds_feat))
        drop = mlp_acc - acc_feat
        add("PyTorch MLP", "feature_sensitivity", feat, acc_feat, acc_feat, f"baseline_drop={drop:.6f}")

    # ------------------------
    # Operating envelope (MLP only) for regression
    # ------------------------
    env_header = ["noise_std", "delay_ms", "effective_accuracy", "pass_fail"]
    env_rows: list[list[object]] = []

    for std in profile.envelope_noise_stds:
        Xn = apply_noise(X_test, std=float(std), seed=0)
        preds = predict_mlp_binary(mlp_res.model, Xn)

        for delay in profile.envelope_delays_ms:
            sim = simulate_latency(
                n_samples=len(y_test),
                delay_ms=int(delay),
                timeout_ms=int(profile.timeout_ms),
                drop_rate=float(profile.drop_rate),
                seed=0,
            )

            burst2 = simulate_packet_burst_loss(
                n_samples=len(y_test),
                burst_ms=int(profile.burst_ms),
                sample_period_ms=int(profile.sample_period_ms),
                seed=0,
            )

            sim["dropped_mask"] = (
                np.asarray(sim["dropped_mask"], dtype=bool)
                | np.asarray(burst2["dropped_mask"], dtype=bool)
            )

            eff = effective_accuracy_from_latency(preds, y_test, sim)
            pf = "PASS" if eff >= float(profile.threshold) else "FAIL"
            env_rows.append([float(std), int(delay), float(eff), pf])

    out_csv = results_dir / "week06_product_demo_results.csv"
    env_csv = results_dir / "week06_operating_envelope.csv"

    write_csv(out_csv, header, rows)
    write_csv(env_csv, env_header, env_rows)

    # ------------------------
    # Grade + Safety gate based on ALL results
    # ------------------------
    eff_vals = [float(r[4]) for r in rows]
    worst_eff = min(eff_vals)
    grade = safety_grade(worst_eff)

    # ------------------------
    # Regression: compare to gold if exists
    # ------------------------
    regression = compare_to_gold_standard(
        gold_path=gold_path,
        profile_name=profile.name,
        threshold=float(profile.threshold),
        baseline_acc=float(mlp_acc),
        worst_eff=float(worst_eff),
        envelope_csv_path=str(env_csv),
    )
    reg_json, reg_md = write_regression_artifacts(results_dir=str(results_dir), regression=regression)

    # Save gold standard if requested
    if save_gold:
        save_gold_standard(
            out_path=gold_path,
            profile_name=profile.name,
            threshold=float(profile.threshold),
            baseline_acc=float(mlp_acc),
            worst_eff=float(worst_eff),
            envelope_csv_path=str(env_csv),
        )

    # ------------------------
    # Fix-it engine (based on MLP rows only)
    # ------------------------
    df_all = pd.read_csv(out_csv)
    df_mlp = df_all[df_all["model"] == "PyTorch MLP"].copy()

    mitigations_md = prescriptive_mitigations(
        df_results=df_mlp,
        threshold=float(profile.threshold),
        profile_name=profile.name,
    )
    mit_path = results_dir / "mitigations.md"
    mit_path.write_text(mitigations_md, encoding="utf-8")

    # ------------------------
    # Report
    # ------------------------
    positives = (
        f"- Profile: **{profile.name}** (threshold={profile.threshold:.2f}, timeout_ms={profile.timeout_ms}, drop_rate={profile.drop_rate:.2f})\n"
        f"- PyTorch MLP baseline accuracy: **{mlp_acc:.4f}**\n"
        f"- Logistic Regression baseline accuracy: **{lr_acc:.4f}**\n"
        "- Includes explicit combined-constraints tests (noise + latency + staleness).\n"
        "- Produced stress tests, operating envelope, regression artifacts, and prescriptive mitigations."
    )

    concerns = (
        f"- Worst effective accuracy observed (global): **{worst_eff:.4f}** (Grade **{grade}**)\n"
        "- Late/missing predictions are treated as failures.\n"
        "- Some latency conditions can collapse effective accuracy."
    )

    regression_text = "No gold standard found yet. Run with `--save-gold` to set V1 baseline."
    if regression is not None:
        regression_text = regression.notes

    artifacts = (
        f"- Results CSV: `{out_csv}`\n"
        f"- Envelope CSV (MLP): `{env_csv}`\n"
        f"- Dashboard: `results/master_dashboard.png`\n"
        f"- Mitigations: `{mit_path}`\n"
        + (f"- Regression JSON: `{reg_json}`\n" if reg_json else "")
        + (f"- Regression Summary: `{reg_md}`\n" if reg_md else "")
        + (f"- Gold Standard: `{gold_path}`\n" if save_gold else "")
    )

    report_md = results_dir / "week06_report.md"
    write_report_md(
        report_md,
        "Week 06 – Smart Factory Model Reliability Report (Startup Demo)",
        [
            ("Summary", f"Model: {mlp_note}\n\nSafety Grade: **{grade}** (worst effective accuracy={worst_eff:.4f})"),
            ("Positives", positives),
            ("Issues / Concerns", concerns),
            ("Operating Envelope", "PASS/FAIL grid saved in `results/week06_operating_envelope.csv`."),
            ("Regression Check (V1 vs V2)", regression_text),
            ("Required Mitigations (Fix-it Engine)", mitigations_md),
            ("Artifacts", artifacts),
        ],
    )

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {env_csv}")
    print(f"Saved: {report_md}")
    if save_gold:
        print(f"Saved: {gold_path}")

    exit_code = apply_safety_gate(rows, threshold=float(profile.threshold), model_name=model_name)
    return exit_code


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV dataset")
    ap.add_argument("--label", required=True, help="Label column name (e.g., 'label')")
    ap.add_argument("--profile", default="industrial_wifi", help="Deployment profile name (configs/profiles.json)")
    ap.add_argument("--model-name", default="MLP_v1", help='Model name used in warnings (default: "MLP_v1")')
    ap.add_argument("--save-gold", action="store_true", help="Save current run as V1 baseline gold_standard.json")
    ap.add_argument("--gold-path", default="results/gold_standard.json", help="Path to gold standard JSON")
    args = ap.parse_args()

    code = run(
        csv_path=args.csv,
        label_col=args.label,
        profile_name=args.profile,
        model_name=args.model_name,
        save_gold=bool(args.save_gold),
        gold_path=args.gold_path,
    )
    sys.exit(code)