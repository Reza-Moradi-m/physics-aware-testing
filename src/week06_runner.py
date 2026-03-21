# src/week06_runner.py
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.constraints import (
    apply_bias_drift,
    apply_intermittent_dropout,
    apply_noise,
    apply_quantization,
    apply_sensor_saturation,
    apply_staleness,
    apply_stuck_at_value,
    simulate_latency,
    simulate_packet_burst_loss,
)
from src.profiles import load_profile


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def fit_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            early_stopping=True,
        ),
    }
    for _, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = model.predict(X)
    return float(accuracy_score(y, preds))


def apply_latency_failure_rows(X: np.ndarray, dropped_mask: np.ndarray, timed_out_mask: np.ndarray) -> np.ndarray:
    X_mod = np.asarray(X, dtype=float).copy()
    bad = dropped_mask | timed_out_mask
    X_mod[bad, :] = 0.0
    return X_mod


def make_operating_envelope(
    model_name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    profile,
) -> pd.DataFrame:
    rows = []
    for noise_std in profile.envelope_noise_stds:
        for delay_ms in profile.envelope_delays_ms:
            X_mod = apply_noise(X_test, noise_std, seed=11)
            X_mod = apply_staleness(X_mod, delay_ms, drift_per_ms=profile.drift_per_ms, seed=12)

            latency_info = simulate_latency(
                n_samples=len(X_mod),
                delay_ms=delay_ms,
                timeout_ms=profile.timeout_ms,
                drop_rate=profile.drop_rate,
                seed=13,
            )
            X_mod = apply_latency_failure_rows(
                X_mod,
                latency_info["dropped_mask"],
                latency_info["timed_out_mask"],
            )

            acc = evaluate_model(model, X_mod, y_test)
            rows.append(
                {
                    "model": model_name,
                    "noise_std": noise_std,
                    "delay_ms": delay_ms,
                    "effective_accuracy": acc,
                }
            )
    return pd.DataFrame(rows)


def feature_sensitivity(
    model_name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    profile,
) -> pd.DataFrame:
    baseline = evaluate_model(model, X_test, y_test)
    rows = []

    for i, fname in enumerate(feature_names):
        X_mod = np.asarray(X_test, dtype=float).copy()
        rng = np.random.default_rng(100 + i)

        shuffled_col = X_mod[:, i].copy()
        rng.shuffle(shuffled_col)
        X_mod[:, i] = shuffled_col

        acc = evaluate_model(model, X_mod, y_test)
        rows.append(
            {
                "model": model_name,
                "feature": fname,
                "baseline_accuracy": baseline,
                "perturbed_accuracy": acc,
                "drop": baseline - acc,
            }
        )

    return pd.DataFrame(rows)


def build_mitigations(results_df: pd.DataFrame) -> list[str]:
    mitigations: list[str] = []

    grouped = (
        results_df.groupby("constraint")["delta"]
        .mean()
        .sort_values(ascending=False)
    )

    for constraint, mean_delta in grouped.items():
        if mean_delta < 0.02:
            continue

        if constraint == "latency":
            mitigations.append(
                "- Latency is a major failure mode. Consider buffering, timeout-aware inference, or fallback logic."
            )
        elif constraint == "staleness":
            mitigations.append(
                "- Staleness is hurting reliability. Consider freshness checks, timestamp validation, or drift-aware preprocessing."
            )
        elif constraint == "noise":
            mitigations.append(
                "- Noise is degrading performance. Consider filtering, robust normalization, or noise-augmented training."
            )
        elif constraint == "intermittent_dropout":
            mitigations.append(
                "- Intermittent dropout is damaging performance. Consider imputation, missingness indicators, or row-level redundancy."
            )
        elif constraint == "stuck_at_value":
            mitigations.append(
                "- Stuck-at-value failures are visible. Consider freeze detection, watchdog logic, or change-rate monitoring."
            )
        elif constraint == "bias_drift":
            mitigations.append(
                "- Bias drift is significant. Consider scheduled recalibration, online monitoring, or adaptive normalization."
            )
        elif constraint == "saturation":
            mitigations.append(
                "- Sensor saturation is a risk. Consider wider sensor range, clipping-aware preprocessing, or saturation alarms."
            )
        elif constraint == "quantization":
            mitigations.append(
                "- Quantization is causing loss. Consider higher-resolution sensors or quantization-aware training."
            )
        elif constraint == "packet_burst_loss":
            mitigations.append(
                "- Burst loss is severe. Consider temporal buffering, retransmission, redundancy, or blackout-tolerant features."
            )

    if not mitigations:
        mitigations.append("- No severe failure mode crossed the mitigation threshold in this run.")

    return mitigations


def write_markdown_report(
    profile,
    dataset_name: str,
    results_df: pd.DataFrame,
    safety_triggered: bool,
    threshold: float,
) -> None:
    worst = results_df.loc[results_df["effective_accuracy"].idxmin()]
    best_clean = results_df[results_df["constraint"] == "clean"]["effective_accuracy"].max()

    lines = [
        "# RobustAI Engine Audit Report",
        "",
        f"**Dataset:** {dataset_name}",
        f"**Profile:** {profile.name}",
        f"**Profile Description:** {profile.desc}",
        f"**Safety Threshold:** {threshold:.2f}",
        f"**Best Clean Accuracy:** {best_clean:.4f}",
        f"**Worst Effective Accuracy:** {worst['effective_accuracy']:.4f}",
        f"**Worst Case:** model={worst['model']}, constraint={worst['constraint']}",
        "",
    ]

    if safety_triggered:
        lines.extend(
            [
                "## CRITICAL",
                "",
                "Safety gate triggered. Deployment is not recommended without mitigation.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## PASS",
                "",
                "Safety gate did not trigger for this run.",
                "",
            ]
        )

    lines.extend(
        [
            "## Result Summary",
            "",
            results_df.to_markdown(index=False),
            "",
        ]
    )

    (RESULTS_DIR / "week06_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_regression_artifacts(results_df: pd.DataFrame, save_gold: bool) -> None:
    gold_path = RESULTS_DIR / "gold_standard.json"
    report_path = RESULTS_DIR / "regression_report.json"
    summary_path = RESULTS_DIR / "regression_summary.md"

    current_summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "rows": results_df.to_dict(orient="records"),
    }

    if save_gold:
        gold_path.write_text(json.dumps(current_summary, indent=2), encoding="utf-8")

    if not gold_path.exists():
        return

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_rows = pd.DataFrame(gold.get("rows", []))

    if gold_rows.empty:
        return

    join_cols = ["model", "constraint"]
    merged = pd.merge(
        gold_rows[join_cols + ["effective_accuracy"]],
        results_df[join_cols + ["effective_accuracy"]],
        on=join_cols,
        suffixes=("_gold", "_current"),
        how="outer",
    )
    merged["delta_vs_gold"] = merged["effective_accuracy_current"] - merged["effective_accuracy_gold"]

    regression_report = {
        "generated_at": datetime.utcnow().isoformat(),
        "comparisons": merged.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(regression_report, indent=2), encoding="utf-8")

    summary_lines = [
        "# Regression Summary",
        "",
        merged.to_markdown(index=False),
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def append_audit_log(
    dataset_name: str,
    profile_name: str,
    threshold: float,
    worst_acc: float,
    safety_triggered: bool,
) -> None:
    log_path = RESULTS_DIR / "audit.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{datetime.utcnow().isoformat()} | "
            f"dataset={dataset_name} | "
            f"profile={profile_name} | "
            f"threshold={threshold:.4f} | "
            f"worst_effective_accuracy={worst_acc:.4f} | "
            f"safety_triggered={safety_triggered}\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="RobustAI Engine audit runner")
    ap.add_argument("--csv", required=True, help="Input CSV path")
    ap.add_argument("--label", default="label", help="Target column")
    ap.add_argument("--profile", default="industrial_wifi", help="Profile name")
    ap.add_argument("--dataset-name", default="custom_dataset", help="Dataset label for reports")
    ap.add_argument("--model-name", default="RobustAI_Audit", help="Unused legacy arg kept for compatibility")
    ap.add_argument("--compare", action="store_true", help="Train multiple models side-by-side")
    ap.add_argument("--save-gold", action="store_true", help="Save this run as the gold standard")
    args = ap.parse_args()

    profile = load_profile(args.profile)

    df = pd.read_csv(args.csv)
    if args.label not in df.columns:
        raise KeyError(f"Label column '{args.label}' not found in {args.csv}")

    feature_cols = [c for c in df.columns if c != args.label]
    # Remove ID-ish columns from training if present
    removable = {"engine_id"}
    feature_cols = [c for c in feature_cols if c not in removable]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[args.label].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = fit_models(X_train_s, y_train) if args.compare else {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=42,
            early_stopping=True,
        ).fit(X_train_s, y_train)
    }

    result_rows = []
    sensitivity_rows = []
    envelope_rows = []

    for model_name, model in models.items():
        clean_acc = evaluate_model(model, X_test_s, y_test)
        result_rows.append(
            {
                "model": model_name,
                "profile": profile.name,
                "constraint": "clean",
                "clean_accuracy": clean_acc,
                "effective_accuracy": clean_acc,
                "delta": 0.0,
            }
        )

        # noise scenarios
        for noise_std in profile.noise_stds:
            if noise_std <= 0:
                continue
            X_mod = apply_noise(X_test_s, noise_std, seed=1)
            acc = evaluate_model(model, X_mod, y_test)
            result_rows.append(
                {
                    "model": model_name,
                    "profile": profile.name,
                    "constraint": "noise",
                    "clean_accuracy": clean_acc,
                    "effective_accuracy": acc,
                    "delta": clean_acc - acc,
                }
            )

        # latency + timeout/drop
        for delay_ms in profile.latency_delays_ms:
            if delay_ms <= 0:
                continue
            latency_info = simulate_latency(
                n_samples=len(X_test_s),
                delay_ms=delay_ms,
                timeout_ms=profile.timeout_ms,
                drop_rate=profile.drop_rate,
                seed=2,
            )
            X_mod = apply_latency_failure_rows(
                X_test_s,
                latency_info["dropped_mask"],
                latency_info["timed_out_mask"],
            )
            acc = evaluate_model(model, X_mod, y_test)
            result_rows.append(
                {
                    "model": model_name,
                    "profile": profile.name,
                    "constraint": "latency",
                    "clean_accuracy": clean_acc,
                    "effective_accuracy": acc,
                    "delta": clean_acc - acc,
                }
            )

            X_stale = apply_staleness(X_test_s, delay_ms, profile.drift_per_ms, seed=3)
            acc_stale = evaluate_model(model, X_stale, y_test)
            result_rows.append(
                {
                    "model": model_name,
                    "profile": profile.name,
                    "constraint": "staleness",
                    "clean_accuracy": clean_acc,
                    "effective_accuracy": acc_stale,
                    "delta": clean_acc - acc_stale,
                }
            )

            X_bias = apply_bias_drift(X_test_s, profile.bias_per_ms, delay_ms)
            acc_bias = evaluate_model(model, X_bias, y_test)
            result_rows.append(
                {
                    "model": model_name,
                    "profile": profile.name,
                    "constraint": "bias_drift",
                    "clean_accuracy": clean_acc,
                    "effective_accuracy": acc_bias,
                    "delta": clean_acc - acc_bias,
                }
            )

        # intermittent dropout
        X_drop = apply_intermittent_dropout(X_test_s, profile.intermittent_dropout_prob, seed=4)
        acc_drop = evaluate_model(model, X_drop, y_test)
        result_rows.append(
            {
                "model": model_name,
                "profile": profile.name,
                "constraint": "intermittent_dropout",
                "clean_accuracy": clean_acc,
                "effective_accuracy": acc_drop,
                "delta": clean_acc - acc_drop,
            }
        )

        # stuck-at-value
        X_stuck = apply_stuck_at_value(X_test_s, profile.stuck_prob, seed=5)
        acc_stuck = evaluate_model(model, X_stuck, y_test)
        result_rows.append(
            {
                "model": model_name,
                "profile": profile.name,
                "constraint": "stuck_at_value",
                "clean_accuracy": clean_acc,
                "effective_accuracy": acc_stuck,
                "delta": clean_acc - acc_stuck,
            }
        )

        # saturation
        X_sat = apply_sensor_saturation(
            X_test_s,
            per_feature=True,
            ref_X=X_train_s,
            clip_percentile=profile.saturation_percentile,
        )
        acc_sat = evaluate_model(model, X_sat, y_test)
        result_rows.append(
            {
                "model": model_name,
                "profile": profile.name,
                "constraint": "saturation",
                "clean_accuracy": clean_acc,
                "effective_accuracy": acc_sat,
                "delta": clean_acc - acc_sat,
            }
        )

        # quantization
        X_quant = apply_quantization(X_test_s, decimals=profile.quantization_decimals)
        acc_quant = evaluate_model(model, X_quant, y_test)
        result_rows.append(
            {
                "model": model_name,
                "profile": profile.name,
                "constraint": "quantization",
                "clean_accuracy": clean_acc,
                "effective_accuracy": acc_quant,
                "delta": clean_acc - acc_quant,
            }
        )

        # packet burst loss
        burst = simulate_packet_burst_loss(
            len(X_test_s),
            burst_ms=profile.burst_ms,
            sample_period_ms=profile.sample_period_ms,
            seed=6,
        )
        X_burst = np.asarray(X_test_s, dtype=float).copy()
        X_burst[burst["dropped_mask"], :] = 0.0
        acc_burst = evaluate_model(model, X_burst, y_test)
        result_rows.append(
            {
                "model": model_name,
                "profile": profile.name,
                "constraint": "packet_burst_loss",
                "clean_accuracy": clean_acc,
                "effective_accuracy": acc_burst,
                "delta": clean_acc - acc_burst,
            }
        )

        sensitivity_rows.append(
            feature_sensitivity(model_name, model, X_test_s, y_test, feature_cols, profile)
        )
        envelope_rows.append(
            make_operating_envelope(model_name, model, X_test_s, y_test, profile)
        )

    results_df = pd.DataFrame(result_rows)
    sensitivity_df = pd.concat(sensitivity_rows, ignore_index=True)
    envelope_df = pd.concat(envelope_rows, ignore_index=True)

    results_df.to_csv(RESULTS_DIR / "week06_product_demo_results.csv", index=False)
    sensitivity_df.to_csv(RESULTS_DIR / "feature_sensitivity.csv", index=False)
    envelope_df.to_csv(RESULTS_DIR / "week06_operating_envelope.csv", index=False)

    mitigations = build_mitigations(results_df)
    (RESULTS_DIR / "mitigations.md").write_text(
        "# Mitigation Recommendations\n\n" + "\n".join(mitigations) + "\n",
        encoding="utf-8",
    )

    worst_acc = float(results_df["effective_accuracy"].min())
    safety_triggered = worst_acc < profile.threshold

    write_markdown_report(
        profile=profile,
        dataset_name=args.dataset_name,
        results_df=results_df,
        safety_triggered=safety_triggered,
        threshold=profile.threshold,
    )
    save_regression_artifacts(results_df, save_gold=args.save_gold)
    append_audit_log(
        dataset_name=args.dataset_name,
        profile_name=profile.name,
        threshold=profile.threshold,
        worst_acc=worst_acc,
        safety_triggered=safety_triggered,
    )

    if safety_triggered:
        print("\n" + "!" * 60)
        print("SAFETY GATE WARNING: DEPLOYMENT RISK DETECTED")
        print(f"Worst effective accuracy: {worst_acc:.4f}")
        print(f"Threshold: {profile.threshold:.4f}")
        print("Review mitigations.md before deployment.")
        print("!" * 60 + "\n")
        raise SystemExit(2)

    print("Audit completed successfully.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()