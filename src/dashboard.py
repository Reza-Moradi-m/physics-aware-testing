# src/dashboard.py
from __future__ import annotations

from pathlib import Path
import re
import json

import pandas as pd
import matplotlib.pyplot as plt


def parse_first_number(text: str) -> float:
    m = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
    return float(m.group()) if m else 0.0


def grade(x: float) -> str:
    if x >= 0.90:
        return "A"
    if x >= 0.80:
        return "B"
    if x >= 0.70:
        return "C"
    if x >= 0.60:
        return "D"
    return "F"


def extract_baseline_drop(note: str) -> float:
    """
    Reads notes like: baseline_drop=0.012345
    """
    m = re.search(r"baseline_drop=([-+]?\d*\.\d+|\d+)", str(note))
    return float(m.group(1)) if m else 0.0


def clean_failure_label(param: str) -> str:
    text = str(param)
    replacements = {
        "intermittent_dropout=0.03": "intermittent_dropout",
        "stuck_at_value=0.03": "stuck_at_value",
        "sensor_saturation(per_feature,p99.5)": "sensor_saturation",
        "quantization(decimals=1)": "quantization",
        "packet_burst_loss(500ms)": "packet_burst_loss",
        "bias_drift(bias_per_ms=0.0008,delay=200)": "bias_drift",
    }
    return replacements.get(text, text)


def safe_fraction_from_env(env_df: pd.DataFrame, threshold: float | None = None) -> float:
    """
    Computes fraction of grid that is "safe".
    Prefers pass_fail column if present; else uses effective_accuracy >= threshold.
    """
    if "pass_fail" in env_df.columns:
        return float((env_df["pass_fail"].astype(str).str.upper() == "PASS").mean())

    if threshold is None:
        threshold = 0.75
    return float((env_df["effective_accuracy"].astype(float) >= float(threshold)).mean())


def load_gold(gold_path: str = "results/gold_standard.json") -> dict | None:
    p = Path(gold_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_dashboard(
    results_csv: str = "results/week06_product_demo_results.csv",
    envelope_csv: str = "results/week06_operating_envelope.csv",
    gold_path: str = "results/gold_standard.json",
    out_path: str = "results/master_dashboard.png",
    primary_model: str = "PyTorch MLP",
) -> None:
    df = pd.read_csv(results_csv)

    # New schema uses "model" column. If missing, treat all rows as primary model.
    if "model" in df.columns:
        df_primary = df[df["model"] == primary_model].copy()
    else:
        df_primary = df.copy()
        df_primary["model"] = primary_model

    env_v2 = pd.read_csv(envelope_csv)

    Path("results").mkdir(exist_ok=True)

    # 5 rows x 2 cols (extra row for regression + V1 envelope)
    fig, axs = plt.subplots(5, 2, figsize=(18, 24))
    fig.suptitle("AI Reliability & Robustness Certificate (Smart Factory Model)", fontsize=18, y=0.985)

    # =========================================================
    # 1) Safety scorecard (PRIMARY MODEL ONLY)
    # =========================================================
    baseline_rows = df_primary[df_primary["test_name"] == "baseline"]
    if len(baseline_rows) == 0:
        baseline_acc = 0.0
        baseline_note = "Baseline row not found."
    else:
        baseline_acc = float(baseline_rows.iloc[0]["raw_accuracy"])
        baseline_note = str(baseline_rows.iloc[0].get("notes", ""))

    worst_eff = float(df_primary["effective_accuracy"].astype(float).min()) if len(df_primary) else 0.0

    score_text = (
        f"Safety Scorecard ({primary_model})\n\n"
        f"Baseline Accuracy: {baseline_acc:.2%}\n"
        f"Worst Effective Acc: {worst_eff:.2%}\n"
        f"Grade: {grade(worst_eff)}\n\n"
        f"Interpretation:\n"
        f"- Raw accuracy = model correctness\n"
        f"- Effective accuracy = correctness after deployment penalties\n"
        f"- Lower effective accuracy = higher real-world risk\n\n"
        f"Model Note:\n{baseline_note}"
    )

    axs[0, 0].text(
        0.5,
        0.5,
        score_text,
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.8", alpha=0.2),
    )
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Safety Summary")

    # =========================================================
    # 2) Noise tolerance (PRIMARY MODEL ONLY)
    # =========================================================
    noise_df = df_primary[df_primary["test_name"] == "noise"].copy()
    if len(noise_df) > 0:
        noise_df["std"] = noise_df["param"].apply(parse_first_number)
        noise_df = noise_df.sort_values("std")
        axs[0, 1].plot(noise_df["std"], noise_df["raw_accuracy"].astype(float), marker="o")
        axs[0, 1].set_title("Noise Tolerance")
        axs[0, 1].set_xlabel("Noise std (Gaussian)")
        axs[0, 1].set_ylabel("Accuracy")
        axs[0, 1].grid(True)
    else:
        axs[0, 1].text(0.5, 0.5, "No noise rows found.", ha="center", va="center")
        axs[0, 1].axis("off")

    # =========================================================
    # 3) Latency vs effective accuracy (PRIMARY MODEL ONLY)
    # =========================================================
    lat_df = df_primary[df_primary["test_name"] == "latency_effects"].copy()
    if len(lat_df) > 0:
        lat_df["delay"] = lat_df["param"].apply(parse_first_number)
        lat_df = lat_df.sort_values("delay")
        axs[1, 0].plot(lat_df["delay"], lat_df["effective_accuracy"].astype(float), marker="o")
        axs[1, 0].set_title("Latency vs Effective Accuracy")
        axs[1, 0].set_xlabel("Delay (ms)")
        axs[1, 0].set_ylabel("Effective Accuracy")
        axs[1, 0].grid(True)
    else:
        axs[1, 0].text(0.5, 0.5, "No latency rows found.", ha="center", va="center")
        axs[1, 0].axis("off")

    # =========================================================
    # 4) V2 Operating envelope heatmap (CURRENT)
    # =========================================================
    if len(env_v2) > 0:
        pivot_v2 = env_v2.pivot(index="noise_std", columns="delay_ms", values="effective_accuracy")
        axs[1, 1].imshow(pivot_v2.values, aspect="auto")
        axs[1, 1].set_title("Operating Envelope V2 (Current) — Noise vs Latency")
        axs[1, 1].set_xlabel("Delay (ms)")
        axs[1, 1].set_ylabel("Noise std")

        axs[1, 1].set_xticks(range(len(pivot_v2.columns)))
        axs[1, 1].set_xticklabels(list(pivot_v2.columns))
        axs[1, 1].set_yticks(range(len(pivot_v2.index)))
        axs[1, 1].set_yticklabels(list(pivot_v2.index))

        for i, ns in enumerate(pivot_v2.index):
            for j, dm in enumerate(pivot_v2.columns):
                val = float(pivot_v2.loc[ns, dm])
                axs[1, 1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    else:
        axs[1, 1].text(0.5, 0.5, "No envelope CSV found / empty.", ha="center", va="center")
        axs[1, 1].axis("off")

    # =========================================================
    # 5) Failure modes impact (PRIMARY MODEL ONLY)
    # =========================================================
    fm = df_primary[df_primary["test_name"] == "failure_mode"].copy()
    if len(fm) > 0:
        fm["label"] = fm["param"].apply(clean_failure_label)
        fm["eff"] = fm["effective_accuracy"].astype(float)

        fm_plot = fm.groupby("label", as_index=False)["eff"].min()
        fm_plot = fm_plot.sort_values("eff", ascending=True)

        axs[2, 0].barh(fm_plot["label"], fm_plot["eff"])
        axs[2, 0].axvline(baseline_acc, linestyle="--")
        axs[2, 0].set_xlim(0.0, 1.0)
        axs[2, 0].set_title("Failure Modes Impact")
        axs[2, 0].set_xlabel("Effective Accuracy (lower = more dangerous)")
        axs[2, 0].grid(True, axis="x")

        for i, row in fm_plot.reset_index(drop=True).iterrows():
            axs[2, 0].text(float(row["eff"]) + 0.01, i, f"{row['eff']:.2f}", va="center", fontsize=9)
    else:
        axs[2, 0].text(0.5, 0.5, "No failure mode rows found.", ha="center", va="center")
        axs[2, 0].axis("off")

    # =========================================================
    # 6) Feature sensitivity profiling (PRIMARY MODEL ONLY)
    # =========================================================
    fs = df_primary[df_primary["test_name"] == "feature_sensitivity"].copy()
    if len(fs) > 0:
        fs["feature"] = fs["param"].astype(str)
        fs["baseline_drop"] = fs["notes"].apply(extract_baseline_drop)
        fs = fs.sort_values("baseline_drop", ascending=False)

        axs[2, 1].barh(fs["feature"], fs["baseline_drop"])
        axs[2, 1].invert_yaxis()
        axs[2, 1].set_title("Feature Sensitivity (Accuracy Drop)")
        axs[2, 1].set_xlabel("Accuracy Drop from Baseline")
        axs[2, 1].grid(True, axis="x")

        for i, row in fs.reset_index(drop=True).iterrows():
            axs[2, 1].text(float(row["baseline_drop"]) + 0.002, i, f"{row['baseline_drop']:.3f}", va="center", fontsize=9)
    else:
        axs[2, 1].text(0.5, 0.5, "No feature sensitivity rows found.", ha="center", va="center")
        axs[2, 1].axis("off")

    # =========================================================
    # 7) Operational model comparison (from baseline + latency_effects@200ms)
    # =========================================================
    if "model" in df.columns:
        base_all = df[df["test_name"] == "baseline"][["model", "raw_accuracy"]].copy()
        lat_all = df[(df["test_name"] == "latency_effects") & (df["param"].astype(str).str.contains("delay_ms=200"))][
            ["model", "effective_accuracy"]
        ].copy()

        if len(base_all) > 0 and len(lat_all) > 0:
            comp = pd.merge(base_all, lat_all, on="model", how="inner")
            comp["raw_accuracy"] = comp["raw_accuracy"].astype(float)
            comp["effective_accuracy"] = comp["effective_accuracy"].astype(float)

            x = range(len(comp))
            width = 0.35

            axs[3, 0].bar([i - width / 2 for i in x], comp["raw_accuracy"], width=width, label="Baseline Acc")
            axs[3, 0].bar([i + width / 2 for i in x], comp["effective_accuracy"], width=width, label="Eff Acc @200ms")

            axs[3, 0].set_xticks(list(x))
            axs[3, 0].set_xticklabels(comp["model"])
            axs[3, 0].set_ylim(0.0, 1.05)
            axs[3, 0].set_title("Operational Model Comparison")
            axs[3, 0].set_ylabel("Accuracy")
            axs[3, 0].legend()
            axs[3, 0].grid(True, axis="y")
        else:
            axs[3, 0].text(0.5, 0.5, "Missing baseline or delay_ms=200 latency rows.", ha="center", va="center")
            axs[3, 0].axis("off")
    else:
        axs[3, 0].text(0.5, 0.5, "No model column in CSV (old schema).", ha="center", va="center")
        axs[3, 0].axis("off")

    # =========================================================
    # 8) Notes / consultant interpretation
    # =========================================================
    notes = (
        "Deployment Interpretation:\n\n"
        "- Noise chart shows robustness to sensor corruption.\n"
        "- Latency chart shows where real-time deadlines start failing.\n"
        "- Operating envelope visualizes safe vs unsafe zones.\n"
        "- Failure modes show realistic breakdown risks.\n"
        "- Feature sensitivity reveals the most critical hardware signal.\n"
        "- Model comparison shows whether a simpler model deploys more reliably."
    )
    axs[3, 1].text(0.03, 0.95, notes, va="top", fontsize=12)
    axs[3, 1].axis("off")
    axs[3, 1].set_title("Consultant View")

    # =========================================================
    # 9) Regression summary (V1 vs V2) + V1 envelope heatmap
    # =========================================================
    gold = load_gold(gold_path)

    if gold is None:
        axs[4, 0].text(0.5, 0.5, "Regression (V1 vs V2): No gold_standard.json found.\nRun: python3 -m src.week06_runner ... --save-gold", ha="center", va="center")
        axs[4, 0].axis("off")

        axs[4, 1].text(0.5, 0.5, "V1 envelope not available.", ha="center", va="center")
        axs[4, 1].axis("off")
    else:
        # Pull V1 info
        v1_env_path = gold.get("envelope_csv_path") or gold.get("envelope_csv") or gold.get("envelope_path")
        v1_threshold = gold.get("threshold", None)
        v1_baseline = float(gold.get("baseline_acc", 0.0))
        v1_worst = float(gold.get("worst_eff", 0.0))

        # Compute V2 info (from current primary model + current env)
        v2_baseline = float(baseline_acc)
        v2_worst = float(worst_eff)

        v2_safe = safe_fraction_from_env(env_v2, threshold=None)
        v1_safe = None
        env_v1 = None

        if v1_env_path and Path(str(v1_env_path)).exists():
            env_v1 = pd.read_csv(str(v1_env_path))
            v1_safe = safe_fraction_from_env(env_v1, threshold=float(v1_threshold) if v1_threshold is not None else None)

        # Regression summary panel
        if v1_safe is None:
            reg_text = (
                "Regression (V1 vs V2)\n\n"
                f"Gold file found: {gold_path}\n"
                "But V1 envelope CSV path is missing or not found.\n\n"
                "Tip: ensure gold_standard.json stores the envelope CSV path."
            )
            axs[4, 0].text(0.03, 0.95, reg_text, va="top", fontsize=12)
            axs[4, 0].axis("off")
        else:
            delta_safe = (v2_safe - v1_safe)
            delta_base = (v2_baseline - v1_baseline)
            delta_worst = (v2_worst - v1_worst)

            reg_text = (
                "Regression (V1 vs V2)\n\n"
                f"V1 (Gold) baseline acc: {v1_baseline:.3f}\n"
                f"V2 (Current) baseline acc: {v2_baseline:.3f}   (Δ {delta_base:+.3f})\n\n"
                f"V1 (Gold) worst eff acc: {v1_worst:.3f}\n"
                f"V2 (Current) worst eff acc: {v2_worst:.3f}   (Δ {delta_worst:+.3f})\n\n"
                f"Safe-zone fraction (PASS area):\n"
                f"V1: {v1_safe:.2%}\n"
                f"V2: {v2_safe:.2%}   (Δ {delta_safe:+.2%})\n\n"
                "Interpretation:\n"
                "- If V2 safe-zone shrinks, the new model is less tolerant to constraints.\n"
                "- If V2 safe-zone grows, robustness improved."
            )
            axs[4, 0].text(0.03, 0.95, reg_text, va="top", fontsize=12)
            axs[4, 0].axis("off")

        # V1 envelope heatmap panel
        if env_v1 is not None and len(env_v1) > 0:
            pivot_v1 = env_v1.pivot(index="noise_std", columns="delay_ms", values="effective_accuracy")
            axs[4, 1].imshow(pivot_v1.values, aspect="auto")
            axs[4, 1].set_title("Operating Envelope V1 (Gold) — Noise vs Latency")
            axs[4, 1].set_xlabel("Delay (ms)")
            axs[4, 1].set_ylabel("Noise std")

            axs[4, 1].set_xticks(range(len(pivot_v1.columns)))
            axs[4, 1].set_xticklabels(list(pivot_v1.columns))
            axs[4, 1].set_yticks(range(len(pivot_v1.index)))
            axs[4, 1].set_yticklabels(list(pivot_v1.index))

            for i, ns in enumerate(pivot_v1.index):
                for j, dm in enumerate(pivot_v1.columns):
                    val = float(pivot_v1.loc[ns, dm])
                    axs[4, 1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        else:
            axs[4, 1].text(0.5, 0.5, "V1 envelope not found/readable.", ha="center", va="center")
            axs[4, 1].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    print(f"✅ Dashboard saved: {out_path}")


if __name__ == "__main__":
    make_dashboard()