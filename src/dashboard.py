# src/dashboard.py
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def parse_first_number(text: str) -> float:
    m = re.search(r"[-+]?\d*\.\d+|\d+", text)
    return float(m.group()) if m else 0.0


def make_dashboard(
    results_csv: str = "results/week06_product_demo_results.csv",
    envelope_csv: str = "results/week06_operating_envelope.csv",
    out_path: str = "results/master_dashboard.png",
) -> None:
    df = pd.read_csv(results_csv)
    env = pd.read_csv(envelope_csv)

    Path("results").mkdir(exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("AI Reliability & Robustness Certificate (Smart Factory Model)", fontsize=18, y=0.98)

    # --- Safety scorecard ---
    baseline_acc = float(df[df["test_name"] == "baseline"]["raw_accuracy"].iloc[0])
    worst_eff = float(df["effective_accuracy"].min())

    def grade(x: float) -> str:
        if x >= 0.90: return "A"
        if x >= 0.80: return "B"
        if x >= 0.70: return "C"
        if x >= 0.60: return "D"
        return "F"

    score_text = (
        f"Safety Scorecard\n\n"
        f"Baseline Accuracy: {baseline_acc:.2%}\n"
        f"Worst Effective Acc: {worst_eff:.2%}\n"
        f"Grade: {grade(worst_eff)}\n\n"
        f"Interpretation:\n"
        f"- Raw accuracy is model correctness.\n"
        f"- Effective accuracy includes missed deadlines.\n"
    )
    axs[0, 0].text(0.5, 0.5, score_text, ha="center", va="center", fontsize=14,
                  bbox=dict(boxstyle="round,pad=0.8", alpha=0.2))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Safety Summary")

    # --- Noise curve ---
    noise_df = df[df["test_name"] == "noise"].copy()
    noise_df["std"] = noise_df["param"].apply(parse_first_number)
    noise_df = noise_df.sort_values("std")

    axs[0, 1].plot(noise_df["std"], noise_df["raw_accuracy"], marker="o")
    axs[0, 1].set_title("Noise Tolerance")
    axs[0, 1].set_xlabel("Noise std (Gaussian)")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].grid(True)

    # --- Latency effective curve ---
    lat_df = df[df["test_name"] == "latency_effects"].copy()
    lat_df["delay"] = lat_df["param"].apply(parse_first_number)
    lat_df = lat_df.sort_values("delay")

    axs[1, 0].plot(lat_df["delay"], lat_df["effective_accuracy"], marker="o")
    axs[1, 0].set_title("Latency vs Effective Accuracy")
    axs[1, 0].set_xlabel("Delay (ms)")
    axs[1, 0].set_ylabel("Effective Accuracy")
    axs[1, 0].grid(True)

    # --- Operating envelope heatmap (Noise vs Delay, PASS/FAIL) ---
    # Pivot to grid
    pivot = env.pivot(index="noise_std", columns="delay_ms", values="effective_accuracy")
    axs[1, 1].imshow(pivot.values, aspect="auto")
    axs[1, 1].set_title("Operating Envelope (Noise vs Latency)")
    axs[1, 1].set_xlabel("Delay (ms)")
    axs[1, 1].set_ylabel("Noise std")

    # tick labels
    axs[1, 1].set_xticks(range(len(pivot.columns)))
    axs[1, 1].set_xticklabels(list(pivot.columns))
    axs[1, 1].set_yticks(range(len(pivot.index)))
    axs[1, 1].set_yticklabels(list(pivot.index))

    # annotate values
    for i, ns in enumerate(pivot.index):
        for j, dm in enumerate(pivot.columns):
            val = pivot.loc[ns, dm]
            axs[1, 1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    print(f"✅ Dashboard saved: {out_path}")


if __name__ == "__main__":
    make_dashboard()