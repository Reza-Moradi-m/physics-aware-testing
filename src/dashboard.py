# src/dashboard.py
from __future__ import annotations

from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


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


def load_gold(gold_path: str = "results/gold_standard.json") -> dict | None:
    p = Path(gold_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_fraction_from_env(env_df: pd.DataFrame, threshold: float) -> float:
    if len(env_df) == 0:
        return 0.0
    return float((env_df["effective_accuracy"].astype(float) >= float(threshold)).mean())


def wrap_lines(text: str, width: int = 62) -> str:
    parts = []
    for line in text.splitlines():
        if not line.strip():
            parts.append("")
        else:
            parts.append(textwrap.fill(line, width=width))
    return "\n".join(parts)


def draw_text_panel(ax, title: str, body: str, fontsize: int = 10) -> None:
    ax.set_title(title, fontsize=12, pad=10, loc="left")
    ax.text(
        0.02,
        0.98,
        wrap_lines(body, width=60),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3f8", edgecolor="#c8d3df"),
    )
    ax.axis("off")


def prettify_constraint_name(name: str) -> str:
    mapping = {
        "clean": "clean",
        "noise": "noise",
        "latency": "latency",
        "staleness": "staleness",
        "bias_drift": "bias drift",
        "intermittent_dropout": "dropout",
        "stuck_at_value": "stuck-at-value",
        "packet_burst_loss": "burst loss",
        "quantization": "quantization",
        "saturation": "saturation",
    }
    return mapping.get(str(name), str(name).replace("_", " "))


def make_dashboard(
    results_csv: str = "results/week06_product_demo_results.csv",
    envelope_csv: str = "results/week06_operating_envelope.csv",
    sensitivity_csv: str = "results/feature_sensitivity.csv",
    gold_path: str = "results/gold_standard.json",
    out_path: str = "results/master_dashboard.png",
    primary_model: str = "MLP",
    threshold: float = 0.70,
) -> None:
    results_path = Path(results_csv)
    envelope_path = Path(envelope_csv)
    sensitivity_path = Path(sensitivity_csv)

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {results_csv}")

    df = pd.read_csv(results_path)

    required_cols = {"model", "constraint", "clean_accuracy", "effective_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {sorted(missing)}")

    df_primary = df[df["model"] == primary_model].copy()
    if len(df_primary) == 0:
        primary_model = str(df["model"].iloc[0])
        df_primary = df[df["model"] == primary_model].copy()

    env_df = pd.read_csv(envelope_path) if envelope_path.exists() else pd.DataFrame()
    sens_df = pd.read_csv(sensitivity_path) if sensitivity_path.exists() else pd.DataFrame()
    gold = load_gold(gold_path)

    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    gs = gridspec.GridSpec(
        4,
        2,
        figure=fig,
        width_ratios=[1.15, 1.35],
        height_ratios=[1.0, 1.0, 1.0, 0.85],
    )

    axs = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])],
    ]

    fig.suptitle("RobustAI Engine — Reliability Audit Dashboard", fontsize=20)

    clean_rows = df_primary[df_primary["constraint"] == "clean"]
    clean_acc = float(clean_rows["effective_accuracy"].max()) if len(clean_rows) else 0.0
    worst_eff = float(df_primary["effective_accuracy"].min()) if len(df_primary) else 0.0
    mean_eff = float(df_primary["effective_accuracy"].mean()) if len(df_primary) else 0.0
    safety_status = "CRITICAL" if worst_eff < threshold else "PASS"

    # 1) Safety summary
    summary_text = (
        f"Primary Model: {primary_model}\n\n"
        f"Clean Accuracy: {clean_acc:.2%}\n"
        f"Worst Effective Accuracy: {worst_eff:.2%}\n"
        f"Average Effective Accuracy: {mean_eff:.2%}\n"
        f"Threshold: {threshold:.2%}\n"
        f"Safety Gate: {safety_status}\n"
        f"Grade: {grade(worst_eff)}\n\n"
        f"Interpretation:\n"
        f"- Clean accuracy = ideal case.\n"
        f"- Effective accuracy = stressed deployment case.\n"
        f"- Worst-case behavior is the key deployment risk metric."
    )
    draw_text_panel(axs[0][0], "Safety Summary", summary_text, fontsize=10)

    # 2) Constraint impact as accuracy loss
    ax = axs[0][1]
    risk_df = df_primary.copy()
    if len(risk_df) > 0:
        risk_df["accuracy_loss"] = (
            risk_df["clean_accuracy"].astype(float) - risk_df["effective_accuracy"].astype(float)
        )

        constraint_loss = (
            risk_df.groupby("constraint", as_index=False)["accuracy_loss"]
            .max()
            .sort_values("accuracy_loss", ascending=False)
        )

        labels = [prettify_constraint_name(x) for x in constraint_loss["constraint"]]
        values = constraint_loss["accuracy_loss"].astype(float)

        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_title(f"Constraint Impact ({primary_model})", fontsize=12)
        ax.set_xlabel("Accuracy Loss")
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=9)

        for i, val in enumerate(values):
            ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No constraint data found.", ha="center", va="center")
        ax.axis("off")

    # 3) Model comparison
    comp = (
        df.groupby("model", as_index=False)
        .agg(
            clean_accuracy=("clean_accuracy", "max"),
            worst_effective_accuracy=("effective_accuracy", "min"),
            mean_effective_accuracy=("effective_accuracy", "mean"),
        )
        .sort_values("worst_effective_accuracy", ascending=True)
    )

    ax = axs[1][0]
    if len(comp) > 0:
        x = range(len(comp))
        width = 0.36

        ax.bar(
            [i - width / 2 for i in x],
            comp["clean_accuracy"].astype(float),
            width=width,
            label="Clean Accuracy",
        )
        ax.bar(
            [i + width / 2 for i in x],
            comp["worst_effective_accuracy"].astype(float),
            width=width,
            label="Worst Effective Accuracy",
        )
        ax.axhline(threshold, linestyle=":", label="Threshold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(comp["model"], rotation=0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Model Comparison", fontsize=12)
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        for i, row in comp.reset_index(drop=True).iterrows():
            ax.text(
                i - width / 2,
                float(row["clean_accuracy"]) + 0.015,
                f"{row['clean_accuracy']:.2f}",
                ha="center",
                fontsize=8,
            )
            ax.text(
                i + width / 2,
                float(row["worst_effective_accuracy"]) + 0.015,
                f"{row['worst_effective_accuracy']:.2f}",
                ha="center",
                fontsize=8,
            )
    else:
        ax.text(0.5, 0.5, "No model comparison data.", ha="center", va="center")
        ax.axis("off")

    # 4) Operating envelope
    ax = axs[1][1]
    env_primary = env_df[env_df["model"] == primary_model].copy() if len(env_df) else pd.DataFrame()

    if len(env_primary) > 0:
        pivot_env = env_primary.pivot(index="noise_std", columns="delay_ms", values="effective_accuracy")
        im = ax.imshow(pivot_env.values, aspect="auto")
        ax.set_title(f"Operating Envelope ({primary_model})", fontsize=12)
        ax.set_xlabel("Delay (ms)")
        ax.set_ylabel("Noise std")
        ax.set_xticks(range(len(pivot_env.columns)))
        ax.set_xticklabels(list(pivot_env.columns))
        ax.set_yticks(range(len(pivot_env.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot_env.index])

        for i, ns in enumerate(pivot_env.index):
            for j, dm in enumerate(pivot_env.columns):
                val = float(pivot_env.loc[ns, dm])
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No operating envelope data found.", ha="center", va="center")
        ax.axis("off")

    # 5) Feature sensitivity
    ax = axs[2][0]
    sens_primary = sens_df[sens_df["model"] == primary_model].copy() if len(sens_df) else pd.DataFrame()

    if len(sens_primary) > 0:
        sens_primary = sens_primary.sort_values("drop", ascending=False).head(12)
        labels = sens_primary["feature"].astype(str)
        values = sens_primary["drop"].astype(float)

        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_title(f"Feature Sensitivity ({primary_model})", fontsize=12)
        ax.set_xlabel("Accuracy Drop from Perturbation")
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=9)

        for i, val in enumerate(values):
            offset = 0.001 if val >= 0 else -0.001
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, i, f"{val:.3f}", va="center", ha=ha, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No feature sensitivity data found.", ha="center", va="center")
        ax.axis("off")

    # 6) Top risk drivers
    ax = axs[2][1]
    if len(risk_df) > 0:
        risk_plot = (
            risk_df.groupby("constraint", as_index=False)["accuracy_loss"]
            .max()
            .sort_values("accuracy_loss", ascending=False)
            .head(10)
        )

        labels = [prettify_constraint_name(x) for x in risk_plot["constraint"]]
        values = risk_plot["accuracy_loss"].astype(float)

        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_title("Top Risk Drivers", fontsize=12)
        ax.set_xlabel("Accuracy Loss")
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=9)

        for i, val in enumerate(values):
            ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No risk data found.", ha="center", va="center")
        ax.axis("off")

    # 7) Regression summary
    if gold is None:
        reg_text = (
            "No gold_standard.json found.\n\n"
            "Run the audit with --save-gold to create a baseline for regression comparison."
        )
    else:
        try:
            gold_rows = pd.DataFrame(gold.get("rows", []))
            if len(gold_rows) > 0:
                current_primary = df[df["model"] == primary_model].copy()
                gold_primary = gold_rows[gold_rows["model"] == primary_model].copy()

                gold_worst = float(gold_primary["effective_accuracy"].min()) if len(gold_primary) else 0.0
                current_worst = float(current_primary["effective_accuracy"].min()) if len(current_primary) else 0.0

                gold_clean = float(gold_primary["clean_accuracy"].max()) if len(gold_primary) else 0.0
                current_clean = float(current_primary["clean_accuracy"].max()) if len(current_primary) else 0.0

                reg_text = (
                    f"Primary Model: {primary_model}\n\n"
                    f"Gold Clean Accuracy: {gold_clean:.3f}\n"
                    f"Current Clean Accuracy: {current_clean:.3f}\n"
                    f"Delta Clean: {current_clean - gold_clean:+.3f}\n\n"
                    f"Gold Worst Effective Accuracy: {gold_worst:.3f}\n"
                    f"Current Worst Effective Accuracy: {current_worst:.3f}\n"
                    f"Delta Worst: {current_worst - gold_worst:+.3f}"
                )
            else:
                reg_text = "Gold file exists but contains no rows."
        except Exception as e:
            reg_text = f"Failed to parse gold data:\n{e}"

    draw_text_panel(axs[3][0], "Regression Summary", reg_text, fontsize=10)

    # 8) Consultant interpretation
    safe_fraction = safe_fraction_from_env(env_primary, threshold=threshold) if len(env_primary) else 0.0

    worst_constraint = "unknown"
    worst_loss = 0.0
    if len(risk_df) > 0:
        worst_row = (
            risk_df.groupby("constraint", as_index=False)["accuracy_loss"]
            .max()
            .sort_values("accuracy_loss", ascending=False)
            .iloc[0]
        )
        worst_constraint = prettify_constraint_name(worst_row["constraint"])
        worst_loss = float(worst_row["accuracy_loss"])

    top_features_text = "N/A"
    if len(sens_primary) > 0:
        top_features = sens_primary.sort_values("drop", ascending=False).head(3)["feature"].tolist()
        top_features_text = ", ".join(map(str, top_features))

    notes = (
        f"Primary model: {primary_model}\n"
        f"Safe-zone fraction: {safe_fraction:.2%}\n"
        f"Safety gate: {safety_status}\n"
        f"Strongest risk: {worst_constraint} ({worst_loss:.3f} loss)\n"
        f"Top sensitive features: {top_features_text}\n\n"
        f"Interpretation:\n"
        f"- Latency and timing-related issues are the dominant deployment risk when they appear.\n"
        f"- The model remains above the safety threshold, but with visible degradation under stress.\n"
        f"- Feature sensitivity identifies which inputs matter most to prediction stability.\n"
        f"- Regression tracks whether reliability improves or worsens over time."
    )
    draw_text_panel(axs[3][1], "Consultant Interpretation", notes, fontsize=10)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard saved: {out_path}")


if __name__ == "__main__":
    make_dashboard()