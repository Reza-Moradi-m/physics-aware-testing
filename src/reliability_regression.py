# src/reliability_regression.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RegressionResult:
    baseline_accuracy_delta: float
    worst_eff_delta: float
    safe_zone_delta: float  # + means bigger safe zone, - means smaller
    notes: str


def _safe_zone_fraction(env_df: pd.DataFrame, threshold: float) -> float:
    # env_df columns: noise_std, delay_ms, effective_accuracy, pass_fail
    passed = (env_df["effective_accuracy"].astype(float) >= float(threshold)).mean()
    return float(passed)


def save_gold_standard(
    *,
    out_path: str,
    profile_name: str,
    threshold: float,
    baseline_acc: float,
    worst_eff: float,
    envelope_csv_path: str,
) -> None:
    env = pd.read_csv(envelope_csv_path)
    safe_frac = _safe_zone_fraction(env, threshold)

    payload = {
        "profile": profile_name,
        "threshold": float(threshold),
        "baseline_accuracy": float(baseline_acc),
        "worst_effective_accuracy": float(worst_eff),
        "safe_zone_fraction": float(safe_frac),
        "envelope_csv": str(envelope_csv_path),
    }

    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compare_to_gold_standard(
    *,
    gold_path: str,
    profile_name: str,
    threshold: float,
    baseline_acc: float,
    worst_eff: float,
    envelope_csv_path: str,
) -> RegressionResult | None:
    p = Path(gold_path)
    if not p.exists():
        return None  # no regression comparison yet

    gold: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))

    # Compute current safe zone
    env = pd.read_csv(envelope_csv_path)
    safe_frac_now = _safe_zone_fraction(env, threshold)

    baseline_delta = float(baseline_acc) - float(gold.get("baseline_accuracy", 0.0))
    worst_eff_delta = float(worst_eff) - float(gold.get("worst_effective_accuracy", 0.0))
    safe_zone_delta = float(safe_frac_now) - float(gold.get("safe_zone_fraction", 0.0))

    # Notes: explain tradeoff clearly
    notes = []
    notes.append(f"Gold profile: {gold.get('profile')}  | Current profile: {profile_name}")
    notes.append(f"Baseline accuracy change: {baseline_delta:+.3f}")
    notes.append(f"Worst effective accuracy change: {worst_eff_delta:+.3f}")
    notes.append(f"Safe operating zone change: {safe_zone_delta:+.3f} (fraction of envelope cells passing)")

    # simple alert heuristic
    if safe_zone_delta < -0.10 and baseline_delta > 0:
        notes.append("⚠️ Regression warning: model improved baseline accuracy but reduced operating envelope noticeably.")
    elif safe_zone_delta < -0.10:
        notes.append("⚠️ Regression warning: operating envelope shrank noticeably.")
    elif safe_zone_delta > 0.05:
        notes.append("✅ Improvement: operating envelope expanded.")
    else:
        notes.append("No major envelope change detected.")

    return RegressionResult(
        baseline_accuracy_delta=baseline_delta,
        worst_eff_delta=worst_eff_delta,
        safe_zone_delta=safe_zone_delta,
        notes="\n".join(notes),
    )


def write_regression_artifacts(
    *,
    results_dir: str,
    regression: RegressionResult | None,
) -> tuple[str | None, str | None]:
    results = Path(results_dir)
    results.mkdir(exist_ok=True)

    if regression is None:
        return None, None

    json_path = results / "regression_report.json"
    md_path = results / "regression_summary.md"

    json_path.write_text(
        json.dumps(
            {
                "baseline_accuracy_delta": regression.baseline_accuracy_delta,
                "worst_effective_accuracy_delta": regression.worst_eff_delta,
                "safe_zone_delta": regression.safe_zone_delta,
                "notes": regression.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path.write_text(
        "\n".join(
            [
                "## Regression Check (V1 vs V2)",
                "",
                regression.notes,
                "",
            ]
        ),
        encoding="utf-8",
    )

    return str(json_path), str(md_path)


def prescriptive_mitigations(
    *,
    df_results: pd.DataFrame,
    threshold: float,
    profile_name: str,
) -> str:
    """
    Uses:
      - latency_effects curve to find first failing delay
      - feature_sensitivity to find the biggest drop feature
    Produces: Markdown bullet list.
    """
    lines: list[str] = []
    lines.append("## Required Mitigations (Fix-it Engine)\n")

    # 1) latency-based: find first delay where eff < threshold
    lat = df_results[df_results["test_name"] == "latency_effects"].copy()
    if len(lat) > 0:
        lat["delay"] = lat["param"].astype(str).str.extract(r"(\d+)").astype(float)
        lat = lat.sort_values("delay")
        failing = lat[lat["effective_accuracy"].astype(float) < float(threshold)]
        if len(failing) > 0:
            first_fail = failing.iloc[0]
            fail_delay = int(first_fail["delay"])
            fail_eff = float(first_fail["effective_accuracy"])

            # simple buffer recommendation: about (fail_delay - timeout) doesn’t make sense,
            # so we recommend a buffer based on approaching the fail point
            buffer_ms = 50 if fail_delay >= 150 else 20
            lines.append(f"- **Option A (Software):** Add a **{buffer_ms}ms prediction buffer** (queue/pipeline) to tolerate real-time jitter.")
            lines.append(f"  - Reason: effective accuracy fell below threshold at ~{fail_delay}ms (eff={fail_eff:.2%}, threshold={threshold:.0%}).")

    # 2) feature sensitivity: biggest baseline drop feature
    sens = df_results[df_results["test_name"] == "feature_sensitivity"].copy()
    if len(sens) > 0:
        # notes contains baseline_drop=...
        sens["drop"] = sens["notes"].astype(str).str.extract(r"baseline_drop=([0-9.\-eE]+)").astype(float)
        worst = sens.sort_values("drop", ascending=False).iloc[0]
        feat = str(worst["param"])
        drop = float(worst["drop"])

        # Translate into a “reduce noise by X sigma” suggestion (rough)
        target_sigma = 0.2 if drop > 0.05 else 0.1
        lines.append(f"- **Option B (Hardware):** Improve **{feat}** sensor stability (reduce noise by ~{target_sigma}σ).")
        lines.append(f"  - Reason: {feat} had the largest accuracy drop during single-feature noise injection (Δ={drop:.3f}).")

    if len(lines) == 1:
        lines.append("- No prescriptive recommendations generated (missing latency_effects or feature_sensitivity rows).")

    lines.append(f"\n(Profile used: `{profile_name}`)\n")
    return "\n".join(lines)