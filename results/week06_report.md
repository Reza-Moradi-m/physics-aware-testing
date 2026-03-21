# Week 06 – Smart Factory Model Reliability Report (Startup Demo)

Generated: 2026-03-11T14:55:06

## Summary

Model: PyTorch MLP (train_acc=0.8807, test_acc=0.8817)

Safety Grade: **F** (worst effective accuracy=0.0000)

## Positives

- Profile: **industrial_wifi** (threshold=0.75, timeout_ms=120, drop_rate=0.05)
- PyTorch MLP baseline accuracy: **0.8817**
- Logistic Regression baseline accuracy: **0.9967**
- Includes explicit combined-constraints tests (noise + latency + staleness).
- Produced stress tests, operating envelope, regression artifacts, and prescriptive mitigations.

## Issues / Concerns

- Worst effective accuracy observed (global): **0.0000** (Grade **F**)
- Late/missing predictions are treated as failures.
- Some latency conditions can collapse effective accuracy.

## Operating Envelope

PASS/FAIL grid saved in `results/week06_operating_envelope.csv`.

## Regression Check (V1 vs V2)

Gold profile: industrial_wifi  | Current profile: industrial_wifi
Baseline accuracy change: +0.000
Worst effective accuracy change: +0.000
Safe operating zone change: +0.000 (fraction of envelope cells passing)
No major envelope change detected.

## Required Mitigations (Fix-it Engine)

## Required Mitigations (Fix-it Engine)

- **Option A (Software):** Add a **50ms prediction buffer** (queue/pipeline) to tolerate real-time jitter.
  - Reason: effective accuracy fell below threshold at ~150ms (eff=0.00%, threshold=75%).
- **Option B (Hardware):** Improve **temp_c** sensor stability (reduce noise by ~0.1σ).
  - Reason: temp_c had the largest accuracy drop during single-feature noise injection (Δ=0.000).

(Profile used: `industrial_wifi`)

## Artifacts

- Results CSV: `results/week06_product_demo_results.csv`
- Envelope CSV (MLP): `results/week06_operating_envelope.csv`
- Dashboard: `results/master_dashboard.png`
- Mitigations: `results/mitigations.md`
- Regression JSON: `results/regression_report.json`
- Regression Summary: `results/regression_summary.md`
