# Week 06 – Smart Factory Model Reliability Report (Startup Demo)

Generated: 2026-02-11T17:07:03

## Summary

Model: PyTorch MLP (train_acc=0.8807, test_acc=0.8817)

Safety Grade: **F** (worst effective accuracy=0.0000)

## Positives

- Baseline accuracy on Smart Factory CSV: **0.8817**
- Runs physics stress tests: noise, latency(drop/timeout), staleness(slow/fast), and failure modes.
- Produces an Operating Envelope grid (noise vs latency) with PASS/FAIL boundary.

## Issues / Concerns

- Worst effective accuracy observed: **0.0000** (Grade **F**)
- Fast-world staleness and sensor failure modes can drop accuracy significantly.
- Effective accuracy is the real deployment metric: late/missing decisions count as failures.

## Operating Envelope

- The Operating Envelope shows safe vs unsafe regions:
  - X-axis: latency delay
  - Y-axis: noise level
  - Cell color (in dashboard) indicates PASS/FAIL based on effective accuracy threshold.

## Artifacts

- Results CSV: `results/week06_product_demo_results.csv`
- Envelope CSV: `results/week06_operating_envelope.csv`
- Dashboard: `results/master_dashboard.png` (generated next)
