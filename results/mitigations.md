## Required Mitigations (Fix-it Engine)

- **Option A (Software):** Add a **50ms prediction buffer** (queue/pipeline) to tolerate real-time jitter.
  - Reason: effective accuracy fell below threshold at ~150ms (eff=0.00%, threshold=75%).
- **Option B (Hardware):** Improve **temp_c** sensor stability (reduce noise by ~0.1σ).
  - Reason: temp_c had the largest accuracy drop during single-feature noise injection (Δ=0.000).

(Profile used: `industrial_wifi`)
