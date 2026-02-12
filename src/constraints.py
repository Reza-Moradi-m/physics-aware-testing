# src/constraints.py
from __future__ import annotations

import numpy as np


# ----------------------------
# Noise
# ----------------------------
def apply_noise(X: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=X.shape)
    return np.asarray(X, dtype=float) + noise


# ----------------------------
# Latency effects (drop + timeout)
# ----------------------------
def simulate_latency(
    n_samples: int,
    delay_ms: int,
    timeout_ms: int,
    drop_rate: float,
    seed: int = 0,
    jitter_frac: float = 0.10,
) -> dict:
    if not (0.0 <= drop_rate <= 1.0):
        raise ValueError("drop_rate must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac, size=n_samples)
    delay_applied_ms = (delay_ms * jitter).astype(int)

    dropped_mask = rng.random(n_samples) < drop_rate
    timed_out_mask = delay_applied_ms > timeout_ms

    return {
        "dropped_mask": dropped_mask,
        "timed_out_mask": timed_out_mask,
        "delay_applied_ms": delay_applied_ms,
    }


# ----------------------------
# Staleness = drift grows with delay
# ----------------------------
def apply_staleness(
    X: np.ndarray,
    delay_ms: int,
    drift_per_ms: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if delay_ms <= 0 or drift_per_ms <= 0:
        return X.copy()

    rng = np.random.default_rng(seed)
    drift_std = float(delay_ms) * float(drift_per_ms)
    drift = rng.normal(loc=0.0, scale=drift_std, size=X.shape)
    return X + drift


# ============================================================
# NEW: "Advanced physics failure modes"
# ============================================================

def apply_intermittent_dropout(
    X: np.ndarray,
    drop_prob: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """
    Randomly zero-out entire rows to simulate missing sensor frames.
    """
    X = np.asarray(X, dtype=float).copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(X)) < drop_prob
    X[mask, :] = 0.0
    return X


def apply_stuck_at_value(
    X: np.ndarray,
    stuck_prob: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """
    Some samples become copies of the previous sample (sensor "stuck" freeze).
    """
    X = np.asarray(X, dtype=float).copy()
    rng = np.random.default_rng(seed)
    if len(X) <= 1:
        return X

    stuck = rng.random(len(X)) < stuck_prob
    for i in range(1, len(X)):
        if stuck[i]:
            X[i] = X[i - 1]
    return X


def apply_bias_drift(
    X: np.ndarray,
    bias_per_ms: float,
    delay_ms: int,
) -> np.ndarray:
    """
    Systematic drift: all features get biased over time (e.g. calibration/thermal drift).
    """
    X = np.asarray(X, dtype=float).copy()
    bias = float(delay_ms) * float(bias_per_ms)
    return X + bias