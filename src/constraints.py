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


import numpy as np

def apply_sensor_saturation(
    X: np.ndarray,
    *,
    clip_max: float | None = None,
    clip_min: float | None = None,
    per_feature: bool = False,
    ref_X: np.ndarray | None = None,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    """
    Sensor Saturation / Flatline:
    Caps values so sensors can't exceed physical range.

    Options:
      - clip_max/clip_min: scalar caps applied to all features
      - per_feature=True with ref_X: derive per-feature caps from ref_X percentile
    """
    X = np.asarray(X, dtype=float)

    if per_feature:
        if ref_X is None:
            raise ValueError("apply_sensor_saturation(per_feature=True) requires ref_X")
        ref_X = np.asarray(ref_X, dtype=float)
        hi = np.percentile(ref_X, clip_percentile, axis=0)
        lo = np.percentile(ref_X, 100.0 - clip_percentile, axis=0)
        return np.clip(X, lo, hi)

    # global caps
    if clip_min is None:
        clip_min = -np.inf
    if clip_max is None:
        clip_max = np.inf
    return np.clip(X, clip_min, clip_max)


def apply_quantization(
    X: np.ndarray,
    *,
    decimals: int = 1,
) -> np.ndarray:
    """
    Quantization Error:
    Simulates low-resolution sensors (ADC precision).
    Example: decimals=1 -> round to 1 decimal place.
    """
    X = np.asarray(X, dtype=float)
    return np.round(X, decimals=decimals)


def simulate_packet_burst_loss(
    n_samples: int,
    *,
    burst_ms: int = 500,
    sample_period_ms: int = 10,
    seed: int = 0,
) -> dict:
    """
    Packet Burst Loss:
    Drops ALL samples for one contiguous window (radio blackout).

    Returns:
      dict with dropped_mask (bool array, True = dropped)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(n_samples)

    if n_samples <= 0:
        return {"dropped_mask": np.zeros(0, dtype=bool)}

    # how many consecutive samples correspond to the blackout window
    window_len = int(np.ceil(burst_ms / max(sample_period_ms, 1)))
    window_len = max(1, min(window_len, n_samples))

    start = int(rng.integers(0, n_samples - window_len + 1))
    mask = np.zeros(n_samples, dtype=bool)
    mask[start : start + window_len] = True
    return {"dropped_mask": mask, "start_idx": start, "window_len": window_len}