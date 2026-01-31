# src/constraints.py

from __future__ import annotations

import time
import numpy as np


def apply_noise(X: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    """
    Adds Gaussian noise to inputs to simulate sensor noise.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=X.shape)
    return X + noise


def simulate_latency(
    n_samples: int,
    delay_ms: int,
    timeout_ms: int,
    drop_rate: float,
    seed: int = 0,
    jitter_frac: float = 0.10,
) -> dict:
    """
    Simulates system-level latency effects.

    Returns:
      - dropped_mask: samples lost (e.g., packet drop / missing sensor read)
      - timed_out_mask: samples that miss the deadline (timeout)
      - delay_applied_ms: per-sample effective delay (with jitter)
    """
    if not (0.0 <= drop_rate <= 1.0):
        raise ValueError("drop_rate must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)

    # Per-sample delay with small jitter (defaults to +/-10%)
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
# Backwards compatibility shims
# ----------------------------

def apply_latency(X=None, delay_ms: int = 50):
    """
    Old function name used by older model.py/tests.
    Kept only to prevent import errors.
    """
    time.sleep(delay_ms / 1000)
    return X

def add_gaussian_noise(X, std: float, seed: int = 0):
    """
    Backwards-compatible name used by older model.py/tests.
    Calls the new function apply_noise().
    """
    return apply_noise(X, std=std, seed=seed)


def apply_staleness(
    X: np.ndarray,
    delay_ms: int,
    drift_per_ms: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate *staleness* by adding drift that grows with delay.

    Physics intuition:
    - If the system evolves over time, older measurements are 'off' from current reality.
    - We approximate that mismatch as additive drift with std proportional to delay.

    drift_std = delay_ms * drift_per_ms

    Params:
      delay_ms: how stale the data is
      drift_per_ms: how fast the world changes (bigger => faster dynamics)
      seed: reproducibility
    """
    X = np.asarray(X, dtype=float)

    if delay_ms <= 0 or drift_per_ms <= 0:
        return X.copy()

    rng = np.random.default_rng(seed)
    drift_std = float(delay_ms) * float(drift_per_ms)

    drift = rng.normal(loc=0.0, scale=drift_std, size=X.shape)
    return X + drift