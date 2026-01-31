# src/model.py
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.constraints import apply_noise, apply_staleness, simulate_latency


def train_and_evaluate(random_state: int = 42) -> float:
    """
    Baseline model under ideal conditions (no physics constraints).
    Returns accuracy on a held-out test set.
    """
    X, y = make_classification(
        n_samples=1500,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        flip_y=0.0,
        class_sep=2.0,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return float(accuracy_score(y_test, preds))


def train_model(random_state: int = 42) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """
    Train the baseline model once and return (model, X_test, y_test).
    """
    X, y = make_classification(
        n_samples=1500,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        flip_y=0.0,
        class_sep=2.0,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)

    return model, np.asarray(X_test), np.asarray(y_test)


# -------------------------
# Noise
# -------------------------
def evaluate_with_noise(std: float, seed: int = 0, random_state: int = 42) -> float:
    model, X_test, y_test = train_model(random_state=random_state)
    X_noisy = apply_noise(X_test, std=std, seed=seed)
    preds = model.predict(X_noisy)
    return float(accuracy_score(y_test, preds))


# -------------------------
# Latency (system-level effects)
# -------------------------
def evaluate_with_latency_effects(
    delay_ms: int,
    timeout_ms: int,
    drop_rate: float,
    seed: int = 0,
    random_state: int = 42,
) -> dict[str, float]:
    model, X_test, y_test = train_model(random_state=random_state)

    preds = model.predict(X_test)
    raw_acc = float(accuracy_score(y_test, preds))

    sim: dict[str, Any] = simulate_latency(
        n_samples=len(y_test),
        delay_ms=delay_ms,
        timeout_ms=timeout_ms,
        drop_rate=drop_rate,
        seed=seed,
    )

    dropped_mask = np.asarray(sim["dropped_mask"], dtype=bool)
    timed_out_mask = np.asarray(sim["timed_out_mask"], dtype=bool)
    failed_mask = dropped_mask | timed_out_mask

    failed_frac = float(failed_mask.mean())

    effective_correct = (preds == y_test).copy()
    effective_correct[failed_mask] = False
    effective_acc = float(effective_correct.mean())

    return {
        "raw_accuracy": raw_acc,
        "effective_accuracy": effective_acc,
        "failed_frac": failed_frac,
        "dropped_frac": float(dropped_mask.mean()),
        "timed_out_frac": float(timed_out_mask.mean()),
    }


# -------------------------
# NEW: Staleness
# -------------------------
def evaluate_with_staleness(
    delay_ms: int,
    drift_per_ms: float = 0.01,
    seed: int = 0,
    random_state: int = 42,
) -> float:
    """
    Staleness = inputs become outdated.
    We model this as drift that grows with delay.
    """
    model, X_test, y_test = train_model(random_state=random_state)

    X_stale = apply_staleness(X_test, delay_ms=delay_ms, drift_per_ms=drift_per_ms, seed=seed)
    preds = model.predict(X_stale)
    return float(accuracy_score(y_test, preds))


# -------------------------
# Combined: Noise + Latency
# -------------------------
def evaluate_with_noise_and_latency(
    noise_std: float,
    delay_ms: int,
    timeout_ms: int,
    drop_rate: float,
    seed: int = 0,
    random_state: int = 42,
) -> dict[str, float]:
    model, X_test, y_test = train_model(random_state=random_state)

    X_noisy = apply_noise(X_test, std=noise_std, seed=seed)
    preds = model.predict(X_noisy)
    raw_acc = float(accuracy_score(y_test, preds))

    sim: dict[str, Any] = simulate_latency(
        n_samples=len(y_test),
        delay_ms=delay_ms,
        timeout_ms=timeout_ms,
        drop_rate=drop_rate,
        seed=seed,
    )

    dropped_mask = np.asarray(sim["dropped_mask"], dtype=bool)
    timed_out_mask = np.asarray(sim["timed_out_mask"], dtype=bool)
    failed_mask = dropped_mask | timed_out_mask

    failed_frac = float(failed_mask.mean())

    effective_correct = (preds == y_test).copy()
    effective_correct[failed_mask] = False
    effective_acc = float(effective_correct.mean())

    return {
        "raw_accuracy": raw_acc,
        "effective_accuracy": effective_acc,
        "failed_frac": failed_frac,
        "dropped_frac": float(dropped_mask.mean()),
        "timed_out_frac": float(timed_out_mask.mean()),
    }


# -------------------------
# Combined: Noise + Latency + Staleness (real stacks)
# -------------------------
def evaluate_with_noise_latency_staleness(
    noise_std: float,
    delay_ms: int,
    drift_per_ms: float,
    timeout_ms: int,
    drop_rate: float,
    seed: int = 0,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Real-world stack:
    - noise corrupts the measurement
    - staleness drifts it away from reality
    - latency can drop/timeout the sample (deadline failures)
    """
    model, X_test, y_test = train_model(random_state=random_state)

    X_phys = apply_noise(X_test, std=noise_std, seed=seed)
    X_phys = apply_staleness(X_phys, delay_ms=delay_ms, drift_per_ms=drift_per_ms, seed=seed)

    preds = model.predict(X_phys)
    raw_acc = float(accuracy_score(y_test, preds))

    sim: dict[str, Any] = simulate_latency(
        n_samples=len(y_test),
        delay_ms=delay_ms,
        timeout_ms=timeout_ms,
        drop_rate=drop_rate,
        seed=seed,
    )

    dropped_mask = np.asarray(sim["dropped_mask"], dtype=bool)
    timed_out_mask = np.asarray(sim["timed_out_mask"], dtype=bool)
    failed_mask = dropped_mask | timed_out_mask

    failed_frac = float(failed_mask.mean())

    effective_correct = (preds == y_test).copy()
    effective_correct[failed_mask] = False
    effective_acc = float(effective_correct.mean())

    return {
        "raw_accuracy": raw_acc,
        "effective_accuracy": effective_acc,
        "failed_frac": failed_frac,
        "dropped_frac": float(dropped_mask.mean()),
        "timed_out_frac": float(timed_out_mask.mean()),
    }