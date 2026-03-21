# src/profiles.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    desc: str
    threshold: float
    timeout_ms: int
    drop_rate: float
    drift_per_ms: float
    bias_per_ms: float
    latency_delays_ms: list[int]
    noise_stds: list[float]
    envelope_noise_stds: list[float]
    envelope_delays_ms: list[int]
    burst_ms: int
    sample_period_ms: int
    sensitivity_noise_std: float
    saturation_percentile: float
    quantization_decimals: int
    intermittent_dropout_prob: float
    stuck_prob: float


def load_profile(profile_name: str, profiles_path: str = "configs/profiles.json") -> Profile:
    p = Path(profiles_path)
    if not p.exists():
        raise FileNotFoundError(f"Profiles file not found: {profiles_path}")

    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    if profile_name not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Unknown profile '{profile_name}'. Available: {available}")

    cfg = data[profile_name]
    return Profile(
        name=profile_name,
        desc=str(cfg["desc"]),
        threshold=float(cfg["threshold"]),
        timeout_ms=int(cfg["timeout_ms"]),
        drop_rate=float(cfg["drop_rate"]),
        drift_per_ms=float(cfg["drift_per_ms"]),
        bias_per_ms=float(cfg["bias_per_ms"]),
        latency_delays_ms=list(map(int, cfg["latency_delays_ms"])),
        noise_stds=list(map(float, cfg["noise_stds"])),
        envelope_noise_stds=list(map(float, cfg["envelope_noise_stds"])),
        envelope_delays_ms=list(map(int, cfg["envelope_delays_ms"])),
        burst_ms=int(cfg["burst_ms"]),
        sample_period_ms=int(cfg["sample_period_ms"]),
        sensitivity_noise_std=float(cfg["sensitivity_noise_std"]),
        saturation_percentile=float(cfg["saturation_percentile"]),
        quantization_decimals=int(cfg["quantization_decimals"]),
        intermittent_dropout_prob=float(cfg["intermittent_dropout_prob"]),
        stuck_prob=float(cfg["stuck_prob"]),
    )