# src/profiles.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Profile:
    name: str
    threshold: float
    timeout_ms: int
    drop_rate: float
    latency_delays_ms: list[int]
    noise_stds: list[float]
    envelope_noise_stds: list[float]
    envelope_delays_ms: list[int]
    burst_ms: int
    sample_period_ms: int
    sensitivity_noise_std: float


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
        threshold=float(cfg["threshold"]),
        timeout_ms=int(cfg["timeout_ms"]),
        drop_rate=float(cfg["drop_rate"]),
        latency_delays_ms=list(map(int, cfg["latency_delays_ms"])),
        noise_stds=list(map(float, cfg["noise_stds"])),
        envelope_noise_stds=list(map(float, cfg["envelope_noise_stds"])),
        envelope_delays_ms=list(map(int, cfg["envelope_delays_ms"])),
        burst_ms=int(cfg["burst_ms"]),
        sample_period_ms=int(cfg["sample_period_ms"]),
        sensitivity_noise_std=float(cfg["sensitivity_noise_std"]),
    )