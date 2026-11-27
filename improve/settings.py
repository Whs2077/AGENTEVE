"""Validate the improve block. A disabled block is not required to be complete."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ImproveSettings:
    """Parameters that apply only after improve is enabled."""

    enabled: bool
    repeat: int
    retain: float
    sample_count: int
    model_name: str
    api_url: str
    api_key_env: str


class GeneratorUnavailable(RuntimeError):
    """The synonym or counterexample model was not called."""


def parse_improve(raw: dict | None) -> ImproveSettings:
    """Check repeat, retain, and sample_count when improve is enabled.

    ``repeat < 0`` selects the four tests. The minus sign is not a repeat count.
    ``retain`` below 0.5 is rejected. ``sample_count`` must be a positive integer.
    """
    if not isinstance(raw, dict):
        raw = {}
    if not raw.get("enabled", False):
        return ImproveSettings(False, 1, 1.0, 1, "", "", "")
    repeat = _integer(raw.get("repeat"), "repeat")
    if repeat == 0:
        raise ValueError("improve.repeat cannot be 0")
    retain = raw.get("retain")
    if isinstance(retain, bool) or not isinstance(retain, (int, float)):
        raise ValueError("improve.retain must belong to [0.5, 1]")
    retain = float(retain)
    if math.isnan(retain) or retain < 0.5 or retain > 1:
        raise ValueError("improve.retain must belong to [0.5, 1]")
    sample_count = _integer(raw.get("sample_count"), "sample_count")
    if sample_count < 1:
        raise ValueError("improve.sample_count must be a positive integer")
    generator = raw.get("generator") if isinstance(raw.get("generator"), dict) else raw
    model_name = str(generator.get("modelName") or "")
    api_url = str(generator.get("apiUrl") or "")
    api_key_env = str(generator.get("apiKeyEnv") or "")
    if not model_name or not api_url or not api_key_env:
        raise ValueError("enabled improve requires generator modelName, apiUrl, and apiKeyEnv")
    return ImproveSettings(True, repeat, retain, sample_count, model_name, api_url, api_key_env)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"improve.{name} must be an integer")
    return value
