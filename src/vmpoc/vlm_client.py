from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptionInputs:
    """
    Minimal inputs used to generate a short moment description.
    In the POC we do NOT call a VLM; we use simple heuristics.
    """
    motion_raw: float
    sharpness_raw: float
    interest_score: float


def describe_moment_poc(inputs: DescriptionInputs) -> str:
    """
    Generate a short, deterministic description (POC fallback).

    Why:
    - Enables end-to-end demo without requiring a local VLM setup.
    - Keeps outputs stable for testing and iteration on scoring/selection.

    This is intentionally simple: later we will replace it with a VLM-based captioner.
    """
    # Coarse thresholds: tuned for relative ordering, not absolute correctness.
    if inputs.interest_score >= 0.75:
        level = "High"
    elif inputs.interest_score >= 0.45:
        level = "Medium"
    else:
        level = "Low"

    # Interpret motion vs sharpness in a practical way.
    if inputs.motion_raw > 12.0 and inputs.sharpness_raw < 80.0:
        hint = "Fast motion with possible blur/shake"
    elif inputs.motion_raw > 12.0:
        hint = "Action / strong motion"
    elif inputs.sharpness_raw > 150.0:
        hint = "Stable and sharp scene"
    else:
        hint = "Moderate activity / context shot"

    return f"[{level} interest] {hint}"