from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .segments import Segment, segment_overlaps
from .features import SegmentFeatures
from .selector import SelectedMoment


@dataclass(frozen=True)
class InspectionConfidence:
    """
    Output of the inspection confidence estimation.

    - score_1_10: integer confidence score (1 = low, 10 = high)
    - coverage_ratio: fraction of total interest covered by selected moments (0..1)
    - borderline_count: number of near-threshold segments not selected
    - reasons: short, human-readable reasons (for UI/debugging)
    """
    score_1_10: int
    coverage_ratio: float
    borderline_count: int
    reasons: List[str]


def _segments_covered_by_moments(
    all_segments: List[SegmentFeatures],
    selected: List[SelectedMoment],
    min_overlap_sec: float = 0.5,
) -> List[bool]:
    """
    Mark each segment as 'covered' if it overlaps with any selected moment
    by at least min_overlap_sec.

    This is used to compute interest-mass coverage.
    """
    covered = [False] * len(all_segments)

    if not selected:
        return covered

    moment_segs = [Segment(m.start_sec, m.end_sec) for m in selected]

    for i, sf in enumerate(all_segments):
        seg = sf.segment
        for mseg in moment_segs:
            if segment_overlaps(seg, mseg) >= min_overlap_sec:
                covered[i] = True
                break

    return covered


def _score_flatness_penalty(scores: np.ndarray) -> float:
    """
    Penalize flat distributions (no clear peaks).

    Heuristic:
    - If score standard deviation is small relative to mean, scores are "flat",
      meaning the video does not have strong standout segments (harder selection).
    - Returns a penalty in [0, 3].
    """
    if scores.size == 0:
        return 3.0

    mean = float(np.mean(scores))
    std = float(np.std(scores))

    # If mean is near zero, the scores carry little signal -> low confidence.
    if mean < 1e-6:
        return 3.0

    ratio = std / mean  # higher is better (peaky), lower is flatter
    # Map ratio to penalty: ratio < 0.25 => strong penalty, ratio > 1.0 => small penalty
    if ratio <= 0.25:
        return 3.0
    if ratio >= 1.0:
        return 0.0
    # Linear interpolation between 0.25 and 1.0
    return float(3.0 * (1.0 - (ratio - 0.25) / (1.0 - 0.25)))


def compute_inspection_confidence(
    all_segment_features: List[SegmentFeatures],
    selected_moments: List[SelectedMoment],
    borderline_factor: float = 0.85,
    max_borderline_penalty: float = 3.0,
) -> InspectionConfidence:
    """
    Estimate how confident we are that we've "covered" the meaningful parts of the video.

    Important:
    - This is NOT a claim of ground-truth completeness.
    - It's a deterministic confidence estimate derived from our own scoring signal.

    Core idea:
    - If selected moments cover most of the total 'interest mass', confidence increases.
    - If many near-threshold segments remain unselected, confidence decreases.
    - If the interest scores are flat (no clear peaks), confidence decreases.

    Returns:
    - InspectionConfidence with score_1_10 and short reasons.
    """
    if not all_segment_features:
        return InspectionConfidence(
            score_1_10=1,
            coverage_ratio=0.0,
            borderline_count=0,
            reasons=["No segments/features were computed (cannot assess coverage)."],
        )

    scores = np.asarray([sf.interest_score for sf in all_segment_features], dtype=np.float64)
    total_interest = float(np.sum(scores))

    # If total interest is ~0, our signal says "nothing interesting" OR "signal failed".
    # For POC we treat this as low confidence and request human review.
    if total_interest < 1e-6:
        return InspectionConfidence(
            score_1_10=2,
            coverage_ratio=0.0,
            borderline_count=0,
            reasons=[
                "Interest signal is near zero across the video (no clear moments detected).",
                "This can happen with very dark/blurred footage or low visual change.",
            ],
        )

    covered_mask = _segments_covered_by_moments(all_segment_features, selected_moments)
    covered_interest = float(np.sum(scores[np.asarray(covered_mask, dtype=bool)]))
    coverage_ratio = float(np.clip(covered_interest / total_interest, 0.0, 1.0))

    # Borderline segments: near the weakest selected moment score.
    borderline_count = 0
    borderline_penalty = 0.0
    if selected_moments:
        weakest_selected = min(m.interest_score for m in selected_moments)
        borderline_threshold = weakest_selected * borderline_factor

        # Count segments not covered that are close to selected threshold.
        for i, sf in enumerate(all_segment_features):
            if covered_mask[i]:
                continue
            if sf.interest_score >= borderline_threshold:
                borderline_count += 1

        # Penalize if there are many borderline segments.
        # Heuristic: 0 -> 0 penalty, 10+ -> max penalty.
        borderline_penalty = float(np.clip(borderline_count / 10.0, 0.0, 1.0) * max_borderline_penalty)
    else:
        # If we selected nothing but total_interest is non-trivial, confidence should be low.
        borderline_penalty = max_borderline_penalty

    # Flatness penalty: if scores are flat, selection is ambiguous.
    flatness_penalty = _score_flatness_penalty(scores)

    # Base confidence from coverage ratio: map [0..1] -> [1..10]
    base = 1.0 + 9.0 * coverage_ratio

    # Apply penalties and clamp.
    final = base - borderline_penalty - flatness_penalty
    final_int = int(np.clip(np.round(final), 1, 10))

    reasons: List[str] = []
    reasons.append(f"Interest coverage ratio: {coverage_ratio:.2f} (covered vs total interest mass).")

    if borderline_count > 0:
        reasons.append(f"Borderline candidates not covered: {borderline_count} (higher means possible missed moments).")
    else:
        reasons.append("Few/no borderline candidates outside selected moments (lower miss risk).")

    if flatness_penalty >= 2.0:
        reasons.append("Interest scores are relatively flat (no strong peaks), selection is less certain.")
    else:
        reasons.append("Interest scores show peaks, selection is more reliable.")

    return InspectionConfidence(
        score_1_10=final_int,
        coverage_ratio=coverage_ratio,
        borderline_count=borderline_count,
        reasons=reasons[:4],  # keep it short for UI
    )