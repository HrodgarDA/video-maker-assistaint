from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .segments import Segment, segment_iou
from .features import SegmentFeatures


@dataclass(frozen=True)
class SelectedMoment:
    """
    A selected moment candidate produced by the POC selector.

    Notes:
    - thumb_timestamp_sec is where we will extract the thumbnail from the master video.
    - In the POC we use the segment midpoint. Later we can pick a better frame.
    """
    start_sec: float
    end_sec: float
    interest_score: float
    thumb_timestamp_sec: float


def _segment_midpoint(seg: Segment) -> float:
    return (seg.start_sec + seg.end_sec) / 2.0


def select_top_moments(
    segment_features: List[SegmentFeatures],
    max_moments: int = 8,
    iou_threshold: float = 0.30,
    min_gap_sec: float = 1.25,
    min_score: float = 0.10,
) -> List[SelectedMoment]:
    """
    Select top moments using a greedy ranking + non-max suppression strategy.

    Why:
    - Sliding windows create many overlapping candidates.
    - We want a compact set of representative moments without duplicates.

    Rules (POC):
    - Sort candidates by interest_score descending.
    - Accept a candidate if:
        * Its score >= min_score, and
        * It does not overlap too much with an already selected moment (IoU), and
        * It is not too close in time to already selected moments (min_gap_sec on midpoints).
    - Stop after max_moments.

    Parameters:
    - iou_threshold: maximum allowed IoU with already selected segments.
    - min_gap_sec: minimum time distance between moment midpoints (extra guard).
    - min_score: discard extremely low-interest segments.

    Returns:
    - List[SelectedMoment] sorted by interest_score descending (selection order).
    """
    if max_moments <= 0 or not segment_features:
        return []

    # Sort candidates by interest score descending.
    candidates = sorted(segment_features, key=lambda x: x.interest_score, reverse=True)

    selected: List[SelectedMoment] = []
    selected_segments: List[Segment] = []

    for cand in candidates:
        if len(selected) >= max_moments:
            break
        if cand.interest_score < min_score:
            # Since candidates are sorted descending, we can stop early.
            break

        seg = cand.segment
        mid = _segment_midpoint(seg)

        # Non-max suppression: reject if too similar to already selected segments.
        reject = False
        for s_seg in selected_segments:
            # IoU overlap check
            if segment_iou(seg, s_seg) >= iou_threshold:
                reject = True
                break

            # Additional temporal distance guard (helps when IoU is small but still redundant)
            s_mid = _segment_midpoint(s_seg)
            if abs(mid - s_mid) < min_gap_sec:
                reject = True
                break

        if reject:
            continue

        selected_segments.append(seg)
        selected.append(
            SelectedMoment(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                interest_score=cand.interest_score,
                thumb_timestamp_sec=mid,
            )
        )

    return selected