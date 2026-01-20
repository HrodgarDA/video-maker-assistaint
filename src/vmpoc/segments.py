from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    """A time interval inside a video."""
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def generate_sliding_segments(
    duration_sec: float,
    window_sec: float,
    step_sec: float,
    min_segment_sec: float = 0.5,
) -> list[Segment]:
    """
    Generate sliding-window segments that cover the full video duration.

    Why sliding windows:
    - We assume raw footage (no edits, no reliable cut boundaries).
    - We want uniform coverage and then rank segments by "interest".

    Parameters:
    - duration_sec: total video length
    - window_sec: length of each candidate segment
    - step_sec: stride between consecutive windows
    - min_segment_sec: minimum segment length we consider valid (guards edge cases)

    Returns:
    - list of Segment(start_sec, end_sec)
    """
    if duration_sec <= 0:
        return []

    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive numbers.")

    segments: list[Segment] = []

    # If video is shorter than the window, return a single segment.
    if duration_sec <= window_sec:
        if duration_sec >= min_segment_sec:
            segments.append(Segment(0.0, duration_sec))
        return segments

    t = 0.0
    while t < duration_sec:
        start = t
        end = min(t + window_sec, duration_sec)

        if (end - start) >= min_segment_sec:
            segments.append(Segment(start, end))

        # Stop when the next window would start beyond the end.
        t += step_sec
        if t >= duration_sec:
            break

    return segments


def segment_overlaps(a: Segment, b: Segment) -> float:
    """
    Compute overlap duration (in seconds) between two segments.
    """
    left = max(a.start_sec, b.start_sec)
    right = min(a.end_sec, b.end_sec)
    return max(0.0, right - left)


def segment_iou(a: Segment, b: Segment) -> float:
    """
    Compute Intersection-over-Union (IoU) between segments.
    Useful later for deduplication / non-max suppression.
    """
    inter = segment_overlaps(a, b)
    union = a.duration_sec + b.duration_sec - inter
    if union <= 0:
        return 0.0
    return inter / union


def clip_segment_to_duration(seg: Segment, duration_sec: float) -> Segment:
    """
    Clamp a segment to [0, duration_sec].
    """
    start = max(0.0, min(seg.start_sec, duration_sec))
    end = max(0.0, min(seg.end_sec, duration_sec))
    if end < start:
        end = start
    return Segment(start, end)