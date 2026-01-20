from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .segments import Segment


@dataclass(frozen=True)
class SegmentFeatures:
    """
    Features computed for a segment.

    - motion_raw: average frame difference magnitude (proxy for action/event)
    - sharpness_raw: Laplacian variance (proxy for focus / usability)
    - interest_score: normalized combined score in [0, 1]
    """
    segment: Segment
    motion_raw: float
    sharpness_raw: float
    interest_score: float


def _read_frame_at_timestamp(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    """
    Seek and read a frame at the given timestamp.

    Notes:
    - CAP_PROP_POS_MSEC is approximate, but good enough for POC scoring.
    - We always read from proxy_quality, so decoding is relatively cheap.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale float32."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    """
    Compute a common sharpness proxy: variance of Laplacian response.
    Higher variance usually means sharper edges (more in focus / less motion blur).
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two grayscale frames."""
    return float(np.mean(np.abs(a - b)))


def _sample_timestamps(segment: Segment, n: int) -> list[float]:
    """
    Sample n timestamps uniformly within (start, end).
    We avoid the exact endpoints to reduce risk of boundary artifacts.
    """
    if n <= 1:
        mid = (segment.start_sec + segment.end_sec) / 2.0
        return [mid]

    eps = 1e-3
    start = segment.start_sec + eps
    end = max(start, segment.end_sec - eps)

    # Uniform sampling.
    ts = np.linspace(start, end, num=n, dtype=np.float64)
    return [float(x) for x in ts]


def _robust_normalize(values: list[float], p_low: float = 10.0, p_high: float = 90.0) -> list[float]:
    """
    Robustly map values to [0, 1] using percentiles (reduces the impact of outliers).

    Example:
    - Anything at/below p10 -> ~0
    - Anything at/above p90 -> ~1
    - Linear scaling in between
    """
    if not values:
        return []

    arr = np.asarray(values, dtype=np.float64)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)

    # Guard against degenerate distributions.
    if hi <= lo + 1e-9:
        # If all values are (almost) equal, return 0.5 everywhere.
        return [0.5 for _ in values]

    norm = (arr - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return [float(x) for x in norm]


def compute_segment_features(
    proxy_quality_path: Path,
    segments: list[Segment],
    frames_per_segment: int = 3,
    motion_weight: float = 0.6,
    sharpness_weight: float = 0.4,
) -> list[SegmentFeatures]:
    """
    Compute motion + sharpness features for each segment and derive an interest score.

    Approach (POC):
    - For each segment, sample a few frames uniformly.
    - motion_raw: average mean-abs-diff between consecutive sampled frames.
    - sharpness_raw: average Laplacian variance over sampled frames.
    - Normalize motion/sharpness across all segments (robust percentiles).
    - interest_score = w_motion * motion_norm + w_sharpness * sharpness_norm

    Why not optical flow (yet):
    - Optical flow is more accurate but heavier and more code.
    - For a POC, frame difference is a reasonable proxy for "something happened".
    """
    if not segments:
        return []

    cap = cv2.VideoCapture(str(proxy_quality_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open proxy file: {proxy_quality_path}")

    motion_raw_list: list[float] = []
    sharpness_raw_list: list[float] = []
    per_segment_raw: list[tuple[Segment, float, float]] = []

    try:
        for seg in segments:
            ts_list = _sample_timestamps(seg, max(2, frames_per_segment))

            frames_gray: list[np.ndarray] = []
            sharpness_vals: list[float] = []

            for ts in ts_list:
                frame = _read_frame_at_timestamp(cap, ts)
                if frame is None:
                    continue
                gray = _to_gray(frame)
                frames_gray.append(gray)
                sharpness_vals.append(_laplacian_variance(gray))

            # If we couldn't read enough frames, assign minimal raw scores.
            if len(frames_gray) < 2:
                motion_raw = 0.0
            else:
                diffs = [
                    _mean_abs_diff(frames_gray[i], frames_gray[i + 1])
                    for i in range(len(frames_gray) - 1)
                ]
                motion_raw = float(np.mean(diffs)) if diffs else 0.0

            sharpness_raw = float(np.mean(sharpness_vals)) if sharpness_vals else 0.0

            per_segment_raw.append((seg, motion_raw, sharpness_raw))
            motion_raw_list.append(motion_raw)
            sharpness_raw_list.append(sharpness_raw)
    finally:
        cap.release()

    # Robust normalization across the whole video (within-video ranking).
    motion_norm = _robust_normalize(motion_raw_list, p_low=10.0, p_high=90.0)
    sharpness_norm = _robust_normalize(sharpness_raw_list, p_low=10.0, p_high=90.0)

    # Combine into final interest score.
    features: list[SegmentFeatures] = []
    for i, (seg, m_raw, s_raw) in enumerate(per_segment_raw):
        score = (motion_weight * motion_norm[i]) + (sharpness_weight * sharpness_norm[i])
        score = float(np.clip(score, 0.0, 1.0))
        features.append(
            SegmentFeatures(
                segment=seg,
                motion_raw=m_raw,
                sharpness_raw=s_raw,
                interest_score=score,
            )
        )

    return features