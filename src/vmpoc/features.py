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

    Raw signals (debuggable):
    - motion_raw: average mean-abs-diff between sampled frames (proxy for activity)
    - sharpness_raw: average Laplacian variance (proxy for focus / blur)

    Normalized signals (within-video, robust):
    - motion_norm: motion_raw mapped to [0, 1]
    - sharpness_norm: sharpness_raw mapped to [0, 1]

    Separated scores (all in [0, 1]):
    - quality_score: technical usability (sharpness + exposure penalty)
    - composition_score: camera smoothness / stability proxy (low jitter -> high score)
    - action_score: perceived action/eventness (motion + peaks, penalized by bad composition)

    Final:
    - interest_score: weighted combination of (quality, composition, action)
    """
    segment: Segment
    motion_raw: float
    sharpness_raw: float
    motion_norm: float
    sharpness_norm: float
    quality_score: float
    composition_score: float
    action_score: float
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


def _exposure_penalty(gray: np.ndarray, t_low: float = 5.0, t_high: float = 250.0) -> float:
    """
    Cheap exposure/clipping proxy:
    - fraction of pixels near black + near white.
    Returns a value in [0, 2] (practically small for good exposures).
    """
    frac_dark = float(np.mean(gray < t_low))
    frac_bright = float(np.mean(gray > t_high))
    return frac_dark + frac_bright


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

    if hi <= lo + 1e-9:
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
    composition_weight: float = 0.2,
) -> list[SegmentFeatures]:
    """
    Compute cheap visual features for each segment and derive an interest score.

    Approach (POC v2):
    - For each segment, sample a few frames uniformly.
    - motion_raw: average mean-abs-diff between consecutive sampled frames.
    - sharpness_raw: average Laplacian variance over sampled frames.
    - exposure_raw: near-black + near-white pixel fraction (clipping proxy)
    - jitter_raw: variance of per-step diffs (proxy for shake vs smooth motion)
    - eventness_raw: peakiness of diffs (max - median)

    Then:
    - Robustly normalize raw signals across segments (within-video).
    - Compute:
        quality_score      = sharpness_norm * (1 - exposure_norm)
        composition_score  = 1 - jitter_norm
        action_score       = (0.7*motion_norm + 0.3*eventness_norm) * (0.5 + 0.5*composition_score)
    - Combine:
        interest_score     = wq*quality_score + wc*composition_score + wa*action_score
      where:
        wq ~= sharpness_weight, wa ~= motion_weight, wc ~= composition_weight (all normalized to sum 1).
    """
    if not segments:
        return []

    cap = cv2.VideoCapture(str(proxy_quality_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open proxy file: {proxy_quality_path}")

    motion_raw_list: list[float] = []
    sharpness_raw_list: list[float] = []
    exposure_raw_list: list[float] = []
    jitter_raw_list: list[float] = []
    eventness_raw_list: list[float] = []

    per_segment_raw: list[tuple[Segment, float, float]] = []
    per_segment_aux: list[tuple[float, float, float]] = []  # exposure_raw, jitter_raw, eventness_raw

    try:
        for seg in segments:
            ts_list = _sample_timestamps(seg, max(2, frames_per_segment))

            frames_gray: list[np.ndarray] = []
            sharpness_vals: list[float] = []
            exposure_vals: list[float] = []

            for ts in ts_list:
                frame = _read_frame_at_timestamp(cap, ts)
                if frame is None:
                    continue
                gray = _to_gray(frame)
                frames_gray.append(gray)
                sharpness_vals.append(_laplacian_variance(gray))
                exposure_vals.append(_exposure_penalty(gray))

            if len(frames_gray) < 2:
                motion_raw = 0.0
                diffs: list[float] = []
            else:
                diffs = [
                    _mean_abs_diff(frames_gray[i], frames_gray[i + 1])
                    for i in range(len(frames_gray) - 1)
                ]
                motion_raw = float(np.mean(diffs)) if diffs else 0.0

            sharpness_raw = float(np.mean(sharpness_vals)) if sharpness_vals else 0.0
            exposure_raw = float(np.mean(exposure_vals)) if exposure_vals else 0.0

            if len(diffs) >= 2:
                jitter_raw = float(np.var(diffs))
            else:
                jitter_raw = 0.0

            if diffs:
                eventness_raw = float(np.max(diffs) - np.median(diffs))
            else:
                eventness_raw = 0.0

            per_segment_raw.append((seg, motion_raw, sharpness_raw))
            per_segment_aux.append((exposure_raw, jitter_raw, eventness_raw))

            motion_raw_list.append(motion_raw)
            sharpness_raw_list.append(sharpness_raw)
            exposure_raw_list.append(exposure_raw)
            jitter_raw_list.append(jitter_raw)
            eventness_raw_list.append(eventness_raw)
    finally:
        cap.release()

    motion_norm = _robust_normalize(motion_raw_list, p_low=10.0, p_high=90.0)
    sharpness_norm = _robust_normalize(sharpness_raw_list, p_low=10.0, p_high=90.0)
    exposure_norm = _robust_normalize(exposure_raw_list, p_low=10.0, p_high=90.0)
    jitter_norm = _robust_normalize(jitter_raw_list, p_low=10.0, p_high=90.0)
    eventness_norm = _robust_normalize(eventness_raw_list, p_low=10.0, p_high=90.0)

    w_quality = float(sharpness_weight)
    w_action = float(motion_weight)
    w_comp = float(composition_weight)
    w_sum = w_quality + w_action + w_comp
    if w_sum <= 1e-9:
        w_quality, w_comp, w_action = 0.4, 0.2, 0.4
        w_sum = 1.0
    w_quality /= w_sum
    w_comp /= w_sum
    w_action /= w_sum

    features: list[SegmentFeatures] = []
    for i, (seg, m_raw, s_raw) in enumerate(per_segment_raw):
        exposure_raw, jitter_raw, eventness_raw = per_segment_aux[i]

        m_n = float(motion_norm[i])
        s_n = float(sharpness_norm[i])
        exp_n = float(exposure_norm[i])
        jit_n = float(jitter_norm[i])
        evt_n = float(eventness_norm[i])

        quality_score = float(np.clip(s_n * (1.0 - exp_n), 0.0, 1.0))
        composition_score = float(np.clip(1.0 - jit_n, 0.0, 1.0))

        action_base = float(np.clip((0.7 * m_n) + (0.3 * evt_n), 0.0, 1.0))
        action_score = float(np.clip(action_base * (0.5 + 0.5 * composition_score), 0.0, 1.0))

        interest_score = (w_quality * quality_score) + (w_comp * composition_score) + (w_action * action_score)
        interest_score = float(np.clip(interest_score, 0.0, 1.0))

        features.append(
            SegmentFeatures(
                segment=seg,
                motion_raw=m_raw,
                sharpness_raw=s_raw,
                motion_norm=m_n,
                sharpness_norm=s_n,
                quality_score=quality_score,
                composition_score=composition_score,
                action_score=action_score,
                interest_score=interest_score,
            )
        )

    return features