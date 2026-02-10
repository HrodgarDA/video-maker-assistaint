from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    - people_raw: average face-presence proxy from sampled frames
    - warm_raw: warm-color ratio in the top third (sunset/scenic proxy)
    - smooth_motion_raw: coherent global motion (camera roll/pan proxy)

    Normalized signals (within-video, robust):
    - motion_norm: motion_raw mapped to [0, 1]
    - sharpness_norm: sharpness_raw mapped to [0, 1]
    - people_norm: people_raw mapped to [0, 1]
    - warm_norm: warm_raw mapped to [0, 1]
    - smooth_motion_norm: smooth_motion_raw mapped to [0, 1]

    Separated scores (all in [0, 1]):
    - quality_score: technical usability (sharpness + exposure penalty)
    - composition_score: stability + smooth camera motion (low jitter + coherent flow)
    - action_score: perceived action/eventness (motion + peaks, penalized by bad composition)
    - people_score: group/people likelihood (faces + sharpness)
    - scenic_score: sunset/scenic likelihood (warmth + low motion + sharpness)
    - camera_motion_score: smooth camera roll/pan likelihood (coherent flow)

    Final:
    - interest_score: weighted combination of all scores
    """
    segment: Segment
    motion_raw: float
    sharpness_raw: float
    people_raw: float
    warm_raw: float
    smooth_motion_raw: float
    motion_norm: float
    sharpness_norm: float
    people_norm: float
    warm_norm: float
    smooth_motion_norm: float
    quality_score: float
    composition_score: float
    action_score: float
    people_score: float
    scenic_score: float
    camera_motion_score: float
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


def _resize_max_width(gray: np.ndarray, max_width: int = 320) -> np.ndarray:
    """Resize grayscale frame to a max width, preserving aspect ratio."""
    h, w = gray.shape[:2]
    if w <= max_width:
        return gray
    scale = float(max_width) / float(w)
    new_w = max_width
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=1)
def _get_face_detector() -> cv2.CascadeClassifier | None:
    """Load a lightweight Haar cascade for face detection, if available."""
    try:
        cascade_dir = getattr(cv2, "data", None)
        if cascade_dir is None or not hasattr(cascade_dir, "haarcascades"):
            return None
        cascade_path = Path(cascade_dir.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            return None
        return detector
    except Exception:
        return None


def _face_presence(gray: np.ndarray, detector: cv2.CascadeClassifier | None) -> tuple[float, float]:
    """
    Return (face_count, face_area_ratio) from a grayscale frame.
    face_area_ratio is total face area / frame area.
    """
    if detector is None:
        return 0.0, 0.0
    small = _resize_max_width(gray, max_width=320)
    gray_u8 = np.clip(small, 0, 255).astype(np.uint8)
    faces = detector.detectMultiScale(
        gray_u8,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return 0.0, 0.0
    frame_area = float(gray_u8.shape[0] * gray_u8.shape[1])
    area = float(np.sum([w * h for (_, _, w, h) in faces]))
    return float(len(faces)), float(area / frame_area) if frame_area > 0 else 0.0


def _warmth_metrics_top_third(frame_bgr: np.ndarray) -> tuple[float, float]:
    """
    Estimate warm-color ratio and average saturation in the top third of the frame.
    Returns (warm_ratio, sat_mean) in [0, 1].
    """
    h = frame_bgr.shape[0]
    if h <= 0:
        return 0.0, 0.0
    top = frame_bgr[: max(1, h // 3), :, :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # Warm hues in OpenCV HSV: reds/oranges/yellows ~ [0..35] and [160..179].
    warm_mask = ((h_ch <= 35) | (h_ch >= 160)) & (s_ch >= 50) & (v_ch >= 50)
    warm_ratio = float(np.mean(warm_mask)) if warm_mask.size else 0.0
    sat_mean = float(np.mean(s_ch) / 255.0) if s_ch.size else 0.0
    return warm_ratio, sat_mean


def _downscale_for_flow(gray: np.ndarray, max_width: int = 160) -> np.ndarray:
    """Downscale a grayscale frame for faster optical flow."""
    h, w = gray.shape[:2]
    if w <= max_width:
        return gray
    scale = float(max_width) / float(w)
    new_w = max_width
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _flow_stats(gray_frames: list[np.ndarray]) -> tuple[float, float]:
    """
    Compute global flow magnitude and directional variance across consecutive frames.
    Returns (mean_mag, dir_var) where dir_var in [0..1] (lower is smoother).
    """
    if len(gray_frames) < 2:
        return 0.0, 1.0

    mags: list[float] = []
    angles: list[float] = []

    for i in range(len(gray_frames) - 1):
        prev = _downscale_for_flow(gray_frames[i])
        nxt = _downscale_for_flow(gray_frames[i + 1])
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            0.5,
            2,
            15,
            3,
            5,
            1.2,
            0,
        )
        mean_flow = flow.reshape(-1, 2).mean(axis=0)
        dx = float(mean_flow[0])
        dy = float(mean_flow[1])
        mag = float(np.hypot(dx, dy))
        angle = float(np.arctan2(dy, dx))
        mags.append(mag)
        angles.append(angle)

    if not mags:
        return 0.0, 1.0

    mean_mag = float(np.mean(mags))
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    r = float(np.hypot(sin_mean, cos_mean))
    dir_var = float(np.clip(1.0 - r, 0.0, 1.0))
    return mean_mag, dir_var


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
    people_weight: float = 0.15,
    scenic_weight: float = 0.15,
    camera_motion_weight: float = 0.10,
) -> list[SegmentFeatures]:
    """
    Compute cheap visual features for each segment and derive an interest score.

    Approach (POC v3):
    - For each segment, sample a few frames uniformly.
    - motion_raw: average mean-abs-diff between consecutive sampled frames.
    - sharpness_raw: average Laplacian variance over sampled frames.
    - exposure_raw: near-black + near-white pixel fraction (clipping proxy)
    - jitter_raw: variance of per-step diffs (proxy for shake vs smooth motion)
    - eventness_raw: peakiness of diffs (max - median)
    - people_raw: face presence (count + area)
    - warm_raw: warm-hue ratio in the top third (sunset/scenic proxy)
    - smooth_motion_raw: coherent global motion (camera roll/pan proxy)

    Then:
    - Robustly normalize raw signals across segments (within-video).
    - Compute:
        quality_score       = sharpness_norm * (1 - exposure_norm)
        camera_motion_score = smooth_motion_norm
        composition_score   = 0.6*(1 - jitter_norm) + 0.4*camera_motion_score
        action_score        = (0.7*motion_norm + 0.3*eventness_norm) * (0.5 + 0.5*composition_score)
        people_score        = people_norm * (0.5 + 0.5*sharpness_norm)
        scenic_score        = warm/sat * low-motion * sharpness * composition
    - Combine:
        interest_score      = weighted sum of all scores (weights normalized to sum 1).
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
    people_raw_list: list[float] = []
    warm_raw_list: list[float] = []
    sat_raw_list: list[float] = []
    smooth_motion_raw_list: list[float] = []

    per_segment_raw: list[tuple[Segment, float, float, float, float, float]] = []
    per_segment_aux: list[tuple[float, float, float]] = []  # exposure_raw, jitter_raw, eventness_raw

    face_detector = _get_face_detector()

    try:
        for seg in segments:
            ts_list = _sample_timestamps(seg, max(2, frames_per_segment))

            frames_gray: list[np.ndarray] = []
            sharpness_vals: list[float] = []
            exposure_vals: list[float] = []
            face_counts: list[float] = []
            face_area_ratios: list[float] = []
            warm_vals: list[float] = []
            sat_vals: list[float] = []

            face_sample_idx = set()
            if ts_list:
                face_sample_idx.add(0)
                face_sample_idx.add(len(ts_list) // 2)

            for idx, ts in enumerate(ts_list):
                frame = _read_frame_at_timestamp(cap, ts)
                if frame is None:
                    continue
                gray = _to_gray(frame)
                frames_gray.append(gray)
                sharpness_vals.append(_laplacian_variance(gray))
                exposure_vals.append(_exposure_penalty(gray))

                if face_detector is not None and idx in face_sample_idx:
                    f_count, f_area = _face_presence(gray, face_detector)
                    face_counts.append(f_count)
                    face_area_ratios.append(f_area)

                warm_ratio, sat_mean = _warmth_metrics_top_third(frame)
                warm_vals.append(warm_ratio)
                sat_vals.append(sat_mean)

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

            if face_counts:
                avg_count = float(np.mean(face_counts))
                avg_area = float(np.mean(face_area_ratios)) if face_area_ratios else 0.0
                people_raw = avg_count + (3.0 * avg_area)
            else:
                people_raw = 0.0

            warm_raw = float(np.mean(warm_vals)) if warm_vals else 0.0
            sat_raw = float(np.mean(sat_vals)) if sat_vals else 0.0

            flow_mag_mean, flow_dir_var = _flow_stats(frames_gray)
            smooth_motion_raw = float(flow_mag_mean * (1.0 - flow_dir_var))

            per_segment_raw.append(
                (seg, motion_raw, sharpness_raw, people_raw, warm_raw, smooth_motion_raw)
            )
            per_segment_aux.append((exposure_raw, jitter_raw, eventness_raw))

            motion_raw_list.append(motion_raw)
            sharpness_raw_list.append(sharpness_raw)
            exposure_raw_list.append(exposure_raw)
            jitter_raw_list.append(jitter_raw)
            eventness_raw_list.append(eventness_raw)
            people_raw_list.append(people_raw)
            warm_raw_list.append(warm_raw)
            sat_raw_list.append(sat_raw)
            smooth_motion_raw_list.append(smooth_motion_raw)
    finally:
        cap.release()

    motion_norm = _robust_normalize(motion_raw_list, p_low=10.0, p_high=90.0)
    sharpness_norm = _robust_normalize(sharpness_raw_list, p_low=10.0, p_high=90.0)
    exposure_norm = _robust_normalize(exposure_raw_list, p_low=10.0, p_high=90.0)
    jitter_norm = _robust_normalize(jitter_raw_list, p_low=10.0, p_high=90.0)
    eventness_norm = _robust_normalize(eventness_raw_list, p_low=10.0, p_high=90.0)
    if max(people_raw_list, default=0.0) <= 0.0:
        people_norm = [0.0 for _ in people_raw_list]
    else:
        people_norm = _robust_normalize(people_raw_list, p_low=10.0, p_high=90.0)
    if max(warm_raw_list, default=0.0) <= 0.0:
        warm_norm = [0.0 for _ in warm_raw_list]
    else:
        warm_norm = _robust_normalize(warm_raw_list, p_low=10.0, p_high=90.0)
    if max(sat_raw_list, default=0.0) <= 0.0:
        sat_norm = [0.0 for _ in sat_raw_list]
    else:
        sat_norm = _robust_normalize(sat_raw_list, p_low=10.0, p_high=90.0)
    if max(smooth_motion_raw_list, default=0.0) <= 0.0:
        smooth_motion_norm = [0.0 for _ in smooth_motion_raw_list]
    else:
        smooth_motion_norm = _robust_normalize(smooth_motion_raw_list, p_low=10.0, p_high=90.0)

    w_quality = float(sharpness_weight)
    w_action = float(motion_weight)
    w_comp = float(composition_weight)
    w_people = float(people_weight)
    w_scenic = float(scenic_weight)
    w_cam = float(camera_motion_weight)
    w_sum = w_quality + w_action + w_comp + w_people + w_scenic + w_cam
    if w_sum <= 1e-9:
        w_quality, w_comp, w_action = 0.4, 0.2, 0.4
        w_people, w_scenic, w_cam = 0.0, 0.0, 0.0
        w_sum = 1.0
    w_quality /= w_sum
    w_comp /= w_sum
    w_action /= w_sum
    w_people /= w_sum
    w_scenic /= w_sum
    w_cam /= w_sum

    features: list[SegmentFeatures] = []
    for i, (seg, m_raw, s_raw, p_raw, w_raw, sm_raw) in enumerate(per_segment_raw):
        exposure_raw, jitter_raw, eventness_raw = per_segment_aux[i]

        m_n = float(motion_norm[i])
        s_n = float(sharpness_norm[i])
        exp_n = float(exposure_norm[i])
        jit_n = float(jitter_norm[i])
        evt_n = float(eventness_norm[i])
        p_n = float(people_norm[i]) if people_norm else 0.0
        warm_n = float(warm_norm[i]) if warm_norm else 0.0
        sat_n = float(sat_norm[i]) if sat_norm else 0.0
        sm_n = float(smooth_motion_norm[i]) if smooth_motion_norm else 0.0

        quality_score = float(np.clip(s_n * (1.0 - exp_n), 0.0, 1.0))
        camera_motion_score = float(np.clip(sm_n, 0.0, 1.0))
        composition_score = float(
            np.clip((0.6 * (1.0 - jit_n)) + (0.4 * camera_motion_score), 0.0, 1.0)
        )

        action_base = float(np.clip((0.7 * m_n) + (0.3 * evt_n), 0.0, 1.0))
        action_score = float(np.clip(action_base * (0.5 + 0.5 * composition_score), 0.0, 1.0))

        people_score = float(np.clip(p_n * (0.5 + 0.5 * s_n), 0.0, 1.0))
        scenic_base = float(np.clip((0.6 * warm_n) + (0.4 * sat_n), 0.0, 1.0))
        scenic_score = float(
            np.clip(
                scenic_base
                * (0.6 + 0.4 * (1.0 - m_n))
                * (0.6 + 0.4 * s_n)
                * (0.6 + 0.4 * composition_score),
                0.0,
                1.0,
            )
        )

        interest_score = (
            (w_quality * quality_score)
            + (w_comp * composition_score)
            + (w_action * action_score)
            + (w_people * people_score)
            + (w_scenic * scenic_score)
            + (w_cam * camera_motion_score)
        )
        interest_score = float(np.clip(interest_score, 0.0, 1.0))

        features.append(
            SegmentFeatures(
                segment=seg,
                motion_raw=m_raw,
                sharpness_raw=s_raw,
                people_raw=p_raw,
                warm_raw=w_raw,
                smooth_motion_raw=sm_raw,
                motion_norm=m_n,
                sharpness_norm=s_n,
                people_norm=p_n,
                warm_norm=warm_n,
                smooth_motion_norm=sm_n,
                quality_score=quality_score,
                composition_score=composition_score,
                action_score=action_score,
                people_score=people_score,
                scenic_score=scenic_score,
                camera_motion_score=camera_motion_score,
                interest_score=interest_score,
            )
        )

    return features
