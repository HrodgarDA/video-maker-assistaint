from __future__ import annotations

from pathlib import Path
from typing import List
from functools import lru_cache

import cv2
import numpy as np

from .ffmpeg_tools import extract_thumbnail
from .selector import SelectedMoment


def _read_frame_at_timestamp(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def _resize_max_width(gray: np.ndarray, max_width: int = 360) -> np.ndarray:
    h, w = gray.shape[:2]
    if w <= max_width:
        return gray
    scale = float(max_width) / float(w)
    new_w = max_width
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


def _exposure_penalty(gray: np.ndarray, t_low: float = 5.0, t_high: float = 250.0) -> float:
    frac_dark = float(np.mean(gray < t_low))
    frac_bright = float(np.mean(gray > t_high))
    return frac_dark + frac_bright


@lru_cache(maxsize=1)
def _get_face_detector() -> cv2.CascadeClassifier | None:
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


def _face_presence(gray: np.ndarray, detector: cv2.CascadeClassifier | None) -> float:
    if detector is None:
        return 0.0
    small = _resize_max_width(gray, max_width=320)
    gray_u8 = np.clip(small, 0, 255).astype(np.uint8)
    faces = detector.detectMultiScale(
        gray_u8,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return 0.0
    frame_area = float(gray_u8.shape[0] * gray_u8.shape[1])
    area = float(np.sum([w * h for (_, _, w, h) in faces]))
    return float(len(faces)) + (3.0 * float(area / frame_area)) if frame_area > 0 else float(len(faces))


def _normalize_list(values: list[float]) -> list[float]:
    if not values:
        return []
    v_min = float(min(values))
    v_max = float(max(values))
    if v_max <= v_min + 1e-9:
        return [0.5 for _ in values]
    return [float((v - v_min) / (v_max - v_min)) for v in values]


def _sample_thumbnail_timestamps(
    start_sec: float,
    end_sec: float,
    n: int = 5,
    offset_sec: float = 0.2,
    tail_margin_sec: float = 0.2,
) -> list[float]:
    if end_sec <= start_sec:
        return [start_sec]

    seg_len = end_sec - start_sec
    if seg_len < (offset_sec + tail_margin_sec):
        return [(start_sec + end_sec) / 2.0]

    safe_start = start_sec + offset_sec
    safe_end = max(safe_start, end_sec - tail_margin_sec)

    if n <= 1:
        return [(safe_start + safe_end) / 2.0]

    ts = np.linspace(safe_start, safe_end, num=n, dtype=np.float64)
    return [float(x) for x in ts]


def _pick_best_thumbnail_timestamp(
    cap: cv2.VideoCapture,
    moment: SelectedMoment,
    face_detector: cv2.CascadeClassifier | None,
    samples: int = 5,
) -> float:
    candidates = _sample_thumbnail_timestamps(moment.start_sec, moment.end_sec, n=samples)
    if moment.thumb_timestamp_sec not in candidates:
        candidates.append(moment.thumb_timestamp_sec)

    sharpness_vals: list[float] = []
    exposure_vals: list[float] = []
    face_vals: list[float] = []
    valid_ts: list[float] = []

    for ts in candidates:
        frame = _read_frame_at_timestamp(cap, ts)
        if frame is None:
            continue
        gray = _to_gray(frame)
        gray = _resize_max_width(gray, max_width=360)

        sharpness_vals.append(_laplacian_variance(gray))
        exposure_vals.append(_exposure_penalty(gray))
        face_vals.append(_face_presence(gray, face_detector))
        valid_ts.append(ts)

    if not valid_ts:
        return moment.thumb_timestamp_sec

    sharp_norm = _normalize_list(sharpness_vals)
    exp_norm = _normalize_list(exposure_vals)
    face_norm = _normalize_list(face_vals) if any(face_vals) else [0.0 for _ in face_vals]

    scores: list[float] = []
    for i in range(len(valid_ts)):
        score = (0.6 * sharp_norm[i]) + (0.3 * face_norm[i]) + (0.1 * (1.0 - exp_norm[i]))
        scores.append(float(score))

    best_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return float(valid_ts[best_idx])


def write_moment_thumbnails(
    master_video_path: Path,
    moments: List[SelectedMoment],
    thumbs_dir: Path,
) -> list[tuple[Path, float]]:
    """
    Extract thumbnails from the master video for the selected moments.

    Notes:
    - We name thumbnails deterministically (moment_01.jpg, moment_02.jpg, ...).
    - We extract from master for UI quality (proxies are intentionally degraded).
    """
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[tuple[Path, float]] = []

    cap = cv2.VideoCapture(str(master_video_path))
    cap_ok = cap.isOpened()
    face_detector = _get_face_detector()

    try:
        for idx, m in enumerate(moments, start=1):
            out_jpg = thumbs_dir / f"moment_{idx:02d}.jpg"
            best_ts = m.thumb_timestamp_sec
            if cap_ok:
                best_ts = _pick_best_thumbnail_timestamp(cap, m, face_detector)
            extract_thumbnail(master_video_path, best_ts, out_jpg)
            out_paths.append((out_jpg, best_ts))
    finally:
        if cap_ok:
            cap.release()

    return out_paths
