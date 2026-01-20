from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class MomentResult:
    """Final moment object shown in the UI and saved to JSON."""
    start_sec: float
    end_sec: float
    thumb_timestamp_sec: float
    thumbnail_path: Path
    interest_score: float
    description: str


@dataclass(frozen=True)
class VideoResult:
    """Per-video output for the POC run."""
    video_path: Path
    video_stem: str
    duration_sec: float
    proxy_quality_path: Path
    proxy_vlm_path: Path
    inspection_confidence_1_10: int
    confidence_reasons: List[str]
    moments: List[MomentResult]


@dataclass(frozen=True)
class RunResult:
    """Batch output for a folder analysis run."""
    run_id: str
    input_dir: Path
    run_dir: Path
    videos: List[VideoResult]