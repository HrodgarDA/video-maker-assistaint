from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List

from .paths import make_new_run_paths, RunPaths
from .ffmpeg_tools import probe_video, build_quality_proxy, build_vlm_proxy
from .segments import generate_sliding_segments
from .features import compute_segment_features, SegmentFeatures
from .selector import select_top_moments, SelectedMoment
from .confidence import compute_inspection_confidence
from .thumbnails import write_moment_thumbnails
from .vlm_client import DescriptionInputs, describe_moment_poc
from .models import RunResult, VideoResult, MomentResult
from .report_store import save_video_result, save_run_index


@dataclass(frozen=True)
class PocSettings:
    """
    Minimal POC settings.

    We keep the surface area small so that changing defaults is easy and the run is reproducible.
    """
    window_sec: float = 3.0
    step_sec: float = 1.0
    frames_per_segment: int = 3
    max_moments_per_video: int = 8

    # Proxy parameters
    quality_proxy_height: int = 720
    quality_proxy_fps: int = 12
    vlm_proxy_height: int = 480
    vlm_proxy_fps: int = 5


def _is_video_file(path: Path) -> bool:
    """Simple extension-based filter (POC)."""
    return path.suffix.lower() in {".mp4", ".mov", ".mkv", ".m4v", ".avi"}


def _scan_videos(input_dir: Path) -> List[Path]:
    """
    Recursively scan for video files.

    POC note:
    - We use extension-based filtering; later we can validate via ffprobe.
    """
    videos = [p for p in input_dir.rglob("*") if p.is_file() and _is_video_file(p)]
    return sorted(videos, key=lambda p: p.name.lower())


def _match_selected_moments_to_features(
    all_features: List[SegmentFeatures],
    selected: List[SelectedMoment],
) -> List[SegmentFeatures]:
    """
    Map each SelectedMoment back to its SegmentFeatures.

    Why:
    - We want motion_raw/sharpness_raw for deterministic POC descriptions.
    - The selection uses exact segment boundaries, so (start,end) is a stable key.
    """
    index = {(sf.segment.start_sec, sf.segment.end_sec): sf for sf in all_features}
    out: List[SegmentFeatures] = []
    for m in selected:
        sf = index.get((m.start_sec, m.end_sec))
        if sf is not None:
            out.append(sf)
    return out


def run_batch_analysis(
    input_dir: Path,
    base_runs_dir: Path,
    settings: PocSettings,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> RunResult:
    """
    Run the full POC pipeline on a folder of videos.

    Contract:
    - input_dir: the folder the user selected in Streamlit
    - base_runs_dir: root folder where we create a run subfolder
    - progress_callback: optional hook for UI updates (done, total, message)
    """
    input_dir = input_dir.expanduser().resolve()
    base_runs_dir = base_runs_dir.expanduser().resolve()
    base_runs_dir.mkdir(parents=True, exist_ok=True)

    run_paths: RunPaths = make_new_run_paths(base_runs_dir)
    videos = _scan_videos(input_dir)
    total = len(videos)

    results: List[VideoResult] = []

    for idx, video_path in enumerate(videos, start=1):
        video_stem = video_path.stem

        if progress_callback:
            progress_callback(idx - 1, total, f"[{idx}/{total}] Probing: {video_path.name}")

        meta = probe_video(video_path)

        # --- Proxies (POC: regenerate each run; caching comes later) ---
        if progress_callback:
            progress_callback(idx - 1, total, f"[{idx}/{total}] Building proxies: {video_path.name}")

        proxy_quality = build_quality_proxy(
            video_path=video_path,
            proxy_path=run_paths.proxy_quality_path(video_stem),
            target_height=settings.quality_proxy_height,
            target_fps=settings.quality_proxy_fps,
        )

        proxy_vlm = build_vlm_proxy(
            video_path=video_path,
            proxy_path=run_paths.proxy_vlm_path(video_stem),
            target_height=settings.vlm_proxy_height,
            target_fps=settings.vlm_proxy_fps,
        )

        # --- Sliding-window segments ---
        segments = generate_sliding_segments(
            duration_sec=meta.duration_sec,
            window_sec=settings.window_sec,
            step_sec=settings.step_sec,
        )

        # --- Feature extraction and per-segment interest scoring ---
        if progress_callback:
            progress_callback(idx - 1, total, f"[{idx}/{total}] Computing features: {video_path.name}")

        all_features = compute_segment_features(
            proxy_quality_path=proxy_quality,
            segments=segments,
            frames_per_segment=settings.frames_per_segment,
        )

        # --- Moment selection (deduplicate overlapping windows) ---
        selected = select_top_moments(
            segment_features=all_features,
            max_moments=settings.max_moments_per_video,
        )

        # --- Inspection confidence (derived from our scoring distribution) ---
        conf = compute_inspection_confidence(all_features, selected)

        # --- Thumbnails from master ---
        if progress_callback:
            progress_callback(idx - 1, total, f"[{idx}/{total}] Extracting thumbnails: {video_path.name}")

        thumbs_dir = run_paths.thumbs_dir(video_stem)
        thumb_paths = write_moment_thumbnails(video_path, selected, thumbs_dir)

        # --- Descriptions (POC fallback) ---
        matched = _match_selected_moments_to_features(all_features, selected)

        moments_out: List[MomentResult] = []
        for m_i, m in enumerate(selected):
            sf = matched[m_i] if m_i < len(matched) else None
            desc_inputs = DescriptionInputs(
                motion_raw=sf.motion_raw if sf else 0.0,
                sharpness_raw=sf.sharpness_raw if sf else 0.0,
                interest_score=m.interest_score,
            )
            description = describe_moment_poc(desc_inputs)

            moments_out.append(
                MomentResult(
                    start_sec=m.start_sec,
                    end_sec=m.end_sec,
                    thumb_timestamp_sec=m.thumb_timestamp_sec,
                    thumbnail_path=thumb_paths[m_i],
                    interest_score=m.interest_score,
                    description=description,
                )
            )

        video_result = VideoResult(
            video_path=video_path,
            video_stem=video_stem,
            duration_sec=meta.duration_sec,
            proxy_quality_path=proxy_quality,
            proxy_vlm_path=proxy_vlm,
            inspection_confidence_1_10=conf.score_1_10,
            confidence_reasons=conf.reasons,
            moments=moments_out,
        )

        # Persist per-video result.
        save_video_result(video_result, run_paths.video_result_json_path(video_stem))
        results.append(video_result)

        if progress_callback:
            progress_callback(idx, total, f"[{idx}/{total}] Done: {video_path.name}")

    run_result = RunResult(
        run_id=run_paths.run_id,
        input_dir=input_dir,
        run_dir=run_paths.run_dir,
        videos=results,
    )

    # Persist run index (single file for the UI).
    save_run_index(run_result, run_paths.run_index_json_path())

    return run_result