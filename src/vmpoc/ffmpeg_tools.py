from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoMeta:
    """Minimal metadata we need to drive segmentation and processing."""
    video_path: Path
    duration_sec: float
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    has_audio: bool


def _run_command(cmd: list[str]) -> None:
    """
    Run a subprocess command with robust error reporting.

    Why:
    - ffmpeg/ffprobe failures are common (codec issues, corrupted files, etc.).
    - We want the caller to receive actionable stderr output.
    """
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        # Typically means ffmpeg/ffprobe is not installed or not in PATH.
        raise RuntimeError(
            f"Command not found: {cmd[0]}. "
            "Ensure ffmpeg/ffprobe are installed and available in your PATH."
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{stderr}") from e


def probe_video(video_path: Path) -> VideoMeta:
    """
    Inspect a video file using ffprobe and return minimal metadata.

    What we use it for:
    - duration_sec: to generate sliding-window segments correctly
    - has_audio: to decide if we can/should compute audio-related features (later)
    - fps/size: for logging and potential heuristics

    Note:
    - fps from ffprobe is a "nominal" stream rate; real CFR/VFR handling is complex.
      For this POC, we use it mainly for informational purposes.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]

    raw = subprocess.check_output(cmd)
    info = json.loads(raw.decode("utf-8", errors="ignore"))

    duration_sec = float(info["format"]["duration"])

    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    has_audio = len(audio_streams) > 0

    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    if video_streams:
        vs = video_streams[0]
        width = int(vs.get("width")) if vs.get("width") is not None else None
        height = int(vs.get("height")) if vs.get("height") is not None else None

        # r_frame_rate is often like "30000/1001"
        rfr = vs.get("r_frame_rate")
        if isinstance(rfr, str) and "/" in rfr:
            num, den = rfr.split("/", 1)
            try:
                fps = float(num) / float(den)
            except Exception:
                fps = None

    return VideoMeta(
        video_path=video_path,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        has_audio=has_audio,
    )


def build_quality_proxy(
    video_path: Path,
    proxy_path: Path,
    target_height: int = 720,
    target_fps: int = 12,
    crf: int = 23,
) -> Path:
    """
    Build the 'quality proxy' for feature extraction (motion/blur/stability).

    Design choices:
    - scale=-2:target_height keeps aspect ratio and makes width divisible by 2.
    - fps=target_fps reduces frame count for faster OpenCV processing.
    - yuv420p enforces 8-bit SDR output for broad compatibility.
    - H.264 (libx264) gives good speed/size tradeoffs for proxies.
    - audio is removed (-an) because this POC does not use audio signals.

    Output:
    - proxy_path is a lightweight mp4 used by OpenCV for analysis.
    """
    proxy_path.parent.mkdir(parents=True, exist_ok=True)

    vf = f"scale=-2:{target_height},fps={target_fps}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(proxy_path),
    ]
    _run_command(cmd)
    return proxy_path


def build_vlm_proxy(
    video_path: Path,
    proxy_path: Path,
    target_height: int = 480,
    target_fps: int = 5,
    crf: int = 28,
) -> Path:
    """
    Build the 'VLM proxy' for semantic descriptions (cheap input to a VLM).

    Design choices:
    - 480p + 5fps dramatically reduces decoding and downstream compute.
    - We keep the same basic encoding setup as quality proxy for simplicity.
    - We can afford a higher CRF (more compression) because we only need
      coarse semantics, not pixel-perfect fidelity.

    Output:
    - proxy_path is used to extract representative frames for the VLM.
    """
    proxy_path.parent.mkdir(parents=True, exist_ok=True)

    vf = f"scale=-2:{target_height},fps={target_fps}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        str(proxy_path),
    ]
    _run_command(cmd)
    return proxy_path


def extract_thumbnail(
    video_path: Path,
    timestamp_sec: float,
    out_jpg_path: Path,
) -> Path:
    """
    Extract a single frame from the *master* video at a given timestamp.

    Why extract from the master:
    - Proxies are intentionally degraded.
    - Thumbnails should look good in the UI and be faithful to the original file.

    Notes:
    - -ss before -i performs a fast seek in many cases (good for batch).
    - We save JPEG with quality controlled by -q:v (lower is better).
    """
    out_jpg_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_jpg_path),
    ]
    _run_command(cmd)
    return out_jpg_path