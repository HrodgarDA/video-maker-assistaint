from __future__ import annotations

from pathlib import Path
from typing import List

from .ffmpeg_tools import extract_thumbnail
from .selector import SelectedMoment


def write_moment_thumbnails(
    master_video_path: Path,
    moments: List[SelectedMoment],
    thumbs_dir: Path,
) -> list[Path]:
    """
    Extract thumbnails from the master video for the selected moments.

    Notes:
    - We name thumbnails deterministically (moment_01.jpg, moment_02.jpg, ...).
    - We extract from master for UI quality (proxies are intentionally degraded).
    """
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    for idx, m in enumerate(moments, start=1):
        out_jpg = thumbs_dir / f"moment_{idx:02d}.jpg"
        extract_thumbnail(master_video_path, m.thumb_timestamp_sec, out_jpg)
        out_paths.append(out_jpg)

    return out_paths