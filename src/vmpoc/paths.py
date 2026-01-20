from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Centralized paths for a single run output directory."""
    run_id: str
    run_dir: Path

    def videos_dir(self) -> Path:
        return self.run_dir / "videos"

    def video_dir(self, video_stem: str) -> Path:
        return self.videos_dir() / video_stem

    def proxy_quality_path(self, video_stem: str) -> Path:
        return self.video_dir(video_stem) / "proxy_quality.mp4"

    def proxy_vlm_path(self, video_stem: str) -> Path:
        return self.video_dir(video_stem) / "proxy_vlm.mp4"

    def thumbs_dir(self, video_stem: str) -> Path:
        return self.video_dir(video_stem) / "thumbs"

    def video_result_json_path(self, video_stem: str) -> Path:
        return self.video_dir(video_stem) / "result.json"

    def run_index_json_path(self) -> Path:
        return self.run_dir / "run_index.json"


def make_new_run_paths(base_runs_dir: Path) -> RunPaths:
    """
    Create a timestamp-based run directory.

    Why:
    - Keeps runs isolated and reproducible.
    - Makes it easy to re-open a past run from the UI.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_runs_dir / run_id
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    return RunPaths(run_id=run_id, run_dir=run_dir)