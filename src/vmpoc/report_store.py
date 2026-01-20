from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .models import RunResult, VideoResult


def _to_jsonable(obj: Any) -> Any:
    """
    Convert dataclasses and Path objects into JSON-serializable values.

    Why:
    - Dataclasses can be converted to dict via asdict(), but Path is not JSON-serializable.
    - We want a single consistent serialization layer for the POC.
    """
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_video_result(video_result: VideoResult, out_json_path: Path) -> None:
    """Save a single video's result to JSON."""
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(video_result)
    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_index(run_result: RunResult, out_json_path: Path) -> None:
    """
    Save a run index JSON.

    Why:
    - The UI needs a single entrypoint file to load the whole run quickly.
    - We persist after the batch so results can be re-opened later.
    """
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(run_result)
    out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_run_index(run_index_path: Path) -> dict:
    """
    Load run_index.json as a dict.

    For the POC UI we keep it as dict to avoid rehydrating dataclasses.
    """
    return json.loads(run_index_path.read_text(encoding="utf-8"))