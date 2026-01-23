from __future__ import annotations

from pathlib import Path
import streamlit as st

import sys
import subprocess

from src.vmpoc.batch_runner import run_batch_analysis, PocSettings
from src.vmpoc.report_store import load_run_index


# -----------------------------
# Utils
# -----------------------------

def _progress_callback(progress_bar, status_box):
    def cb(done: int, total: int, message: str) -> None:
        if total > 0:
            progress_bar.progress(min(done / total, 1.0))
        status_box.write(message)
    return cb


def _pick_directory_dialog() -> str:
    """
    Native folder picker.
    - macOS: AppleScript (stable with Streamlit)
    - other OS: empty string (manual paste fallback)
    """
    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "Select video folder")'
        try:
            out = subprocess.check_output(
                ["osascript", "-e", script],
                text=True
            ).strip()
            return out or ""
        except Exception:
            return ""
    return ""


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(page_title="Video Moments POC", layout="wide")

st.title("Video Moments POC")
st.caption("Batch analysis: sliding-window moments + thumbnails + inspection confidence (local).")


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("Settings")

    # ---------- 1) Run settings ----------
    st.subheader("💾 I/O Paths")

    # --- App state (source of truth) ---
    if "input_dir_value" not in st.session_state:
        st.session_state["input_dir_value"] = str(Path.home())

    # Apply pending update BEFORE widget creation
    pending = st.session_state.pop("input_dir_pending", None)
    if pending:
        st.session_state["input_dir_value"] = pending

    # --- Widget (UI state) ---
    st.text_input(
        "Input folder (videos)",
        value=st.session_state["input_dir_value"],
        key="input_dir_widget",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        browse_btn = st.button("Browse…", use_container_width=True)
    with col_b:
        home_btn = st.button("Home", use_container_width=True)

    # Sync manual edits → app state
    st.session_state["input_dir_value"] = st.session_state.get(
        "input_dir_widget",
        st.session_state["input_dir_value"],
    )

    # Button actions (schedule update for next rerun)
    if browse_btn:
        chosen = _pick_directory_dialog()
        if chosen:
            st.session_state["input_dir_pending"] = chosen
            st.rerun()
        else:
            st.warning("Folder picker cancelled. Paste the path manually.")

    if home_btn:
        st.session_state["input_dir_pending"] = str(Path.home())
        st.rerun()

    runs_dir_str = st.text_input(
        "Runs output folder",
        value=str(Path.cwd() / "runs")
    )

    st.divider()

    # ---------- 2) Scoring ----------
    st.subheader("💯 Scoring")

    preset = st.selectbox(
        "Preset",
        options=["Custom", "Drone", "Action"],
        index=0,
        help="Presets change the balance between quality, composition and action.",
    )

    if preset == "Drone":
        default_quality, default_composition, default_action = 0.70, 0.70, 0.15
    elif preset == "Action":
        default_quality, default_composition, default_action = 0.60, 0.20, 0.75
    else:
        default_quality, default_composition, default_action = 0.50, 0.50, 0.50

    quality_weight = st.slider("Quality", 0.0, 1.0, default_quality, 0.05)
    composition_weight = st.slider("Composition", 0.0, 1.0, default_composition, 0.05)
    action_weight = st.slider("Action", 0.0, 1.0, default_action, 0.05)

    w_sum = quality_weight + composition_weight + action_weight
    if w_sum <= 1e-9:
        quality_weight, composition_weight, action_weight = 0.4, 0.2, 0.4
        w_sum = 1.0

    quality_weight /= w_sum
    composition_weight /= w_sum
    action_weight /= w_sum

    st.caption(
        f"Normalized → Q={quality_weight:.2f} | C={composition_weight:.2f} | A={action_weight:.2f}"
    )

    st.divider()

    max_moments = st.slider("Max moments per video", 1, 20, 8)

    st.divider()

    # ---------- Advanced ----------
    advanced = st.toggle("Advanced settings", value=False)

    # Defaults
    window_sec = 3.0
    step_sec = 1.0
    bucket_sec = 12.0
    max_per_bucket = 1
    min_gap_sec = 1.25
    iou_threshold = 0.30
    min_score = 0.10
    q_h, q_fps = 720, 12
    v_h, v_fps = 480, 5

    if advanced:
        st.subheader("📽️ Segmentation")
        window_sec = st.number_input("Window (sec)", 1.0, 10.0, 3.0, 0.5)
        step_sec = st.number_input("Step (sec)", 0.25, 5.0, 1.0, 0.25)

        st.divider()

        st.subheader("🔢 Selection")
        bucket_sec = st.number_input("Diversity bucket (sec)", 0.0, 120.0, 12.0, 1.0)
        max_per_bucket = st.slider("Max per bucket", 1, 5, 1)
        min_gap_sec = st.number_input("Min gap (sec)", 0.0, 30.0, 1.25, 0.25)
        iou_threshold = st.slider("Max IoU", 0.0, 0.95, 0.30, 0.05)
        min_score = st.slider("Min score", 0.0, 1.0, 0.10, 0.01)

        st.divider()

        st.subheader("©️ Proxy")
        q_h = st.selectbox("Quality height", [540, 720, 1080], index=1)
        q_fps = st.selectbox("Quality fps", [8, 10, 12, 15], index=2)
        v_h = st.selectbox("VLM height", [360, 480, 540], index=1)
        v_fps = st.selectbox("VLM fps", [3, 5, 8], index=1)

    run_btn = st.button("Analyze folder", type="primary")


# -----------------------------
# Run
# -----------------------------

if run_btn:
    input_dir = Path(st.session_state["input_dir_value"]).expanduser()
    runs_dir = Path(runs_dir_str).expanduser()

    settings = PocSettings(
        window_sec=window_sec,
        step_sec=step_sec,
        max_moments_per_video=max_moments,
        preset=preset,
        quality_weight=quality_weight,
        composition_weight=composition_weight,
        action_weight=action_weight,
        iou_threshold=iou_threshold,
        min_gap_sec=min_gap_sec,
        min_score=min_score,
        bucket_sec=bucket_sec,
        max_per_bucket=max_per_bucket,
        quality_proxy_height=q_h,
        quality_proxy_fps=q_fps,
        vlm_proxy_height=v_h,
        vlm_proxy_fps=v_fps,
    )

    st.subheader("Progress")
    progress_bar = st.progress(0.0)
    status_box = st.empty()

    run_result = run_batch_analysis(
        input_dir=input_dir,
        base_runs_dir=runs_dir,
        settings=settings,
        progress_callback=_progress_callback(progress_bar, status_box),
    )

    st.success(f"Run completed: {run_result.run_id}")
    st.session_state["last_run_dir"] = str(run_result.run_dir)


# -----------------------------
# Results
# -----------------------------

st.divider()
st.subheader("Results")

last_run_dir = st.session_state.get("last_run_dir")

if not last_run_dir:
    st.info("No runs yet. Click 'Analyze folder' to generate a new run.")
else:
    run_dir = Path(last_run_dir)
    index_path = run_dir / "run_index.json"

    if not index_path.exists():
        st.warning("run_index.json not found.")
    else:
        data = load_run_index(index_path)
        st.caption(f"Run ID: {data['run_id']}")

        for v in data.get("videos", []):
            name = Path(v["video_path"]).name
            conf = v["inspection_confidence_1_10"]

            with st.expander(f"{name} | confidence {conf}/10"):
                for m in v.get("moments", []):
                    st.image(
                        m["thumbnail_path"],
                        caption=(
                            f"{m['description']}\n"
                            f"{m['start_sec']:.1f}s → {m['end_sec']:.1f}s | "
                            f"score={m['interest_score']:.2f}"
                        ),
                        use_container_width=True,
                    )