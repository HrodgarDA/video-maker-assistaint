from __future__ import annotations

from pathlib import Path
import streamlit as st

from src.vmpoc.batch_runner import run_batch_analysis, PocSettings
from src.vmpoc.report_store import load_run_index


def _progress_callback(progress_bar, status_box):
    def cb(done: int, total: int, message: str) -> None:
        if total > 0:
            progress_bar.progress(min(done / total, 1.0))
        status_box.write(message)
    return cb


st.set_page_config(page_title="Video Moments POC", layout="wide")

st.title("Video Moments POC")
st.caption("Batch analysis: sliding-window moments + thumbnails + inspection confidence (local).")

with st.sidebar:
    st.header("Run settings")

    input_dir_str = st.text_input("Input folder path (videos)", value=str(Path.home()))
    runs_dir_str = st.text_input("Runs output folder", value=str(Path.cwd() / "runs"))

    st.divider()
    st.subheader("Segmentation")
    window_sec = st.number_input("Window (sec)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    step_sec = st.number_input("Step (sec)", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    max_moments = st.slider("Max moments per video", min_value=1, max_value=20, value=8)

    st.divider()
    st.subheader("Proxy")
    st.caption("Quality proxy drives scoring; VLM proxy is prepared for later.")
    q_h = st.selectbox("Quality proxy height", options=[540, 720, 1080], index=1)
    q_fps = st.selectbox("Quality proxy fps", options=[8, 10, 12, 15], index=2)
    v_h = st.selectbox("VLM proxy height", options=[360, 480, 540], index=1)
    v_fps = st.selectbox("VLM proxy fps", options=[3, 5, 8], index=1)

    run_btn = st.button("Analyze folder", type="primary")


if run_btn:
    input_dir = Path(input_dir_str).expanduser()
    runs_dir = Path(runs_dir_str).expanduser()

    settings = PocSettings(
        window_sec=float(window_sec),
        step_sec=float(step_sec),
        max_moments_per_video=int(max_moments),
        quality_proxy_height=int(q_h),
        quality_proxy_fps=int(q_fps),
        vlm_proxy_height=int(v_h),
        vlm_proxy_fps=int(v_fps),
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


st.divider()
st.subheader("Results")

last_run_dir = st.session_state.get("last_run_dir", "")
run_dir_str = st.text_input("Load an existing run (run folder path)", value=last_run_dir)

if run_dir_str:
    run_dir = Path(run_dir_str).expanduser()
    index_path = run_dir / "run_index.json"

    if index_path.exists():
        data = load_run_index(index_path)
        st.caption(f"Run ID: {data['run_id']} | Input: {data['input_dir']}")

        videos = data.get("videos", [])
        st.write(f"Videos analyzed: {len(videos)}")

        for v in videos:
            video_name = Path(v["video_path"]).name
            conf = v["inspection_confidence_1_10"]

            with st.expander(f"{video_name}  |  inspection_confidence: {conf}/10", expanded=False):
                st.write("Confidence reasons:")
                for r in v.get("confidence_reasons", [])[:4]:
                    st.write(f"- {r}")

                moments = v.get("moments", [])
                if not moments:
                    st.info("No moments selected for this video.")
                    continue

                # Show thumbnails in a grid
                cols = st.columns(min(4, len(moments)))
                for idx, m in enumerate(moments):
                    col = cols[idx % len(cols)]
                    thumb_path = m["thumbnail_path"]
                    caption = (
                        f"{m['description']}\n"
                        f"{m['start_sec']:.1f}s → {m['end_sec']:.1f}s | score={m['interest_score']:.2f}"
                    )
                    col.image(thumb_path, caption=caption, use_container_width=True)
    else:
        st.warning("run_index.json not found in the provided run folder.")