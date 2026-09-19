"""
Real-time Dual Hand Tracking and Finger Counter Web Application.
Tracks both hands, counts extended fingers (0-5 per hand, 0-10 total), and classifies gestures.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from hand_detector import HandDetector
from visualizer import draw_hands_frame

try:
    st.set_page_config(page_title="Dual Hand Finger Counter", layout="wide")
except Exception:
    pass

st.markdown("""
<style>
    .stApp { background-color: #0f141c; color: #f1f5f9; }
    .card { background: #161e2b; border: 1px solid #283548; border-radius: 8px; padding: 14px; text-align: center; }
    .card-val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .card-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector():
    return HandDetector()


detector = get_detector()

st.sidebar.markdown("### Camera Settings")
cam_idx = st.sidebar.selectbox("Select Camera Device", [0, 1, 2], index=0)
conf_val = st.sidebar.slider("Detection Confidence", 0.20, 0.80, 0.35, 0.05)
detector.min_confidence = conf_val

st.title("Dual-Hand Tracking and Finger Counter")
st.caption("Real-time computer vision system tracking Left and Right hands with 0 to 10 finger counts.")

tab1, tab2, tab3, tab4 = st.tabs(["Live Camera Feed", "Image Upload", "Video File", "Sample Dataset"])


def render_metrics(summary, fps):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><div class='card-lbl'>Left Hand</div><div class='card-val'>{summary.get('left_hand_fingers', 0)} / 5</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='card-lbl'>Right Hand</div><div class='card-val'>{summary.get('right_hand_fingers', 0)} / 5</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><div class='card-lbl'>Total Count</div><div class='card-val' style='color:#00e676;'>{summary.get('total_fingers', 0)} / 10</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><div class='card-lbl'>Speed</div><div class='card-val'>{fps:.1f} FPS</div></div>", unsafe_allow_html=True)


# TAB 1: Live Camera Feed
with tab1:
    st.markdown("#### Real-Time Camera Stream")
    st.write("For high-performance desktop window execution at 30+ FPS, run in terminal:")
    st.code("python live_cam.py", language="bash")

    live_active = st.checkbox("Start Live In-Browser Video Stream", value=False, key="hand_live_cb")
    if live_active:
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)

        if not cap.isOpened():
            st.error(f"Could not open camera at device index {cam_idx}. Select another index from the sidebar.")
        else:
            metrics_box = st.empty()
            image_box = st.empty()
            stop_btn = st.button("Stop Camera Feed", key="hand_stop_btn")

            fps = 0.0
            while cap.isOpened() and not stop_btn:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                hands, summary = detector.process(frame, is_mirrored=True)

                elapsed = time.time() - t0
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                with metrics_box.container():
                    render_metrics(summary, fps)

                annotated = draw_hands_frame(frame, hands, summary, fps=fps)
                image_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            cap.release()

# TAB 2: Image Upload
with tab2:
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        up_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"])
    with up_col2:
        path_input = st.text_input("Or local file path:", placeholder="e.g. D:\path\to\hand.jpg")

    target_img = None
    if up_file:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        target_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif path_input.strip() and os.path.exists(path_input.strip()):
        target_img = cv2.imread(path_input.strip())

    if target_img is not None:
        t0 = time.time()
        hands, summary = detector.process(target_img)
        fps = 1.0 / (time.time() - t0)

        render_metrics(summary, fps)
        annotated = draw_hands_frame(target_img, hands, summary, fps=fps)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if hands:
            st.markdown("#### Detected Hands Breakdown")
            table_data = []
            for h in hands:
                table_data.append({
                    "Hand": h["handedness"],
                    "Fingers Counted": f"{h['finger_count']} / 5",
                    "Classified Gesture": h["gesture"],
                    "Confidence": f"{int(h['confidence'] * 100)}%",
                    "Bounding Box": str(h["bbox"])
                })
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

# TAB 3: Video File
with tab3:
    vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if vid_file:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        run_btn = st.button("Process Video")
        if run_btn:
            f_count = 0
            while cap.isOpened() and f_count < 100:
                ret, frame = cap.read()
                if not ret:
                    break
                hands, summary = detector.process(frame)
                annotated = draw_hands_frame(frame, hands, summary, fps=30.0)
                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                f_count += 1
            cap.release()
            st.success("Video processing completed.")

# TAB 4: Sample Benchmark
with tab4:
    sample_path = os.path.join(MODULE_DIR, "data", "sample.jpg")
    if os.path.exists(sample_path):
        frame = cv2.imread(sample_path)
        hands, summary = detector.process(frame)
        render_metrics(summary, 30.0)
        annotated = draw_hands_frame(frame, hands, summary, fps=30.0)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
