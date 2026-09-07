import os
import tempfile

import streamlit as st
from ultralytics import YOLO

from backend.processor import SmartReframer
from backend.subject_discovery import discover_subjects

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Video Reframing · CV", layout="centered")

st.markdown("""
<style>
body{background-color:#0b0f17;color:#e8edf7}
.subject-card{border-radius:14px;padding:10px;background:#161b28;
  transition:transform .2s ease,box-shadow .2s ease;border:1px solid #232b3d}
.subject-card:hover{transform:scale(1.03);box-shadow:0 10px 30px rgba(0,0,0,.4)}
.state-strip{display:flex;height:26px;border-radius:8px;overflow:hidden;margin-top:4px}
.seg-LOCKED{background:#2ecf7f;flex:1}
.seg-SEARCHING{background:#4a5568;flex:1}
.seg-RELOCKED{background:#4f8cff;flex:1}
.small{font-size:12px;color:#9aa7bd}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Smart Video Reframing")
st.caption("Detect people & animals → pick one → the camera follows them. "
           "If they leave frame, the scene pans normally and re-locks when they return.")

DETECT_ANIMALS = st.checkbox("🐾 Also detect animals (cats, dogs, birds…)", value=False)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()
reframer = SmartReframer(model)

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()

    with st.expander("▶ Original video", expanded=False):
        st.video(uploaded)

    st.subheader("Detected subjects")
    with st.spinner("Scanning video for people & animals…"):
        subjects, n_samples = discover_subjects(
            model, tfile.name,
            sample_rate=15,
            classes=(0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23) if DETECT_ANIMALS else (0,),
        )

    if not subjects:
        st.warning("No stable subjects detected.")
        os.remove(tfile.name)
        st.stop()

    # subject cards with presence timeline
    per_row = 3
    for row_start in range(0, len(subjects), per_row):
        cols = st.columns(per_row)
        for i in range(row_start, min(row_start + per_row, len(subjects))):
            s = subjects[i]
            with cols[i % per_row]:
                st.markdown('<div class="subject-card">', unsafe_allow_html=True)
                st.image(s["thumbnail"], use_container_width=True)
                label = f"**{s['class_label']}** · {s['conf']:.2f} conf"
                st.markdown(label)
                st.markdown(
                    f'<div class="state-strip">'
                    + "".join(f'<div class="seg-{"LOCKED" if v else "SEARCHING"}"></div>'
                              for v in s["timeline"])
                    + "</div>", unsafe_allow_html=True)
                st.markdown(f'<div class="small">present in {s["presence_pct"]}% of sampled frames'
                            f' (frames {s["first_frame"]}–{s["last_frame"]})</div>',
                            unsafe_allow_html=True)
                if st.button("Focus on this", key=f"sel_{i}"):
                    st.session_state.selected_idx = i
                st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.selected_idx is not None:
        sel = subjects[st.session_state.selected_idx]
        st.success(f"Focusing {sel['class_label']} (present {sel['presence_pct']}% of the video — "
                   f"the camera will hold and pan normally while they're off-screen, then re-lock on return)")

        if st.button("🎬 Generate focused vertical video"):
            output_path = "focused_short.mp4"
            progress = st.progress(0.0)
            status = st.empty()

            def cb(p):
                progress.progress(min(p, 1.0))
                status.text(f"processing… {p*100:.0f}%")

            stats = reframer.process_video(
                tfile.name, output_path, sel["box"], progress_cb=cb)
            progress.progress(1.0)

            # state strip of the run
            st.markdown("**Focus timeline** (green = following subject, blue = re-locked, grey = scene pan)")
            st.markdown(
                '<div class="state-strip">'
                + "".join(f'<div class="seg-{"SEARCHING" if v == "SEARCHING" else "RELOCKED" if v == "RELOCKED" else "LOCKED"}"></div>'
                          for v in stats["state_timeline"][:: max(1, len(stats["state_timeline"]) // 120)])
                + "</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="small">locked {100 - stats["search_pct"]:.0f}% · '
                        f'scene-pan {stats["search_pct"]}% · re-locks: {stats["relocks"]}</div>',
                        unsafe_allow_html=True)

            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("⬇ Download focused video", f, file_name="focused_short.mp4")

    os.remove(tfile.name)
