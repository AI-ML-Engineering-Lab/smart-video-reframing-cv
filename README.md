# Smart Video Reframing — CV

<div align="center">

**Pick a person or animal → the camera follows them for the whole video.**
**Subject leaves frame? The scene pans normally. They return? It re-locks — automatically.**

YOLOv8 · IoU + appearance-based re-identification · Streamlit

</div>

---

Instagram-style "focus mode" for any landscape video: detect every person and
animal on screen, choose one, and get a stable 9:16 vertical crop that tracks
them — with graceful behaviour when they exit and re-enter the scene.

## How it works

```
                 ┌─────────────────────────────────────────────┐
                 │  1. DISCOVERY  (sampled scan)               │
                 │     YOLOv8 → people + animals               │
                 │     cluster by IoU → identity tracks        │
                 │     presence % + timeline per subject       │
                 └──────────────────┬──────────────────────────┘
                                    │  you pick a subject
                 ┌──────────────────▼──────────────────────────┐
                 │  2. FOCUS MODE  (state machine)             │
                 │                                             │
                 │   LOCKED ──── subject visible               │
                 │      │  crop follows (EMA-smoothed,         │
                 │      │  dead-zone kills jitter)             │
                 │      ▼                                      │
                 │   SEARCHING ── subject gone > grace period  │
                 │      │  slow cinematic pan of the scene     │
                 │      │  (normal viewing, no frozen frame)   │
                 │      ▼                                      │
                 │   RE-ENTRY ─── re-ID by HSV appearance      │
                 │      │  similarity vs. locked reference     │
                 │      └───── smooth glide back to RELOCKED   │
                 └──────────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────────────┐
                 │  3. RENDER                                  │
                 │     vertical crop + original audio merged   │
                 │     focus-timeline strip (locked/pan/relock)│
                 └─────────────────────────────────────────────┘
```

### The hard part: re-identification
Simple trackers lose the subject the moment they leave frame. This one keeps a
**HSV colour-histogram signature** captured at lock time, so when someone
re-enters from the other side of the scene, `cv2.compareHist` decides *"that's
the same person"* — and the camera glides back instead of jumping or tracking
a stranger. A decoy in different clothing is rejected (`reid_hist=0.35`
threshold, verified in unit tests).

## Run it

```bash
git clone https://github.com/AI-ML-Engineering-Lab/smart-video-reframing-cv.git
cd smart-video-reframing-cv
pip install -r requirements.txt
streamlit run app.py
```

Upload a video → review detected subjects with presence timelines →
**Focus on this** → 🎬 Generate → download the focused vertical cut.

## UI

- Subject cards with **enhanced thumbnails** (Lanczos upscale + bilateral denoise + unsharp mask)
- **Presence timeline** per subject: green segments = frames where they're on screen
- **Animal mode** toggle: also detect cats, dogs, birds, horses… (10 COCO classes)
- **Focus timeline** on the result: green = following, grey = scene pan, blue = re-lock

## Project structure

```
backend/
  subject_discovery.py   # multi-class detection, IoU track clustering, presence timelines
  tracker.py             # FocusTracker state machine (LOCKED ⇄ SEARCHING) + HSV re-ID
  processor.py           # crop-window rendering loop, stats, focus timeline
  video_io.py            # OpenCV I/O + audio merge (MoviePy)
  smoother.py            # exponential smoothing helper
app.py                   # Streamlit UI
docs/                    # design notes & failure cases
```

## Verified behaviour

On a 768×432 test clip with two people entering/leaving at different times:

| Metric | Result |
|---|---|
| Frames processed | 1,394 → 1,394 (9:16 crop, audio preserved) |
| Subject locked | 68.9% of frames |
| Scene pan (subject absent) | 31.1% |
| Automatic re-locks | 6 |

## Known limits

- Histogram re-ID can confuse people wearing similar colours (see `docs/failure_cases.md`)
- Detection sampling during discovery is every 15th frame — very brief appearances (<0.5s) may not be listed
- Single-subject focus; multi-subject "group framing" is future work

## License

MIT
