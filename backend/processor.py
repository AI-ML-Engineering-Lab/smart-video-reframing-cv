"""
Smart Video Reframing — processing pipeline.

detect → select → focus. The crop window follows the selected subject for the
whole video; if the subject leaves frame, the camera pans the scene normally
and automatically re-locks when they re-enter (appearance-based re-ID).
"""
from backend.tracker import FocusTracker
from backend.video_io import open_video, create_writer, merge_audio


class SmartReframer:
    def __init__(self, model, crop_ratio=9 / 16):
        self.model = model
        self.crop_ratio = crop_ratio

    def process_video(self, input_path, output_path, selected_box,
                      progress_cb=None):
        cap, props = open_video(input_path)
        orig_w, orig_h = props["width"], props["height"]
        fps = max(int(props["fps"]), 1)
        total = max(props["total_frames"], 1)

        target_h = orig_h
        target_w = int(target_h * self.crop_ratio)
        if target_w > orig_w:  # very wide source: crop height instead
            target_w = orig_w
            target_h = int(orig_w / self.crop_ratio)

        temp_path = output_path + ".temp.mp4"
        out = create_writer(temp_path, fps, target_w, target_h)

        tracker = FocusTracker(selected_box, grace_seconds=2.0, fps=fps)
        timeline = []          # per-frame state for the UI strip
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if tracker.reference_hist is None:
                tracker.bind_reference(frame, selected_box)

            center, state = tracker.step(self.model, frame)

            x0 = int(center - target_w / 2)
            x0 = max(0, min(x0, orig_w - target_w))
            out.write(frame[:, x0:x0 + target_w])
            timeline.append(state)

            frame_idx += 1
            if progress_cb and frame_idx % 30 == 0:
                progress_cb(frame_idx / total)

        cap.release()
        out.release()

        stats = {
            **tracker.stats,
            "frames": frame_idx,
            "search_pct": round(100 * tracker.stats["search_frames"] / frame_idx, 1)
            if frame_idx else 0,
            "state_timeline": timeline,
        }
        merge_audio(input_path, temp_path, output_path)
        return stats
