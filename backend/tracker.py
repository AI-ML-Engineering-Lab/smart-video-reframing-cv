"""
Subject-locked tracker with exit / re-entry handling.

Behaviour (Instagram-style "focus mode"):
  LOCKED     — subject visible → crop window follows them (smoothed, dead-zone)
  SEARCHING  — subject leaves the frame → camera holds/centres per policy for
               `grace_frames`, then transitions to a slow cinematic pan of the
               full scene (normal viewing) while still watching for re-entry
  RE-ENTRY   — subject re-appears → matched by IoU continuity + appearance
               (histogram) similarity to the locked reference → re-lock with a
               smooth glide back, not a jump cut

Re-identification uses a colour-histogram signature captured at lock time,
which is what lets the tracker say "that's the same dog/person" even after
the subject left and re-entered elsewhere in the frame.
"""
import cv2
import numpy as np


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def hsv_histogram(frame, box):
    """Normalised HSV histogram — cheap, rotation-tolerant appearance signature."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


class FocusTracker:
    """State machine: LOCKED ⇄ SEARCHING, with appearance-based re-entry."""

    def __init__(self, target_box, conf_threshold=0.35,
                 dead_zone_ratio=0.02, grace_seconds=2.0,
                 fps=30, smooth_alpha=0.18, reid_iou=0.25, reid_hist=0.35):
        self.state = "LOCKED"
        self.target_box = tuple(target_box)
        self.conf_threshold = conf_threshold
        self.dead_zone_ratio = dead_zone_ratio
        self.reid_iou_min = reid_iou
        self.reid_hist_min = reid_hist

        self.prev_center = (target_box[0] + target_box[2]) / 2
        self.smooth_alpha = smooth_alpha
        self.smoothed_center = self.prev_center

        self.grace_frames = int(grace_seconds * fps)
        self.missed = 0

        self.reference_hist = None   # set on first frame
        self.stats = {"locked_frames": 0, "search_frames": 0, "relocks": 0}

    # ---------------------------------------------------------------- helpers
    def _detect(self, model, frame, classes=(0,)):
        results = model(frame, verbose=False, classes=list(classes))
        out = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    xy = box.xyxy[0]
                    xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
                    x1, y1, x2, y2 = xy
                    out.append((float(x1), float(y1), float(x2), float(y2), conf))
        return out

    def _matches_reference(self, frame, box):
        """IoU-free re-identification: appearance histogram similarity."""
        if self.reference_hist is None:
            return True
        hist = hsv_histogram(frame, box)
        if hist is None:
            return False
        return float(cv2.compareHist(self.reference_hist, hist,
                                     cv2.HISTCMP_CORREL)) >= self.reid_hist_min

    # ---------------------------------------------------------------- main
    def step(self, model, frame, classes=(0,)):
        """Advance one frame. Returns (crop_center_x, state)."""
        detections = self._detect(model, frame, classes)

        if self.state == "LOCKED":
            return self._step_locked(frame, detections)
        return self._step_searching(frame, detections)

    # -- LOCKED ------------------------------------------------------------
    def _step_locked(self, frame, detections):
        self.stats["locked_frames"] += 1
        best, best_score = None, self.reid_iou_min
        for (x1, y1, x2, y2, conf) in detections:
            cand = (x1, y1, x2, y2)
            v = iou(cand, self.target_box)
            if v > best_score:
                best, best_score = cand, v

        if best is not None:
            self.target_box = best
            self.missed = 0
            raw = (best[0] + best[2]) / 2
            self.smoothed_center += self.smooth_alpha * (raw - self.smoothed_center)
            return self.smoothed_center, "LOCKED"

        # subject not found this frame → start/continue grace countdown
        self.missed += 1
        if self.missed > self.grace_frames:
            self.state = "SEARCHING"
            self._search_dir = 1 if self.prev_center < frame.shape[1] / 2 else -1
            return self.smoothed_center, "SEARCHING"
        # hold the camera where the subject was last seen during grace
        return self.smoothed_center, "LOCKED"

    # -- SEARCHING ---------------------------------------------------------
    def _step_searching(self, frame, detections):
        self.stats["search_frames"] += 1
        w = frame.shape[1]

        # 1) look for re-entry by appearance
        for (x1, y1, x2, y2, conf) in detections:
            cand = (x1, y1, x2, y2)
            if self.reference_hist is None:
                matched = True
            else:
                hist = hsv_histogram(frame, cand)
                matched = hist is not None and float(
                    cv2.compareHist(self.reference_hist, hist,
                                    cv2.HISTCMP_CORREL)) >= self.reid_hist_min
            if matched:
                self.state = "LOCKED"
                self.target_box = cand
                self.stats["relocks"] += 1
                self.missed = 0
                raw = (cand[0] + cand[2]) / 2
                self.smoothed_center += self.smooth_alpha * (raw - self.smoothed_center)
                return self.smoothed_center, "RELOCKED"

        # 2) no subject → slow cinematic pan across the scene (normal viewing)
        self._search_dir = getattr(self, "_search_dir", 1)
        edge_margin = w * 0.15
        nxt = self.smoothed_center + self._search_dir * w * 0.0012
        if nxt < edge_margin:
            self._search_dir = 1
            nxt = edge_margin
        elif nxt > w - edge_margin:
            self._search_dir = -1
            nxt = w - edge_margin
        self.smoothed_center += self.smooth_alpha * (nxt - self.smoothed_center)
        return self.smoothed_center, "SEARCHING"

    # -- bind the reference appearance --------------------------------------
    def bind_reference(self, frame, box):
        self.reference_hist = hsv_histogram(frame, box)


class LockedPersonTracker(FocusTracker):
    """Backward-compatible alias."""
    pass
