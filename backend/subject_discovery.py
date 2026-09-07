"""
Subject discovery — multi-class detection (people + animals) with
per-subject presence timelines.

Every detected subject gets a thumbnail + a frame-by-frame presence record
so the UI can show "on screen 74% of the video" — and so the tracker knows
what to do when the subject leaves and returns.
"""
import cv2

# COCO ids for "subjects a user might follow"
PERSON_CLASS = 0
ANIMAL_CLASSES = {
    14: "Bird", 15: "Cat", 16: "Dog", 17: "Horse", 18: "Sheep",
    19: "Cow", 20: "Elephant", 21: "Bear", 22: "Zebra", 23: "Giraffe",
}
CLASS_LABELS = {0: "Person", **ANIMAL_CLASSES}


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


# ----------------- IMAGE ENHANCEMENT (thumbnails) -----------------
def enhance_thumbnail(crop, target_size=256):
    """Classical upscale + edge-preserving denoise + unsharp mask."""
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    h, w, _ = crop.shape
    if h < 10 or w < 10:
        return crop
    scale = target_size / max(h, w)
    crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LANCZOS4)
    crop = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
    gaussian = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(crop, 1.5, gaussian, -0.5, 0)


# ----------------- SUBJECT DISCOVERY -----------------
def discover_subjects(model, video_path, sample_rate=15, conf_threshold=0.35,
                      max_subjects=8, classes=(PERSON_CLASS,), progress_cb=None):
    """
    Scan the video at a sampling interval, cluster detections into identity
    tracks, and return subject cards with:
      - thumbnail + class label + confidence
      - presence_pct  (share of sampled frames in which they appear)
      - presence_timeline (list of 0/1 per sampled frame, for the UI strip)
      - first/last frame indices
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    tracks = []
    frame_id = 0
    n_samples = 0
    class_ids = list(classes)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % sample_rate == 0:
            n_samples += 1
            results = model(frame, verbose=False, classes=class_ids)
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    curr = (x1, y1, x2, y2)

                    best, best_iou = None, 0.35
                    for t in tracks:
                        v = iou(t["box"], curr)
                        if v > best_iou:
                            best, best_iou = t, v
                    if best is not None:
                        best["box"] = curr
                        best["hits"] += 1
                        best["area_sum"] += (x2 - x1) * (y2 - y1)
                        best["timeline"].append(1)
                        best["last_frame"] = frame_id
                        if conf > best["conf"] and frame_id - best["thumb_frame"] > sample_rate * 3:
                            best["conf"] = conf
                            best["thumbnail"] = enhance_thumbnail(frame[y1:y2, x1:x2])
                            best["thumb_frame"] = frame_id
                    else:
                        tracks.append({
                            "box": curr, "cls": cls,
                            "class_label": CLASS_LABELS.get(cls, "Subject"),
                            "conf": conf,
                            "hits": 1,
                            "area_sum": (x2 - x1) * (y2 - y1),
                            "thumbnail": enhance_thumbnail(frame[y1:y2, x1:x2]),
                            "thumb_frame": frame_id,
                            "first_frame": frame_id,
                            "last_frame": frame_id,
                            "timeline": [0] * (n_samples - 1) + [1],
                        })
            # backfill 0s for tracks not seen this sample
            for t in tracks:
                if len(t["timeline"]) < n_samples:
                    t["timeline"].append(0)
        frame_id += 1
        if progress_cb and total and frame_id % (sample_rate * 10) == 0:
            progress_cb(min(frame_id / total, 1.0))
    cap.release()

    subjects = []
    for t in tracks:
        if t["hits"] < 3:               # require ≥3 sightings to be selectable
            continue
        presence = sum(t["timeline"]) / max(len(t["timeline"]), 1)
        t["presence_pct"] = round(presence * 100)
        t["avg_area"] = t["area_sum"] / t["hits"]
        t["score"] = presence * 0.6 + min(t["avg_area"] / 100_000, 1.0) * 0.4
        subjects.append(t)

    subjects.sort(key=lambda s: s["score"], reverse=True)
    return subjects[:max_subjects], n_samples
