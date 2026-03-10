"""
Real-Time Video Viewer — Steps 2–7
ROI selection → MOG2 background subtraction → findContours →
shape + intensity + rim-score features → cross-frame tracking →
rule-based classification → CSV logging (one row per track)
"""

import cv2
import sys
import csv
import os
import numpy as np
from pathlib import Path


roi_selecting   = False
roi_start       = None
roi_end         = None
roi_complete    = False
roi_params      = None


class ContourParams:
    min_area        = 400
    max_area        = 5000
    min_circularity = 0.20
    max_circularity = 1.00


contour_params  = ContourParams()
tuning_mode     = False
display_zoom    = 2.0
INFO_BAR_HEIGHT = 100

RIM_THICKNESS_PX = 3

# ── CLAHE settings ────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

# ── Tracker settings ──────────────────────────────────────────────────────────
MAX_MATCH_DIST    = 50    # ← tune: max pixel distance to associate detection→track
MAX_AREA_RATIO    = 2.5   # ← tune: max area ratio between detection and track
MAX_MISSING_FRAMES = 8    # ← tune: frames without a match before track is closed

# ── Classifier thresholds ─────────────────────────────────────────────────────
CLS_MIN_CIRCULARITY = 0.45   # ← tune: minimum avg circularity for "good" droplet
CLS_MIN_RIM_SCORE   = 5.0    # ← tune: minimum avg rim_score for confirmed droplet
CLS_MIN_AREA        = 300    # ← tune: minimum avg area_px for confirmed droplet


# ── helpers ───────────────────────────────────────────────────────────────────

def circularity(area: float, perimeter: float) -> float:
    """4πA/P², clamped to [0, 1]."""
    if perimeter < 1e-6:
        return 0.0
    return min(1.0, (4 * np.pi * area) / (perimeter ** 2))


def compute_features(cnt, gray_roi: np.ndarray) -> dict:
    h, w = gray_roi.shape[:2]

    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, closed=True)
    circ  = circularity(area, perim)
    eq_d  = np.sqrt(4 * area / np.pi)

    mask_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_filled, [cnt], -1, 255, thickness=cv2.FILLED)

    mean_val, std_val  = cv2.meanStdDev(gray_roi, mask=mask_filled)
    mean_intensity     = float(mean_val[0][0])
    std_intensity      = float(std_val[0][0])

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * RIM_THICKNESS_PX + 1, 2 * RIM_THICKNESS_PX + 1)
    )
    mask_interior = cv2.erode(mask_filled, k, iterations=1)
    mask_rim      = cv2.subtract(mask_filled, mask_interior)

    rim_pixels      = gray_roi[mask_rim      > 0]
    interior_pixels = gray_roi[mask_interior > 0]

    if rim_pixels.size > 0 and interior_pixels.size > 0:
        rim_mean      = float(rim_pixels.mean())
        interior_mean = float(interior_pixels.mean())
        rim_score     = rim_mean - interior_mean
    else:
        rim_mean      = mean_intensity
        interior_mean = mean_intensity
        rim_score     = 0.0

    return {
        "area_px":        area,
        "perimeter_px":   perim,
        "eq_diameter_px": eq_d,
        "circularity":    circ,
        "mean_intensity": mean_intensity,
        "std_intensity":  std_intensity,
        "rim_mean":       rim_mean,
        "interior_mean":  interior_mean,
        "rim_score":      rim_score,
    }


# ── classifier ────────────────────────────────────────────────────────────────

def classify_track(avg: dict) -> str:
    """
    Rule-based label from averaged track features.
    Returns 'droplet' or 'noise'.
    Tune the three threshold constants at the top of the file.
    """
    if (avg["circularity"]    >= CLS_MIN_CIRCULARITY and
        avg["rim_score"]      >= CLS_MIN_RIM_SCORE   and
        avg["area_px"]        >= CLS_MIN_AREA):
        return "droplet"
    return "noise"


# ── tracker ───────────────────────────────────────────────────────────────────

class Track:
    _id_counter = 0

    def __init__(self, cx, cy, feats, frame_num):
        Track._id_counter += 1
        self.track_id      = Track._id_counter
        self.cx            = cx
        self.cy            = cy
        self.area          = feats["area_px"]
        self.missing       = 0
        self.alive         = True
        self.frame_start   = frame_num
        self.frame_end     = frame_num
        self.snapshots     = [feats.copy()]   # one dict per matched frame

    def update(self, cx, cy, feats, frame_num):
        self.cx         = cx
        self.cy         = cy
        self.area       = feats["area_px"]
        self.missing    = 0
        self.frame_end  = frame_num
        self.snapshots.append(feats.copy())

    def averaged_features(self) -> dict:
        keys = self.snapshots[0].keys()
        return {k: float(np.mean([s[k] for s in self.snapshots])) for k in keys}

    @property
    def lifespan(self):
        return self.frame_end - self.frame_start + 1


class DropletTracker:
    def __init__(self):
        self.tracks      = []   # active tracks
        self.closed      = []   # completed tracks ready to log

    def reset(self):
        self.tracks  = []
        self.closed  = []
        Track._id_counter = 0

    def update(self, detections: list, frame_num: int):
        """
        Match detections to existing tracks (nearest centre + area gate).
        Unmatched detections → new tracks.
        Tracks missing too long → closed.
        """
        matched_track_ids = set()
        matched_det_ids   = set()

        # build distance matrix
        for t in self.tracks:
            best_dist = MAX_MATCH_DIST
            best_d    = None
            best_di   = None
            for di, d in enumerate(detections):
                if di in matched_det_ids:
                    continue
                dist = np.hypot(d["cx"] - t.cx, d["cy"] - t.cy)
                area_ratio = max(d["area_px"], t.area) / max(min(d["area_px"], t.area), 1)
                if dist < best_dist and area_ratio < MAX_AREA_RATIO:
                    best_dist = dist
                    best_d    = d
                    best_di   = di

            if best_d is not None:
                t.update(best_d["cx"], best_d["cy"], best_d, frame_num)
                matched_track_ids.add(t.track_id)
                matched_det_ids.add(best_di)

        # unmatched detections → new tracks
        for di, d in enumerate(detections):
            if di not in matched_det_ids:
                self.tracks.append(Track(d["cx"], d["cy"], d, frame_num))

        # age unmatched tracks; close if too old
        still_alive = []
        for t in self.tracks:
            if t.track_id not in matched_track_ids:
                t.missing += 1
            if t.missing > MAX_MISSING_FRAMES:
                t.alive = False
                self.closed.append(t)
            else:
                still_alive.append(t)
        self.tracks = still_alive

    def flush_closed(self) -> list:
        """Return and clear the list of newly closed tracks."""
        out = self.closed[:]
        self.closed = []
        return out

    def close_all(self):
        """Force-close all remaining active tracks (call at end of video)."""
        for t in self.tracks:
            t.alive = False
            self.closed.append(t)
        self.tracks = []


# ── CSV writer ────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "track_id", "label",
    "frame_start", "frame_end", "lifespan_frames",
    "avg_area_px", "avg_eq_diameter_px", "avg_circularity",
    "avg_mean_intensity", "avg_std_intensity",
    "avg_rim_score", "avg_rim_mean", "avg_interior_mean",
]

def write_tracks_to_csv(csv_writer, tracks: list):
    for t in tracks:
        avg   = t.averaged_features()
        label = classify_track(avg)
        csv_writer.writerow({
            "track_id":           t.track_id,
            "label":              label,
            "frame_start":        t.frame_start,
            "frame_end":          t.frame_end,
            "lifespan_frames":    t.lifespan,
            "avg_area_px":        round(avg["area_px"],        1),
            "avg_eq_diameter_px": round(avg["eq_diameter_px"], 2),
            "avg_circularity":    round(avg["circularity"],    3),
            "avg_mean_intensity": round(avg["mean_intensity"], 1),
            "avg_std_intensity":  round(avg["std_intensity"],  1),
            "avg_rim_score":      round(avg["rim_score"],      2),
            "avg_rim_mean":       round(avg["rim_mean"],       1),
            "avg_interior_mean":  round(avg["interior_mean"],  1),
        })


# ── mouse callback ────────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    global roi_selecting, roi_start, roi_end, roi_complete, roi_params

    if not roi_selecting:
        return

    y_adj = y - INFO_BAR_HEIGHT

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start    = (x, y_adj)
        roi_end      = None
        roi_complete = False
        print(f"ROI start: ({x}, {y_adj})")

    elif event == cv2.EVENT_MOUSEMOVE and roi_start is not None:
        roi_end = (x, y_adj)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end      = (x, y_adj)
        roi_complete = True
        x1, y1 = roi_start
        x2, y2 = roi_end
        roi_params = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        print(f"\nROI COMPLETE: {roi_params[2]}x{roi_params[3]} at ({roi_params[0]}, {roi_params[1]})")
        print("Switching to contour detection mode...")


# ── info bar ──────────────────────────────────────────────────────────────────

def create_info_bar(width, status, frame_count, total_frames, speed,
                    contour_count=None, track_count=None, zoom=None):
    bar    = np.zeros((INFO_BAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    cv2.line(bar, (0, 0), (width, 0), (100, 100, 100), 2)

    s_col = (0, 200, 0) if status == "PLAYING" else (0, 150, 255)
    cv2.putText(bar, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_col, 2)
    cv2.putText(bar, f"Frame: {frame_count}/{total_frames}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, f"Speed: {speed:.2f}x",
                (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "+/- to change speed",
                (250, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    if contour_count is not None:
        cv2.putText(bar, f"Det: {contour_count}  Trk: {track_count}",
                    (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(bar, f"Area: {contour_params.min_area}-{contour_params.max_area}px",
                    (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(bar, f"Circ: {contour_params.min_circularity:.2f}-{contour_params.max_circularity:.2f}",
                    (450, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if zoom is not None and zoom != 1.0:
        cv2.putText(bar, f"Zoom: {zoom:.1f}x",
                    (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.line(bar, (0, INFO_BAR_HEIGHT - 1), (width, INFO_BAR_HEIGHT - 1),
             (100, 100, 100), 2)
    return bar


# ── core detection + feature extraction ──────────────────────────────────────

def process_frame_for_contours(frame, roi_params, back_sub,
                                kernel_open, kernel_close, clahe,
                                tracker, frame_num,
                                update_bg=True):
    roi_x, roi_y, roi_w, roi_h = roi_params
    cropped  = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray_roi = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_roi = clahe.apply(gray_roi)

    lr     = 0
    fg_raw = back_sub.apply(gray_roi, learningRate=lr)

    _, fg_thresh = cv2.threshold(fg_raw, 127, 255, cv2.THRESH_BINARY)
    fg_opened    = cv2.morphologyEx(fg_thresh, cv2.MORPH_OPEN,  kernel_open)
    fg_cleaned   = cv2.morphologyEx(fg_opened, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        fg_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (contour_params.min_area <= area <= contour_params.max_area):
            continue

        perim = cv2.arcLength(cnt, closed=True)
        circ  = circularity(area, perim)
        if not (contour_params.min_circularity <= circ <= contour_params.max_circularity):
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        feats = compute_features(cnt, gray_roi)
        detections.append({"cx": cx, "cy": cy, "contour": cnt, **feats})

    # ── update tracker ────────────────────────────────────────────────────
    tracker.update(detections, frame_num)

    # build lookup: which track_id covers each detection centre?
    track_id_map = {}
    for t in tracker.tracks:
        track_id_map[(t.cx, t.cy)] = t.track_id

    # ── draw overlay ──────────────────────────────────────────────────────
    vis = cropped.copy()
    for d in detections:
        cv2.drawContours(vis, [d["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(vis, (d["cx"], d["cy"]), 4, (0, 0, 255), -1)

        tid = track_id_map.get((d["cx"], d["cy"]), "?")
        line0 = f"ID:{tid}"
        line1 = f"A={int(d['area_px'])}px2 c={d['circularity']:.2f}"
        line2 = f"rim={d['rim_score']:+.1f} I={d['mean_intensity']:.0f}"
        cv2.putText(vis, line0,
                    (d["cx"] + 6, d["cy"] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        cv2.putText(vis, line1,
                    (d["cx"] + 6, d["cy"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        cv2.putText(vis, line2,
                    (d["cx"] + 6, d["cy"] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)

    return vis, detections


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global roi_selecting, roi_start, roi_end, roi_complete, roi_params
    global contour_params, tuning_mode, display_zoom

    if len(sys.argv) < 2:
        print("Usage: python MainCode.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # auto-name CSV next to the video file
    csv_path = str(Path(video_path).with_suffix("")) + "_tracks.csv"

    print("=" * 70)
    print(f"Video: {video_path} | FPS: {fps:.2f} | Frames: {total_frames}")
    print(f"CSV output: {csv_path}")
    print("Controls: SPACE=Pause | q=Quit | r=ROI | t=Tune | [/]=Zoom")
    print("          a/d=Step frames | +/-=Speed")
    print("=" * 70)
    print("\n>>> Press 'r' then click-drag to select ROI <<<\n")

    paused           = True
    frame_count      = 0
    saved_count      = 0
    viewing_zoomed   = False
    speed_multiplier = 1.0
    total_logged     = 0

    cv2.namedWindow("Video Viewer")
    cv2.setMouseCallback("Video Viewer", mouse_callback, video_path)

    current_frame = None
    back_sub      = None
    tracker       = DropletTracker()

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clahe        = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                    tileGridSize=CLAHE_TILE_GRID)

    frame_history     = []
    left_arrow_codes  = [81, 2, 0]
    right_arrow_codes = [83, 3, 1]

    # open CSV for writing immediately; rows are flushed as tracks close
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()

    try:
        while True:
            if not roi_selecting and not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count   = 0
                    frame_history = []
                    tracker.reset()
                    if back_sub:
                        back_sub = cv2.createBackgroundSubtractorMOG2(
                            history=150, varThreshold=12, detectShadows=False)
                    continue

                frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_frame = frame.copy()
                frame_history.append(frame.copy())
                if len(frame_history) > 150:
                    frame_history.pop(0)

            else:
                if current_frame is None:
                    ret, frame = cap.read()
                    if ret:
                        current_frame = frame.copy()
                        frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if not frame_history:
                            frame_history.append(frame.copy())

            if current_frame is not None:
                contour_count = None
                track_count   = None

                if viewing_zoomed and roi_params and back_sub:
                    vis, detections = process_frame_for_contours(
                        current_frame, roi_params, back_sub,
                        kernel_open, kernel_close, clahe,
                        tracker, frame_count,
                        update_bg=False,
                    )
                    contour_count = len(detections)
                    track_count   = len(tracker.tracks)
                    video_frame   = vis

                    # flush any tracks that just closed → write to CSV
                    closed = tracker.flush_closed()
                    if closed:
                        write_tracks_to_csv(csv_writer, closed)
                        csv_file.flush()
                        total_logged += len(closed)
                        for t in closed:
                            avg   = t.averaged_features()
                            label = classify_track(avg)
                            print(f"  Track {t.track_id:>4} closed | {label:8} | "
                                  f"frames {t.frame_start}-{t.frame_end} "
                                  f"({t.lifespan} fr) | "
                                  f"circ={avg['circularity']:.2f} "
                                  f"rim={avg['rim_score']:+.1f} "
                                  f"area={avg['area_px']:.0f}")

                    if display_zoom != 1.0:
                        h, w = video_frame.shape[:2]
                        video_frame = cv2.resize(
                            video_frame,
                            (int(w * display_zoom), int(h * display_zoom)),
                            interpolation=cv2.INTER_NEAREST,
                        )
                else:
                    video_frame = current_frame.copy()

                h, w     = video_frame.shape[:2]
                status   = "PAUSED" if paused else "PLAYING"
                info_bar = create_info_bar(
                    w, status, frame_count, total_frames, speed_multiplier,
                    contour_count=contour_count,
                    track_count=track_count,
                    zoom=display_zoom if viewing_zoomed else None,
                )
                display_frame = np.vstack([info_bar, video_frame])

                if roi_selecting:
                    dh, dw = display_frame.shape[:2]
                    cv2.rectangle(display_frame,
                                  (0, INFO_BAR_HEIGHT),
                                  (dw, INFO_BAR_HEIGHT + 50),
                                  (0, 0, 200), -1)
                    cv2.putText(display_frame,
                                "ROI SELECTION: click and drag",
                                (10, INFO_BAR_HEIGHT + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if roi_start and roi_end:
                        s = (roi_start[0], roi_start[1] + INFO_BAR_HEIGHT)
                        e = (roi_end[0],   roi_end[1]   + INFO_BAR_HEIGHT)
                        cv2.rectangle(display_frame, s, e, (0, 255, 0), 2)
                        dim_txt = f"{abs(roi_end[0]-roi_start[0])} x {abs(roi_end[1]-roi_start[1])} px"
                        cv2.putText(display_frame, dim_txt,
                                    (roi_start[0], roi_start[1] + INFO_BAR_HEIGHT - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Video Viewer", display_frame)

            if roi_complete and not viewing_zoomed and roi_params:
                back_sub = cv2.createBackgroundSubtractorMOG2(
                    history=150, varThreshold=20, detectShadows=False)

                print("Training background model on recent frames...")
                roi_x, roi_y, roi_w, roi_h = roi_params
                for hf in frame_history[-150:]:
                    roi_crop = hf[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                    gray_hf  = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
                    gray_hf  = clahe.apply(gray_hf)
                    back_sub.apply(gray_hf, learningRate=-1)

                tracker.reset()
                viewing_zoomed = True
                roi_complete   = False
                roi_selecting  = False
                print("CONTOUR DETECTION + TRACKING ACTIVE\n")

            delay = max(1, int(1000 / fps / speed_multiplier)) if not paused else 1
            key   = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                paused = not paused
                print(f"{'PAUSED' if paused else 'PLAYING'}")

            elif key == ord('+') or key == ord('='):
                speed_multiplier = min(8.0, round(speed_multiplier + 0.25, 2))
                print(f"Speed: {speed_multiplier:.2f}x")

            elif key == ord('-'):
                speed_multiplier = max(0.05, round(speed_multiplier - 0.25, 2))
                print(f"Speed: {speed_multiplier:.2f}x")

            elif key == ord('s') and current_frame is not None:
                out = f"frame_{frame_count:06d}.png"
                cv2.imwrite(out, video_frame)
                saved_count += 1
                print(f"Saved: {out}")

            elif key == ord('['):
                display_zoom = max(1.0, display_zoom - 0.5)
                print(f"Zoom: {display_zoom:.1f}x")

            elif key == ord(']'):
                display_zoom = min(8.0, display_zoom + 0.5)
                print(f"Zoom: {display_zoom:.1f}x")

            elif key == ord('r'):
                roi_selecting = not roi_selecting
                if roi_selecting:
                    paused         = True
                    roi_start      = None
                    roi_end        = None
                    roi_complete   = False
                    viewing_zoomed = False
                    print("\n>>> ROI SELECTION MODE — click and drag <<<")
                else:
                    print(">>> ROI selection cancelled <<<\n")

            elif (key == ord('a') or key in left_arrow_codes) and paused:
                new_pos = max(0, frame_count - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret:
                    current_frame = frame.copy()
                    frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"← Frame: {frame_count}/{total_frames}")

            elif (key == ord('d') or key in right_arrow_codes) and paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if ret:
                    current_frame = frame.copy()
                    frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"→ Frame: {frame_count}/{total_frames}")

            elif key == ord('t'):
                tuning_mode = not tuning_mode
                if tuning_mode:
                    print("\n" + "=" * 70)
                    print("TUNING MODE  —  1/2=MinArea  3/4=MaxArea  5/6=MinCirc  t=Exit")
                    print(f"Current: area {contour_params.min_area}-{contour_params.max_area} | "
                          f"circ {contour_params.min_circularity:.2f}-{contour_params.max_circularity:.2f}")
                    print("=" * 70)
                else:
                    print(">>> TUNING MODE EXITED <<<\n")

            elif tuning_mode:
                changed = False
                if key == ord('1'):
                    contour_params.min_area = max(10, contour_params.min_area - 10);   changed = True
                elif key == ord('2'):
                    contour_params.min_area += 10;                                     changed = True
                elif key == ord('3'):
                    contour_params.max_area = max(100, contour_params.max_area - 100); changed = True
                elif key == ord('4'):
                    contour_params.max_area += 100;                                    changed = True
                elif key == ord('5'):
                    contour_params.min_circularity = max(0.0, contour_params.min_circularity - 0.05); changed = True
                elif key == ord('6'):
                    contour_params.min_circularity = min(1.0, contour_params.min_circularity + 0.05); changed = True

                if changed:
                    print(f"Params: area {contour_params.min_area}-{contour_params.max_area} | "
                          f"circ {contour_params.min_circularity:.2f}")

            elif key != 255 and paused:
                print(f"Unrecognized key: {key}")

    finally:
        # flush any tracks still active when the user quits
        if viewing_zoomed:
            tracker.close_all()
            closed = tracker.flush_closed()
            if closed:
                write_tracks_to_csv(csv_writer, closed)
                total_logged += len(closed)

        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Frames analyzed : {frame_count}/{total_frames} | Saved: {saved_count}")
        print(f"Tracks logged   : {total_logged}  →  {csv_path}")
        print("\nFINAL CONTOUR DETECTION PARAMETERS:")
        print("-" * 70)
        print(f"class ContourParams:")
        print(f"    min_area        = {contour_params.min_area}")
        print(f"    max_area        = {contour_params.max_area}")
        print(f"    min_circularity = {contour_params.min_circularity:.2f}")
        print(f"    max_circularity = {contour_params.max_circularity:.2f}")
        print(f"\nRIM_THICKNESS_PX   = {RIM_THICKNESS_PX}")
        print(f"CLAHE_CLIP_LIMIT   = {CLAHE_CLIP_LIMIT}")
        print(f"CLAHE_TILE_GRID    = {CLAHE_TILE_GRID}")
        print(f"\nMAX_MATCH_DIST     = {MAX_MATCH_DIST}")
        print(f"MAX_AREA_RATIO     = {MAX_AREA_RATIO}")
        print(f"MAX_MISSING_FRAMES = {MAX_MISSING_FRAMES}")
        print(f"\nCLS_MIN_CIRCULARITY = {CLS_MIN_CIRCULARITY}")
        print(f"CLS_MIN_RIM_SCORE   = {CLS_MIN_RIM_SCORE}")
        print(f"CLS_MIN_AREA        = {CLS_MIN_AREA}")
        print("-" * 70)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
