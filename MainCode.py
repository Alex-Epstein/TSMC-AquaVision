import cv2
import sys
import numpy as np


roi_selecting = False
roi_start     = None
roi_end       = None
roi_complete  = False
roi_params    = None


class ContourParams:
    min_area        = 50
    max_area        = 5000
    min_circularity = 0.3
    max_circularity = 1.0


contour_params = ContourParams()
tuning_mode    = False
display_zoom   = 2.0
INFO_BAR_HEIGHT = 100


# ── helpers ──────────────────────────────────────────────────────────────────

def circularity(area: float, perimeter: float) -> float:
    """Returns 4πA/P²; clamps to [0,1] to handle numeric noise."""
    if perimeter < 1e-6:
        return 0.0
    return min(1.0, (4 * np.pi * area) / (perimeter ** 2))


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


def create_info_bar(width, status, frame_count, total_frames, speed,
                    contour_count=None, zoom=None):
    bar        = np.zeros((INFO_BAR_HEIGHT, width, 3), dtype=np.uint8)
    bar[:]     = (40, 40, 40)
    cv2.line(bar, (0, 0), (width, 0), (100, 100, 100), 2)

    s_col = (0, 200, 0) if status == "PLAYING" else (0, 150, 255)
    cv2.putText(bar, status,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_col, 2)
    cv2.putText(bar, f"Frame: {frame_count}/{total_frames}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, f"Speed: {speed:.2f}x",
                (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "+/- to change speed",
                (250, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    if contour_count is not None:
        cv2.putText(bar, f"Contours: {contour_count}",
                    (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(bar,
                    f"Area: {contour_params.min_area}-{contour_params.max_area}px",
                    (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(bar,
                    f"Circ: {contour_params.min_circularity:.2f}-{contour_params.max_circularity:.2f}",
                    (450, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if zoom is not None and zoom != 1.0:
        cv2.putText(bar, f"Zoom: {zoom:.1f}x",
                    (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.line(bar, (0, INFO_BAR_HEIGHT - 1), (width, INFO_BAR_HEIGHT - 1),
             (100, 100, 100), 2)
    return bar


# ── core detection ────────────────────────────────────────────────────────────

def process_frame_for_contours(frame, roi_params, back_sub,
                                kernel_open, kernel_close,
                                update_bg=True):
    """
    Returns (annotated_roi_bgr, list_of_dicts).
    Each dict: {cx, cy, area, perimeter, circularity, contour}
    """
    roi_x, roi_y, roi_w, roi_h = roi_params
    cropped = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # ── background subtraction ────────────────────────────────────────────
    lr = -1 if update_bg else 0
    fg_raw = back_sub.apply(cropped, learningRate=lr)

    _, fg_thresh  = cv2.threshold(fg_raw, 127, 255, cv2.THRESH_BINARY)
    fg_opened     = cv2.morphologyEx(fg_thresh,  cv2.MORPH_OPEN,  kernel_open)
    fg_cleaned    = cv2.morphologyEx(fg_opened,  cv2.MORPH_CLOSE, kernel_close)

    # ── contour detection ─────────────────────────────────────────────────
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

        M  = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        detections.append({
            "cx": cx, "cy": cy,
            "area": area, "perimeter": perim,
            "circularity": circ,
            "contour": cnt,
        })

    # ── draw overlay ──────────────────────────────────────────────────────
    vis = cropped.copy()
    for d in detections:
        cv2.drawContours(vis, [d["contour"]], -1, (0, 255, 0), 2)
        cv2.circle(vis, (d["cx"], d["cy"]), 4, (0, 0, 255), -1)

        eq_diam = int(np.sqrt(4 * d["area"] / np.pi))
        label   = f"d={eq_diam}px c={d['circularity']:.2f}"
        cv2.putText(vis, label,
                    (d["cx"] + 6, d["cy"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)

    return vis, detections


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global roi_selecting, roi_start, roi_end, roi_complete, roi_params
    global contour_params, tuning_mode, display_zoom

    if len(sys.argv) < 2:
        print("Usage: python video_viewer.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 70)
    print(f"Video: {video_path} | FPS: {fps:.2f} | Frames: {total_frames}")
    print("Controls: SPACE=Pause | q=Quit | r=ROI | t=Tune | [/]=Zoom")
    print("          a/d=Step frames | +/-=Speed")
    print("=" * 70)
    print("\n>>> Press 'r' then click-drag to select ROI <<<\n")

    paused           = True
    frame_count      = 0
    saved_count      = 0
    viewing_zoomed   = False
    speed_multiplier = 1.0          # start at 1× — use +/- to change

    cv2.namedWindow("Video Viewer")
    cv2.setMouseCallback("Video Viewer", mouse_callback, video_path)

    current_frame = None
    back_sub      = None

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frame_history     = []
    left_arrow_codes  = [81, 2, 0]
    right_arrow_codes = [83, 3, 1]

    try:
        while True:
            # ── frame acquisition ─────────────────────────────────────────
            if not roi_selecting and not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count   = 0
                    frame_history = []
                    if back_sub:
                        back_sub = cv2.createBackgroundSubtractorMOG2(
                            history=500, varThreshold=16, detectShadows=False)
                    continue

                frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_frame = frame.copy()
                frame_history.append(frame.copy())
                if len(frame_history) > 500:
                    frame_history.pop(0)

            else:
                if current_frame is None:
                    ret, frame = cap.read()
                    if ret:
                        current_frame = frame.copy()
                        frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if not frame_history:
                            frame_history.append(frame.copy())

            # ── render ────────────────────────────────────────────────────
            if current_frame is not None:
                contour_count = None

                if viewing_zoomed and roi_params and back_sub:
                    vis, detections = process_frame_for_contours(
                        current_frame, roi_params, back_sub,
                        kernel_open, kernel_close,
                        update_bg=(not paused),
                    )
                    contour_count = len(detections)
                    video_frame   = vis

                    if display_zoom != 1.0:
                        h, w = video_frame.shape[:2]
                        video_frame = cv2.resize(
                            video_frame,
                            (int(w * display_zoom), int(h * display_zoom)),
                            interpolation=cv2.INTER_NEAREST,
                        )
                else:
                    video_frame = current_frame.copy()

                h, w    = video_frame.shape[:2]
                status  = "PAUSED" if paused else "PLAYING"
                info_bar = create_info_bar(
                    w, status, frame_count, total_frames, speed_multiplier,
                    contour_count=contour_count,
                    zoom=display_zoom if viewing_zoomed else None,
                )
                display_frame = np.vstack([info_bar, video_frame])

                # ROI selection overlay
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

            # ── activate contour mode once ROI is drawn ───────────────────
            if roi_complete and not viewing_zoomed and roi_params:
                back_sub = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False)

                print("Training background model on recent frames...")
                roi_x, roi_y, roi_w, roi_h = roi_params
                for hf in frame_history[-100:]:
                    back_sub.apply(hf[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])

                viewing_zoomed = True
                roi_complete   = False
                roi_selecting  = False
                print("CONTOUR DETECTION ACTIVE\n")

            # ── key handling ──────────────────────────────────────────────
            delay = max(1, int(1000 / fps / speed_multiplier)) if not paused else 1
            key   = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                paused = not paused
                print(f"{'PAUSED' if paused else 'PLAYING'}")

            # speed up / slow down
            elif key == ord('+') or key == ord('='):   # '=' shares key with '+' (no shift)
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

            # step backward
            elif (key == ord('a') or key in left_arrow_codes) and paused:
                new_pos = max(0, frame_count - 2)   # -2 because read() advances by 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                ret, frame = cap.read()
                if ret:
                    current_frame = frame.copy()
                    frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"← Frame: {frame_count}/{total_frames}")

            # step forward
            elif (key == ord('d') or key in right_arrow_codes) and paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if ret:
                    current_frame = frame.copy()
                    frame_count   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"→ Frame: {frame_count}/{total_frames}")

            # tuning mode toggle
            elif key == ord('t'):
                tuning_mode = not tuning_mode
                if tuning_mode:
                    print("\n" + "=" * 70)
                    print("TUNING MODE  —  keys: 1/2=MinArea  3/4=MaxArea  5/6=MinCirc  t=Exit")
                    print(f"Current: area {contour_params.min_area}-{contour_params.max_area} | "
                          f"circ {contour_params.min_circularity:.2f}-{contour_params.max_circularity:.2f}")
                    print("=" * 70)
                else:
                    print(">>> TUNING MODE EXITED <<<\n")

            elif tuning_mode:
                changed = False
                if key == ord('1'):
                    contour_params.min_area = max(10, contour_params.min_area - 10);  changed = True
                elif key == ord('2'):
                    contour_params.min_area += 10;                                    changed = True
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
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Frames analyzed: {frame_count}/{total_frames} | Saved: {saved_count}")
        print("\nFINAL CONTOUR DETECTION PARAMETERS:")
        print("-" * 70)
        print(f"class ContourParams:")
        print(f"    min_area        = {contour_params.min_area}")
        print(f"    max_area        = {contour_params.max_area}")
        print(f"    min_circularity = {contour_params.min_circularity:.2f}")
        print(f"    max_circularity = {contour_params.max_circularity:.2f}")
        print("-" * 70)
        print("Paste this into your next script to hard-code tuned parameters.")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
