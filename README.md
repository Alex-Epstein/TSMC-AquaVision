# AquaVision

Computer vision that watches high-speed video of fluid flow, tracks every droplet, and measures the shape and brightness signals that give bubbles and irregular flow away. Built with TSMC Arizona.

## Why

Semiconductor fabs depend on steady delivery of chemicals and deionized water. A micro-bubble or a partly clogged nozzle can cause wafer defects and cost yield. These events happen in milliseconds. That is too fast for a person to see and too local for pressure or flow sensors to catch. AquaVision films the flow at 240 fps and measures every droplet that passes through the frame.

We built a small, repeatable test rig: an Air-Logic needle valve (F282240B85SS) on a lab stand, a 3D-printed funnel and 1/8" ID silicone tubing. Video was captured with an iPhone 14 in 240 fps slow-motion mode.

## How it works

All processing lives in `MainCode.py` and uses Python, OpenCV and NumPy.

1. **Region of interest.** Press `r` and drag a box around the stream. Only that crop is processed.
2. **Contrast and background.** Each crop is converted to grayscale and passed through CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`). A MOG2 background model (`history=150`, `varThreshold=20`, `detectShadows=False`) is trained on up to the last 150 frames, then frozen (`learningRate=0`) so moving droplets never get absorbed into the background.
3. **Mask cleanup.** The foreground mask is thresholded at 127, then cleaned with `MORPH_OPEN` (3x3 ellipse) to remove speckle and `MORPH_CLOSE` (5x5 ellipse) to fill the hollow centers that bubbles leave in the mask.
4. **Contours and gates.** `cv2.findContours` with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE` finds each object. Objects outside the area gate (400 to 5000 px) or the circularity gate (0.20 to 1.00) are dropped. The centroid comes from `cv2.moments`.
5. **Features.** `compute_features()` measures each object:
   - Circularity, `C = 4πA / P²`, clamped to [0, 1]. A perfect circle scores 1.
   - Mean and standard deviation of intensity inside the contour (`cv2.meanStdDev`). The paper calls high internal variance "optical chaos": air-liquid interfaces inside a bubble scatter light, while a clean droplet looks uniform.
   - Rim Score, `ΔI = μ_rim − μ_interior`. The filled mask is eroded by `RIM_THICKNESS_PX = 3` to split each object into a rim band and an interior.
   - Area, perimeter and equivalent diameter.
6. **Tracking.** `DropletTracker` matches each detection to the nearest live track within 50 px, as long as the area ratio stays under 2.5. Unmatched detections start new tracks. A track closes after 8 frames without a match. Each `Track` keeps a feature snapshot per frame and averages them when it closes.
7. **Classification.** `classify_track()` labels a closed track `droplet` when its averages meet all three thresholds: circularity ≥ 0.45, rim score ≥ 5.0 and area ≥ 300 px. Everything else is `noise`.
8. **Logging.** `write_tracks_to_csv()` writes one row per closed track to `<video>_tracks.csv`, next to the input video.

```mermaid
flowchart LR
    A[Video frame] --> B[ROI crop]
    B --> C[Grayscale + CLAHE]
    C --> D[MOG2 foreground mask]
    D --> E[Threshold + open/close]
    E --> F[Contours]
    F --> G[Area and circularity gates]
    G --> H["Features: C, ΔI, μ, σ"]
    H --> I[Centroid tracker]
    I --> J[Rule-based label]
    J --> K[Per-track CSV]
```

All thresholds are constants at the top of `MainCode.py`, so they can be tuned without touching the logic.

## Results

These numbers come from the team's paper, [`TSMC_AquaVision_Paper.pdf`](TSMC_AquaVision_Paper.pdf).

- Feature extraction, including `cv2.moments` and `cv2.meanStdDev`, stayed under 4 ms per frame.
- The system processed the full 240 fps stream with no dropped frames.
- `CHAIN_APPROX_SIMPLE` cut contour vertex count by about 70%.
- Steady droplets held a Rim Score near zero. Aerated inclusions showed a bright "halo" with ΔI > 15 and about 40% higher intensity standard deviation.

The paper also lists two limits. The Rim Score is sensitive to big shifts in ambient lighting, and a back-lit LED panel is planned to fix that. The method also assumes transparent or semi-transparent fluid. Opaque slurries would need a boundary-only approach.

## Run it

Requires Python 3 and a desktop session, since the viewer opens an OpenCV window.

```bash
git clone https://github.com/Alex-Epstein/TSMC-AquaVision.git
cd TSMC-AquaVision

# Videos are stored in Git LFS
git lfs install
git lfs pull --include "TestVideo.mp4"   # about 49 MB
# git lfs pull                           # all videos, about 1.7 GB

pip install opencv-python numpy
python MainCode.py TestVideo.mp4
```

The video opens paused. Press `r`, drag a box around the stream and release. Detection and tracking start right away. Press `q` to quit. Remaining tracks are flushed to the CSV and the final tuning values print to the console.

### Controls

| Key | Action |
| --- | --- |
| `Space` | Pause or resume |
| `+` / `=` and `-` | Speed up or slow down by 0.25x (range 0.05x to 8x) |
| `]` and `[` | Zoom the ROI view in or out by 0.5x (range 1x to 8x, starts at 2x) |
| `a` / `d` or arrow keys | Step one frame back or forward while paused |
| `r` | Start or cancel ROI selection |
| `t` | Toggle tuning mode |
| `1` / `2` | In tuning mode: lower or raise `min_area` by 10 |
| `3` / `4` | In tuning mode: lower or raise `max_area` by 100 |
| `5` / `6` | In tuning mode: lower or raise `min_circularity` by 0.05 |
| `s` | Save the current view as `frame_XXXXXX.png` |
| `q` | Quit and write remaining tracks |

### Output

Each row in `<video>_tracks.csv` is one tracked object:

`track_id, label, frame_start, frame_end, lifespan_frames, avg_area_px, avg_eq_diameter_px, avg_circularity, avg_mean_intensity, avg_std_intensity, avg_rim_score, avg_rim_mean, avg_interior_mean`

`TestVideo_tracks.csv` is a sample run on `TestVideo.mp4`.

## Repo layout

| File | What it is |
| --- | --- |
| `MainCode.py` | The full pipeline and interactive viewer |
| `FunctionDescriptions.txt` | Key bindings and a tuning guide for each parameter |
| `TestVideo_tracks.csv` | Sample per-track output |
| `TSMC_AquaVision_Paper.pdf` | Team paper with method and results |
| `TestVideo.mp4` | Short test clip (Git LFS) |
| `TSMCVideo1.mp4` to `TSMCVideo5.mp4` | Recorded flow videos (Git LFS, about 225 to 420 MB each) |
| `open-tap-middle-screen.mp4` | Additional sample clip |

## Team

Built January to April 2026 as a project for TSMC Arizona.

- Alex Epstein
- Emily Chu
- Rishab Subramanya
- Ameer Salim

Mentor: Sai Aruru, Equipment Engineer, TSMC Arizona.

Thanks to Georgia Tech ASME for project oversight.

## Paper

[AquaVision Anomaly Detection: A Smart System for Fluid Flow Monitoring and Irregularity Identification](TSMC_AquaVision_Paper.pdf)
