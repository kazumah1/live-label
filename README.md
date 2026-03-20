<<<<<<< HEAD
# LiveLabel
=======
# LiveLabel: a Local Real-Time Computer Vision Pipeline
A comprehensive real-time computer vision system with two main approaches: object detection with YOLO/SAM and motion-based scene analysis, both powered by local VLMs via Ollama.
>>>>>>> bd2f5edca06516198cb437b7d547d30a554f741b

Most computer vision demos either run in the cloud — introducing latency and privacy concerns — or require writing custom integration code to wire up a specific model. LiveLabel was built to answer a simpler question: *what is the camera looking at, right now, with no internet required?*

It is a real-time computer vision web app that streams an annotated camera feed to a browser dashboard, using any local vision-language model (VLM) served via Ollama. No cloud dependency. No API keys. Runs on a laptop.

---

## How it works

Two modes are available and switchable at runtime from the dashboard sidebar:

**Detector mode** — Uses FastSAM or YOLO to detect and segment objects in every frame. Each detected object is assigned a persistent UID and individually labeled by the VLM (crop-and-ask). Labels are generated asynchronously so the video stream stays smooth.

**Analyzer mode** — Monitors for motion using a combination of background subtraction, frame differencing, and structural similarity. When significant motion is detected, the full frame is sent to the VLM for a scene-level description.

The dashboard includes: live MJPEG stream, mode toggle (Detector/Analyzer), source toggle (Camera/Screen), detection controls (boxes, labels, overlays, confidence threshold), model selector, live scene summary, and per-object list.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a vision model pulled
- Model weights in a `models/` directory (default: `FastSAM-s.pt`)
- macOS camera and screen recording permissions

Recommended VLMs:

```
ollama pull qwen2.5vl:7b    # best quality
ollama pull moondream:latest # much faster (~2-5s vs 15-30s per analysis)
```

---

## Quick start

```
git clone <repo>
cd cv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Place your detection weights in `models/` (e.g., `models/FastSAM-s.pt`).

Make sure Ollama is running:

```
ollama serve
```

Start the web app:

```
python web_server.py
```

Open http://localhost:5000. Select a mode from the sidebar and use the controls to adjust detection behavior.

---

## CLI arguments (web_server.py)

| Argument | Default | Description |
|---|---|---|
| `--mode` | `detector` | `detector` or `analyzer` |
| `--source` | `0` | `0` = camera, `1` = screen capture |
| `--vlm` | `qwen2.5vl:7b` | Ollama model name |
| `--detection-model` | `FastSAM-s.pt` | Path to weights file in `models/` |
| `--server` | `http://localhost:11434` | Ollama server URL |
| `--confidence` | `0.3` | Detection confidence threshold |
| `--label-workers` | `1` | Parallel VLM labeling threads |
| `--port` | `5000` | Port for the Flask server |

Examples:

```
# Analyzer mode with a faster model
python web_server.py --mode analyzer --vlm moondream:latest

# Detector mode with parallel labeling (set OLLAMA_NUM_PARALLEL to match)
OLLAMA_NUM_PARALLEL=2 python web_server.py --label-workers 2

# Screen capture source
python web_server.py --source 1
```

---

## Standalone CLI scripts

The detector and analyzer can be run independently without the web server, for scripting or testing:

```
python object_detector.py
python analyzer.py
```

Both accept similar arguments (`--vlm`, `--server`, `--confidence`, `--source`). Run with `--help` to see options.

---

## Optional: Electron overlay (screen mode)

When using screen capture as the source, the browser window itself will be captured, creating an infinite mirror loop. The optional Electron overlay solves this by floating the annotated feed and scene caption in a separate window that is excluded from screen recording via macOS `setContentProtection`.

Setup:

```
cd overlay
npm install
npm start
```

The overlay reads the MJPEG stream from the running Flask server. Start `web_server.py --source 1` first, then launch the overlay.

---

## Performance

| Mode | Display FPS | VLM cadence |
|---|---|---|
| Detector | ~15-30 FPS | Once per object (async, background) |
| Analyzer | ~15-30 FPS | On motion detection (~5-30s per analysis) |

Using `moondream:latest` reduces analyzer latency to ~2-5s. For detector mode with many objects, `--label-workers 2` with `OLLAMA_NUM_PARALLEL=2` enables parallel labeling.

---

## Project structure

```
web_server.py        # Flask server + dashboard backend
object_detector.py   # RealTimeDetector (FastSAM/YOLO + VLM labeling)
analyzer.py          # RealTimeLabeler (motion detection + scene VLM)
templates/
  dashboard.html     # Single-page dashboard UI
overlay/
  main.js            # Electron main process
  index.html         # Overlay window (stream + caption)
  package.json
models/              # Detection model weights (gitignored)
requirements.txt
```
<<<<<<< HEAD
=======

## Performance Comparison

| Mode | FPS | Use Case | Resource Usage |
|------|-----|----------|----------------|
| Object Detection | ~0.3-0.5 | Precise object identification | High (YOLO + VLM) |
| Motion Analysis | ~15-30 | Scene understanding | Low (Motion detection + VLM) |

## Motion Detection Details

The motion analysis mode uses three complementary detection methods:

### 1. Background Subtraction
- **MOG2**: Mixture of Gaussians, adaptive to lighting changes
- **KNN**: K-Nearest Neighbors, better for complex backgrounds
- **Purpose**: Detects new objects appearing in frame

### 2. Frame Difference
- Compares consecutive frames pixel-by-pixel
- **Purpose**: Detects immediate movements and gestures
- **Threshold**: Uses half the motion area threshold for higher sensitivity

### 3. Structural Similarity
- Compares frames over longer time periods using template matching
- **Purpose**: Detects overall scene changes and camera movement
- **Threshold**: Default 0.9 (90% similarity required to avoid analysis)

### Motion Detection Parameters
- `motion_area_threshold`: Minimum contour area to trigger analysis (default: 5000 pixels)
- `ssim_threshold`: Structural similarity threshold (default: 0.9)
- `buffer_size`: Number of frames to keep for comparison (default: 10)

## Notes
- Ensure your VLM supports image inputs via Ollama's `/api/generate` with `images` payloads
- Object detection mode will automatically download YOLO weights if not present
- Motion analysis mode is more responsive and suitable for real-time applications
- Both modes support webcam and screen capture sources
- Asynchronous processing ensures smooth video streams during VLM analysis
>>>>>>> bd2f5edca06516198cb437b7d547d30a554f741b
