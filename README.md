# LiveLabel: a Local Real-Time Computer Vision Pipeline
A comprehensive real-time computer vision system with two main approaches: object detection with YOLO/SAM and motion-based scene analysis, both powered by local VLMs via Ollama.

## Features
- **Object Detection Mode**: YOLO/SAM-based detection with bounding boxes and per-object labeling
- **Motion Analysis Mode**: Intelligent motion detection with full-scene VLM analysis
- Face/person censoring (object detection mode)
- Webcam and screen capture streaming
- Compatibility with any Ollama VLM that supports images
- Fully local setup (optionally use remote models)
- Asynchronous processing for smooth real-time performance
- Advanced motion detection with multiple algorithms

## Two Operating Modes

### 1. Object Detection Mode (`object_detector.py`)
**Best for**: Precise object identification, multi-object scenes, object tracking

**How it Works:**
- **Detection:** YOLO/SAM infers every `--frame-delay` frames and filters by `--confidence`
- **Storing:** Object is given a UID stored in `prev_box_ids`, and bounding boxes are stored in `prev_boxes`
- **Tracking heuristic:** Centers + IoU matching update `prev_boxes` and stable UIDs
- **Labeling:** Async queue crops frame per stored box -> base64 -> POST to Ollama `--server` with `--vlm` to generate text label
- **Censoring:** Pixelates class person when `--censor` is True, skips labeling and boundary box drawing

### 2. Motion Analysis Mode (`analyzer.py`)
**Best for**: Scene understanding, activity recognition, real-time responsiveness

**How it Works:**
- **Triple Motion Detection:**
  - **Background Subtraction (MOG2/KNN)**: Detects new objects appearing in frame
  - **Frame Difference**: Detects immediate movements and gestures
  - **Structural Similarity**: Detects overall scene changes and camera movement
- **Smart Triggering**: Only analyzes when significant motion is detected
- **Full Scene Analysis**: VLM analyzes entire frame for comprehensive scene understanding
- **Asynchronous Processing**: Motion detection runs in real-time while VLM analysis happens in background
## Next Steps
- SAM integration for better object segmentation

## Prerequisites
1. Python 3.10+ (project currently uses a venv in `venv/`)
2. Ollama installed with a VLM pulled (e.g., `llava:7b`)
3. Model weights available locally (default: `yolo11n.pt` in repo root)
4. Screen and camera permissions if using those sources

## Getting Started
1. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Start your Ollama server (default API at `http://localhost:11434`)
```
ollama serve
```
4. Choose your mode and run:

**Object Detection Mode:**
```
python object_detector.py [args]
```

**Motion Analysis Mode:**
```
python analyzer.py [args]
```

## Arguments

### Object Detection Mode (`object_detector.py`)
- `--vlm` (str): VLM model name. Default: `llava:7b`.
- `--detection-model` (str): Path to YOLO `.pt` weights. Default: `yolo11n.pt`.
- `--server` (str): VLM server URL. Default: `http://localhost:11434`.
- `--confidence` (float): Confidence threshold for object detection filtering. Default: `0.3`.
- `--frame-delay` (int): Frame interval between detection passes. Default: `5`.
- `--censor` (bool): Enable face/person pixelation. Default: `False`.
- `--source` (int): Video source: `0` = webcam, `1` = screen capture. Default: `0`.

### Motion Analysis Mode (`analyzer.py`)
- `--vlm` (str): VLM model name. Default: `qwen2.5vl:7b`.
- `--subtractor` (str): Background subtractor algorithm. Options: `MOG2`, `KNN`. Default: `KNN`.
- `--server` (str): VLM server URL. Default: `http://localhost:11434`.
- `--motion` (float): Motion sensitivity threshold. Default: `0.3`.
- `--frame-delay` (int): Frame interval between analysis passes. Default: `1`.
- `--source` (int): Video source: `0` = webcam, `1` = screen capture. Default: `0`.

## Usage Examples

### Object Detection Mode
Run with defaults (webcam, local Ollama, YOLO `yolo11n.pt`):
```
python object_detector.py
```

Enable censoring and use screen capture:
```
python object_detector.py --censor True --source 1
```

Use a different VLM and model weights:
```
python object_detector.py --vlm llava:13b --detection-model face/last.pt
```

### Motion Analysis Mode
Run with defaults (webcam, KNN subtractor, qwen2.5vl:7b):
```
python analyzer.py
```

Use MOG2 background subtractor with higher motion sensitivity:
```
python analyzer.py --subtractor MOG2 --motion 0.5
```

Use screen capture with different VLM:
```
python analyzer.py --source 1 --vlm llava:7b
```

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
