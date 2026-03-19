"""
LiveLabel web server.

Run:
    python web_server.py [--mode detector|analyzer] [--port 5000] [other args]

Then open http://localhost:5000 in a browser.
"""

import argparse
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from flask import Flask, Response, render_template, jsonify, request
import requests
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from object_detector import RealTimeDetector
from analyzer import RealTimeLabeler


app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared dashboard state
# ---------------------------------------------------------------------------

@dataclass
class DashboardState:
    event_log: deque = field(default_factory=lambda: deque(maxlen=50))
    scene_summary: str = "Analyzing scene..."
    last_summary_time: float = 0.0
    motion_detected: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, uid: str, label: str):
        with self._lock:
            self.event_log.appendleft({
                "time": time.strftime("%H:%M:%S"),
                "uid": uid,
                "label": label,
            })

    def set_summary(self, summary: str):
        with self._lock:
            self.scene_summary = summary
            self.last_summary_time = time.time()

    def set_motion(self, detected: bool):
        with self._lock:
            self.motion_detected = detected

    def snapshot(self):
        with self._lock:
            return {
                "scene_summary": self.scene_summary,
                "event_log": list(self.event_log),
                "motion": self.motion_detected,
            }


dashboard = DashboardState()


@dataclass
class RuntimeConfig:
    show_boxes: bool = True
    show_overlays: bool = True
    show_labels: bool = True
    confidence: float = 0.3
    vlm: str = "llava:7b"
    source: int = 0
    mode: str = "detector"  # read-only from UI; set at startup
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def as_dict(self):
        with self._lock:
            return {
                "show_boxes": self.show_boxes,
                "show_overlays": self.show_overlays,
                "show_labels": self.show_labels,
                "confidence": self.confidence,
                "vlm": self.vlm,
                "source": self.source,
                "mode": self.mode,
            }

    def update(self, data: dict):
        with self._lock:
            if "show_boxes" in data:
                self.show_boxes = bool(data["show_boxes"])
            if "show_overlays" in data:
                self.show_overlays = bool(data["show_overlays"])
            if "show_labels" in data:
                self.show_labels = bool(data["show_labels"])
            if "confidence" in data:
                self.confidence = max(0.05, min(1.0, float(data["confidence"])))
            if "vlm" in data:
                self.vlm = str(data["vlm"])
            if "source" in data:
                self.source = int(data["source"])


config = RuntimeConfig()

# Populated in main() before the Flask app starts.
detector: RealTimeDetector = None
labeler: RealTimeLabeler = None
known_labels: dict = {}  # uid -> label, tracks already-logged events
startup_args = None  # stored in main() for backend recreation on mode switch


# ---------------------------------------------------------------------------
# Scene summary worker (detector mode only — uses raw frame)
# ---------------------------------------------------------------------------

def _scene_summary_worker(server_url: str):
    """Motion-triggered scene summary for detector mode.

    Polls the detector's raw frame, runs a lightweight background subtractor to
    detect motion, and fires a VLM scene description only when something changes.
    This mirrors the analyzer's approach but runs as a sidecar to the detector.
    """
    subtractor = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion_area_threshold = 3000  # px²
    cooldown = 4.0                # min seconds between VLM calls
    last_call_time = 0.0

    while True:
        time.sleep(0.1)
        if detector is None:
            continue

        frame = detector.get_raw_frame()
        if frame is None:
            continue

        # Motion detection — background subtractor + contour area
        mask = subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_motion = sum(cv2.contourArea(c) for c in contours) > motion_area_threshold
        dashboard.set_motion(has_motion)

        now = time.time()
        if not has_motion or (now - last_call_time) < cooldown:
            continue

        # Motion detected and cooldown elapsed — ask VLM for a scene description
        jpeg = detector.get_raw_jpeg()
        if jpeg is None:
            continue
        last_call_time = now
        try:
            img_b64 = base64.b64encode(jpeg).decode("utf-8")
            payload = {
                "model": config.vlm,
                "prompt": "Describe what is happening in this scene in one sentence.",
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 30},
                "images": [img_b64],
            }
            resp = requests.post(f"{server_url}/api/generate", json=payload, timeout=60)
            if resp.status_code == 200:
                summary = resp.json().get("response", "").strip()
                if summary:
                    dashboard.set_summary(summary)
        except Exception as e:
            print(f"[summary] error: {e}")


def _event_watcher():
    """Poll detector.labels for newly labeled objects and log them."""
    while True:
        time.sleep(0.5)
        if detector is None:
            continue
        current_labels = dict(detector.labels)
        for uid, label in current_labels.items():
            if uid not in known_labels:
                known_labels[uid] = label
                dashboard.add_event(uid, label)


def _analyzer_summary_watcher():
    """Push labeler state (label + motion) into dashboard whenever it changes."""
    last = ""
    while True:
        time.sleep(0.5)
        if labeler is None:
            continue
        dashboard.set_motion(bool(labeler.motion_detected))
        current = labeler.label
        if current and current != last:
            dashboard.set_summary(current)
            last = current


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


def _mjpeg_generator():
    while True:
        backend = detector if config.mode == "detector" else labeler
        jpeg = backend.get_annotated_jpeg() if backend else None
        if jpeg is None:
            time.sleep(0.033)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(0.033)


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/state")
def api_state():
    state = dashboard.snapshot()
    objects = []
    if config.mode == "detector" and detector:
        with detector.detection_lock:
            boxes = dict(detector.prev_boxes)
        labels = dict(detector.labels)
        metadata = dict(detector.object_metadata)
        for uid, coords in boxes.items():
            label = labels.get(uid, "[labeling...]")
            conf = metadata.get(uid, {}).get("conf", 0)
            objects.append({"uid": uid, "label": label, "conf": round(float(conf), 2)})
    state["objects"] = objects
    return jsonify(state)


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify(config.as_dict())


@app.route("/api/config", methods=["POST"])
def api_config_post():
    data = request.get_json(silent=True) or {}
    config.update(data)
    # Apply immediately to running backend
    if detector:
        if "show_boxes" in data:
            detector.show_boxes = config.show_boxes
        if "show_overlays" in data:
            detector.show_overlays = config.show_overlays
        if "show_labels" in data:
            detector.show_labels = config.show_labels
        if "confidence" in data:
            detector.confidence_level = config.confidence
        if "source" in data:
            detector.source = config.source
    if labeler and "source" in data:
        labeler.source = config.source
    return jsonify(config.as_dict())


@app.route("/api/mode", methods=["POST"])
def api_mode():
    global detector, labeler
    data = request.get_json(silent=True) or {}
    new_mode = data.get("mode")
    if new_mode not in ("detector", "analyzer"):
        return jsonify({"error": "invalid mode"}), 400
    if new_mode == config.mode:
        return jsonify(config.as_dict())
    # Stop current backend and wait for camera to be released
    if detector:
        detector.stop()
        detector._stopped.wait(timeout=3.0)
        detector = None
    if labeler:
        labeler.stop()
        labeler._stopped.wait(timeout=3.0)
        labeler = None
    config.mode = new_mode
    # Start new backend
    if new_mode == "detector":
        detector = RealTimeDetector(
            vlm=config.vlm,
            detection_model=startup_args.detection_model,
            llm_server=startup_args.server,
            confidence_level=config.confidence,
            frame_delay=startup_args.frame_delay,
            min_box_area=startup_args.min_area,
            containment_threshold=startup_args.containment,
            max_box_fraction=startup_args.max_box_fraction,
            web_mode=True,
            n_label_workers=startup_args.label_workers,
        )
        threading.Thread(
            target=detector.run,
            kwargs={"censor": False, "source": config.source},
            daemon=True,
        ).start()
    else:
        labeler = RealTimeLabeler(vlm=config.vlm, llm_server=startup_args.server, web_mode=True)
        threading.Thread(target=labeler.run, kwargs={"source": config.source}, daemon=True).start()
    return jsonify(config.as_dict())


@app.route("/api/models")
def api_models():
    detection_models = []
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith((".pt", ".pth")):
                detection_models.append(f)
    known_vlms = [
        "llava:7b",
        "qwen2.5vl:7b",
        "llava:13b",
        "llava-llama3:8b",
        "moondream:latest",
        "minicpm-v:8b",
    ]
    return jsonify({"detection_models": detection_models, "vlm_models": known_vlms})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LiveLabel Web Server")
    parser.add_argument("--mode", default="detector", choices=["detector", "analyzer"],
                        help="detector: object tracking + labeling; analyzer: motion-triggered scene analysis")
    parser.add_argument("--source", default=0, type=int, choices=[0, 1])
    parser.add_argument("--vlm", default="llava:7b", type=str)
    parser.add_argument("--detection-model", default="FastSAM-s.pt", type=str)
    parser.add_argument("--server", default="http://localhost:11434", type=str)
    parser.add_argument("--confidence", default=0.3, type=float)
    parser.add_argument("--frame-delay", default=1, type=int)
    parser.add_argument("--min-area", default=4000, type=int)
    parser.add_argument("--containment", default=0.92, type=float)
    parser.add_argument("--max-box-fraction", default=0.6, type=float)
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--label-workers", default=1, type=int,
                        help="parallel VLM labeling threads (set OLLAMA_NUM_PARALLEL to same value)")
    args = parser.parse_args()

    global detector, labeler, startup_args
    startup_args = args
    config.vlm = args.vlm
    config.confidence = args.confidence
    config.source = args.source
    config.mode = args.mode

    if args.mode == "detector":
        detector = RealTimeDetector(
            vlm=args.vlm,
            detection_model=args.detection_model,
            llm_server=args.server,
            confidence_level=args.confidence,
            frame_delay=args.frame_delay,
            min_box_area=args.min_area,
            containment_threshold=args.containment,
            max_box_fraction=args.max_box_fraction,
            web_mode=True,
            n_label_workers=args.label_workers,
        )
        threading.Thread(
            target=detector.run,
            kwargs={"censor": False, "source": args.source},
            daemon=True,
        ).start()
    else:
        labeler = RealTimeLabeler(
            vlm=args.vlm,
            llm_server=args.server,
            web_mode=True,
        )
        threading.Thread(target=labeler.run, kwargs={"source": args.source}, daemon=True).start()

    # Start all watcher threads — each null-checks its global, safe regardless of mode
    threading.Thread(target=_scene_summary_worker, args=(args.server,), daemon=True).start()
    threading.Thread(target=_event_watcher, daemon=True).start()
    threading.Thread(target=_analyzer_summary_watcher, daemon=True).start()

    print(f"LiveLabel running at http://localhost:{args.port}  [mode: {args.mode}]")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
