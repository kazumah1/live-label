import cv2
import cvzone
from PIL import Image
import base64
import requests
from io import BytesIO
import numpy as np
import threading
import time
import pyautogui
import argparse
from collections import deque


class RealTimeLabeler:
    def __init__(self, vlm="llava:7b", subtractor="MOG2", llm_server="http://localhost:11434", motion_threshold=0.3, web_mode=False):
        self.vlm = vlm
        self.server = llm_server
        self.current_frame = None
        self.motion_threshold = motion_threshold
        # Scale area threshold: higher motion_threshold = less sensitive (requires larger movement).
        # Default 0.3 -> 3000px²; increase --motion to require larger movement.
        self.motion_area_threshold = int(10000 * max(motion_threshold, 0.01))
        self.ssim_threshold = 0.75
        if subtractor == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN()
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2()
        self.frame = 0

        self.label = "thinking..."
        self.motion_detected = False
        self.web_mode = web_mode
        self.source = 0
        self._latest_jpeg = None
        self._jpeg_lock = threading.Lock()
        self._stopped = threading.Event()

        self.frame_buffer = deque()
        self.buffer_size = 10

        self._pending_frame = None
        self._pending_lock = threading.Lock()
        self.analysis_running = True
        self.running = True
        self.last_analysis_time = 0
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_thread.start()

    def _analysis_worker(self):
        while self.analysis_running:
            frame = None
            with self._pending_lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                img_b64 = self.process_img(frame)
                self.label = self._get_label_from_llm(img_b64)
            except Exception as e:
                print(f"[analyzer] worker error: {e}")

    def _get_label_from_llm(self, img_base64):
        payload = {
            "model": self.vlm,
            "prompt": "Describe the main subject in detail in 3-10 words",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 48,
            },
            "images": [img_base64]
        }

        response = requests.post(
            f"{self.server}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            label = result.get('response', '').strip()
            return label if label else "unknown object"
        else:
            print(f"[analyzer] Ollama error {response.status_code}: {response.text[:200]}")
            return "unknown object"

    def process_img(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_PIL = Image.fromarray(img_rgb)
        buffer = BytesIO()
        crop_PIL.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def detect_motion(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.popleft()
        if len(self.frame_buffer) < 2:
            return True

        curr_mask = self.subtractor.apply(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        curr_mask = cv2.morphologyEx(curr_mask, cv2.MORPH_OPEN, kernel=kernel)
        curr_mask = cv2.morphologyEx(curr_mask, cv2.MORPH_CLOSE, kernel=kernel)
        contours, _ = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        subtractor_motion = sum(cv2.contourArea(c) for c in contours) > self.motion_area_threshold

        frame_diff = cv2.absdiff(self.frame_buffer[-1], self.frame_buffer[-2])
        _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        diff_contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buffer_motion = sum(cv2.contourArea(c) for c in diff_contours) > (self.motion_area_threshold / 2)

        return subtractor_motion or buffer_motion

    def get_annotated_jpeg(self):
        with self._jpeg_lock:
            return self._latest_jpeg

    def stop(self):
        self.running = False
        self.analysis_running = False

    def run(self, source=0):
        self.source = source
        active_source = source
        cap = None
        if source == 0:
            cap = cv2.VideoCapture(0)
            cap.set(3, 320)
            cap.set(4, 320)
        analysis_cooldown_s = 0.5

        while self.running:
            if self.source != active_source:
                if cap is not None:
                    cap.release()
                    cap = None
                active_source = self.source
                if active_source == 0:
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 320)
                    cap.set(4, 320)

            if active_source == 1:
                ss = pyautogui.screenshot()
                raw_frame = np.array(ss)
                img = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            else:
                if cap is None:
                    continue
                success, img = cap.read()
                raw_frame = img

            # Use raw frame for motion detection to avoid label overlay bleeding into signal.
            motion_frame = raw_frame.copy() if active_source == 1 else img

            now = time.time()
            has_motion = self.detect_motion(motion_frame)
            self.motion_detected = has_motion

            if has_motion and (now - self.last_analysis_time) >= analysis_cooldown_s:
                with self._pending_lock:
                    self._pending_frame = img.copy()
                self.last_analysis_time = now

            if self.label:
                cvzone.putTextRect(img, self.label, (10, 30), scale=1, thickness=1,
                                   colorR=(0, 0, 0), colorB=(0, 0, 0))
            if not self.web_mode:
                cv2.imshow("image", img)
                cv2.waitKey(1)
            else:
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()

            self.frame += 1

        if cap is not None:
            cap.release()
        self._stopped.set()


def main():
    parser = argparse.ArgumentParser(description="Real Time Scene Labeler")
    parser.add_argument('--source', default=0, type=int, choices=[0, 1], help="video source [0:webcam, 1:screen capture]")
    parser.add_argument('--vlm', default='llava:7b', type=str, help='VLM model name')
    parser.add_argument('--subtractor', default='MOG2', type=str, help='background subtractor type')
    parser.add_argument('--server', default='http://localhost:11434', type=str, help='Ollama server URL')
    parser.add_argument('--motion', default=0.3, type=float, help="motion sensitivity threshold (higher = less sensitive)")

    args = parser.parse_args()
    labeler = RealTimeLabeler(
        vlm=args.vlm,
        subtractor=args.subtractor,
        llm_server=args.server,
        motion_threshold=args.motion,
    )
    labeler.run(args.source)


if __name__ == "__main__":
    main()
