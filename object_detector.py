from ultralytics import YOLO, FastSAM, SAM
import cv2
import math
import cvzone
import random
from PIL import Image
import base64
import requests
from io import BytesIO
import numpy as np
import threading
from queue import PriorityQueue, Empty
import time
import pyautogui
import argparse
import colorsys

# model = YOLO('yolo11n.pt')
# model = YOLO('face/last.pt')


# class_names = model.names
'''['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']'''
# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(class_names))]

def uid_color(uid):
    """Return a deterministic vivid BGR color for a given UID string."""
    h = (hash(uid) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))

class RealTimeDetector:
    def __init__(self, vlm="llava:7b", detection_model="yolo11n.pt", llm_server="http://localhost:11434", confidence_level=0.3, frame_delay=5, min_box_area=4000, containment_threshold=0.92, max_box_fraction=0.6, web_mode=False, show_boxes=True, show_overlays=True, show_labels=True, n_label_workers=1):
        self.vlm = vlm
        if detection_model[:4] == "yolo":
            self.detection_model = YOLO(f"models/{detection_model}")
        elif detection_model[:7] == "FastSAM":
            self.detection_model = FastSAM(f"models/{detection_model}")
        elif detection_model[:3] == "sam":
            # "sam" alone isn't a valid path — fall back to the base SAM model.
            if not detection_model.endswith((".pt", ".pth")):
                detection_model = "sam_b.pt"
            self.detection_model = SAM(f"models/{detection_model}")
        else:
            raise ValueError(f"Invalid detection model: {detection_model}")
        try:
            self.class_names = self.detection_model.names
        except AttributeError:
            self.class_names = {0: 'object'}
        n_colors = max(len(self.class_names), 1)
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_colors)]
        self.server = llm_server
        self.confidence_level = confidence_level
        self.current_frame = None
        self.frame_delay = frame_delay
        self.frame = 0
        # Boxes smaller than this (px²) are dropped before tracking.
        self.min_box_area = min_box_area
        # If a box's area overlapping a larger box exceeds this fraction of its
        # own area, it's considered contained and suppressed.
        self.containment_threshold = containment_threshold
        # Boxes covering more than this fraction of the total frame area are
        # treated as background/whole-scene segments and dropped.
        self.max_box_fraction = max_box_fraction
        self.web_mode = web_mode
        self.show_boxes = show_boxes
        self.show_overlays = show_overlays
        self.show_labels = show_labels
        self.source = 0  # runtime-switchable: 0=webcam, 1=screen

        # Keep VLM request size manageable and avoid client-side timeouts.
        self.image_max_dim = 384
        self.jpeg_quality = 80
        self.request_timeout_s = 60

        # need a way to store previous bounding boxes for object tracking
        self.prev_box_ids = set()
        self.prev_boxes = {}
        self.curr_box_ids = []
        self.tmp_coords = {}
        # segmentation mask polygons: uid -> np.ndarray of shape (N,2) int32, or None
        self.masks = {}
        self.tmp_masks = {}
        # need a way to store labels per bounding box
        self.labels = {}
        self.label_failures = {}  # uid -> failure count
        self.cancelled_labels = set()  # UIDs removed from tracking; skip their queued VLM calls

        # async labeling
        self.running = True
        self.max_label_retries = 3
        self.labeling_queue = PriorityQueue()
        self.labeling_in_progress = set()
        self.labeling_lock = threading.Lock()
        self._queue_counter = 0  # tiebreaker so PriorityQueue never compares uid strings

        # Spawn N workers — each sends independent requests to Ollama.
        # Set OLLAMA_NUM_PARALLEL=<n_workers> so Ollama processes them in parallel.
        self.n_label_workers = n_label_workers
        for _ in range(self.n_label_workers):
            threading.Thread(target=self._labeling_worker, daemon=True).start()

        # Store class and confidence for each tracked object
        self.object_metadata = {}  # {uid: {'class': b_class, 'conf': conf}}

        self.uid = 0

        # Web mode: JPEG buffer shared between detector thread and Flask.
        self._latest_jpeg = None
        self._jpeg_lock = threading.Lock()
        self._stopped = threading.Event()

        # Background detection: inference runs in its own thread so the display
        # loop is never blocked waiting for the model.
        self._detection_running = False
        self.detection_lock = threading.Lock()
        self.censor_regions = []  # (x1,y1,x2,y2) list updated by detection thread

        # Tracking tuning: higher = more stable IDs, but may drop/merge tracks.
        self.iou_match_threshold = 0.30
        # If IoU is low, we fall back to matching based on center distance + size similarity.
        self.iou_match_threshold_fallback = 0.15
        self.size_ratio_threshold = 0.25
        self.center_distance_fraction = 0.30
        self.center_distance_fraction_fallback = 0.18

        # Track persistence: keeps last known boxes for a few frames when detections flicker.
        # This reduces UID churn -> fewer relabels when you're motionless.
        self.max_track_age_frames = 5
        self.track_age = {}  # uid -> age in frames
        
    
    def _suppress_contained_boxes(self, detections):
        """Remove boxes that are mostly contained within a larger box.

        Standard IoU-based NMS keeps nested boxes because their IoU with the
        parent is low. This pass explicitly checks containment ratio: if more
        than `containment_threshold` of a box's area lies inside a larger box,
        the smaller one is dropped.

        `detections` is a list of (x1, y1, x2, y2, conf, b_class, polygon_pts).
        Returns the filtered list, sorted largest-area first.
        """
        if len(detections) <= 1:
            return detections

        # Sort largest area first so we always compare against the bigger box.
        detections = sorted(detections, key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        keep = [True] * len(detections)

        for i in range(len(detections)):
            if not keep[i]:
                continue
            ax1, ay1, ax2, ay2 = detections[i][:4]
            area_a = (ax2 - ax1) * (ay2 - ay1)
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                bx1, by1, bx2, by2 = detections[j][:4]
                area_b = (bx2 - bx1) * (by2 - by1)

                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                intersection = (ix2 - ix1) * (iy2 - iy1)

                # Suppress the smaller box if it's mostly inside the larger one.
                if intersection / area_b >= self.containment_threshold:
                    keep[j] = False

        return [d for d, k in zip(detections, keep) if k]

    def _update_detections(self, img, censor):
        """Run model inference + tracking. Executes in a background thread."""
        try:
            self.current_frame = img.copy()
            results = self.detection_model(img, stream=True, imgsz=640, conf=0.5, iou=0.6, retina_masks=True, max_det=100)
            new_censor_regions = []
            raw_detections = []
            frame_area = img.shape[0] * img.shape[1]

            for r in results:
                boxes = r.boxes
                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = b.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil(b.conf[0]*100)/100
                    b_class = int(b.cls[0])

                    # Extract polygon for this detection (None if model has no masks).
                    polygon_pts = None
                    if r.masks is not None and i < len(r.masks.xy):
                        pts = r.masks.xy[i]
                        if len(pts) >= 3:
                            polygon_pts = np.array(pts, dtype=np.int32)

                    if b_class == 0 and censor:
                        cx1 = max(0, x1 - 10)
                        cy1 = max(0, y1 - 10)
                        cx2 = min(img.shape[1], x2 + 10)
                        cy2 = min(img.shape[0], y2 + 10)
                        new_censor_regions.append((cx1, cy1, cx2, cy2))
                    elif conf > self.confidence_level:
                        area = (x2 - x1) * (y2 - y1)
                        if self.min_box_area <= area <= self.max_box_fraction * frame_area:
                            raw_detections.append((x1, y1, x2, y2, conf, b_class, polygon_pts))

            for x1, y1, x2, y2, conf, b_class, polygon_pts in self._suppress_contained_boxes(raw_detections):
                uid = self._generate_uid()
                self.curr_box_ids.append(uid)
                self.tmp_coords[uid] = [x1, y1, x2, y2]
                self.tmp_masks[uid] = polygon_pts
                self.object_metadata[uid] = {'class': b_class, 'conf': conf}

            self._vectorize_ious(self.curr_box_ids, self.prev_box_ids)

            # Collect UIDs that need labeling, then crop outside the lock so
            # JPEG encoding doesn't block _draw_boxes on every display frame.
            with self.detection_lock:
                self.censor_regions = new_censor_regions
                unlabeled = [
                    (uid, coords) for uid, coords in self.prev_boxes.items()
                    if uid not in self.labels
                ]

            for uid, (x1, y1, x2, y2) in unlabeled:
                with self.labeling_lock:
                    already = uid in self.labeling_in_progress
                if already:
                    continue
                try:
                    img_base64 = self.crop_and_process_coords(img, x1, y1, x2, y2)
                except ValueError:
                    continue
                area = (x2 - x1) * (y2 - y1)
                with self.labeling_lock:
                    # Re-check inside lock in case another thread beat us.
                    if uid not in self.labeling_in_progress:
                        self._queue_counter += 1
                        self.labeling_queue.put(
                            (time.time(), -area, self._queue_counter, uid, img_base64)
                        )
                        self.labeling_in_progress.add(uid)
                        print(f"[queue] added {uid}  area={area}  queue size={self.labeling_queue.qsize()}")
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            # Always release the flag so the next frame can start a new thread.
            self.tmp_coords = {}
            self.tmp_masks = {}
            self.curr_box_ids = []
            self._detection_running = False

    def _draw_boxes(self, img):
        """Draw current tracked boxes onto img. Runs in the main display loop."""
        with self.detection_lock:
            censor_regions = list(self.censor_regions)
            boxes_snapshot = list(self.prev_boxes.items())
            masks_snapshot = dict(self.masks)

        for (cx1, cy1, cx2, cy2) in censor_regions:
            roi = img[cy1:cy2, cx1:cx2]
            h, w = roi.shape[:2]
            if h > 0 and w > 0:
                temp = cv2.resize(roi, (max(1, w // 25), max(1, h // 25)), interpolation=cv2.INTER_LINEAR)
                img[cy1:cy2, cx1:cx2] = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        # Render semi-transparent segmentation masks before drawing boxes on top.
        if self.show_overlays:
            overlay = img.copy()
            drew_any_mask = False
            for uid, _ in boxes_snapshot:
                poly = masks_snapshot.get(uid)
                if poly is not None and len(poly) >= 3:
                    cv2.fillPoly(overlay, [poly], uid_color(uid))
                    drew_any_mask = True
            if drew_any_mask:
                np.copyto(img, cv2.addWeighted(overlay, 0.35, img, 0.65, 0))

        for uid, coords in boxes_snapshot:
            x1, y1, x2, y2 = coords
            metadata = self.object_metadata.get(uid, {'class': 0, 'conf': 0.3})
            obj_conf = metadata['conf']
            label = self.labels.get(uid, '[labeling...]')
            color = uid_color(uid)
            if self.show_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if self.show_labels:
                cvzone.putTextRect(img, f'{label} {obj_conf}', (max(0, x1), max(35, y1 - 20)),
                                   scale=2, thickness=2,
                                   colorR=color, colorB=color)
    
    def _labeling_worker(self):
        while self.running:
            try:
                _time, _neg_area, _counter, uid, img_base64 = self.labeling_queue.get(timeout=1.0)
                if uid in self.cancelled_labels:
                    with self.labeling_lock:
                        self.labeling_in_progress.discard(uid)
                    self.cancelled_labels.discard(uid)
                    print(f"[label] skipped stale {uid}  (queue remaining: {self.labeling_queue.qsize()})")
                    continue
                print(f"[label] calling VLM for {uid}  (queue remaining: {self.labeling_queue.qsize()})")
                try:
                    label = self._get_label_from_llm(img_base64)
                    print(f"[label] {uid} -> '{label}'")
                    with self.labeling_lock:
                        self.labels[uid] = label
                        self.labeling_in_progress.discard(uid)
                        self.label_failures.pop(uid, None)
                except Exception as e:
                    print(f"[label] ERROR for {uid}: {type(e).__name__}: {e}")
                    with self.labeling_lock:
                        failures = self.label_failures.get(uid, 0) + 1
                        self.label_failures[uid] = failures
                        self.labeling_in_progress.discard(uid)
                        if failures >= self.max_label_retries:
                            # Give up and set a permanent fallback so this UID
                            # stops retrying and blocking the queue.
                            fallback = "unknown object"
                            try:
                                obj_class = self.object_metadata[uid]['class']
                                fallback = self.class_names[obj_class]
                            except (KeyError, IndexError):
                                pass
                            self.labels[uid] = fallback
                            print(f"Gave up labeling {uid} after {failures} attempts, using '{fallback}'")
            except Empty:
                continue
            except Exception as e:
                print(f"Labeling worker unexpected error: {e}")
                continue
    
    def _get_current_frame(self):
        return getattr(self, 'current_frame', None)

    def label_box_from_coords(self, img, x1, y1, x2, y2):
        """Generate label from coordinates instead of box object"""
        img_base64 = self.crop_and_process_coords(img, x1, y1, x2, y2)
        return self._get_label_from_llm(img_base64)
    
    def _get_label_from_llm(self, img_base64):
        """Common method to get label from LLM"""
        prompt = "Describe this object in 3-5 words"

        payload = {
            "model": self.vlm,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048
            },
            "images": [img_base64]
        }

        response = requests.post(
            f"{self.server}/api/generate",
            json=payload,
            timeout=self.request_timeout_s
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'unknown object').strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return "unknown object"
    
    def crop_and_process_coords(self, img, x1, y1, x2, y2):
        """Crop and process image using coordinates directly"""
        # Clamp to valid image bounds to avoid empty/invalid crops.
        h, w = img.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            # If crop is empty, let the caller retry via exception handling.
            raise ValueError(f"Empty crop: {(x1, y1, x2, y2)}")

        crop = img[y1:y2, x1:x2]

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Downscale large crops (common with screen capture) to prevent huge base64/JPEG payloads.
        ch, cw = crop_rgb.shape[:2]
        max_side = max(ch, cw)
        if max_side > self.image_max_dim:
            scale = self.image_max_dim / float(max_side)
            new_w = max(1, int(cw * scale))
            new_h = max(1, int(ch * scale))
            crop_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        crop_PIL = Image.fromarray(crop_rgb)
        buffer = BytesIO()
        crop_PIL.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_base64
    
    def _vectorize_ious(self, curr_box_ids, prev_box_ids):
        # Stable one-to-one matching between previous tracks and new detections.
        # This prevents UID churn (and therefore label churn) when you're motionless.
        if len(self.tmp_coords) == 0:
            # No new detections: age existing tracks and keep them briefly.
            for prev_id in list(self.prev_boxes.keys()):
                self.track_age[prev_id] = self.track_age.get(prev_id, 0) + 1
                if self.track_age[prev_id] > self.max_track_age_frames:
                    self.prev_boxes.pop(prev_id, None)
                    self.masks.pop(prev_id, None)
                    self.track_age.pop(prev_id, None)
                    self.cancelled_labels.add(prev_id)
            self.prev_box_ids = set(self.prev_boxes.keys())
            return

        if len(self.prev_boxes) == 0:
            self.prev_boxes = self.tmp_coords.copy()
            self.masks = self.tmp_masks.copy()
            self.prev_box_ids = set(self.prev_boxes.keys())
            self.track_age = {uid: 0 for uid in self.prev_boxes.keys()}
            return

        curr_ids = list(self.tmp_coords.keys())
        prev_ids = list(self.prev_boxes.keys())

        # Scale the center-distance gate with the current frame size.
        frame_h, frame_w = self.current_frame.shape[:2]
        center_distance_threshold = self.center_distance_fraction * float(max(frame_w, frame_h))

        candidates = []
        for ci, curr_id in enumerate(curr_ids):
            x1, y1, x2, y2 = self.tmp_coords[curr_id]
            c_cx = (x1 + x2) / 2.0
            c_cy = (y1 + y2) / 2.0

            c_w = max(1.0, float(x2 - x1))
            c_h = max(1.0, float(y2 - y1))
            for pi, prev_id in enumerate(prev_ids):
                px1, py1, px2, py2 = self.prev_boxes[prev_id]
                p_cx = (px1 + px2) / 2.0
                p_cy = (py1 + py2) / 2.0

                dist = math.hypot(c_cx - p_cx, c_cy - p_cy)
                if dist > center_distance_threshold:
                    continue

                p_w = max(1.0, float(px2 - px1))
                p_h = max(1.0, float(py2 - py1))

                iou = self._calculate_iou(self.tmp_coords[curr_id], self.prev_boxes[prev_id])

                # Size similarity helps when box dims flicker slightly.
                w_rel = abs(c_w - p_w) / max(c_w, p_w)
                h_rel = abs(c_h - p_h) / max(c_h, p_h)
                size_similar = (w_rel <= self.size_ratio_threshold) and (h_rel <= self.size_ratio_threshold)

                # 1) Strong match: good IoU.
                if iou >= self.iou_match_threshold:
                    candidates.append((iou, ci, pi))
                    continue

                # 2) Flicker match: similar size + not-too-far center distance,
                # even if IoU is low due to small shifts.
                fallback_center_threshold = self.center_distance_fraction_fallback * float(max(frame_w, frame_h))
                if size_similar and dist <= fallback_center_threshold:
                    # Score uses size similarity implicitly; still prefer higher IoU when available.
                    size_bonus = 0.05 if iou < self.iou_match_threshold_fallback else 0.0
                    score = iou + size_bonus
                    candidates.append((score, ci, pi))

        # Greedy: assign highest-IoU matches first, ensuring each track and detection is used once.
        candidates.sort(reverse=True, key=lambda x: x[0])

        assigned_curr = set()
        assigned_prev = set()
        new_prev_boxes = {}
        new_masks = {}

        for iou, ci, pi in candidates:
            if ci in assigned_curr or pi in assigned_prev:
                continue
            curr_id = curr_ids[ci]
            prev_id = prev_ids[pi]
            new_prev_boxes[prev_id] = self.tmp_coords[curr_id]
            new_masks[prev_id] = self.tmp_masks.get(curr_id)
            # Propagate fresh detection metadata to the stable tracking UID.
            if curr_id in self.object_metadata:
                self.object_metadata[prev_id] = self.object_metadata[curr_id]
            assigned_curr.add(ci)
            assigned_prev.add(pi)

        # Unmatched previous tracks persist briefly (reduces box flicker + relabel churn).
        for pi, prev_id in enumerate(prev_ids):
            if pi in assigned_prev:
                # Matched -> reset age.
                self.track_age[prev_id] = 0
                continue
            self.track_age[prev_id] = self.track_age.get(prev_id, 0) + 1
            if self.track_age[prev_id] <= self.max_track_age_frames:
                new_prev_boxes[prev_id] = self.prev_boxes[prev_id]
                new_masks[prev_id] = self.masks.get(prev_id)
            else:
                self.cancelled_labels.add(prev_id)

        # Anything not matched becomes a new track (new UID from this frame).
        # Try to inherit a label from the nearest unmatched old labeled track so
        # that brief tracking gaps don't cause a full re-label.
        # One-to-one: each old UID can donate its label to at most one new UID.
        frame_diag = math.hypot(frame_w, frame_h)
        inherit_threshold = 0.15 * frame_diag
        inherited_from = set()

        for ci, curr_id in enumerate(curr_ids):
            if ci not in assigned_curr:
                new_prev_boxes[curr_id] = self.tmp_coords[curr_id]
                new_masks[curr_id] = self.tmp_masks.get(curr_id)
                self.track_age[curr_id] = 0

                cx1, cy1, cx2, cy2 = self.tmp_coords[curr_id]
                ccx, ccy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
                new_area = max(1.0, float((cx2 - cx1) * (cy2 - cy1)))

                best_uid, best_dist = None, float('inf')
                for pi, old_uid in enumerate(prev_ids):
                    if pi in assigned_prev or old_uid not in self.labels:
                        continue
                    if old_uid in inherited_from:
                        continue  # already donated to another new track
                    ox1, oy1, ox2, oy2 = self.prev_boxes[old_uid]
                    old_area = max(1.0, float((ox2 - ox1) * (oy2 - oy1)))
                    # Skip if the boxes are very different in size — different objects.
                    if max(new_area, old_area) / min(new_area, old_area) > 2.5:
                        continue
                    dist = math.hypot(ccx - (ox1 + ox2) / 2.0, ccy - (oy1 + oy2) / 2.0)
                    if dist < best_dist:
                        best_dist, best_uid = dist, old_uid

                if best_uid and best_dist < inherit_threshold:
                    self.labels[curr_id] = self.labels[best_uid]
                    inherited_from.add(best_uid)

        self.prev_boxes = new_prev_boxes
        self.masks = new_masks
        self.prev_box_ids = set(self.prev_boxes.keys())

    
    def _calculate_iou(self, curr_box, prev_box):
        x1 = max(curr_box[0], prev_box[0])
        y1 = max(curr_box[1], prev_box[1])
        x2 = min(curr_box[2], prev_box[2])
        y2 = min(curr_box[3], prev_box[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
        area2 = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
    
    def _generate_uid(self):
        self.uid += 1
        return "obj_" + str(self.uid)

    def get_annotated_jpeg(self):
        """Return the latest annotated frame as JPEG bytes (for web streaming)."""
        with self._jpeg_lock:
            return self._latest_jpeg

    def get_raw_frame(self):
        """Return the latest raw (unannotated) frame as a numpy array."""
        frame = self.current_frame
        return frame.copy() if frame is not None else None

    def get_raw_jpeg(self):
        """Return the latest raw (unannotated) frame as JPEG bytes."""
        frame = self.current_frame
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def stop(self):
        self.running = False

    def run(self, censor=False, source=0):
        self.source = source
        active_source = source
        cap = None
        if source == 0:
            cap = cv2.VideoCapture(0)
            cap.set(3, 320)
            cap.set(4, 320)

        while self.running:
            # Hot-switch source if changed via UI
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
                img = cv2.cvtColor(np.array(ss), cv2.COLOR_RGB2BGR)
            else:
                if cap is None:
                    continue
                success, img = cap.read()

            # Kick off detection in background whenever the previous run finished.
            if not self._detection_running and self.frame % self.frame_delay == 0:
                self._detection_running = True
                threading.Thread(
                    target=self._update_detections,
                    args=(img.copy(), censor),
                    daemon=True
                ).start()

            # Draw last known boxes every frame — never blocks on inference.
            self._draw_boxes(img)

            self.frame += 1
            if self.web_mode:
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
            else:
                cv2.imshow("image", img)
                cv2.waitKey(1)

        if cap is not None:
            cap.release()
        self._stopped.set()

def main():
    parser = argparse.ArgumentParser(description="Real Time Object Detector + Identifier")
    parser.add_argument('--censor', action='store_true', help="pixelate faces/people (class=0) instead of labeling")
    parser.add_argument('--source', default=0, type=int, choices=[0, 1], help="video source [0:webcam, 1:screen capture]")
    parser.add_argument('--vlm', default='qwen2.5vl:7b', type=str, help='VLM model name')
    parser.add_argument('--detection-model', default='FastSAM-s.pt', type=str, help='object detection model file (e.g. FastSAM-s.pt, yolo11n.pt, sam_b.pt, sam_l.pt)')
    parser.add_argument('--server', default='http://localhost:11434', type=str, help='VLM model server URL')
    parser.add_argument('--confidence', default=0.3, type=float, help="confidence threshold for object detection")
    parser.add_argument('--frame-delay', default=1, type=int, help='object detection intervals')
    parser.add_argument('--min-area', default=4000, type=int, help='minimum bounding box area in pixels² (filters tiny segments)')
    parser.add_argument('--containment', default=0.92, type=float, help='suppress a box if this fraction of it lies inside a larger box')
    parser.add_argument('--max-box-fraction', default=0.6, type=float, help='drop boxes covering more than this fraction of the frame (background segments)')
    parser.add_argument('--web', action='store_true', help='enable web dashboard mode (no cv2.imshow)')
    parser.add_argument('--port', default=5000, type=int, help='port for web dashboard (used by web_server.py)')

    args = parser.parse_args()
    detector = RealTimeDetector(
        vlm=args.vlm,
        detection_model=args.detection_model,
        llm_server=args.server,
        confidence_level=args.confidence,
        frame_delay=args.frame_delay,
        min_box_area=args.min_area,
        containment_threshold=args.containment,
        max_box_fraction=args.max_box_fraction,
        web_mode=args.web,
        )
    detector.run(args.censor, args.source)

if __name__ == "__main__":
    main()