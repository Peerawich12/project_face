import hashlib
import json
import logging
import math
import shutil
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

from data_sync import AttendanceStore, attendance_store

logger = logging.getLogger("school_attendance.vision")

FACES_DIR = Path("faces")
FACE_DATA_FILE = Path("face_data.json")
YUNET_MODEL_PATH = Path("face_detection_yunet.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes cooldown
DEFAULT_MATCH_TOLERANCE = 0.44  # Base threshold for close-range faces


def _load_face_data(path: Path) -> Dict[str, List[List[float]]]:
    """Load face encoding arrays from face_data.json."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        result: Dict[str, List[List[float]]] = {}
        for sid, encodings in data.items():
            if isinstance(encodings, list) and len(encodings) > 0:
                if isinstance(encodings[0], (int, float)):
                    result[str(sid)] = [encodings]
                else:
                    result[str(sid)] = [
                        e for e in encodings if isinstance(e, list) and len(e) >= 128
                    ]
        return result
    except Exception as exc:
        logger.error("Failed to load face_data.json: %s", exc)
        return {}


def _save_face_data(path: Path, data: Dict[str, List[List[float]]]) -> None:
    """Save face encoding arrays to face_data.json."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved face arrays to %s (%d labels)", path, len(data))
    except Exception as exc:
        logger.error("Failed to save face_data.json: %s", exc)


class TrackedFace:
    """Tracks a face bounding box and its label smoothly across frames."""

    def __init__(
        self,
        box_full: Tuple[int, int, int, int],
        label: str = "Unknown",
        distance: float = 1.0,
        last_encoded_ts: float = 0.0,
    ) -> None:
        self.box_full = box_full  # (top, right, bottom, left) in full frame coordinates
        self.label = label
        self.distance = distance
        self.last_encoded_ts = last_encoded_ts

    @property
    def center(self) -> Tuple[float, float]:
        top, right, bottom, left = self.box_full
        return ((left + right) * 0.5, (top + bottom) * 0.5)

    @property
    def width(self) -> int:
        top, right, bottom, left = self.box_full
        return max(1, right - left)


class VisionEngine:
    """
    Long-Distance School Vision Engine:
    1. Long-Distance Face Detection (0.3m to 6m+):
       Uses OpenCV YuNet (`cv2.FaceDetectorYN`) on the FULL resolution frame (7ms)
       instead of shrinking the frame to 1/4 size, detecting even small distant faces
       down to 16x16 pixels across a classroom or hallway.
    2. Long-Distance Face Array Extraction (`_encode_face_box`):
       Crops the detected face from the full-resolution frame and upscales small/distant
       faces with Lanczos-4 interpolation to a normalized 160x160 patch before extracting
       the 128-d face array, so distant faces match close-up registered arrays accurately.
    3. Thread-Safe Capture -> Save in `faces/` -> Process Array -> Real-Time Detection:
       Protected by `_dlib_lock` so pressing Capture never crashes or exits.
    """

    def __init__(
        self,
        store: AttendanceStore = attendance_store,
        faces_dir: Path = FACES_DIR,
        face_data_file: Path = FACE_DATA_FILE,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        camera_index: int = 1,  # Default to 1 (USB Webcam), fallback to 0 (PC Camera)
        match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
    ) -> None:
        self.store = store
        self.faces_dir = faces_dir
        self.face_data_file = face_data_file
        self.cooldown_seconds = cooldown_seconds
        self.camera_index = camera_index
        self.match_tolerance = match_tolerance
        self.show_window = True

        self._switch_camera_requested: Optional[int] = None

        # Locks for thread safety
        self._dlib_lock = threading.RLock()
        self._yunet_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._track_lock = threading.Lock()

        # Cooldown dictionary: last_seen[student_id] -> monotonic timestamp
        self.last_seen: Dict[str, float] = {}

        # Known face arrays per person: student_id -> List[np.ndarray(128,)]
        self.known_person_arrays: Dict[str, List[np.ndarray]] = {}
        self.known_face_encodings: List[np.ndarray] = []
        self.known_student_ids: List[str] = []

        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._vision_thread: Optional[threading.Thread] = None

        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._camera_active = False

        # Current tracked faces drawn on the video feed (box + label around face)
        self._tracked_faces: List[TrackedFace] = []

        # Folder watcher state
        self._last_folder_signature: str = ""
        self._last_folder_check_ts: float = 0.0

        self._quick_capture_counter: int = 1
        self._last_detected_id: Optional[str] = None
        self._last_detected_time: Optional[str] = None
        self._status_banner: str = ""
        self._status_banner_until: float = 0.0

        # Initialize YuNet long-distance face detector (detects faces up to 5-6m away in ~7ms)
        self._yunet_detector = None
        self._yunet_input_size: Tuple[int, int] = (0, 0)
        self._init_yunet_detector()

        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.sync_folder_to_array(force=False)
        self.load_known_faces()

    def _init_yunet_detector(self) -> None:
        """Initialize OpenCV YuNet deep face detector for long-distance school detection."""
        if not hasattr(cv2, "FaceDetectorYN"):
            return
        try:
            if not YUNET_MODEL_PATH.exists():
                logger.info("Downloading YuNet long-distance face detection model...")
                urllib.request.urlretrieve(YUNET_MODEL_URL, str(YUNET_MODEL_PATH))
            self._yunet_detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH),
                "",
                (640, 480),
                0.48,  # Low enough confidence threshold to catch small faces far away (3-6m)
                0.3,
                5000,
            )
            self._yunet_input_size = (640, 480)
            logger.info("YuNet long-distance face detector initialized successfully.")
        except Exception as exc:
            logger.warning("Could not initialize YuNet detector (falling back to HOG): %s", exc)
            self._yunet_detector = None

    def _detect_faces_full_frame(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces across the FULL resolution frame so students standing far away
        (2m - 6m from the camera) are detected reliably.
        Returns list of (top, right, bottom, left) in full-frame pixel coordinates.
        """
        h, w = frame_bgr.shape[:2]

        # 1. Primary: YuNet on full frame (~7ms, detects faces down to 15x15 px at 5m+ distance)
        if self._yunet_detector is not None:
            try:
                with self._yunet_lock:
                    if self._yunet_input_size != (w, h):
                        self._yunet_detector.setInputSize((w, h))
                        self._yunet_input_size = (w, h)
                    _, faces = self._yunet_detector.detect(frame_bgr)

                boxes: List[Tuple[int, int, int, int]] = []
                if faces is not None:
                    for det in faces:
                        fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                        if fw < 14 or fh < 14:
                            continue
                        # Make bounding box a clean square-ish region around the face
                        top = max(0, fy)
                        left = max(0, fx)
                        bottom = min(h, fy + fh)
                        right = min(w, fx + fw)
                        if (right - left) >= 14 and (bottom - top) >= 14:
                            boxes.append((top, right, bottom, left))
                return boxes
            except Exception as exc:
                logger.debug("YuNet detect error: %s", exc)

        # 2. Fallback if YuNet unavailable: dlib HOG at 0.5x scale (2x more range than 0.25x)
        if FACE_RECOGNITION_AVAILABLE:
            half = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
            rgb_half = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
            with self._dlib_lock:
                locs = face_recognition.face_locations(rgb_half, number_of_times_to_upsample=1, model="hog")
            return [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in locs]

        return []

    def _encode_face_box(
        self,
        frame_bgr: np.ndarray,
        box_full: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract a 128-d face array from `box_full = (top, right, bottom, left)` at ANY distance!
        If the person is far from the camera (small face box, e.g. 20x20 to 120x120 px),
        crops the face with context padding from the full frame and upscales it with
        Lanczos-4 interpolation to a normalized ~160x160 face size before running dlib's
        68-point landmark alignment & 128-d ResNet encoder.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return None

        h, w = frame_bgr.shape[:2]
        top, right, bottom, left = box_full
        fw = max(1, right - left)
        fh = max(1, bottom - top)

        # Add 35% context margin around the face so forehead, chin, and ears are included
        pad_x = int(fw * 0.35)
        pad_y = int(fh * 0.35)
        crop_top = max(0, top - pad_y)
        crop_bottom = min(h, bottom + pad_y)
        crop_left = max(0, left - pad_x)
        crop_right = min(w, right + pad_x)

        crop_bgr = frame_bgr[crop_top:crop_bottom, crop_left:crop_right]
        if crop_bgr.size == 0:
            return None

        # Relative face box inside crop_bgr
        rel_top = top - crop_top
        rel_left = left - crop_left
        rel_bottom = bottom - crop_top
        rel_right = right - crop_left

        # Normalize face scale to ~160px wide so distant small faces and close-up faces
        # are encoded at the exact same resolution!
        target_face_size = 160.0
        scale = target_face_size / float(max(fw, fh))

        if scale > 1.15 or scale < 0.85:
            new_w = max(64, int(crop_bgr.shape[1] * scale))
            new_h = max(64, int(crop_bgr.shape[0] * scale))
            interp = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
            crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=interp)
            rel_top = int(rel_top * scale)
            rel_bottom = int(rel_bottom * scale)
            rel_left = int(rel_left * scale)
            rel_right = int(rel_right * scale)

        rgb_patch = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        scaled_box = (
            max(0, rel_top),
            min(rgb_patch.shape[1], rel_right),
            min(rgb_patch.shape[0], rel_bottom),
            max(0, rel_left),
        )

        with self._dlib_lock:
            encodings = face_recognition.face_encodings(rgb_patch, [scaled_box])
            if not encodings:
                return None
            return encodings[0]

    def _extract_array_from_bgr(self, bgr_img: np.ndarray) -> Optional[List[float]]:
        """Detect the primary face in `bgr_img` at any distance and extract its 128-d array."""
        boxes = self._detect_faces_full_frame(bgr_img)
        if not boxes:
            return None
        largest_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        enc = self._encode_face_box(bgr_img, largest_box)
        if enc is None:
            return None
        return np.round(enc, 6).tolist()

    def _compute_folder_signature(self) -> str:
        """Compute hash of all images in faces/ to detect newly saved images."""
        if not self.faces_dir.exists():
            return ""
        supported = {".jpg", ".jpeg", ".png", ".webp"}
        items: List[str] = []
        for p in sorted(self.faces_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in supported:
                try:
                    st = p.stat()
                    items.append(f"{p.relative_to(self.faces_dir)}:{st.st_size}:{int(st.st_mtime_ns)}")
                except OSError:
                    pass
        return hashlib.md5("|".join(items).encode("utf-8")).hexdigest()

    def sync_folder_to_array(self, force: bool = False) -> int:
        """
        Automatically import images saved in `faces/` folder, process their face into a 128-d array,
        and save to `face_data.json`.
        """
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        sig = self._compute_folder_signature()
        if not force and sig == self._last_folder_signature:
            return len(self.known_person_arrays)

        self._last_folder_signature = sig
        if not FACE_RECOGNITION_AVAILABLE:
            return 0

        supported = {".jpg", ".jpeg", ".png", ".webp"}
        grouped_images: Dict[str, List[Path]] = {}

        for item in sorted(self.faces_dir.iterdir()):
            if item.is_dir():
                label = item.name.strip()
                if not label:
                    continue
                imgs = [
                    p for p in sorted(item.iterdir())
                    if p.is_file() and p.suffix.lower() in supported
                ]
                if imgs:
                    grouped_images.setdefault(label, []).extend(imgs)

        for item in sorted(self.faces_dir.iterdir()):
            if item.is_file() and item.suffix.lower() in supported:
                label = item.stem.split("_")[0].strip()
                if not label:
                    continue
                if label not in grouped_images:
                    grouped_images.setdefault(label, []).append(item)

        if not grouped_images:
            return len(self.known_person_arrays)

        face_data = _load_face_data(self.face_data_file)
        updated = False

        for label, img_paths in grouped_images.items():
            existing = face_data.get(label, [])
            if not force and len(existing) >= len(img_paths) and len(existing) > 0:
                continue

            extracted_arrays: List[List[float]] = []
            for img_path in img_paths:
                try:
                    bgr = cv2.imread(str(img_path))
                    if bgr is None:
                        continue
                    arr = self._extract_array_from_bgr(bgr)
                    if arr is not None:
                        extracted_arrays.append(arr)
                except Exception as exc:
                    logger.warning("Failed to process image %s to array: %s", img_path, exc)

            if extracted_arrays:
                face_data[label] = extracted_arrays
                updated = True
                logger.info(
                    "Imported %d image(s) from faces/ folder into array for '%s'",
                    len(extracted_arrays),
                    label,
                )

        if updated:
            _save_face_data(self.face_data_file, face_data)
            self.load_known_faces()

        return len(self.known_person_arrays)

    def load_known_faces(self) -> int:
        """Load face arrays from face_data.json into memory."""
        self.known_person_arrays.clear()
        self.known_face_encodings.clear()
        self.known_student_ids.clear()

        face_data = _load_face_data(self.face_data_file)
        for student_id, encodings_list in face_data.items():
            self.store.register_person(student_id)
            person_vecs: List[np.ndarray] = []
            for vec in encodings_list:
                arr = np.array(vec[:128], dtype=np.float64)
                if arr.shape == (128,):
                    person_vecs.append(arr)
                    self.known_face_encodings.append(arr)
                    self.known_student_ids.append(student_id)
            if person_vecs:
                self.known_person_arrays[student_id] = person_vecs

        return len(self.known_person_arrays)

    def register_face_from_frame(
        self,
        student_id: str,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Capture current frame (works at close range OR at a distance!),
        save image into `faces/`, extract normalized 128-d array, and save to `face_data.json`.
        """
        try:
            student_id = str(student_id).strip()
            if not student_id:
                return False

            if frame is None:
                with self._frame_lock:
                    if self._latest_raw_frame is not None:
                        frame = self._latest_raw_frame.copy()

            if frame is None or not FACE_RECOGNITION_AVAILABLE:
                return False

            boxes = self._detect_faces_full_frame(frame)
            if not boxes:
                self._set_banner("No face detected - please look at camera", duration=3.0)
                return False

            target_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
            enc = self._encode_face_box(frame, target_box)
            if enc is None:
                return False

            encoding_array = np.round(enc, 6).tolist()

            # 1. Save image to folder: faces/<student_id>.jpg and faces/<student_id>/pic_N.jpg
            self.faces_dir.mkdir(parents=True, exist_ok=True)
            person_dir = self.faces_dir / student_id
            person_dir.mkdir(parents=True, exist_ok=True)

            existing_pics = sorted(person_dir.glob("pic_*.jpg"))
            next_idx = len(existing_pics) + 1
            pic_path = person_dir / f"pic_{next_idx}.jpg"

            cv2.imwrite(str(pic_path), frame)
            cv2.imwrite(str(self.faces_dir / f"{student_id}.jpg"), frame)

            # 2. Save extracted array to face_data.json
            face_data = _load_face_data(self.face_data_file)
            current_arrays = face_data.get(student_id, [])
            current_arrays.append(encoding_array)
            face_data[student_id] = current_arrays
            _save_face_data(self.face_data_file, face_data)

            # 3. Update folder signature and reload arrays into memory
            self._last_folder_signature = self._compute_folder_signature()
            self.store.register_person(student_id)
            self.load_known_faces()
            self.last_seen.pop(student_id, None)

            # 4. Update tracked face label immediately
            top, right, bottom, left = target_box
            cx, cy = (left + right) * 0.5, (top + bottom) * 0.5
            with self._track_lock:
                for tf in self._tracked_faces:
                    tcx, tcy = tf.center
                    if math.hypot(cx - tcx, cy - tcy) < 120:
                        tf.label = student_id
                        tf.distance = 0.0
                        tf.last_encoded_ts = time.monotonic()

            self.record_detection(student_id)
            self._set_banner(
                f"CAPTURED '{student_id}' -> SAVED IN faces/ & ARRAY ({len(current_arrays)} samples)",
                duration=3.5,
            )
            return True

        except Exception as exc:
            logger.error("Capture error for %s: %s", student_id, exc)
            self._set_banner(f"Capture error: {exc}", duration=3.0)
            return False

    capture_student_picture = register_face_from_frame

    def register_face_from_upload(self, student_id: str, image_bytes: bytes) -> bool:
        try:
            student_id = str(student_id).strip()
            if not student_id:
                return False

            arr_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr_bytes, cv2.IMREAD_COLOR)
            if bgr is None or not FACE_RECOGNITION_AVAILABLE:
                return False

            face_arr = self._extract_array_from_bgr(bgr)
            if face_arr is None:
                return False

            self.faces_dir.mkdir(parents=True, exist_ok=True)
            person_dir = self.faces_dir / student_id
            person_dir.mkdir(parents=True, exist_ok=True)
            existing_pics = sorted(person_dir.glob("pic_*.jpg"))
            next_idx = len(existing_pics) + 1

            cv2.imwrite(str(person_dir / f"pic_{next_idx}.jpg"), bgr)
            cv2.imwrite(str(self.faces_dir / f"{student_id}.jpg"), bgr)

            face_data = _load_face_data(self.face_data_file)
            current_list = face_data.get(student_id, [])
            current_list.append(face_arr)
            face_data[student_id] = current_list
            _save_face_data(self.face_data_file, face_data)

            self._last_folder_signature = self._compute_folder_signature()
            self.store.register_person(student_id)
            self.load_known_faces()
            self.last_seen.pop(student_id, None)
            return True
        except Exception as exc:
            logger.error("Upload error for %s: %s", student_id, exc)
            return False

    save_uploaded_student_picture = register_face_from_upload

    def delete_face_data(self, student_id: str) -> bool:
        student_id = str(student_id).strip()
        face_data = _load_face_data(self.face_data_file)
        existed = student_id in face_data
        if existed:
            del face_data[student_id]
            _save_face_data(self.face_data_file, face_data)

        person_dir = self.faces_dir / student_id
        if person_dir.exists():
            shutil.rmtree(person_dir, ignore_errors=True)
            existed = True

        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.faces_dir / f"{student_id}{ext}"
            if p.exists():
                p.unlink()
                existed = True

        self._last_folder_signature = self._compute_folder_signature()
        self.store.remove_person(student_id)
        self.load_known_faces()
        with self._track_lock:
            for tf in self._tracked_faces:
                if tf.label == student_id:
                    tf.label = "Unknown"
        return existed

    def is_on_cooldown(self, student_id: str, now_ts: Optional[float] = None) -> bool:
        now_ts = now_ts if now_ts is not None else time.monotonic()
        last_ts = self.last_seen.get(student_id)
        if last_ts is None:
            return False
        return (now_ts - last_ts) < self.cooldown_seconds

    def record_detection(self, student_id: str) -> bool:
        now_ts = time.monotonic()
        self._last_detected_id = student_id
        self._last_detected_time = datetime.now().strftime("%H:%M:%S")

        if self.is_on_cooldown(student_id, now_ts):
            return False

        self.last_seen[student_id] = now_ts
        self.store.mark_student_present(student_id, datetime.now())
        self._set_banner(f"DETECTED: {student_id}", duration=3.0)
        return True

    def _set_banner(self, text: str, duration: float = 3.0) -> None:
        self._status_banner = text
        self._status_banner_until = time.monotonic() + duration

    def _match_encoding(
        self,
        face_encoding: np.ndarray,
        face_width_px: int = 120,
    ) -> Tuple[str, float]:
        """
        Compare a live 128-d face array against stored person arrays.
        Includes distance-adaptive tolerance: when a student is far from the camera
        (small face width in pixels, e.g. < 75px at 2-5m distance), allows a slight
        optical-blur margin (+0.04) so distant students are recognized smoothly.
        """
        if not self.known_person_arrays:
            return "Unknown", 1.0

        # Distance-adaptive threshold:
        # Close range (>= 90px face): self.match_tolerance (0.44)
        # Medium/Far range (20px..85px face): smoothly up to +0.045 (0.485)
        extra_far_margin = 0.0
        if face_width_px < 90:
            extra_far_margin = min(0.045, (90 - face_width_px) / 90.0 * 0.05)
        effective_tolerance = self.match_tolerance + extra_far_margin

        best_person = "Unknown"
        best_dist = 1.0

        for person_id, vec_list in self.known_person_arrays.items():
            if not vec_list:
                continue
            dists = [float(np.linalg.norm(face_encoding - v[:128])) for v in vec_list]
            dists.sort()
            min_d = dists[0]
            top_avg = sum(dists[: min(3, len(dists))]) / min(3, len(dists))
            score = 0.65 * min_d + 0.35 * top_avg

            if min_d <= effective_tolerance and score < best_dist:
                best_dist = score
                best_person = person_id

        return best_person, best_dist

    def _vision_worker_loop(self) -> None:
        """
        Dedicated background thread for Long-Distance Face Detection & Array Matching.
        - Uses YuNet on the FULL frame (~7ms) to detect faces even 4-6 meters away.
        - Upscales small distant face crops with Lanczos-4 before computing 128-d arrays.
        """
        while self._running:
            now_ts = time.monotonic()
            if (now_ts - self._last_folder_check_ts) > 2.0:
                self._last_folder_check_ts = now_ts
                try:
                    self.sync_folder_to_array(force=False)
                except Exception:
                    pass

            with self._frame_lock:
                frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None

            if frame is None or not FACE_RECOGNITION_AVAILABLE:
                time.sleep(0.03)
                continue

            try:
                # 1. Detect faces at ANY distance on full-resolution frame (~7ms with YuNet)
                boxes_full = self._detect_faces_full_frame(frame)

                if not boxes_full:
                    with self._track_lock:
                        self._tracked_faces.clear()
                    time.sleep(0.02)
                    continue

                with self._track_lock:
                    old_tracked = list(self._tracked_faces)

                new_tracked: List[TrackedFace] = []
                used_old = set()
                boxes_to_encode: List[Tuple[int, Tuple[int, int, int, int]]] = []

                for idx_box, box_full in enumerate(boxes_full):
                    top, right, bottom, left = box_full
                    cx = (left + right) * 0.5
                    cy = (top + bottom) * 0.5
                    box_w = max(right - left, 24)

                    best_old_idx = -1
                    best_dist = float("inf")
                    for o_idx, old_tf in enumerate(old_tracked):
                        if o_idx in used_old:
                            continue
                        ocx, ocy = old_tf.center
                        d = math.hypot(cx - ocx, cy - ocy)
                        # Allow proportional movement threshold (works for both small distant & large close boxes)
                        if d < best_dist and d < max(box_w * 0.85, 45.0):
                            best_dist = d
                            best_old_idx = o_idx

                    if best_old_idx >= 0:
                        used_old.add(best_old_idx)
                        prev = old_tracked[best_old_idx]
                        tf = TrackedFace(
                            box_full=box_full,
                            label=prev.label,
                            distance=prev.distance,
                            last_encoded_ts=prev.last_encoded_ts,
                        )
                    else:
                        tf = TrackedFace(
                            box_full=box_full,
                            label="Unknown",
                            distance=1.0,
                            last_encoded_ts=0.0,
                        )

                    new_tracked.append(tf)

                    if self.known_person_arrays and (
                        tf.label == "Unknown" or (now_ts - tf.last_encoded_ts) > 1.2
                    ):
                        boxes_to_encode.append((idx_box, box_full))

                # Immediately publish updated bounding boxes so labels move smoothly at 30 FPS
                with self._track_lock:
                    self._tracked_faces = new_tracked

                # 2. Extract 128-d array from upscaled full-res face crops for any face needing ID
                if boxes_to_encode and self.known_person_arrays:
                    enc_ts = time.monotonic()
                    for target_idx, box_full in boxes_to_encode:
                        enc = self._encode_face_box(frame, box_full)
                        if enc is None:
                            continue
                        face_w = max(1, box_full[1] - box_full[3])
                        matched_id, dist_score = self._match_encoding(enc, face_width_px=face_w)
                        if matched_id != "Unknown":
                            self.record_detection(matched_id)
                        if target_idx < len(new_tracked):
                            new_tracked[target_idx].label = matched_id
                            new_tracked[target_idx].distance = dist_score
                            new_tracked[target_idx].last_encoded_ts = enc_ts

                    with self._track_lock:
                        self._tracked_faces = new_tracked

            except Exception as exc:
                logger.debug("Vision worker iteration error: %s", exc)

            time.sleep(0.015)

    def _draw_annotations(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]:
        """
        Draw ONLY the clean bounding box and label around each detected face
        (NO lines drawn inside the face).
        """
        annotated = frame.copy()
        detections: List[Tuple[str, Tuple[int, int, int, int]]] = []

        with self._track_lock:
            current_tracked = list(self._tracked_faces)

        for tf in current_tracked:
            top, right, bottom, left = tf.box_full
            student_id = tf.label
            is_known = student_id != "Unknown"
            color = (0, 200, 83) if is_known else (0, 165, 255)

            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

            label_text = student_id if is_known else "Unknown"
            (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            label_w = max(right - left, tw + 16)

            cv2.rectangle(annotated, (left, bottom), (left + label_w, bottom + 28), color, cv2.FILLED)
            cv2.putText(
                annotated,
                label_text,
                (left + 7, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            detections.append((student_id, (top, right, bottom, left)))

        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 34), (30, 27, 75), cv2.FILLED)
        cam_label = (
            f"WEBCAM [Index {self.camera_index}]"
            if self.camera_index == 1
            else f"CAMERA [Index {self.camera_index}]"
        )
        hud_text = (
            f"{cam_label} (Long-Range) | Saved: {len(self.known_person_arrays)} "
            f"| Press 'C': Capture | 'W': Switch Cam"
        )
        cv2.putText(
            annotated,
            hud_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if time.monotonic() < self._status_banner_until and self._status_banner:
            cv2.rectangle(annotated, (0, h - 36), (w, h), (16, 185, 129), cv2.FILLED)
            cv2.putText(
                annotated,
                self._status_banner,
                (12, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated, detections

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]:
        with self._frame_lock:
            self._latest_raw_frame = frame.copy()
        return self._draw_annotations(frame)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def start_background_capture(self) -> None:
        if self._running:
            return
        self._running = True
        self._vision_thread = threading.Thread(target=self._vision_worker_loop, daemon=True)
        self._vision_thread.start()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("VisionEngine started (Long-Distance YuNet + Lanczos Upscale Array Encoder).")

    def stop_background_capture(self) -> None:
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._vision_thread and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=2.0)

    def switch_camera(self, new_index: int) -> bool:
        new_index = int(new_index)
        logger.info("Switching camera to index %d requested.", new_index)
        self._switch_camera_requested = new_index
        return True

    def _open_camera_pc(self) -> cv2.VideoCapture:
        """
        Open webcam at self.camera_index.
        Avoids calling CAP_PROP_FRAME_WIDTH/HEIGHT on MSMF so USB webcams never hang on startup.
        """
        logger.info("Opening webcam at index %d...", self.camera_index)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            fallback_idx = 0 if self.camera_index != 0 else 1
            logger.warning(
                "Camera index %d not opened. Trying fallback index %d...",
                self.camera_index,
                fallback_idx,
            )
            cap = cv2.VideoCapture(fallback_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(fallback_idx)
            if cap.isOpened():
                self.camera_index = fallback_idx

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info("Webcam opened successfully at index %d.", self.camera_index)
        return cap

    def _handle_window_key(self, key: int, raw_frame: np.ndarray) -> None:
        if key == 255 or key < 0:
            return

        if key == 27:  # ESC only to hide window
            self.show_window = False
            cv2.destroyAllWindows()
        elif key in (ord("w"), ord("W"), 9):  # 'W' or TAB -> switch camera
            next_idx = 0 if self.camera_index == 1 else 1
            self.switch_camera(next_idx)
        elif key in (ord("c"), ord("C")):
            sid = f"Person_{self._quick_capture_counter}"
            self._quick_capture_counter += 1
            frame_copy = raw_frame.copy()
            threading.Thread(
                target=self.register_face_from_frame,
                args=(sid, frame_copy),
                daemon=True,
            ).start()
        elif ord("1") <= key <= ord("7"):
            sid = f"6600{key - ord('0')}"
            frame_copy = raw_frame.copy()
            threading.Thread(
                target=self.register_face_from_frame,
                args=(sid, frame_copy),
                daemon=True,
            ).start()

    def _capture_loop(self) -> None:
        if self._vision_thread is None or not self._vision_thread.is_alive():
            self._running = True
            self._vision_thread = threading.Thread(target=self._vision_worker_loop, daemon=True)
            self._vision_thread.start()

        cap = self._open_camera_pc()
        if not cap.isOpened():
            logger.warning("No camera detected at index %d.", self.camera_index)
            self._camera_active = False
            cap.release()
            self._generate_standby_frame()
            return

        self._camera_active = True
        logger.info("Camera active at index %d.", self.camera_index)

        try:
            while self._running:
                if self._switch_camera_requested is not None:
                    target_idx = self._switch_camera_requested
                    self._switch_camera_requested = None
                    if target_idx != self.camera_index:
                        self.camera_index = target_idx
                        cap.release()
                        time.sleep(0.1)
                        cap = self._open_camera_pc()
                        if cap.isOpened():
                            cam_desc = (
                                "Webcam USB (Index 1)"
                                if self.camera_index == 1
                                else f"Camera (Index {self.camera_index})"
                            )
                            self._set_banner(f"Switched to {cam_desc}", duration=3.0)
                        else:
                            self._set_banner(f"Could not open Camera {self.camera_index}", duration=3.0)

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                with self._frame_lock:
                    self._latest_raw_frame = frame.copy()

                annotated_frame, _ = self._draw_annotations(frame)

                if self.show_window:
                    try:
                        cv2.imshow(
                            "camera_pc - Long-Distance Face Detector (C: Capture | W: Switch Cam | ESC: Hide)",
                            annotated_frame,
                        )
                        key = cv2.waitKey(1) & 0xFF
                        self._handle_window_key(key, frame)
                    except Exception as exc:
                        logger.warning("OpenCV window error: %s", exc)
                        self.show_window = False

                ok, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = buffer.tobytes()

                time.sleep(0.015)
        finally:
            self._camera_active = False
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _generate_standby_frame(self) -> None:
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas[:] = (30, 41, 59)
        cv2.putText(
            canvas,
            f"Webcam not detected (Index {self.camera_index})",
            (110, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        ok, buffer = cv2.imencode(".jpg", canvas)
        if ok:
            with self._frame_lock:
                self._latest_jpeg = buffer.tobytes()

    def get_status(self) -> dict:
        face_data = _load_face_data(self.face_data_file)
        registered_ids = sorted(face_data.keys())
        roster_ids = sorted(set(self.store._roster.keys()) | set(registered_ids))
        registered_files = {sid: sid in face_data for sid in roster_ids}

        array_previews: Dict[str, dict] = {}
        for sid, enc_list in face_data.items():
            first_vec = enc_list[0] if enc_list else []
            preview_nums = ", ".join(f"{v:.3f}" for v in first_vec[:4])
            person_dir = self.faces_dir / sid
            saved_pics = (
                len(list(person_dir.glob("pic_*.jpg")))
                if person_dir.exists()
                else (1 if (self.faces_dir / f"{sid}.jpg").exists() else 0)
            )
            array_previews[sid] = {
                "samples": len(enc_list),
                "saved_pics": saved_pics,
                "dims": min(128, len(first_vec)),
                "preview": f"[{preview_nums}, ...]",
            }

        return {
            "engine": "YuNet Long-Distance Detector + Lanczos Upscale 128-d Array",
            "face_recognition_installed": FACE_RECOGNITION_AVAILABLE,
            "camera_active": self._camera_active,
            "camera_index": self.camera_index,
            "camera_name": "Webcam (Index 1)" if self.camera_index == 1 else "PC Camera (Index 0)",
            "known_faces_count": len(face_data),
            "total_arrays_count": len(self.known_face_encodings),
            "known_student_ids": registered_ids,
            "registered_files": registered_files,
            "array_previews": array_previews,
            "last_detected_id": self._last_detected_id,
            "last_detected_time": self._last_detected_time,
            "cooldown_seconds": self.cooldown_seconds,
            "active_cooldowns": {
                sid: max(0, int(self.cooldown_seconds - (time.monotonic() - ts)))
                for sid, ts in self.last_seen.items()
                if (time.monotonic() - ts) < self.cooldown_seconds
            },
        }


vision_engine = VisionEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    vision_engine.show_window = True
    vision_engine._capture_loop()
