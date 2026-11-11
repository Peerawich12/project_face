import hashlib
import json
import logging
import math
import shutil
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

from data_sync import AttendanceStore, attendance_store
from supabase_sync import parse_faceshape_to_samples, supabase_manager

logger = logging.getLogger("school_attendance.vision")

FACES_DIR = Path("faces")
FACE_DATA_FILE = Path("face_data.json")
YUNET_MODEL_PATH = Path("face_detection_yunet.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes cooldown
DEFAULT_MATCH_TOLERANCE = 0.42  # Base threshold for close-range faces (tighter to separate similar arrays)
DEFAULT_MIN_MATCH_MARGIN = 0.032  # Required separation margin between #1 and #2 closest people
DEFAULT_STABILITY_FRAMES = 7  # Consecutive/consistent frames required before confirming identity
MAX_MISSED_TRACK_FRAMES = 3  # Grace frames before dropping a tracked face when detector misses a frame


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


def _filter_and_dedup_person_arrays(
    vecs: List[np.ndarray],
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Clean a person's stored 128-d face arrays to prevent similar/ambiguous cross-person matches:
    1. Rejects extreme outlier arrays (e.g. 90-degree 1-eye side profiles or false background
       detections whose distance from the person's core medoid cluster exceeds 0.48).
    2. Deduplicates near-identical arrays (distance < 0.16) so duplicate captures do not
       unfairly skew top-k distance scoring.
    3. Computes a norm-preserved inlier centroid for hybrid centroid + cosine disambiguation.
    """
    if not vecs:
        return [], None
    if len(vecs) == 1:
        return [vecs[0]], vecs[0].copy()

    arr = np.array(vecs, dtype=np.float64)
    n = len(arr)
    dist_mat = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)

    # 1. Identify the core medoid (sample with the most neighbors within 0.46, breaking ties by median distance)
    neighbor_counts = np.sum(dist_mat < 0.46, axis=1)
    median_dists = np.median(dist_mat, axis=1)
    best_idx = int(
        min(
            range(n),
            key=lambda i: (-int(neighbor_counts[i]), float(median_dists[i])),
        )
    )

    inlier_indices = [
        i
        for i in range(n)
        if dist_mat[best_idx, i] <= 0.48
        or any(
            dist_mat[best_idx, k] <= 0.38 and dist_mat[k, i] <= 0.42
            for k in range(n)
        )
    ]
    if not inlier_indices:
        inlier_indices = [best_idx]

    inliers = [arr[i] for i in inlier_indices]

    # 2. Merge/deduplicate near-identical vectors (distance < 0.16)
    deduped: List[np.ndarray] = []
    for v in inliers:
        merged = False
        for idx_d, existing in enumerate(deduped):
            if float(np.linalg.norm(v - existing)) < 0.16:
                deduped[idx_d] = 0.5 * (existing + v)
                merged = True
                break
        if not merged:
            deduped.append(v)

    if not deduped:
        deduped = [inliers[0]]

    # 3. Compute norm-preserved centroid of clean inliers
    raw_mean = np.mean(deduped, axis=0)
    mean_norm = float(np.mean([np.linalg.norm(v) for v in deduped]))
    raw_norm = float(np.linalg.norm(raw_mean))
    centroid = raw_mean * (mean_norm / raw_norm) if raw_norm > 1e-9 else raw_mean.copy()

    return deduped, centroid


class TrackedFace:
    """
    Tracks a face bounding box and stabilizes identity across `stability_frames` (default: 7 frames).
    Prevents 1-frame flickers or false matches caused by similar arrays across different people.
    """

    def __init__(
        self,
        box_full: Tuple[int, int, int, int],
        label: str = "Unknown",
        distance: float = 1.0,
        last_encoded_ts: float = 0.0,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
    ) -> None:
        self.box_full = box_full  # (top, right, bottom, left) in full frame coordinates
        self.label = label  # Confirmed label (remains "Unknown" until stable across `stability_frames`)
        self.distance = distance
        self.last_encoded_ts = last_encoded_ts
        self.stability_frames = max(1, int(stability_frames))

        # 7-frame stabilization state
        self.candidate_label: str = label if label != "Unknown" else "Unknown"
        self.stable_frames: int = self.stability_frames if label != "Unknown" else 0
        self.challenger_label: str = "Unknown"
        self.challenger_frames: int = 0
        self.missed_frames: int = 0

        self.encoding_history: Deque[np.ndarray] = deque(maxlen=self.stability_frames)
        self.frame_votes: Deque[str] = deque(maxlen=self.stability_frames)

    @property
    def is_stable(self) -> bool:
        return self.label != "Unknown" and self.stable_frames >= self.stability_frames

    @property
    def center(self) -> Tuple[float, float]:
        top, right, bottom, left = self.box_full
        return ((left + right) * 0.5, (top + bottom) * 0.5)

    @property
    def width(self) -> int:
        top, right, bottom, left = self.box_full
        return max(1, right - left)

    def update_box_smooth(self, new_box: Tuple[int, int, int, int], alpha: float = 0.68) -> None:
        """Smooth bounding box coordinates across frames while resetting missed-frame counter."""
        pt, pr, pb, pl = self.box_full
        nt, nr, nb, nl = new_box
        self.box_full = (
            int(round(alpha * nt + (1.0 - alpha) * pt)),
            int(round(alpha * nr + (1.0 - alpha) * pr)),
            int(round(alpha * nb + (1.0 - alpha) * pb)),
            int(round(alpha * nl + (1.0 - alpha) * pl)),
        )
        self.missed_frames = 0

    def get_smoothed_encoding(self, new_enc: np.ndarray) -> np.ndarray:
        """
        Return the temporally smoothed 128-d encoding across recent frames in the 7-frame window,
        preserving vector norm so random single-frame landmark noise cancels out.
        """
        if not self.encoding_history:
            return new_enc
        recent = [e for e in self.encoding_history if float(np.linalg.norm(new_enc - e)) <= 0.38]
        recent.append(new_enc)
        if len(recent) == 1:
            return new_enc
        weights = np.linspace(0.6, 1.4, num=len(recent))
        weighted_mean = np.average(np.array(recent, dtype=np.float64), axis=0, weights=weights)
        mean_norm = float(np.mean([np.linalg.norm(v) for v in recent]))
        w_norm = float(np.linalg.norm(weighted_mean))
        return weighted_mean * (mean_norm / w_norm) if w_norm > 1e-9 else new_enc

    def record_frame_match(
        self,
        matched_id: str,
        dist_score: float,
        encoding: np.ndarray,
        now_ts: float,
    ) -> bool:
        """
        Update the 7-frame stability state machine with a new frame's match result.
        Returns True if the identity is confirmed (or re-confirmed) across `self.stability_frames` frames.
        """
        self.last_encoded_ts = now_ts
        self.distance = dist_score
        self.frame_votes.append(matched_id)

        # Case 1: Not yet confirmed (building up to 7 stable frames)
        if self.label == "Unknown":
            if matched_id != "Unknown":
                if matched_id == self.candidate_label:
                    self.encoding_history.append(encoding)
                    self.stable_frames = min(self.stability_frames, self.stable_frames + 1)
                else:
                    # Switched candidate: require a fresh 7-frame streak for the new candidate
                    self.candidate_label = matched_id
                    self.stable_frames = 1
                    self.encoding_history.clear()
                    self.encoding_history.append(encoding)

                if self.stable_frames >= self.stability_frames:
                    self.label = self.candidate_label
                    self.challenger_label = "Unknown"
                    self.challenger_frames = 0
                    return True
            else:
                # Ambiguous / Unknown frame: decay stability counter so 1-2 frame false matches never confirm
                self.stable_frames = max(0, self.stable_frames - 1)
                if self.stable_frames == 0:
                    self.candidate_label = "Unknown"
                    self.encoding_history.clear()
            return False

        # Case 2: Already confirmed across 7 frames -> keep rock-solid unless 7 frames disagree
        if matched_id == self.label:
            self.encoding_history.append(encoding)
            self.stable_frames = self.stability_frames
            self.challenger_label = "Unknown"
            self.challenger_frames = 0
            return True

        # A different label or "Unknown" appeared on this frame while already confirmed
        if matched_id == self.challenger_label:
            self.challenger_frames += 1
        else:
            self.challenger_label = matched_id
            self.challenger_frames = 1

        if self.challenger_frames >= self.stability_frames:
            # The new state persisted for a full 7 frames -> transition state
            self.label = self.challenger_label
            self.candidate_label = self.challenger_label
            self.encoding_history.clear()
            if self.label != "Unknown":
                self.stable_frames = self.stability_frames
                self.encoding_history.append(encoding)
                self.challenger_label = "Unknown"
                self.challenger_frames = 0
                return True
            else:
                self.stable_frames = 0
                self.challenger_label = "Unknown"
                self.challenger_frames = 0
        return False


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
    3. Disambiguation for Similar Arrays & 7-Frame Temporal Stability:
       Filters outlier/duplicate arrays, checks centroid + cosine + second-best margin,
       and requires 7 consistent frames before confirming a detection.
    """

    def __init__(
        self,
        store: AttendanceStore = attendance_store,
        faces_dir: Path = FACES_DIR,
        face_data_file: Path = FACE_DATA_FILE,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
        camera_index: int = 1,  # Default to 1 (USB Webcam), fallback to 0 (PC Camera)
        match_tolerance: float = DEFAULT_MATCH_TOLERANCE,
        min_match_margin: float = DEFAULT_MIN_MATCH_MARGIN,
        stability_frames: int = DEFAULT_STABILITY_FRAMES,
    ) -> None:
        self.store = store
        self.faces_dir = faces_dir
        self.face_data_file = face_data_file
        self.cooldown_seconds = cooldown_seconds
        self.camera_index = camera_index
        self.match_tolerance = match_tolerance
        self.min_match_margin = min_match_margin
        self.stability_frames = max(1, int(stability_frames))
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
        self.known_person_centroids: Dict[str, np.ndarray] = {}
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
                0.55,  # Balanced confidence to avoid wall-art false positives while catching distant faces
                0.3,
                5000,
            )
            self._yunet_input_size = (640, 480)
            logger.info("YuNet long-distance face detector initialized successfully.")
        except Exception as exc:
            logger.warning("Could not initialize YuNet detector (falling back to HOG): %s", exc)
            self._yunet_detector = None

    def _detect_faces_with_landmarks(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float, float]]:
        """
        Detect faces across the frame and return `(box_full, score, eye_ratio)` for each face.
        For ultra-high-resolution images (e.g. > 800px phone uploads), normalizes YuNet input
        scale so close-up faces fit YuNet's anchor receptive field accurately.
        """
        h, w = frame_bgr.shape[:2]

        if self._yunet_detector is not None:
            try:
                max_dim = max(h, w)
                det_scale = 640.0 / float(max_dim) if max_dim > 800 else 1.0
                if det_scale < 1.0:
                    det_w = max(64, int(round(w * det_scale)))
                    det_h = max(64, int(round(h * det_scale)))
                    det_img = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
                else:
                    det_w, det_h = w, h
                    det_img = frame_bgr

                with self._yunet_lock:
                    if self._yunet_input_size != (det_w, det_h):
                        self._yunet_detector.setInputSize((det_w, det_h))
                        self._yunet_input_size = (det_w, det_h)
                    _, faces = self._yunet_detector.detect(det_img)

                results: List[Tuple[Tuple[int, int, int, int], float, float]] = []
                if faces is not None:
                    inv_scale = 1.0 / det_scale
                    for det in faces:
                        fx = float(det[0]) * inv_scale
                        fy = float(det[1]) * inv_scale
                        fw = float(det[2]) * inv_scale
                        fh = float(det[3]) * inv_scale
                        if fw < 14 or fh < 14:
                            continue

                        re_x = float(det[4]) * inv_scale
                        re_y = float(det[5]) * inv_scale
                        le_x = float(det[6]) * inv_scale
                        le_y = float(det[7]) * inv_scale
                        score = float(det[14]) if len(det) > 14 else 1.0
                        eye_dist = math.hypot(le_x - re_x, le_y - re_y)
                        eye_ratio = eye_dist / max(fw, 1.0)

                        top = max(0, int(round(fy)))
                        left = max(0, int(round(fx)))
                        bottom = min(h, int(round(fy + fh)))
                        right = min(w, int(round(fx + fw)))
                        if (right - left) >= 14 and (bottom - top) >= 14:
                            results.append(((top, right, bottom, left), score, eye_ratio))
                return results
            except Exception as exc:
                logger.debug("YuNet detect error: %s", exc)

        if FACE_RECOGNITION_AVAILABLE:
            half = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
            rgb_half = cv2.cvtColor(half, cv2.COLOR_BGR2RGB)
            with self._dlib_lock:
                locs = face_recognition.face_locations(rgb_half, number_of_times_to_upsample=1, model="hog")
            return [((t * 2, r * 2, b * 2, l * 2), 1.0, 0.42) for (t, r, b, l) in locs]

        return []

    def _detect_faces_full_frame(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces across the FULL resolution frame so students standing far away
        (2m - 6m from the camera) are detected reliably.
        Returns list of (top, right, bottom, left) in full-frame pixel coordinates.
        """
        detections = self._detect_faces_with_landmarks(frame_bgr)
        # Filter out extreme 90-degree 1-eye profile false/unalignable boxes (eye_ratio < 0.16)
        return [box for (box, _score, eye_ratio) in detections if eye_ratio >= 0.16]

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

    def _extract_array_from_bgr(
        self,
        bgr_img: np.ndarray,
        min_eye_ratio: float = 0.22,
    ) -> Optional[List[float]]:
        """
        Detect the primary face in `bgr_img` and extract its 128-d array.
        Rejects extreme 90-degree single-eye side profiles (`eye_ratio < min_eye_ratio`)
        so distorted profile encodings do not pollute `face_data.json`.
        """
        detections = self._detect_faces_with_landmarks(bgr_img)
        if not detections:
            return None

        # Prefer high-confidence detections with both eyes visible
        valid = [d for d in detections if d[1] >= 0.60 and d[2] >= min_eye_ratio]
        if not valid:
            valid = [d for d in detections if d[2] >= min_eye_ratio]
        if not valid:
            return None

        largest_box, _score, _er = max(
            valid,
            key=lambda item: (item[0][2] - item[0][0]) * (item[0][1] - item[0][3]),
        )
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
        clean outlier/duplicate samples, and save to `face_data.json`.
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
                raw_vecs = [np.array(v[:128], dtype=np.float64) for v in extracted_arrays]
                clean_vecs, _ = _filter_and_dedup_person_arrays(raw_vecs)
                face_data[label] = [np.round(v, 6).tolist() for v in clean_vecs]
                updated = True
                logger.info(
                    "Imported %d clean array(s) (from %d image(s)) for '%s'",
                    len(clean_vecs),
                    len(img_paths),
                    label,
                )

        if updated:
            _save_face_data(self.face_data_file, face_data)
            self.load_known_faces()

        return len(self.known_person_arrays)

    def load_known_faces(self) -> int:
        """Load face arrays from face_data.json into memory, filtering outliers & deduplicating."""
        self.known_person_arrays.clear()
        self.known_person_centroids.clear()
        self.known_face_encodings.clear()
        self.known_student_ids.clear()

        face_data = _load_face_data(self.face_data_file)
        for student_id, encodings_list in face_data.items():
            self.store.register_person(student_id)
            raw_vecs: List[np.ndarray] = []
            for vec in encodings_list:
                arr = np.array(vec[:128], dtype=np.float64)
                if arr.shape == (128,):
                    raw_vecs.append(arr)

            if raw_vecs:
                clean_vecs, centroid = _filter_and_dedup_person_arrays(raw_vecs)
                self.known_person_arrays[student_id] = clean_vecs
                if centroid is not None:
                    self.known_person_centroids[student_id] = centroid
                for arr in clean_vecs:
                    self.known_face_encodings.append(arr)
                    self.known_student_ids.append(student_id)

        return len(self.known_person_arrays)

    def extract_faceshape_from_current_frame(self) -> Optional[List[float]]:
        """Extract 128-d face-shape array from the current webcam frame without saving yet."""
        with self._frame_lock:
            frame = self._latest_raw_frame.copy() if self._latest_raw_frame is not None else None
        if frame is None or not FACE_RECOGNITION_AVAILABLE:
            return None
        return self._extract_array_from_bgr(frame)

    def extract_faceshape_from_image_bytes(self, image_bytes: bytes) -> Optional[List[float]]:
        """Extract 128-d face-shape array from uploaded image bytes."""
        if not image_bytes or not FACE_RECOGNITION_AVAILABLE:
            return None
        arr_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return self._extract_array_from_bgr(bgr)

    def save_custom_faceshape_record(
        self,
        id_name: str,
        m: str,
        array_faceshape: List[List[float]],
        append_sample: bool = False,
    ) -> List[List[float]]:
        """
        Save `(id_name, m, array_faceshape)` into local `face_data.json` and `AttendanceStore`,
        and reload known faces for real-time recognition.
        """
        sid = str(id_name).strip()
        m_clean = str(m or "").strip() or "-"
        new_samples = parse_faceshape_to_samples(array_faceshape)
        if not sid or not new_samples:
            return []

        face_data = _load_face_data(self.face_data_file)
        if append_sample and sid in face_data:
            combined = face_data[sid] + new_samples
        else:
            combined = new_samples

        face_data[sid] = combined
        _save_face_data(self.face_data_file, face_data)

        self.store.register_person(sid, classroom=m_clean)
        self.load_known_faces()
        self.last_seen.pop(sid, None)
        return combined

    def push_all_local_to_supabase(self) -> Tuple[int, List[str]]:
        """Push all local records in `face_data.json` to Supabase."""
        face_data = _load_face_data(self.face_data_file)
        pushed_count = 0
        errors: List[str] = []
        for sid, samples in face_data.items():
            if not samples:
                continue
            m_val = self.store.get_person_m(sid)
            ok, msg, _ = supabase_manager.upsert_record(
                id_name=sid,
                m=m_val,
                array_faceshape=samples,
                append_sample=False,
            )
            if ok:
                pushed_count += 1
            else:
                errors.append(f"{sid}: {msg}")
        return pushed_count, errors

    def register_face_from_frame(
        self,
        student_id: str,
        frame: Optional[np.ndarray] = None,
        m: str = "-",
    ) -> bool:
        """
        Capture current frame (works at close range OR at a distance!),
        save image into `faces/`, extract normalized 128-d array, save to `face_data.json`,
        and sync `(id_name, m, array_faceshape)` to Supabase if configured.
        """
        try:
            student_id = str(student_id).strip()
            m_clean = str(m or "").strip() or "-"
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
            self.store.register_person(student_id, classroom=m_clean)
            self.load_known_faces()
            self.last_seen.pop(student_id, None)

            # 4. Sync to Supabase if configured on default store
            if self.face_data_file == FACE_DATA_FILE and supabase_manager.is_configured:
                resolved_m = self.store.get_person_m(student_id)
                supabase_manager.upsert_record(
                    id_name=student_id,
                    m=resolved_m,
                    array_faceshape=current_arrays,
                    append_sample=False,
                )

            # 5. Update tracked face label immediately
            top, right, bottom, left = target_box
            cx, cy = (left + right) * 0.5, (top + bottom) * 0.5
            with self._track_lock:
                for tf in self._tracked_faces:
                    tcx, tcy = tf.center
                    if math.hypot(cx - tcx, cy - tcy) < 120:
                        tf.label = student_id
                        tf.candidate_label = student_id
                        tf.stable_frames = self.stability_frames
                        tf.challenger_label = "Unknown"
                        tf.challenger_frames = 0
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

    def register_face_from_upload(
        self,
        student_id: str,
        image_bytes: bytes,
        m: str = "-",
    ) -> bool:
        try:
            student_id = str(student_id).strip()
            m_clean = str(m or "").strip() or "-"
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
            self.store.register_person(student_id, classroom=m_clean)
            self.load_known_faces()
            self.last_seen.pop(student_id, None)

            if self.face_data_file == FACE_DATA_FILE and supabase_manager.is_configured:
                resolved_m = self.store.get_person_m(student_id)
                supabase_manager.upsert_record(
                    id_name=student_id,
                    m=resolved_m,
                    array_faceshape=current_list,
                    append_sample=False,
                )

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

        if self.face_data_file == FACE_DATA_FILE and supabase_manager.is_configured:
            ok_del, _ = supabase_manager.delete_record(student_id)
            if ok_del:
                existed = True

        self._last_folder_signature = self._compute_folder_signature()
        self.store.remove_person(student_id)
        self.load_known_faces()
        with self._track_lock:
            for tf in self._tracked_faces:
                if tf.label == student_id or tf.candidate_label == student_id:
                    tf.label = "Unknown"
                    tf.candidate_label = "Unknown"
                    tf.stable_frames = 0
                    tf.encoding_history.clear()
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
        self._set_banner(
            f"DETECTED: {student_id} (Stable {self.stability_frames}/{self.stability_frames} frames)",
            duration=3.0,
        )
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
        Compare a live 128-d face array against stored person arrays with disambiguation
        for similar arrays across different people:
        1. Hybrid score combining closest sample (`min_d`), top-2 sample average (`top2_avg`),
           and norm-preserved inlier centroid distance (`centroid_d`).
        2. Cosine direction consistency check against the person's centroid.
        3. Second-best margin check (`self.min_match_margin`) so frames that fall ambiguously
           between two people with similar arrays are rejected as "Unknown" instead of flipping.
        """
        if not self.known_person_arrays:
            return "Unknown", 1.0

        # Distance-adaptive threshold:
        # Close range (>= 90px face): self.match_tolerance (0.42)
        # Medium/Far range (20px..89px face): conservative margin up to +0.025 (max 0.445)
        extra_far_margin = 0.0
        if face_width_px < 90:
            extra_far_margin = min(0.025, (90 - face_width_px) / 90.0 * 0.03)
        effective_tolerance = self.match_tolerance + extra_far_margin

        enc_vec = np.asarray(face_encoding[:128], dtype=np.float64)
        enc_norm = float(np.linalg.norm(enc_vec))

        # Candidates list: (hybrid_score, min_d, cos_dist, person_id)
        candidates: List[Tuple[float, float, float, str]] = []

        for person_id, vec_list in self.known_person_arrays.items():
            if not vec_list:
                continue
            dists = [float(np.linalg.norm(enc_vec - v[:128])) for v in vec_list]
            dists.sort()
            min_d = dists[0]
            top2_avg = sum(dists[: min(2, len(dists))]) / min(2, len(dists))

            centroid = self.known_person_centroids.get(person_id)
            if centroid is not None:
                c_vec = centroid[:128]
                centroid_d = float(np.linalg.norm(enc_vec - c_vec))
                c_norm = float(np.linalg.norm(c_vec))
                denom = enc_norm * c_norm
                cos_sim = float(np.dot(enc_vec, c_vec) / denom) if denom > 1e-9 else 0.0
                cos_dist = 1.0 - cos_sim
            else:
                centroid_d = top2_avg
                cos_dist = 0.0

            score = 0.58 * min_d + 0.24 * top2_avg + 0.18 * centroid_d
            candidates.append((score, min_d, cos_dist, person_id))

        if not candidates:
            return "Unknown", 1.0

        candidates.sort(key=lambda item: item[0])
        best_score, best_min_d, best_cos_d, best_person = candidates[0]

        # 1. Must be within distance tolerance and directionally consistent with the person's cluster
        if (
            best_min_d > effective_tolerance
            or best_score > (effective_tolerance + 0.018)
            or best_cos_d > 0.065
        ):
            return "Unknown", 1.0

        # 2. Distinctiveness / Ambiguity margin check when multiple people are registered:
        #    If a second person's arrays are nearly as close as the best match, reject this frame
        #    unless best_min_d is an unmistakable close match (<= 0.34).
        if len(candidates) >= 2 and best_min_d > 0.34:
            second_score, second_min_d, _, _ = candidates[1]
            score_margin = second_score - best_score
            min_d_margin = second_min_d - best_min_d
            if score_margin < self.min_match_margin and min_d_margin < (self.min_match_margin * 0.85):
                return "Unknown", 1.0

        return best_person, best_score

    def _vision_worker_loop(self) -> None:
        """
        Dedicated background thread for Long-Distance Face Detection, Similar-Array Disambiguation,
        and 7-Frame Temporal Stability (`self.stability_frames = 7`).
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
                        kept_tracks: List[TrackedFace] = []
                        for tf in self._tracked_faces:
                            tf.missed_frames += 1
                            if tf.missed_frames <= MAX_MISSED_TRACK_FRAMES:
                                kept_tracks.append(tf)
                        self._tracked_faces = kept_tracks
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
                        if d < best_dist and d < max(box_w * 0.90, 50.0):
                            best_dist = d
                            best_old_idx = o_idx

                    if best_old_idx >= 0:
                        used_old.add(best_old_idx)
                        tf = old_tracked[best_old_idx]
                        tf.stability_frames = self.stability_frames
                        tf.update_box_smooth(box_full)
                    else:
                        tf = TrackedFace(
                            box_full=box_full,
                            label="Unknown",
                            distance=1.0,
                            last_encoded_ts=0.0,
                            stability_frames=self.stability_frames,
                        )

                    new_tracked.append(tf)

                    # Encode every frame while building 7-frame stability (or resolving a challenger),
                    # and re-verify every 0.45s once already confirmed across 7 frames.
                    needs_encoding = (
                        not tf.is_stable
                        or tf.challenger_frames > 0
                        or (now_ts - tf.last_encoded_ts) > 0.45
                    )
                    if self.known_person_arrays and needs_encoding:
                        boxes_to_encode.append((idx_box, box_full))

                # Keep any recently tracked face that momentarily missed 1-3 detector frames
                for o_idx, old_tf in enumerate(old_tracked):
                    if o_idx not in used_old:
                        old_tf.missed_frames += 1
                        if old_tf.missed_frames <= MAX_MISSED_TRACK_FRAMES:
                            ocx, ocy = old_tf.center
                            overlaps_new = any(
                                math.hypot(ocx - tf.center[0], ocy - tf.center[1]) < max(tf.width * 0.7, 40.0)
                                for tf in new_tracked
                            )
                            if not overlaps_new:
                                new_tracked.append(old_tf)

                # Immediately publish updated bounding boxes so labels move smoothly at 30 FPS
                with self._track_lock:
                    self._tracked_faces = new_tracked

                # 2. Extract 128-d array and update 7-frame stability state
                if boxes_to_encode and self.known_person_arrays:
                    enc_ts = time.monotonic()
                    for target_idx, box_full in boxes_to_encode:
                        if target_idx >= len(new_tracked):
                            continue
                        tf = new_tracked[target_idx]
                        enc = self._encode_face_box(frame, box_full)
                        if enc is None:
                            continue

                        face_w = max(1, box_full[1] - box_full[3])
                        raw_id, raw_score = self._match_encoding(enc, face_width_px=face_w)

                        # Temporal smoothing across up to 7 frames in `tf.encoding_history`
                        smooth_enc = tf.get_smoothed_encoding(enc)
                        smooth_id, smooth_score = self._match_encoding(smooth_enc, face_width_px=face_w)

                        if raw_id != "Unknown" and smooth_id == raw_id:
                            matched_id = raw_id
                            dist_score = min(raw_score, smooth_score)
                        elif raw_id != "Unknown" and len(tf.encoding_history) == 0:
                            matched_id = raw_id
                            dist_score = raw_score
                        elif raw_id != "Unknown" and raw_score <= 0.33:
                            matched_id = raw_id
                            dist_score = raw_score
                        else:
                            matched_id = "Unknown"
                            dist_score = 1.0

                        confirmed = tf.record_frame_match(
                            matched_id=matched_id,
                            dist_score=dist_score,
                            encoding=enc,
                            now_ts=enc_ts,
                        )
                        if confirmed and tf.label != "Unknown":
                            self.record_detection(tf.label)

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
        (NO lines drawn inside the face), including 7-frame stability progress.
        """
        annotated = frame.copy()
        detections: List[Tuple[str, Tuple[int, int, int, int]]] = []

        with self._track_lock:
            current_tracked = list(self._tracked_faces)

        for tf in current_tracked:
            top, right, bottom, left = tf.box_full
            student_id = tf.label
            is_known = student_id != "Unknown"
            is_stabilizing = (
                not is_known
                and tf.candidate_label != "Unknown"
                and 0 < tf.stable_frames < tf.stability_frames
            )

            if is_known:
                color = (0, 200, 83)  # Green: confirmed across 7 frames
                label_text = f"{student_id} ({tf.stable_frames}/{tf.stability_frames})"
            elif is_stabilizing:
                color = (235, 168, 14)  # Cyan/Amber: stabilizing across 7 frames
                label_text = f"Verifying... ({tf.stable_frames}/{tf.stability_frames})"
            else:
                color = (0, 165, 255)  # Orange: Unknown
                label_text = "Unknown"

            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

            (tw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            label_w = max(right - left, tw + 16)

            cv2.rectangle(annotated, (left, bottom), (left + label_w, bottom + 28), color, cv2.FILLED)
            cv2.putText(
                annotated,
                label_text,
                (left + 7, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
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
            f"{cam_label} ({self.stability_frames}-Frame Stable) | Saved: {len(self.known_person_arrays)} "
            f"| 'C': Capture | 'W': Switch Cam"
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
                "id_name": sid,
                "m": self.store.get_person_m(sid),
                "samples": len(enc_list),
                "saved_pics": saved_pics,
                "dims": min(128, len(first_vec)),
                "preview": f"[{preview_nums}, ...]",
                "array_faceshape": enc_list,
            }

        return {
            "engine": f"YuNet Long-Distance + {self.stability_frames}-Frame Stability + Similar-Array Disambiguation",
            "face_recognition_installed": FACE_RECOGNITION_AVAILABLE,
            "camera_active": self._camera_active,
            "camera_index": self.camera_index,
            "camera_name": "Webcam (Index 1)" if self.camera_index == 1 else "PC Camera (Index 0)",
            "known_faces_count": len(face_data),
            "total_arrays_count": len(self.known_face_encodings),
            "stability_frames": self.stability_frames,
            "match_tolerance": self.match_tolerance,
            "min_match_margin": self.min_match_margin,
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
