import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_sync import AttendanceStore
from supabase_sync import normalize_classroom_code, parse_faceshape_to_samples
from vision import FaceCropSample, TrackedFace, VisionEngine


class TestSchoolAttendanceSystem(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmp_dir.name)
        self.excel_path = self.base_path / "test_students.xlsx"
        self.log_path = self.base_path / "test_attendance.json"
        self.faces_dir = self.base_path / "faces"
        self.face_data_file = self.base_path / "test_face_data.json"

        self.store = AttendanceStore(
            excel_path=self.excel_path,
            attendance_log_path=self.log_path,
            face_data_path=self.face_data_file,
        )
        self.vision = VisionEngine(
            store=self.store,
            faces_dir=self.faces_dir,
            face_data_file=self.face_data_file,
            cooldown_seconds=300,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_standalone_without_excel(self):
        ok = self.store.sync_from_excel()
        self.assertFalse(ok)
        metrics = self.store.get_public_metrics()
        self.assertEqual(metrics["total_students"], 0)
        self.assertIsNone(metrics["last_sync_error"])

    def test_array_matching_and_false_match_rejection(self):
        rng = np.random.default_rng(42)
        base_encoding = rng.normal(0, 0.1, 128)

        samples = [(base_encoding + rng.normal(0, 0.015, 128)).tolist() for _ in range(5)]
        with open(self.face_data_file, "w", encoding="utf-8") as f:
            json.dump({"101": samples}, f)

        loaded_people = self.vision.load_known_faces()
        self.assertEqual(loaded_people, 1)

        # Same person matches
        live_same = base_encoding + rng.normal(0, 0.015, 128)
        matched_id, dist = self.vision._match_encoding(live_same)
        self.assertEqual(matched_id, "101")
        self.assertLess(dist, 0.35)

        # Different person (~0.48 away) is rejected as Unknown
        direction = rng.normal(0, 1, 128)
        direction /= np.linalg.norm(direction)
        other_person = base_encoding + direction * 0.48
        matched_other, _ = self.vision._match_encoding(other_person)
        self.assertEqual(matched_other, "Unknown")

    def test_cooldown_debouncing(self):
        self.store.register_person("101")
        self.assertTrue(self.vision.record_detection("101"))
        self.assertFalse(self.vision.record_detection("101"))

    def test_supabase_faceshape_parsing_and_custom_record(self):
        rng = np.random.default_rng(7)
        vec_128 = rng.normal(0, 0.1, 128).tolist()

        # Test 1D list, 2D list, and JSON string parsing
        parsed_1d = parse_faceshape_to_samples(vec_128)
        self.assertEqual(len(parsed_1d), 1)
        self.assertEqual(len(parsed_1d[0]), 128)

        parsed_json = parse_faceshape_to_samples(json.dumps([vec_128]))
        self.assertEqual(len(parsed_json), 1)

        # Save record with id_name, m, array_faceshape
        saved = self.vision.save_custom_faceshape_record(
            id_name="09627 สมชาย",
            m="ม.4/1",
            array_faceshape=parsed_1d,
            append_sample=False,
        )
        self.assertEqual(len(saved), 1)
        self.assertEqual(self.store.get_person_m("09627 สมชาย"), "ม.4/1")

        # Verify live matching works with the saved record
        matched_id, _ = self.vision._match_encoding(np.array(vec_128))
        self.assertEqual(matched_id, "09627 สมชาย")

    def test_similar_arrays_disambiguation_and_outlier_filtering(self):
        rng = np.random.default_rng(123)
        base_a = rng.normal(0, 0.1, 128)
        direction = rng.normal(0, 1, 128)
        direction /= np.linalg.norm(direction)
        # Person B is 0.48 away from Person A
        base_b = base_a + direction * 0.48

        samples_a = [(base_a + rng.normal(0, 0.012, 128)).tolist() for _ in range(4)]
        # Add one corrupted outlier array to Person A (0.75 away)
        samples_a.append((base_a + direction * 0.75).tolist())
        samples_b = [(base_b + rng.normal(0, 0.012, 128)).tolist() for _ in range(4)]

        with open(self.face_data_file, "w", encoding="utf-8") as f:
            json.dump({"PersonA": samples_a, "PersonB": samples_b}, f)

        self.vision.load_known_faces()
        # Corrupted outlier in PersonA should be filtered out
        self.assertEqual(len(self.vision.known_person_arrays["PersonA"]), 4)

        # Midpoint vector equidistant (~0.24 / ~0.39) between PersonA and PersonB is rejected as ambiguous
        midpoint = 0.5 * (base_a + base_b) + rng.normal(0, 0.025, 128)
        ambiguous_id, _ = self.vision._match_encoding(midpoint)
        self.assertEqual(ambiguous_id, "Unknown")

        # Clear match to PersonA succeeds
        matched_a, _ = self.vision._match_encoding(base_a + rng.normal(0, 0.01, 128))
        self.assertEqual(matched_a, "PersonA")

    def test_seven_frame_stability(self):
        rng = np.random.default_rng(99)
        enc_a = rng.normal(0, 0.1, 128)
        enc_b = rng.normal(0, 0.1, 128)

        tf = TrackedFace(box_full=(100, 260, 260, 100), stability_frames=7)
        self.assertEqual(tf.stability_frames, 7)

        # A 2-frame false match to a similar array ("09627") must NOT confirm the face
        for i in range(2):
            confirmed = tf.record_frame_match("09627", 0.36, enc_b, float(i))
            self.assertFalse(confirmed)
            self.assertEqual(tf.label, "Unknown")

        # Switching to the true identity ("10511") requires a full 7 consistent frames
        for i in range(1, 7):
            confirmed = tf.record_frame_match("10511", 0.25, enc_a, float(i + 2))
            self.assertFalse(confirmed)
            self.assertEqual(tf.label, "Unknown")
            self.assertEqual(tf.stable_frames, i)

        # 7th consistent frame confirms "10511"
        confirmed_7th = tf.record_frame_match("10511", 0.25, enc_a, 10.0)
        self.assertTrue(confirmed_7th)
        self.assertEqual(tf.label, "10511")
        self.assertTrue(tf.is_stable)

        # Once confirmed across 7 frames, a transient 1-frame glitch does NOT flip the label
        tf.record_frame_match("Unknown", 1.0, enc_a, 11.0)
        self.assertEqual(tf.label, "10511")
        tf.record_frame_match("09627", 0.39, enc_b, 12.0)
        self.assertEqual(tf.label, "10511")

    def test_fast_walking_verification_across_7_frames(self):
        rng = np.random.default_rng(77)
        enc = rng.normal(0, 0.1, 128)

        # Student enters camera frame and walks rapidly horizontally (+45px per frame)
        tf = TrackedFace(box_full=(120, 220, 220, 120), stability_frames=7)
        tf.pre_encoded_track_frames = 1

        # 3 camera frames tracked at 45 FPS while background encoder computes 1st 128-d vector
        for step in range(1, 4):
            shift = step * 45
            tf.update_box_smooth((120, 220 + shift, 220, 120 + shift))
            self.assertFalse(tf.advance_tracked_frame(0.02 * step))

        self.assertGreater(tf.vx, 20.0)

        # Background encoder finishes with a strong match (dist = 0.29 <= 0.38)
        # Pre-tracked frames credit up to 3 frames -> reaches 4/7
        confirmed = tf.record_frame_match("09584", 0.29, enc, 0.18)
        self.assertFalse(confirmed)
        self.assertEqual(tf.stable_frames, 4)

        # Next 3 tracked camera frames advance 5/7 -> 6/7 -> 7/7 and confirm in ~0.24s total!
        self.assertFalse(tf.advance_tracked_frame(0.20))
        self.assertEqual(tf.stable_frames, 5)
        self.assertFalse(tf.advance_tracked_frame(0.22))
        self.assertEqual(tf.stable_frames, 6)
        self.assertTrue(tf.advance_tracked_frame(0.24))
        self.assertEqual(tf.stable_frames, 7)
        self.assertEqual(tf.label, "09584")
        self.assertTrue(tf.is_stable)

        # Verify borderline match (dist = 0.40 > 0.38) caps at 6/7 until 2nd encoding confirms
        tf_borderline = TrackedFace(box_full=(100, 200, 200, 100), stability_frames=7)
        tf_borderline.record_frame_match("09627", 0.40, enc, 1.0)
        for step in range(10):
            self.assertFalse(tf_borderline.advance_tracked_frame(1.0 + 0.02 * step))
        self.assertEqual(tf_borderline.stable_frames, 6)
        self.assertEqual(tf_borderline.label, "Unknown")
        # 2nd encoding confirms borderline match -> reaches 7/7
        self.assertTrue(tf_borderline.record_frame_match("09627", 0.39, enc, 1.25))
        self.assertEqual(tf_borderline.label, "09627")

    def test_classroom_normalization_and_scan_classification(self):
        self.assertEqual(normalize_classroom_code("ม.5/1"), "5/1")
        self.assertEqual(normalize_classroom_code("5/3"), "5/3")
        self.assertEqual(normalize_classroom_code("ม. 2 / 4"), "2/4")
        self.assertEqual(normalize_classroom_code("-"), "-")

        self.assertEqual(self.store._classify_scan_status("07:45:00"), "present")
        self.assertEqual(self.store._classify_scan_status("08:00:00"), "late")
        self.assertEqual(self.store._classify_scan_status("08:30:12"), "late")

    def test_paired_crop_sharpness_beats_motion_blur(self):
        tf = TrackedFace(box_full=(100, 200, 200, 100), stability_frames=5)
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)

        sharp_crop = FaceCropSample(
            crop_bgr=dummy,
            rel_box=(10, 90, 90, 10),
            rel_landmarks=None,
            face_w=80,
            det_score=0.92,
            sharpness=240.0,
            quality_score=0.88,
            pitch_severity=0.4,
            captured_ts=1.00,
            frame_seq=10,
        )
        blurry_crop = FaceCropSample(
            crop_bgr=dummy,
            rel_box=(10, 90, 90, 10),
            rel_landmarks=None,
            face_w=80,
            det_score=0.70,
            sharpness=12.0,
            quality_score=0.40,
            pitch_severity=0.4,
            captured_ts=1.05,
            frame_seq=11,
        )

        tf.offer_crop_sample(sharp_crop)
        tf.offer_crop_sample(blurry_crop)
        # Sharp frame must be retained instead of overwritten by a blurry walking frame
        chosen = tf.pop_pending_crop()
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.frame_seq, 10)
        self.assertIsNone(tf.pop_pending_crop())

    def test_high_camera_pitch_and_margin_disambiguation(self):
        rng = np.random.default_rng(321)
        base_a = rng.normal(0, 0.1, 128)
        d_vec = rng.normal(0, 1, 128)
        d_vec /= np.linalg.norm(d_vec)
        base_b = base_a + d_vec * 0.58

        samples_a = [(base_a + rng.normal(0, 0.012, 128)).tolist() for _ in range(4)]
        samples_b = [(base_b + rng.normal(0, 0.012, 128)).tolist() for _ in range(4)]
        with open(self.face_data_file, "w", encoding="utf-8") as f:
            json.dump({"StudentA": samples_a, "StudentB": samples_b}, f)
        self.vision.load_known_faces()

        # Construct a pitched overhead vector ~0.44 away from StudentA (and far from StudentB)
        ortho = rng.normal(0, 1, 128)
        ortho -= np.dot(ortho, d_vec) * d_vec
        ortho /= np.linalg.norm(ortho)
        pitched_a = base_a + ortho * 0.435

        # At close frontal eye-level (pitch=0, width=140), strict 0.42 rejects it
        pred_frontal, _ = self.vision._match_encoding(pitched_a, face_width_px=140, pitch_severity=0.0)
        self.assertEqual(pred_frontal, "Unknown")

        # Under high-mounted camera (pitch=0.7, distant width=55) with clear separation from StudentB, matches StudentA
        pred_high, _, margin = self.vision._match_encoding_detailed(
            pitched_a, face_width_px=55, pitch_severity=0.7
        )
        self.assertEqual(pred_high, "StudentA")
        self.assertGreater(margin, 0.05)

        # Ambiguous midpoint vector between StudentA and StudentB is STILL rejected even in high-camera mode
        midpoint = 0.5 * (base_a + base_b)
        pred_mid, _ = self.vision._match_encoding(midpoint, face_width_px=55, pitch_severity=0.7)
        self.assertEqual(pred_mid, "Unknown")

    def test_high_camera_fast_walk_coasting_and_low_quality_resilience(self):
        rng = np.random.default_rng(55)
        enc = rng.normal(0, 0.1, 128)
        tf = TrackedFace(box_full=(100, 180, 180, 100), stability_frames=5)
        tf.pre_encoded_track_frames = 3

        # High-margin overhead match (dist=0.395, strong_margin=True) gets pre-track credit -> 4/5
        confirmed = tf.record_frame_match("09584", 0.395, enc, 0.20, strong_margin=True)
        self.assertFalse(confirmed)
        self.assertEqual(tf.stable_frames, 4)

        # A motion-blurred unreadable walking frame (is_low_quality=True) does NOT erase progress
        tf.record_frame_match("Unknown", 1.0, enc, 0.24, is_low_quality=True)
        self.assertEqual(tf.stable_frames, 4)
        self.assertEqual(tf.candidate_label, "09584")

        # Velocity coasting across a missed detection frame keeps bounding box moving smoothly
        tf.vx = 20.0
        tf.vy = 4.0
        old_cx = tf.center[0]
        tf.coast_missed_frame(frame_w=1280, frame_h=720)
        self.assertEqual(tf.missed_frames, 1)
        self.assertGreater(tf.center[0], old_cx)

        # Next tracked camera frame completes 5/5 confirmation immediately!
        self.assertTrue(tf.advance_tracked_frame(0.28))
        self.assertEqual(tf.label, "09584")
        self.assertTrue(tf.is_stable)

    def test_vision_config_toggle(self):
        status_default = self.vision.get_status()
        self.assertTrue(status_default["high_camera_mode"])
        self.assertEqual(status_default["stability_frames"], 5)

        updated = self.vision.configure_vision(high_camera_mode=False, stability_frames=4)
        self.assertFalse(updated["high_camera_mode"])
        self.assertEqual(updated["stability_frames"], 4)


if __name__ == "__main__":
    unittest.main()
