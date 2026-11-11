import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_sync import AttendanceStore
from supabase_sync import parse_faceshape_to_samples
from vision import TrackedFace, VisionEngine


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


if __name__ == "__main__":
    unittest.main()
