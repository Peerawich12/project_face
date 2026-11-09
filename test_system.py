import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_sync import AttendanceStore
from supabase_sync import parse_faceshape_to_samples
from vision import VisionEngine


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


if __name__ == "__main__":
    unittest.main()
