import json
import logging
import os
import shutil
import tempfile
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger("school_attendance.data_sync")

EXCEL_FILE_PATH = Path("students.xlsx")
FACE_DATA_PATH = Path("face_data.json")
ATTENDANCE_LOG_PATH = Path("attendance_today.json")

# Excused / sick leave keywords (English & Thai)
EXCUSED_STATUS_VALUES = {
    "excused",
    "sick",
    "sick leave",
    "excused leave",
    "leave",
    "ลา",
    "ลาป่วย",
    "ลากิจ",
}


class AttendanceStore:
    """Thread-safe state store for student roster, attendance records, and metrics.

    Works standalone WITHOUT Excel (using registered face-shape labels from face_data.json)
    for testing, and also seamlessly merges rows from students.xlsx if the user adds Excel data later.
    """

    def __init__(
        self,
        excel_path: Path = EXCEL_FILE_PATH,
        attendance_log_path: Path = ATTENDANCE_LOG_PATH,
        face_data_path: Path = FACE_DATA_PATH,
    ) -> None:
        self.excel_path = excel_path
        self.attendance_log_path = attendance_log_path
        self.face_data_path = face_data_path
        self._lock = threading.RLock()

        # student_id -> dict(student_id, full_name, classroom, leave_status, is_excused)
        self._roster: Dict[str, dict] = {}
        # student_id -> HH:MM:SS timestamp string of first verification today
        self._present_today: Dict[str, str] = {}
        self._current_date: str = date.today().isoformat()
        self._last_sync_time: Optional[str] = None
        self._last_sync_error: Optional[str] = None
        self.sync_interval_seconds: int = 60

        self._scheduler: Optional[BackgroundScheduler] = None
        self._load_attendance_log()
        self.sync_from_face_data()

    def _check_date_rollover(self) -> None:
        today_str = date.today().isoformat()
        if today_str != self._current_date:
            logger.info("Date changed from %s to %s. Resetting daily attendance.", self._current_date, today_str)
            self._current_date = today_str
            self._present_today.clear()
            self._save_attendance_log()

    def _load_attendance_log(self) -> None:
        with self._lock:
            if not self.attendance_log_path.exists():
                return
            try:
                data = json.loads(self.attendance_log_path.read_text(encoding="utf-8"))
                if data.get("date") == self._current_date:
                    self._present_today = dict(data.get("present_today", {}))
            except Exception as exc:
                logger.warning("Could not load attendance log: %s", exc)

    def _save_attendance_log(self) -> None:
        try:
            payload = {
                "date": self._current_date,
                "present_today": self._present_today,
            }
            self.attendance_log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist attendance log: %s", exc)

    def reset_today_attendance(self) -> None:
        """Clear today's attendance records (useful during testing)."""
        with self._lock:
            self._present_today.clear()
            self._save_attendance_log()
            logger.info("Reset today's attendance records.")

    def register_person(
        self,
        student_id: str,
        full_name: Optional[str] = None,
        classroom: str = "-",
    ) -> None:
        """Register or update a person directly in memory (no Excel required)."""
        sid = str(student_id).strip()
        if not sid:
            return
        with self._lock:
            existing = self._roster.get(sid, {})
            self._roster[sid] = {
                "student_id": sid,
                "full_name": full_name or existing.get("full_name") or sid,
                "classroom": existing.get("classroom") or classroom,
                "leave_status": existing.get("leave_status", ""),
                "is_excused": existing.get("is_excused", False),
            }

    def remove_person(self, student_id: str) -> None:
        """Remove a person from the in-memory roster if not backed by Excel."""
        sid = str(student_id).strip()
        with self._lock:
            self._roster.pop(sid, None)
            self._present_today.pop(sid, None)
            self._save_attendance_log()

    def sync_from_face_data(self) -> None:
        """Ensure any label saved in face_data.json is represented in the roster (even without Excel)."""
        if not self.face_data_path.exists():
            return
        try:
            data = json.loads(self.face_data_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            with self._lock:
                for sid, val in data.items():
                    sid = str(sid).strip()
                    if not sid:
                        continue
                    if isinstance(val, dict):
                        name = val.get("name") or sid
                    else:
                        name = sid
                    if sid not in self._roster:
                        self._roster[sid] = {
                            "student_id": sid,
                            "full_name": name,
                            "classroom": "-",
                            "leave_status": "",
                            "is_excused": False,
                        }
        except Exception as exc:
            logger.debug("Could not sync from face_data.json: %s", exc)

    def sync_from_excel(self) -> bool:
        """
        Optional non-blocking read of students.xlsx.
        If students.xlsx is not used yet (missing or empty), the system works seamlessly
        using face_data.json and in-memory registrations.
        """
        self.sync_from_face_data()

        if not self.excel_path.exists():
            with self._lock:
                self._last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._last_sync_error = None
            return False

        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            shutil.copy2(self.excel_path, tmp_path)
            df = pd.read_excel(tmp_path, engine="openpyxl", dtype=str)
            df = df.fillna("")

            excel_roster: Dict[str, dict] = {}
            for _, row in df.iterrows():
                student_id = str(row.get("student_id", "")).strip()
                if not student_id:
                    continue

                status_raw = str(row.get("status", "Active")).strip().lower()
                if status_raw in {"inactive", "dropped", "ลาออก", "พักการเรียน"}:
                    continue

                full_name = str(row.get("full_name", "")).strip() or student_id
                classroom = str(row.get("classroom", "")).strip() or "-"
                leave_status = str(row.get("leave_status", "")).strip()
                is_excused = leave_status.lower() in EXCUSED_STATUS_VALUES

                excel_roster[student_id] = {
                    "student_id": student_id,
                    "full_name": full_name,
                    "classroom": classroom,
                    "leave_status": leave_status,
                    "is_excused": is_excused,
                }

            with self._lock:
                self._check_date_rollover()
                if excel_roster:
                    # Merge Excel roster while preserving any extra test labels from face_data.json
                    for sid, info in excel_roster.items():
                        self._roster[sid] = info
                self._last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._last_sync_error = None

            self.sync_from_face_data()
            return len(excel_roster) > 0

        except Exception as exc:
            logger.error("Excel sync error: %s", exc)
            with self._lock:
                self._last_sync_error = str(exc)
            return False

        finally:
            if tmp_path and tmp_path.exists():
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def mark_student_present(self, student_id: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Record a student/label as present today.
        Automatically adds the label to the roster if testing without Excel.
        """
        student_id = str(student_id).strip()
        if not student_id or student_id == "Unknown":
            return False

        ts_str = (timestamp or datetime.now()).strftime("%H:%M:%S")

        with self._lock:
            self._check_date_rollover()
            if student_id not in self._roster:
                self.register_person(student_id)

            if student_id not in self._present_today:
                self._present_today[student_id] = ts_str
                self._save_attendance_log()
                logger.info("Student/Label %s marked present at %s", student_id, ts_str)
            return True

    def get_public_metrics(self) -> dict:
        """Calculate aggregated attendance metrics."""
        with self._lock:
            self._check_date_rollover()
            active_ids: Set[str] = set(self._roster.keys())
            total_students = len(active_ids)

            present_ids = {sid for sid in self._present_today.keys() if sid in active_ids}
            present_students = len(present_ids)

            excused_absent_ids = {
                sid
                for sid, info in self._roster.items()
                if info.get("is_excused") and sid not in present_ids
            }
            excused_students = len(excused_absent_ids)

            missing_or_skipping = max(0, total_students - present_students - excused_students)

            attendance_rate = (
                round((present_students / total_students) * 100, 1) if total_students > 0 else 0.0
            )

            return {
                "date": self._current_date,
                "total_students": total_students,
                "present_students": present_students,
                "excused_students": excused_students,
                "missing_or_skipping": missing_or_skipping,
                "attendance_rate": attendance_rate,
                "last_sync_time": self._last_sync_time,
                "last_sync_error": self._last_sync_error,
            }

    def get_student_roster_details(self) -> List[dict]:
        """Return student/label records and their live detection status."""
        with self._lock:
            self._check_date_rollover()
            rows: List[dict] = []
            for sid, info in sorted(self._roster.items(), key=lambda item: item[0]):
                checkin_time = self._present_today.get(sid)
                if checkin_time:
                    attendance_status = "มาเรียน (สแกนแล้ว)"
                    status_code = "present"
                elif info.get("is_excused"):
                    attendance_status = f"ลา ({info.get('leave_status')})"
                    status_code = "excused"
                else:
                    attendance_status = "รอสแกน / ยังไม่พบใบหน้า"
                    status_code = "missing"

                rows.append(
                    {
                        "student_id": sid,
                        "full_name": info["full_name"],
                        "classroom": info["classroom"],
                        "leave_status": info["leave_status"],
                        "checkin_time": checkin_time or "-",
                        "attendance_status": attendance_status,
                        "status_code": status_code,
                    }
                )
            return rows

    get_teacher_roster_details = get_student_roster_details

    def start_scheduler(self, interval_seconds: Optional[int] = None) -> None:
        """Start the background scheduler."""
        if self._scheduler and self._scheduler.running:
            return

        if interval_seconds is not None:
            self.sync_interval_seconds = interval_seconds

        self.sync_from_excel()

        self._scheduler = BackgroundScheduler(daemon=True)
        self._scheduler.add_job(
            self.sync_from_excel,
            trigger="interval",
            seconds=self.sync_interval_seconds,
            id="excel_roster_sync",
            replace_existing=True,
        )
        self._scheduler.start()

    def stop_scheduler(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)


# Singleton store shared across modules
attendance_store = AttendanceStore()
