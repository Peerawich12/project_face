import json
import logging
import os
import shutil
import tempfile
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from supabase_sync import supabase_manager

logger = logging.getLogger("school_attendance.data_sync")

EXCEL_FILE_PATH = Path("students.xlsx")
FACE_DATA_PATH = Path("face_data.json")
FACE_META_PATH = Path("face_meta.json")
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
        face_meta_path: Optional[Path] = None,
    ) -> None:
        self.excel_path = excel_path
        self.attendance_log_path = attendance_log_path
        self.face_data_path = face_data_path
        self.face_meta_path = face_meta_path or (
            FACE_META_PATH if face_data_path == FACE_DATA_PATH else face_data_path.parent / "face_meta.json"
        )
        self._is_default_store = face_data_path == FACE_DATA_PATH
        self._lock = threading.RLock()

        # student_id -> dict(student_id, full_name, classroom, leave_status, is_excused)
        self._roster: Dict[str, dict] = {}
        # student_id -> dict(m, full_name) persisted locally
        self._meta: Dict[str, dict] = {}
        # student_id -> HH:MM:SS timestamp string of first verification today
        self._present_today: Dict[str, str] = {}
        # student_id -> remote/teacher override dict(status, first_scan_at, is_finalized, audit_entries)
        self._attendance_overrides: Dict[str, dict] = {}
        self._current_date: str = date.today().isoformat()
        self._last_sync_time: Optional[str] = None
        self._last_sync_error: Optional[str] = None
        self.sync_interval_seconds: int = 60

        self._scheduler: Optional[BackgroundScheduler] = None
        self._on_faces_updated_callbacks: List[Callable[[], None]] = []
        self._load_face_meta()
        self._load_attendance_log()
        self.sync_from_face_data()

    def register_faces_updated_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked whenever face_data.json is updated from Supabase."""
        if callback not in self._on_faces_updated_callbacks:
            self._on_faces_updated_callbacks.append(callback)

    def _check_date_rollover(self) -> None:
        today_str = date.today().isoformat()
        if today_str != self._current_date:
            logger.info("Date changed from %s to %s. Resetting daily attendance.", self._current_date, today_str)
            self._current_date = today_str
            self._present_today.clear()
            self._attendance_overrides.clear()
            self._save_attendance_log()

    def _load_attendance_log(self) -> None:
        with self._lock:
            if not self.attendance_log_path.exists():
                return
            try:
                data = json.loads(self.attendance_log_path.read_text(encoding="utf-8"))
                if data.get("date") == self._current_date:
                    self._present_today = dict(data.get("present_today", {}))
                    raw_overrides = data.get("overrides", {})
                    if isinstance(raw_overrides, dict):
                        self._attendance_overrides = raw_overrides
            except Exception as exc:
                logger.warning("Could not load attendance log: %s", exc)

    def _save_attendance_log(self) -> None:
        try:
            payload = {
                "date": self._current_date,
                "present_today": self._present_today,
                "overrides": self._attendance_overrides,
            }
            self.attendance_log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to persist attendance log: %s", exc)

    def _load_face_meta(self) -> None:
        with self._lock:
            if not self.face_meta_path.exists():
                return
            try:
                data = json.loads(self.face_meta_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._meta = data
            except Exception as exc:
                logger.debug("Could not load face_meta.json: %s", exc)

    def _save_face_meta(self) -> None:
        try:
            self.face_meta_path.write_text(
                json.dumps(self._meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not save face_meta.json: %s", exc)

    def get_person_m(self, student_id: str) -> str:
        """Return the grade/classroom `m` for `student_id`."""
        sid = str(student_id).strip()
        with self._lock:
            if sid in self._roster and self._roster[sid].get("classroom") not in (None, "", "-"):
                return str(self._roster[sid]["classroom"])
            if sid in self._meta and self._meta[sid].get("m") not in (None, "", "-"):
                return str(self._meta[sid]["m"])
            return "-"

    def get_person_name(self, student_id: str) -> str:
        """Return the display `full_name` for `student_id` (falling back to `student_id`)."""
        sid = str(student_id).strip()
        with self._lock:
            if sid in self._roster and str(self._roster[sid].get("full_name") or "").strip():
                return str(self._roster[sid]["full_name"]).strip()
            if sid in self._meta and str(self._meta[sid].get("full_name") or "").strip():
                return str(self._meta[sid]["full_name"]).strip()
            return sid

    def reset_today_attendance(self) -> None:
        """Clear today's attendance records (useful during testing)."""
        with self._lock:
            self._present_today.clear()
            self._attendance_overrides.clear()
            self._save_attendance_log()
            logger.info("Reset today's attendance records.")

    def register_person(
        self,
        student_id: str,
        full_name: Optional[str] = None,
        classroom: str = "-",
    ) -> None:
        """Register or update a person directly in memory (and persist `m`/classroom metadata)."""
        sid = str(student_id).strip()
        if not sid:
            return
        name_clean = str(full_name).strip() if full_name is not None else ""
        m_clean = str(classroom or "").strip() or "-"
        with self._lock:
            existing = self._roster.get(sid, {})
            existing_meta = self._meta.get(sid, {})

            resolved_name = (
                name_clean
                or str(existing.get("full_name") or "").strip()
                or str(existing_meta.get("full_name") or "").strip()
                or sid
            )
            if m_clean != "-":
                resolved_classroom = m_clean
            else:
                resolved_classroom = (
                    existing.get("classroom")
                    if existing.get("classroom") not in (None, "", "-")
                    else (existing_meta.get("m") or "-")
                )

            self._roster[sid] = {
                "student_id": sid,
                "full_name": resolved_name,
                "classroom": resolved_classroom,
                "leave_status": existing.get("leave_status", ""),
                "is_excused": existing.get("is_excused", False),
            }

            if resolved_classroom != "-" or full_name:
                self._meta[sid] = {
                    "m": resolved_classroom,
                    "full_name": resolved_name,
                }
                self._save_face_meta()

    def remove_person(self, student_id: str) -> None:
        """Remove a person from the in-memory roster if not backed by Excel."""
        sid = str(student_id).strip()
        with self._lock:
            self._roster.pop(sid, None)
            self._present_today.pop(sid, None)
            if sid in self._meta:
                self._meta.pop(sid, None)
                self._save_face_meta()
            self._save_attendance_log()

    def rename_or_update_person(
        self,
        old_student_id: str,
        new_student_id: str,
        full_name: Optional[str] = None,
        classroom: str = "-",
    ) -> None:
        """Update `full_name` / `classroom` and optionally migrate `student_id` in roster, meta, and today's log."""
        old_sid = str(old_student_id).strip()
        new_sid = str(new_student_id).strip()
        if not old_sid or not new_sid:
            return
        name_clean = str(full_name).strip() if full_name is not None else ""
        m_clean = str(classroom or "").strip() or "-"

        with self._lock:
            existing = self._roster.pop(old_sid, {}) if old_sid != new_sid else self._roster.get(old_sid, {})
            existing_meta = self._meta.pop(old_sid, {}) if old_sid != new_sid else self._meta.get(old_sid, {})

            if old_sid != new_sid:
                if old_sid in self._present_today:
                    self._present_today[new_sid] = self._present_today.pop(old_sid)
                if old_sid in self._attendance_overrides:
                    self._attendance_overrides[new_sid] = self._attendance_overrides.pop(old_sid)
                self._save_attendance_log()

            prev_name = str(existing.get("full_name") or existing_meta.get("full_name") or "").strip()
            if name_clean:
                resolved_name = name_clean
            elif old_sid != new_sid and (not prev_name or prev_name == old_sid):
                resolved_name = new_sid
            else:
                resolved_name = prev_name or new_sid

            resolved_classroom = (
                m_clean
                if m_clean != "-"
                else (existing.get("classroom") or existing_meta.get("m") or "-")
            )

            self._roster[new_sid] = {
                "student_id": new_sid,
                "full_name": resolved_name,
                "classroom": resolved_classroom,
                "leave_status": existing.get("leave_status", ""),
                "is_excused": existing.get("is_excused", False),
            }
            self._meta[new_sid] = {
                "m": resolved_classroom,
                "full_name": resolved_name,
            }
            self._save_face_meta()

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
                    meta_info = self._meta.get(sid, {})
                    if isinstance(val, dict):
                        name = val.get("name") or meta_info.get("full_name") or sid
                        m_val = val.get("m") or meta_info.get("m") or "-"
                    else:
                        name = meta_info.get("full_name") or sid
                        m_val = meta_info.get("m") or "-"
                    if sid not in self._roster:
                        self._roster[sid] = {
                            "student_id": sid,
                            "full_name": name,
                            "classroom": m_val,
                            "leave_status": "",
                            "is_excused": False,
                        }
                    else:
                        if name and name != sid and self._roster[sid].get("full_name") in (None, "", sid):
                            self._roster[sid]["full_name"] = name
                        if m_val != "-" and self._roster[sid].get("classroom") in (None, "", "-"):
                            self._roster[sid]["classroom"] = m_val
        except Exception as exc:
            logger.debug("Could not sync from face_data.json: %s", exc)

    def sync_from_supabase(self) -> bool:
        """
        Pull records (`id_name`, `m`, `array_faceshape`) from Supabase into local
        face_data.json and student roster so live face recognition stays in sync.
        """
        if not self._is_default_store or not supabase_manager.is_configured:
            return False

        ok, records, err = supabase_manager.fetch_all_records()
        if not ok:
            return False

        ok_students, remote_students, _ = supabase_manager.fetch_students()
        remote_student_map: Dict[str, dict] = {}
        if ok_students:
            for rs in remote_students:
                rsid = str(rs.get("student_id") or "").strip()
                if rsid:
                    remote_student_map[rsid] = rs

        face_data: Dict[str, list] = {}
        if self.face_data_path.exists():
            try:
                loaded = json.loads(self.face_data_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    face_data = loaded
            except Exception:
                face_data = {}

        face_data_changed = False
        meta_changed = False
        remote_ids: Set[str] = set()

        with self._lock:
            for rec in records:
                id_name = rec.get("id_name", "").strip()
                if not id_name:
                    continue
                remote_ids.add(id_name)
                m_val = rec.get("m", "-") or "-"
                samples = rec.get("array_faceshape") or []

                existing = self._roster.get(id_name, {})
                existing_meta = self._meta.get(id_name, {})
                remote_stu = remote_student_map.get(id_name, {})
                face_name = str(rec.get("name_student") or "").strip()
                remote_name = str(remote_stu.get("full_name") or face_name or "").strip()
                local_name = str(existing.get("full_name") or existing_meta.get("full_name") or "").strip()

                if local_name and local_name != id_name:
                    resolved_name = local_name
                elif face_name and face_name != id_name:
                    resolved_name = face_name
                elif remote_name and remote_name != id_name:
                    resolved_name = remote_name
                else:
                    resolved_name = local_name or remote_name or id_name

                resolved_m = m_val if m_val != "-" else (existing.get("classroom") or existing_meta.get("m") or "-")
                self._roster[id_name] = {
                    "student_id": id_name,
                    "full_name": resolved_name,
                    "classroom": resolved_m,
                    "leave_status": existing.get("leave_status", ""),
                    "is_excused": existing.get("is_excused", False),
                }

                if existing_meta.get("m") != resolved_m or existing_meta.get("full_name") != resolved_name:
                    self._meta[id_name] = {
                        "m": resolved_m,
                        "full_name": resolved_name,
                    }
                    meta_changed = True

                if samples and face_data.get(id_name) != samples:
                    face_data[id_name] = samples
                    face_data_changed = True

            self._last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if meta_changed:
            self._save_face_meta()

        if face_data_changed:
            try:
                self.face_data_path.write_text(
                    json.dumps(face_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.error("Failed to update face_data.json from Supabase: %s", exc)
            for cb in list(self._on_faces_updated_callbacks):
                try:
                    cb()
                except Exception as exc:
                    logger.warning("Faces-updated callback failed: %s", exc)

        # Auto-push any local face arrays that do not exist in Supabase yet
        for sid, local_samples in face_data.items():
            sid_clean = str(sid).strip()
            if sid_clean and sid_clean not in remote_ids and local_samples:
                supabase_manager.upsert_record(
                    id_name=sid_clean,
                    m=self.get_person_m(sid_clean),
                    array_faceshape=local_samples,
                    append_sample=False,
                    name_student=self.get_person_name(sid_clean),
                )

        return True

    @staticmethod
    def _classify_scan_status(ts_str: str) -> str:
        """Classify scan timestamp (HH:MM:SS): before 08:00 -> present, >= 08:00 -> late."""
        return "present" if ts_str < "08:00:00" else "late"

    @staticmethod
    def _build_scan_iso(date_str: str, ts_str: str) -> str:
        dt = datetime.strptime(f"{date_str} {ts_str}", "%Y-%m-%d %H:%M:%S")
        return dt.astimezone().isoformat()

    def _push_checkin_async(
        self,
        student_info: dict,
        date_str: str,
        ts_str: str,
        total_students: int,
        present_students: int,
    ) -> None:
        def _worker() -> None:
            sid = student_info["student_id"]
            status_val = self._classify_scan_status(ts_str)
            scan_iso = self._build_scan_iso(date_str, ts_str)
            supabase_manager.sync_students([student_info])
            supabase_manager.upsert_attendance_records(
                [
                    {
                        "student_id": sid,
                        "attendance_date": date_str,
                        "status": status_val,
                        "first_scan_at": scan_iso,
                        "is_finalized": True,
                        "audit_entries": [],
                    }
                ]
            )
            status_thai = "มาเรียน" if status_val == "present" else "มาสาย"
            hhmm = ts_str[:5]
            supabase_manager.insert_parent_notification(
                {
                    "id": f"scan-{sid}-{date_str}",
                    "student_id": sid,
                    "student_name": student_info.get("full_name") or sid,
                    "title": "บันทึกสถานะการมาเรียนของนักเรียนแล้ว",
                    "body": f"นักเรียนสแกนเข้าโรงเรียนเวลา {hhmm} น. (สถานะ{status_thai})",
                    "target_date": date_str,
                    "new_status": status_val,
                    "previous_status": None,
                    "created_at": scan_iso,
                    "is_read": False,
                }
            )
            supabase_manager.update_sync_heartbeat(total_students, present_students)

        threading.Thread(target=_worker, daemon=True).start()

    def sync_attendance_with_supabase(self) -> bool:
        """
        Two-way attendance sync between the primary PC and Supabase Cloud:
        1. Pushes current student roster (`attendance_students`).
        2. Pulls today's `attendance_records` to capture teacher edits made in Student_APP.
        3. Pushes local scans (and post-09:00 absences / excused leaves) that aren't finalized by teacher edit.
        4. Updates `sync_status` heartbeat for Student_APP.
        """
        if not self._is_default_store or not supabase_manager.is_configured:
            return False

        with self._lock:
            self._check_date_rollover()
            today_str = self._current_date
            roster_snapshot = [dict(v) for v in self._roster.values()]
            present_snapshot = dict(self._present_today)

        ok_students, _ = supabase_manager.sync_students(roster_snapshot)
        ok_fetch, remote_rows, _ = supabase_manager.fetch_attendance_records(today_str)

        remote_by_sid: Dict[str, dict] = {}
        if ok_fetch:
            for row in remote_rows:
                sid = str(row.get("student_id") or "").strip()
                if sid:
                    remote_by_sid[sid] = row

        overrides_changed = False
        now_time_str = datetime.now().strftime("%H:%M:%S")
        records_to_push: List[dict] = []

        with self._lock:
            for sid, info in self._roster.items():
                remote = remote_by_sid.get(sid)
                remote_audits = remote.get("audit_entries") if remote else None
                has_teacher_edit = bool(remote_audits)

                if remote and has_teacher_edit:
                    if self._attendance_overrides.get(sid) != remote:
                        self._attendance_overrides[sid] = remote
                        overrides_changed = True
                    continue

                if sid in present_snapshot:
                    ts_str = present_snapshot[sid]
                    expected_status = self._classify_scan_status(ts_str)
                    scan_iso = self._build_scan_iso(today_str, ts_str)
                    if (
                        not remote
                        or remote.get("status") != expected_status
                        or not remote.get("first_scan_at")
                    ):
                        records_to_push.append(
                            {
                                "student_id": sid,
                                "attendance_date": today_str,
                                "status": expected_status,
                                "first_scan_at": scan_iso,
                                "is_finalized": True,
                                "audit_entries": [],
                            }
                        )
                elif info.get("is_excused"):
                    if not remote or remote.get("status") != "leave":
                        records_to_push.append(
                            {
                                "student_id": sid,
                                "attendance_date": today_str,
                                "status": "leave",
                                "first_scan_at": None,
                                "is_finalized": True,
                                "audit_entries": [],
                            }
                        )
                elif now_time_str >= "09:00:00":
                    if not remote or remote.get("status") == "pending":
                        records_to_push.append(
                            {
                                "student_id": sid,
                                "attendance_date": today_str,
                                "status": "absent",
                                "first_scan_at": None,
                                "is_finalized": True,
                                "audit_entries": [],
                            }
                        )

            if overrides_changed:
                self._save_attendance_log()

        ok_push = True
        if records_to_push:
            ok_push, _ = supabase_manager.upsert_attendance_records(records_to_push)

        metrics = self.get_public_metrics()
        supabase_manager.update_sync_heartbeat(
            metrics["total_students"],
            metrics["present_students"],
        )
        return ok_students and ok_push

    def sync_from_excel(self) -> bool:
        """
        Optional non-blocking read of students.xlsx, plus Supabase and face_data.json sync.
        """
        if self._is_default_store and supabase_manager.is_configured:
            self.sync_from_supabase()
        self.sync_from_face_data()

        if not self.excel_path.exists():
            if self._is_default_store and supabase_manager.is_configured:
                self.sync_attendance_with_supabase()
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
            if self._is_default_store and supabase_manager.is_configured:
                self.sync_attendance_with_supabase()
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

        now_dt = timestamp or datetime.now()
        ts_str = now_dt.strftime("%H:%M:%S")
        should_push_remote = False
        student_info_copy: Optional[dict] = None
        today_str = self._current_date
        total_students = 0
        present_students = 0

        with self._lock:
            self._check_date_rollover()
            today_str = self._current_date
            if student_id not in self._roster:
                self.register_person(student_id)

            if student_id not in self._present_today:
                self._present_today[student_id] = ts_str
                self._save_attendance_log()
                logger.info("Student/Label %s marked present at %s", student_id, ts_str)
                if (
                    self._is_default_store
                    and supabase_manager.is_configured
                    and student_id not in self._attendance_overrides
                ):
                    should_push_remote = True
                    student_info_copy = dict(self._roster[student_id])
                    total_students = len(self._roster)
                    present_students = len(self._present_today)

        if should_push_remote and student_info_copy is not None:
            self._push_checkin_async(
                student_info_copy,
                today_str,
                ts_str,
                total_students,
                present_students,
            )
        return True

    def get_public_metrics(self) -> dict:
        """Calculate aggregated attendance metrics."""
        with self._lock:
            self._check_date_rollover()
            active_ids: Set[str] = set(self._roster.keys())
            total_students = len(active_ids)

            present_ids: Set[str] = set()
            excused_absent_ids: Set[str] = set()
            for sid in active_ids:
                override = self._attendance_overrides.get(sid)
                if override and override.get("status"):
                    st = override["status"]
                    if st in ("present", "late"):
                        present_ids.add(sid)
                    elif st == "leave":
                        excused_absent_ids.add(sid)
                elif sid in self._present_today:
                    present_ids.add(sid)
                elif self._roster[sid].get("is_excused"):
                    excused_absent_ids.add(sid)

            present_students = len(present_ids)
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
                override = self._attendance_overrides.get(sid)
                override_status = override.get("status") if override else None

                if override_status == "present":
                    attendance_status = "มาเรียน (แก้ไขโดยครู)" if not checkin_time else "มาเรียน (สแกนแล้ว)"
                    status_code = "present"
                elif override_status == "late":
                    attendance_status = "มาสาย"
                    status_code = "present"
                elif override_status == "leave":
                    attendance_status = "ลา (ยืนยันจากแอป)"
                    status_code = "excused"
                elif override_status == "absent":
                    attendance_status = "ขาดเรียน"
                    status_code = "missing"
                elif checkin_time:
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
