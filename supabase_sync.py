import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except ImportError:
    Client = Any  # type: ignore
    create_client = None
    SUPABASE_AVAILABLE = False

logger = logging.getLogger("school_attendance.supabase")

# ==============================================================================
# SUPABASE CONFIGURATION — ใส่ SUPABASE_URL และ SUPABASE_KEY ที่นี่โดยตรง
# ==============================================================================
SUPABASE_URL = (
    os.environ.get("SUPABASE_URL", "").strip()
    or "https://lujiagaxncgrbuvdrccp.supabase.co"
)
SUPABASE_KEY = (
    os.environ.get("SUPABASE_KEY", "").strip()
    or "sb_publishable_mBYAfdUbByxcq0v7N9NueA_J3I0Dato"
)

SUPABASE_TABLE = "face_data"
SUPABASE_STUDENTS_TABLE = "attendance_students"
SUPABASE_ATTENDANCE_TABLE = "attendance_records"
SUPABASE_NOTIFICATIONS_TABLE = "parent_notifications"
SUPABASE_SYNC_STATUS_TABLE = "sync_status"


def normalize_classroom_code(raw: Any) -> str:
    """Normalize classroom strings like 'ม.5/1', '5/3', 'ม. 4 / 2' into '5/1', or '-' if unknown."""
    text = str(raw or "").strip()
    if not text or text == "-":
        return "-"
    match = re.search(r"([1-6])\s*[/\-]\s*([1-9]\d*)", text)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return text


def parse_faceshape_to_samples(raw_val: Any) -> List[List[float]]:
    """
    Normalize `array_faceshape` from any format (JSON string, Postgres array string,
    1D float list [128], or 2D float list [[128], ...]) into List[List[float]].
    """
    if raw_val is None:
        return []

    data = raw_val
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return []
        if s.startswith("{") and s.endswith("}"):
            s = s.replace("{", "[").replace("}", "]")
        try:
            data = json.loads(s)
        except Exception:
            return []

    if not isinstance(data, list) or len(data) == 0:
        return []

    # Case 1: 1D vector of numbers -> e.g. [-0.146, 0.081, ...]
    if isinstance(data[0], (int, float)):
        try:
            vec = [round(float(x), 6) for x in data[:128]]
            if len(vec) == 128:
                return [vec]
        except (ValueError, TypeError):
            return []
        return []

    # Case 2: 2D list of vectors -> e.g. [[-0.146, ...], [-0.142, ...]]
    samples: List[List[float]] = []
    for item in data:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                continue
        if isinstance(item, list) and len(item) >= 128:
            try:
                vec = [round(float(x), 6) for x in item[:128]]
                if len(vec) == 128:
                    samples.append(vec)
            except (ValueError, TypeError):
                continue
    return samples


class SupabaseManager:
    """
    Manages connection and CRUD operations for the Supabase table with columns:
    - `id_name`: Student ID / Name identifier
    - `m`: Grade / Classroom level (e.g., ม.1 - ม.6, ม.4/1)
    - `array_faceshape`: 128-d face encoding array(s)
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        self._lock = threading.RLock()

        self.url: str = (url if url is not None else (SUPABASE_URL or os.environ.get("SUPABASE_URL", ""))).strip()
        self.key: str = (key if key is not None else (SUPABASE_KEY or os.environ.get("SUPABASE_KEY", ""))).strip()
        self.table_name: str = (
            table_name if table_name is not None else (SUPABASE_TABLE or os.environ.get("SUPABASE_TABLE", "face_data"))
        ).strip() or "face_data"

        self._client: Optional[Client] = None
        self.connected: bool = False
        self.last_sync_time: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_attendance_sync_time: Optional[str] = None
        self.last_attendance_error: Optional[str] = None
        self._cached_records: List[dict] = []

        self._init_client()

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.key and SUPABASE_AVAILABLE)

    def _init_client(self) -> bool:
        with self._lock:
            if not SUPABASE_AVAILABLE:
                self.connected = False
                self.last_error = "ยังไม่ได้ติดตั้งแพ็กเกจ supabase ในสภาพแวดล้อม Python"
                return False

            if not self.url or not self.key:
                self._client = None
                self.connected = False
                self.last_error = None
                return False

            try:
                self._client = create_client(self.url, self.key)
                self.connected = True
                self.last_error = None
                return True
            except Exception as exc:
                logger.error("Failed to initialize Supabase client: %s", exc)
                self._client = None
                self.connected = False
                self.last_error = str(exc)
                return False

    def fetch_all_records(self) -> Tuple[bool, List[dict], Optional[str]]:
        """Fetch all rows (`id_name`, `m`, `array_faceshape`) from the Supabase table."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, [], "กรุณาใส่ SUPABASE_URL และ SUPABASE_KEY ในไฟล์ supabase_sync.py"

            try:
                response = self._client.table(self.table_name).select("*").execute()
                rows = getattr(response, "data", []) or []

                normalized: List[dict] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    id_name = str(row.get("id_name") or "").strip()
                    if not id_name:
                        continue
                    raw_m = row.get("m")
                    m_val = str(raw_m).strip() if raw_m is not None and str(raw_m).strip() != "" else "-"
                    name_student = str(row.get("name_student") or row.get("full_name") or "").strip()
                    samples = parse_faceshape_to_samples(row.get("array_faceshape"))
                    first_vec = samples[0] if samples else []
                    preview_nums = ", ".join(f"{v:.4f}" for v in first_vec[:4])

                    normalized.append(
                        {
                            "id_name": id_name,
                            "name_student": name_student,
                            "m": m_val,
                            "array_faceshape": samples,
                            "samples_count": len(samples),
                            "dims": len(first_vec),
                            "preview": f"[{preview_nums}, ...]" if first_vec else "[]",
                        }
                    )

                self._cached_records = normalized
                self.connected = True
                self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_error = None
                return True, normalized, None

            except Exception as exc:
                err_msg = str(exc)
                logger.error("Supabase fetch_all_records error: %s", err_msg)
                self.connected = False
                self.last_error = err_msg
                return False, self._cached_records, err_msg

    def _execute_write_with_fallbacks(
        self,
        is_update: bool,
        id_name: str,
        m_val: str,
        samples: List[List[float]],
        name_student: Optional[str] = None,
    ) -> Any:
        """
        Execute insert or update against Supabase with automatic fallbacks for
        different PostgreSQL column types for `array_faceshape` (jsonb, float8[], text)
        and `m` (text vs integer), and optional `name_student` column.
        """
        assert self._client is not None

        array_candidates: List[Any] = [
            samples,
            samples[-1] if samples else [],
            json.dumps(samples),
        ]

        m_candidates: List[Any] = [m_val]
        digits = re.findall(r"\d+", str(m_val))
        if digits:
            m_candidates.append(int(digits[0]))

        clean_name = str(name_student).strip() if name_student is not None else ""
        include_name_options = [True, False] if clean_name else [False]

        last_exc: Optional[Exception] = None
        for include_name in include_name_options:
            for m_try in m_candidates:
                for arr_try in array_candidates:
                    try:
                        if is_update:
                            payload: Dict[str, Any] = {"m": m_try, "array_faceshape": arr_try}
                            if include_name:
                                payload["name_student"] = clean_name
                            return (
                                self._client.table(self.table_name)
                                .update(payload)
                                .eq("id_name", id_name)
                                .execute()
                            )
                        else:
                            payload = {
                                "id_name": id_name,
                                "m": m_try,
                                "array_faceshape": arr_try,
                            }
                            if include_name:
                                payload["name_student"] = clean_name
                            return self._client.table(self.table_name).insert(payload).execute()
                    except Exception as exc:
                        last_exc = exc
                        err_lower = str(exc).lower()
                        if include_name and "name_student" in err_lower:
                            break
                        if "integer" in err_lower and isinstance(m_try, str):
                            break
                        continue
                else:
                    continue
                if include_name and last_exc and "name_student" in str(last_exc).lower():
                    break

        if last_exc:
            raise last_exc

    def upsert_record(
        self,
        id_name: str,
        m: str,
        array_faceshape: Any,
        append_sample: bool = False,
        name_student: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        Insert or update a record (`id_name`, `name_student`, `m`, `array_faceshape`) in Supabase.
        """
        id_name = str(id_name or "").strip()
        m_val = str(m or "").strip() or "-"
        clean_name = str(name_student or "").strip()

        if not id_name:
            return False, "กรุณาระบุ id_name (รหัสหรือชื่อนักเรียน)", None

        new_samples = parse_faceshape_to_samples(array_faceshape)
        if not new_samples:
            return False, "ข้อมูล array_faceshape ไม่ถูกต้อง (ต้องเป็นเวกเตอร์ใบหน้า 128 มิติ)", None

        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "กรุณาใส่ SUPABASE_URL และ SUPABASE_KEY ในไฟล์ supabase_sync.py", None

            try:
                existing_res = (
                    self._client.table(self.table_name)
                    .select("*")
                    .eq("id_name", id_name)
                    .execute()
                )
                existing_rows = getattr(existing_res, "data", []) or []

                if existing_rows:
                    existing_row = existing_rows[0]
                    if append_sample:
                        old_samples = parse_faceshape_to_samples(existing_row.get("array_faceshape"))
                        final_samples = old_samples + new_samples
                    else:
                        final_samples = new_samples

                    if m_val == "-" and existing_row.get("m"):
                        m_val = str(existing_row.get("m")).strip() or "-"

                    if not clean_name and existing_row.get("name_student"):
                        clean_name = str(existing_row.get("name_student")).strip()

                    self._execute_write_with_fallbacks(
                        is_update=True,
                        id_name=id_name,
                        m_val=m_val,
                        samples=final_samples,
                        name_student=clean_name or None,
                    )
                    action_text = "อัปเดตข้อมูลใน Supabase"
                else:
                    final_samples = new_samples
                    self._execute_write_with_fallbacks(
                        is_update=False,
                        id_name=id_name,
                        m_val=m_val,
                        samples=final_samples,
                        name_student=clean_name or None,
                    )
                    action_text = "เพิ่มข้อมูลใหม่ลง Supabase"

                self.connected = True
                self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_error = None

                record_summary = {
                    "id_name": id_name,
                    "name_student": clean_name or id_name,
                    "m": m_val,
                    "array_faceshape": final_samples,
                    "samples_count": len(final_samples),
                    "dims": len(final_samples[0]) if final_samples else 128,
                }
                return (
                    True,
                    f"{action_text} สำเร็จ: {id_name} (ชั้น {m_val}, {len(final_samples)} ตัวอย่างใบหน้า)",
                    record_summary,
                )

            except Exception as exc:
                err_msg = str(exc)
                logger.error("Supabase upsert_record error for %s: %s", id_name, err_msg)
                self.last_error = err_msg
                return False, f"บันทึกข้อมูลลง Supabase ไม่สำเร็จ: {err_msg}", None

    def delete_record(self, id_name: str) -> Tuple[bool, str]:
        """Delete a record by `id_name` from Supabase."""
        id_name = str(id_name or "").strip()
        if not id_name:
            return False, "ไม่พบ id_name ที่ต้องการลบ"

        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "กรุณาใส่ SUPABASE_URL และ SUPABASE_KEY ในไฟล์ supabase_sync.py"

            try:
                self._client.table(self.table_name).delete().eq("id_name", id_name).execute()
                self._cached_records = [
                    r for r in self._cached_records if r.get("id_name") != id_name
                ]
                self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_error = None
                return True, f"ลบข้อมูล '{id_name}' ออกจาก Supabase เรียบร้อยแล้ว"
            except Exception as exc:
                err_msg = str(exc)
                logger.error("Supabase delete_record error for %s: %s", id_name, err_msg)
                self.last_error = err_msg
                return False, f"ลบข้อมูลจาก Supabase ไม่สำเร็จ: {err_msg}"

    def update_student_profile(
        self,
        old_id_name: str,
        new_id_name: str,
        full_name: str,
        classroom: str,
        samples: Optional[List[List[float]]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Update (and optionally rename) a student across `face_data` and `attendance_students`
        in Supabase.
        """
        old_sid = str(old_id_name or "").strip()
        new_sid = str(new_id_name or "").strip()
        clean_name = str(full_name or "").strip() or new_sid
        clean_m = str(classroom or "").strip() or "-"
        if not old_sid or not new_sid:
            return False, "กรุณาระบุ id_name"

        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "Supabase is not configured"

            errors: List[str] = []

            # 1. Sync attendance_students (new or updated row)
            ok_stu, err_stu = self.sync_students(
                [
                    {
                        "student_id": new_sid,
                        "full_name": clean_name,
                        "classroom": clean_m,
                        "leave_status": "",
                        "is_excused": False,
                    }
                ]
            )
            if not ok_stu and err_stu:
                errors.append(err_stu)

            # 2. Update or rename in face_data
            try:
                final_samples = samples or []
                if not final_samples:
                    existing_res = (
                        self._client.table(self.table_name)
                        .select("*")
                        .eq("id_name", old_sid)
                        .execute()
                    )
                    existing_rows = getattr(existing_res, "data", []) or []
                    if existing_rows:
                        final_samples = parse_faceshape_to_samples(
                            existing_rows[0].get("array_faceshape")
                        )

                if old_sid == new_sid:
                    if final_samples:
                        ok_up, msg_up, _ = self.upsert_record(
                            id_name=new_sid,
                            m=clean_m,
                            array_faceshape=final_samples,
                            append_sample=False,
                            name_student=clean_name,
                        )
                        if not ok_up:
                            errors.append(msg_up)
                    else:
                        try:
                            self._client.table(self.table_name).update(
                                {"m": clean_m, "name_student": clean_name}
                            ).eq("id_name", new_sid).execute()
                        except Exception as exc:
                            if "name_student" in str(exc).lower():
                                self._client.table(self.table_name).update(
                                    {"m": clean_m}
                                ).eq("id_name", new_sid).execute()
                            else:
                                raise
                else:
                    if final_samples:
                        ok_up, msg_up, _ = self.upsert_record(
                            id_name=new_sid,
                            m=clean_m,
                            array_faceshape=final_samples,
                            append_sample=False,
                            name_student=clean_name,
                        )
                        if ok_up:
                            self.delete_record(old_sid)
                        else:
                            errors.append(msg_up)
                    else:
                        # Update id_name directly if row exists without parsed samples
                        try:
                            self._client.table(self.table_name).update(
                                {"id_name": new_sid, "m": clean_m, "name_student": clean_name}
                            ).eq("id_name", old_sid).execute()
                        except Exception:
                            pass

                    # Migrate attendance_records & deactivate/delete old attendance_students row
                    try:
                        att_res = (
                            self._client.table(SUPABASE_ATTENDANCE_TABLE)
                            .select("*")
                            .eq("student_id", old_sid)
                            .execute()
                        )
                        old_att_rows = getattr(att_res, "data", []) or []
                        if old_att_rows:
                            migrated = []
                            for r in old_att_rows:
                                att_date = str(r.get("attendance_date") or "").strip()
                                if not att_date:
                                    continue
                                migrated.append(
                                    {
                                        "student_id": new_sid,
                                        "attendance_date": att_date,
                                        "status": r.get("status") or "pending",
                                        "first_scan_at": r.get("first_scan_at"),
                                        "is_finalized": bool(r.get("is_finalized", False)),
                                        "audit_entries": r.get("audit_entries") or [],
                                    }
                                )
                            if migrated:
                                self.upsert_attendance_records(migrated)
                    except Exception as exc:
                        logger.debug("Migrate attendance_records on rename skipped: %s", exc)

                    try:
                        self._client.table(SUPABASE_STUDENTS_TABLE).update(
                            {"active": False}
                        ).eq("student_id", old_sid).execute()
                        self._client.table(SUPABASE_STUDENTS_TABLE).delete().eq(
                            "student_id", old_sid
                        ).execute()
                    except Exception:
                        pass
            except Exception as exc:
                errors.append(str(exc))

            if errors:
                err_text = "; ".join(errors)
                self.last_attendance_error = err_text
                return False, err_text

            self.connected = True
            self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_attendance_error = None
            return True, None

    def fetch_students(self) -> Tuple[bool, List[dict], Optional[str]]:
        """Fetch student roster rows from `attendance_students`."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, [], "Supabase is not configured"
            try:
                response = (
                    self._client.table(SUPABASE_STUDENTS_TABLE)
                    .select("student_id,full_name,classroom,leave_status,is_excused,active")
                    .execute()
                )
                rows = getattr(response, "data", []) or []
                return True, [r for r in rows if isinstance(r, dict)], None
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Supabase fetch_students error: %s", err_msg)
                return False, [], err_msg

    def sync_students(self, students: List[dict]) -> Tuple[bool, Optional[str]]:
        """Upsert normalized student roster rows into `attendance_students`."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "Supabase is not configured"
            if not students:
                return True, None
            now_iso = datetime.now().astimezone().isoformat()
            payload = []
            for s in students:
                sid = str(s.get("student_id") or "").strip()
                if not sid:
                    continue
                payload.append(
                    {
                        "student_id": sid,
                        "full_name": str(s.get("full_name") or sid).strip() or sid,
                        "classroom": normalize_classroom_code(s.get("classroom", "-")),
                        "leave_status": str(s.get("leave_status") or "").strip(),
                        "is_excused": bool(s.get("is_excused", False)),
                        "active": True,
                        "updated_at": now_iso,
                    }
                )
            if not payload:
                return True, None
            try:
                self._client.table(SUPABASE_STUDENTS_TABLE).upsert(
                    payload, on_conflict="student_id"
                ).execute()
                self.last_attendance_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_attendance_error = None
                return True, None
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Supabase sync_students error: %s", err_msg)
                self.last_attendance_error = err_msg
                return False, err_msg

    def fetch_attendance_records(
        self, attendance_date: str
    ) -> Tuple[bool, List[dict], Optional[str]]:
        """Fetch attendance records for a specific YYYY-MM-DD date from `attendance_records`."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, [], "Supabase is not configured"
            try:
                response = (
                    self._client.table(SUPABASE_ATTENDANCE_TABLE)
                    .select("*")
                    .eq("attendance_date", attendance_date)
                    .execute()
                )
                rows = getattr(response, "data", []) or []
                self.last_attendance_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_attendance_error = None
                return True, [r for r in rows if isinstance(r, dict)], None
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Supabase fetch_attendance_records error: %s", err_msg)
                self.last_attendance_error = err_msg
                return False, [], err_msg

    def upsert_attendance_records(self, records: List[dict]) -> Tuple[bool, Optional[str]]:
        """Upsert daily attendance rows into `attendance_records`."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "Supabase is not configured"
            if not records:
                return True, None
            now_iso = datetime.now().astimezone().isoformat()
            payload = []
            for rec in records:
                sid = str(rec.get("student_id") or "").strip()
                att_date = str(rec.get("attendance_date") or "").strip()
                if not sid or not att_date:
                    continue
                payload.append(
                    {
                        "id": f"{sid}_{att_date}",
                        "student_id": sid,
                        "attendance_date": att_date,
                        "status": str(rec.get("status") or "pending"),
                        "first_scan_at": rec.get("first_scan_at"),
                        "is_finalized": bool(rec.get("is_finalized", False)),
                        "audit_entries": rec.get("audit_entries") or [],
                        "updated_at": now_iso,
                    }
                )
            if not payload:
                return True, None
            try:
                self._client.table(SUPABASE_ATTENDANCE_TABLE).upsert(
                    payload, on_conflict="student_id,attendance_date"
                ).execute()
                self.last_attendance_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_attendance_error = None
                return True, None
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Supabase upsert_attendance_records error: %s", err_msg)
                self.last_attendance_error = err_msg
                return False, err_msg

    def insert_parent_notification(self, notification: dict) -> Tuple[bool, Optional[str]]:
        """Upsert a parent notification into `parent_notifications`."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "Supabase is not configured"
            try:
                self._client.table(SUPABASE_NOTIFICATIONS_TABLE).upsert(
                    notification, on_conflict="id"
                ).execute()
                return True, None
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Supabase insert_parent_notification error: %s", err_msg)
                return False, err_msg

    def update_sync_heartbeat(
        self, total_students: int, present_students: int, note: str = "primary-pc"
    ) -> Tuple[bool, Optional[str]]:
        """Update primary PC heartbeat in `sync_status` so Student_APP sees fresh sync time."""
        with self._lock:
            if not self.is_configured or self._client is None:
                return False, "Supabase is not configured"
            try:
                self._client.table(SUPABASE_SYNC_STATUS_TABLE).upsert(
                    {
                        "device_id": "primary-pc",
                        "last_synced_at": datetime.now().astimezone().isoformat(),
                        "total_students": int(total_students),
                        "present_students": int(present_students),
                        "note": note,
                    },
                    on_conflict="device_id",
                ).execute()
                return True, None
            except Exception as exc:
                err_msg = str(exc)
                logger.warning("Supabase update_sync_heartbeat error: %s", err_msg)
                return False, err_msg

    def get_status(self) -> dict:
        with self._lock:
            return {
                "supabase_available": SUPABASE_AVAILABLE,
                "configured": self.is_configured,
                "connected": self.connected,
                "table_name": self.table_name,
                "records_count": len(self._cached_records),
                "last_sync_time": self.last_sync_time,
                "last_error": self.last_error,
                "attendance_last_sync_time": self.last_attendance_sync_time,
                "attendance_last_error": self.last_attendance_error,
            }


supabase_manager = SupabaseManager()
