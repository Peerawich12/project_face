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
#SUPABASE_URL = ""
#SUPABASE_KEY = ""

SUPABASE_TABLE = "face_data"


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
                    samples = parse_faceshape_to_samples(row.get("array_faceshape"))
                    first_vec = samples[0] if samples else []
                    preview_nums = ", ".join(f"{v:.4f}" for v in first_vec[:4])

                    normalized.append(
                        {
                            "id_name": id_name,
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
    ) -> Any:
        """
        Execute insert or update against Supabase with automatic fallbacks for
        different PostgreSQL column types for `array_faceshape` (jsonb, float8[], text)
        and `m` (text vs integer).
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

        last_exc: Optional[Exception] = None
        for m_try in m_candidates:
            for arr_try in array_candidates:
                try:
                    if is_update:
                        payload = {"m": m_try, "array_faceshape": arr_try}
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
                        return self._client.table(self.table_name).insert(payload).execute()
                except Exception as exc:
                    last_exc = exc
                    err_lower = str(exc).lower()
                    if "integer" in err_lower and isinstance(m_try, str):
                        break
                    continue

        if last_exc:
            raise last_exc

    def upsert_record(
        self,
        id_name: str,
        m: str,
        array_faceshape: Any,
        append_sample: bool = False,
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        Insert or update a record (`id_name`, `m`, `array_faceshape`) in Supabase.
        """
        id_name = str(id_name or "").strip()
        m_val = str(m or "").strip() or "-"

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

                    self._execute_write_with_fallbacks(
                        is_update=True,
                        id_name=id_name,
                        m_val=m_val,
                        samples=final_samples,
                    )
                    action_text = "อัปเดตข้อมูลใน Supabase"
                else:
                    final_samples = new_samples
                    self._execute_write_with_fallbacks(
                        is_update=False,
                        id_name=id_name,
                        m_val=m_val,
                        samples=final_samples,
                    )
                    action_text = "เพิ่มข้อมูลใหม่ลง Supabase"

                self.connected = True
                self.last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.last_error = None

                record_summary = {
                    "id_name": id_name,
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
            }


supabase_manager = SupabaseManager()
