import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from data_sync import attendance_store
from supabase_sync import parse_faceshape_to_samples, supabase_manager
from vision import FACES_DIR, vision_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("school_attendance.app")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Scheduler, sync with Supabase, and start VisionEngine on startup."""
    attendance_store.start_scheduler()
    vision_engine.load_known_faces()
    vision_engine.start_background_capture()
    yield
    vision_engine.stop_background_capture()
    attendance_store.stop_scheduler()


app = FastAPI(
    title="Project Face — ระบบเช็คชื่อและเก็บข้อมูลใบหน้าด้วย Supabase",
    description="Real-time face recognition and attendance system synced with Supabase (id_name, m, array_faceshape).",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class SimulateScanRequest(BaseModel):
    student_id: str
    bypass_cooldown: bool = False


class SupabaseRecordRequest(BaseModel):
    id_name: str
    m: str = "-"
    array_faceshape: Optional[Any] = None
    append_sample: bool = False
    capture_from_camera: bool = False


class SupabaseSyncRequest(BaseModel):
    push_local: bool = False


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    metrics = attendance_store.get_public_metrics()
    roster = attendance_store.get_student_roster_details()
    vision_status = vision_engine.get_status()
    supabase_status = supabase_manager.get_status()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "roster": roster,
            "vision_status": vision_status,
            "supabase_status": supabase_status,
        },
    )


@app.get("/api/metrics", response_class=JSONResponse)
async def api_public_metrics():
    return {
        **attendance_store.get_public_metrics(),
        "roster": attendance_store.get_student_roster_details(),
        "vision": vision_engine.get_status(),
        "supabase": supabase_manager.get_status(),
    }


async def _mjpeg_generator():
    while True:
        jpeg_bytes = vision_engine.get_latest_jpeg()
        if jpeg_bytes:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        await asyncio.sleep(0.04)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/student_pic/{student_id}")
async def get_student_pic(student_id: str):
    """Serve the saved face photo from faces/<student_id>/pic_1.jpg or faces/<student_id>.jpg."""
    safe_id = "".join(c for c in student_id if c.isalnum() or c in ("_", "-", " "))
    safe_id = safe_id.strip() or student_id
    person_dir = FACES_DIR / safe_id
    if person_dir.exists():
        for p in sorted(person_dir.glob("pic_*.jpg")):
            return FileResponse(p)
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = FACES_DIR / f"{safe_id}{ext}"
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Picture not saved yet")


@app.post("/api/register-face/{student_id}")
@app.post("/api/capture-pic/{student_id}")
async def api_register_face(
    student_id: str,
    m: str = Query(default="-", description="Grade/classroom level (e.g. ม.1 - ม.6, ม.4/1)"),
):
    """
    Capture the person's face from camera, save to faces/<student_id>/,
    extract 128-d Face-Shape Array, and sync (id_name, m, array_faceshape) to Supabase.
    """
    ok = await asyncio.to_thread(vision_engine.register_face_from_frame, student_id, None, m)
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ไม่พบใบหน้าในกล้อง — กรุณาหันหน้าเข้าหากล้องให้ชัดเจน",
            },
        )
    return {
        "success": True,
        "student_id": student_id,
        "m": attendance_store.get_person_m(student_id),
        "message": f"บันทึกใบหน้า '{student_id}' (ชั้น {attendance_store.get_person_m(student_id)}) และแปลงเป็น array_faceshape สำเร็จ!",
        "vision": vision_engine.get_status(),
        "supabase": supabase_manager.get_status(),
    }


@app.post("/api/upload-face/{student_id}")
@app.post("/api/upload-pic/{student_id}")
async def api_upload_face(
    student_id: str,
    file: UploadFile = File(...),
    m: str = Form(default="-"),
    m_query: Optional[str] = Query(default=None, alias="m"),
):
    """
    Save uploaded image to faces/<student_id>/, import into Face-Shape Array,
    and sync (id_name, m, array_faceshape) to Supabase.
    """
    resolved_m = m_query if (m_query and m_query != "-") else m
    image_bytes = await file.read()
    ok = await asyncio.to_thread(
        vision_engine.register_face_from_upload,
        student_id,
        image_bytes,
        resolved_m,
    )
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ไม่พบใบหน้าในรูปภาพที่อัปโหลด หรือไฟล์ไม่ถูกต้อง",
            },
        )
    return {
        "success": True,
        "student_id": student_id,
        "m": attendance_store.get_person_m(student_id),
        "message": f"อัปโหลดรูปภาพ '{student_id}' และแปลงเป็น array_faceshape เรียบร้อย",
        "vision": vision_engine.get_status(),
        "supabase": supabase_manager.get_status(),
    }


@app.delete("/api/face-data/{student_id}")
async def api_delete_face_data(student_id: str):
    ok = await asyncio.to_thread(vision_engine.delete_face_data, student_id)
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": f"ไม่พบข้อมูลของ {student_id}"},
        )
    return {
        "success": True,
        "student_id": student_id,
        "message": f"ลบข้อมูลและ array_faceshape ของ {student_id} เรียบร้อย",
        "vision": vision_engine.get_status(),
        "supabase": supabase_manager.get_status(),
    }


# ==============================================================================
# SUPABASE DATA COLLECTION & EXTRACTION ENDPOINTS (id_name, m, array_faceshape)
# ==============================================================================


@app.post("/api/extract-faceshape/camera")
async def api_extract_faceshape_camera():
    """Extract 128-d `array_faceshape` from the current live webcam frame for preview/collection."""
    vec = await asyncio.to_thread(vision_engine.extract_faceshape_from_current_frame)
    if vec is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ไม่พบใบหน้าในกล้องสด กรุณามองตรงที่กล้องแล้วลองอีกครั้ง",
            },
        )
    return {
        "success": True,
        "dims": len(vec),
        "array_faceshape": vec,
        "message": f"ดึงค่า array_faceshape จากกล้องสำเร็จ ({len(vec)} มิติ)",
    }


@app.post("/api/extract-faceshape/upload")
async def api_extract_faceshape_upload(file: UploadFile = File(...)):
    """Extract 128-d `array_faceshape` from an uploaded photo for preview/collection."""
    image_bytes = await file.read()
    vec = await asyncio.to_thread(vision_engine.extract_faceshape_from_image_bytes, image_bytes)
    if vec is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ไม่พบใบหน้าในไฟล์รูปภาพที่อัปโหลด",
            },
        )
    return {
        "success": True,
        "dims": len(vec),
        "array_faceshape": vec,
        "message": f"สกัดค่า array_faceshape จากรูปภาพสำเร็จ ({len(vec)} มิติ)",
    }


@app.get("/api/supabase/status")
async def api_supabase_status():
    return supabase_manager.get_status()


@app.get("/api/supabase/records")
async def api_get_supabase_records():
    """
    Return all records (`id_name`, `m`, `array_faceshape`).
    If Supabase is connected, fetches live from Supabase and merges local status.
    If Supabase is not configured yet, returns local records so the UI always works.
    """
    sb_records = []
    sb_ok = False
    sb_err = None

    if supabase_manager.is_configured:
        sb_ok, sb_records, sb_err = await asyncio.to_thread(supabase_manager.fetch_all_records)
        if sb_ok:
            await asyncio.to_thread(attendance_store.sync_from_supabase)
            await asyncio.to_thread(vision_engine.load_known_faces)

    vision_status = vision_engine.get_status()
    local_previews = vision_status.get("array_previews", {})

    # Build unified list showing source (supabase / local / synced)
    merged_map = {}
    for rec in sb_records:
        sid = rec["id_name"]
        merged_map[sid] = {
            **rec,
            "synced_supabase": True,
            "in_local": sid in local_previews,
            "saved_pics": local_previews.get(sid, {}).get("saved_pics", 0),
        }

    for sid, lrec in local_previews.items():
        if sid not in merged_map:
            samples = lrec.get("array_faceshape", [])
            merged_map[sid] = {
                "id_name": sid,
                "m": lrec.get("m") or attendance_store.get_person_m(sid),
                "array_faceshape": samples,
                "samples_count": lrec.get("samples", len(samples)),
                "dims": lrec.get("dims", 128),
                "preview": lrec.get("preview", "[]"),
                "synced_supabase": False,
                "in_local": True,
                "saved_pics": lrec.get("saved_pics", 0),
            }

    records_list = sorted(merged_map.values(), key=lambda x: x["id_name"])
    return {
        "success": True,
        "supabase_ok": sb_ok,
        "supabase_error": sb_err,
        "supabase": supabase_manager.get_status(),
        "records": records_list,
    }


@app.post("/api/supabase/records")
async def api_save_supabase_record(payload: SupabaseRecordRequest):
    """
    Add or update a record (`id_name`, `m`, `array_faceshape`) via the Web Interface.
    Supports either:
    - `capture_from_camera=True`: captures directly from the live camera
    - `array_faceshape`: 128-d array (from camera extract, photo upload, or manual JSON)
    Saves to Supabase AND updates local VisionEngine + AttendanceStore for immediate recognition.
    """
    id_name = payload.id_name.strip()
    m_val = (payload.m or "-").strip() or "-"

    if not id_name:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "กรุณาระบุ id_name (รหัสหรือชื่อ-นามสกุล)"},
        )

    if payload.capture_from_camera and not payload.array_faceshape:
        ok_cap = await asyncio.to_thread(vision_engine.register_face_from_frame, id_name, None, m_val)
        if not ok_cap:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "message": "ไม่พบใบหน้าในกล้องสด กรุณาหันหน้าเข้าหากล้องให้ชัดเจน",
                },
            )
        return {
            "success": True,
            "id_name": id_name,
            "m": m_val,
            "message": (
                f"ถ่ายภาพและบันทึก '{id_name}' (ชั้น {m_val}) ลง Supabase และระบบตรวจจับเรียบร้อย!"
                if supabase_manager.is_configured
                else f"ถ่ายภาพและบันทึก '{id_name}' (ชั้น {m_val}) ในเครื่องเรียบร้อย (ยังไม่ได้เชื่อมต่อ Supabase)"
            ),
            "supabase": supabase_manager.get_status(),
            "vision": vision_engine.get_status(),
        }

    samples = parse_faceshape_to_samples(payload.array_faceshape)
    if not samples:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ไม่พบค่า array_faceshape ที่ถูกต้อง (ต้องเป็นเวกเตอร์ 128 มิติ) — กรุณาดึงค่าจากกล้อง อัปโหลดรูป หรือวาง JSON Array",
            },
        )

    combined_local = await asyncio.to_thread(
        vision_engine.save_custom_faceshape_record,
        id_name,
        m_val,
        samples,
        payload.append_sample,
    )

    if supabase_manager.is_configured:
        ok_sb, msg_sb, rec_sb = await asyncio.to_thread(
            supabase_manager.upsert_record,
            id_name,
            m_val,
            combined_local,
            False,
        )
        if not ok_sb:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "message": f"บันทึกในเครื่องแล้ว แต่บันทึกลง Supabase ไม่สำเร็จ: {msg_sb}",
                    "supabase": supabase_manager.get_status(),
                },
            )
        return {
            "success": True,
            "id_name": id_name,
            "m": m_val,
            "samples_count": len(combined_local),
            "message": msg_sb,
            "record": rec_sb,
            "supabase": supabase_manager.get_status(),
            "vision": vision_engine.get_status(),
        }

    return {
        "success": True,
        "id_name": id_name,
        "m": m_val,
        "samples_count": len(combined_local),
        "message": f"บันทึก '{id_name}' (ชั้น {m_val}, {len(combined_local)} ตัวอย่าง) ในเครื่องเรียบร้อย (ตั้งค่า Supabase เพื่อซิงค์ขึ้น Cloud)",
        "supabase": supabase_manager.get_status(),
        "vision": vision_engine.get_status(),
    }


@app.delete("/api/supabase/records/{id_name}")
async def api_delete_supabase_record(id_name: str):
    """Delete a record (`id_name`) from both Supabase and local storage."""
    ok_local = await asyncio.to_thread(vision_engine.delete_face_data, id_name)
    if not ok_local:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": f"ไม่พบข้อมูลของ '{id_name}'"},
        )
    return {
        "success": True,
        "id_name": id_name,
        "message": f"ลบข้อมูล '{id_name}' ออกจากระบบและ Supabase เรียบร้อยแล้ว",
        "supabase": supabase_manager.get_status(),
        "vision": vision_engine.get_status(),
    }


@app.post("/api/supabase/sync")
async def api_sync_supabase(payload: Optional[SupabaseSyncRequest] = None):
    """
    Two-way sync between local storage and Supabase:
    - If `push_local=True`, pushes all local records (`id_name`, `m`, `array_faceshape`) to Supabase.
    - Pulls all records from Supabase into local `face_data.json` & `AttendanceStore`.
    """
    if not supabase_manager.is_configured:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ยังไม่ได้ตั้งค่า SUPABASE_URL และ SUPABASE_KEY",
                "supabase": supabase_manager.get_status(),
            },
        )

    pushed = 0
    push_errors = []
    if payload and payload.push_local:
        pushed, push_errors = await asyncio.to_thread(vision_engine.push_all_local_to_supabase)

    pulled_ok = await asyncio.to_thread(attendance_store.sync_from_supabase)
    await asyncio.to_thread(vision_engine.load_known_faces)

    if not pulled_ok and push_errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": f"ซิงค์ข้อมูลกับ Supabase ไม่สำเร็จ: {'; '.join(push_errors[:2])}",
                "supabase": supabase_manager.get_status(),
            },
        )

    msg = (
        f"อัปโหลดข้อมูลขึ้น Supabase {pushed} รายการ และซิงค์ข้อมูลล่าสุดสำเร็จ!"
        if (payload and payload.push_local)
        else "ดึงและซิงค์ข้อมูลจาก Supabase สำเร็จ!"
    )
    return {
        "success": True,
        "pushed_count": pushed,
        "message": msg,
        "supabase": supabase_manager.get_status(),
        "vision": vision_engine.get_status(),
        "metrics": attendance_store.get_public_metrics(),
    }


@app.post("/api/simulate-scan")
async def api_simulate_scan(payload: SimulateScanRequest):
    student_id = payload.student_id.strip()
    if payload.bypass_cooldown:
        vision_engine.last_seen.pop(student_id, None)

    if vision_engine.is_on_cooldown(student_id):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "logged": False,
                "reason": f"รหัส {student_id} อยู่ในช่วง Cooldown 5 นาที",
            },
        )

    logged = vision_engine.record_detection(student_id)
    return {
        "logged": logged,
        "student_id": student_id,
        "metrics": attendance_store.get_public_metrics(),
    }


@app.post("/api/reset-test")
async def api_reset_test():
    attendance_store.reset_today_attendance()
    vision_engine.last_seen.clear()
    return {
        "success": True,
        "metrics": attendance_store.get_public_metrics(),
    }


@app.post("/api/sync-now")
async def api_sync_now():
    """Force an immediate import of saved images in faces/, Supabase, and Excel."""
    await asyncio.to_thread(vision_engine.sync_folder_to_array, True)
    await asyncio.to_thread(attendance_store.sync_from_excel)
    await asyncio.to_thread(vision_engine.load_known_faces)
    return {
        "success": True,
        "metrics": attendance_store.get_public_metrics(),
        "vision": vision_engine.get_status(),
        "supabase": supabase_manager.get_status(),
    }


@app.post("/api/switch-camera/{camera_index}")
async def api_switch_camera(camera_index: int):
    """Switch camera device (e.g. 1 for USB Webcam, 0 for PC Camera)."""
    ok = vision_engine.switch_camera(camera_index)
    return {
        "success": ok,
        "camera_index": camera_index,
        "camera_name": "Webcam (Index 1)" if camera_index == 1 else f"Camera (Index {camera_index})",
        "vision": vision_engine.get_status(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
