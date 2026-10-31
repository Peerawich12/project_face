import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from data_sync import attendance_store
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
    """Start VisionEngine (with 7-pic burst & folder-to-array watcher) on startup."""
    attendance_store.start_scheduler()
    vision_engine.start_background_capture()
    yield
    vision_engine.stop_background_capture()
    attendance_store.stop_scheduler()


app = FastAPI(
    title="ระบบเช็คชื่อด้วยกล้อง PC — 7-Picture Burst to Face-Shape Array",
    description="Takes 7 photos of a face, saves to faces/ folder, and immediately imports them into a 7-sample Face-Shape Array in face_data.json.",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class SimulateScanRequest(BaseModel):
    student_id: str
    bypass_cooldown: bool = False


@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    metrics = attendance_store.get_public_metrics()
    roster = attendance_store.get_student_roster_details()
    vision_status = vision_engine.get_status()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "roster": roster,
            "vision_status": vision_status,
        },
    )


@app.get("/api/metrics", response_class=JSONResponse)
async def api_public_metrics():
    return {
        **attendance_store.get_public_metrics(),
        "roster": attendance_store.get_student_roster_details(),
        "vision": vision_engine.get_status(),
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
    safe_id = "".join(c for c in student_id if c.isalnum() or c in ("_", "-"))
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
async def api_register_face(student_id: str):
    """
    Take 7 burst pictures of the person's face from camera_pc, save them to faces/<student_id>/,
    and immediately import all 7 pictures into the Face-Shape Array in face_data.json.
    """
    ok = await asyncio.to_thread(vision_engine.register_face_from_frame, student_id)
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "message": "ไม่สามารถถ่ายครบ 7 รูปได้ — กรุณาหันหน้าเข้าหากล้องให้ชัดเจน",
            },
        )
    return {
        "success": True,
        "student_id": student_id,
        "message": f"ถ่ายภาพใบหน้า 7 รูปลงโฟลเดอร์ faces/{student_id}/ และแปลงเป็น Face-Shape Array สำเร็จ!",
        "vision": vision_engine.get_status(),
    }


@app.post("/api/upload-face/{student_id}")
@app.post("/api/upload-pic/{student_id}")
async def api_upload_face(student_id: str, file: UploadFile = File(...)):
    """
    Save uploaded image to faces/<student_id>/ and immediately import it into the Face-Shape Array.
    """
    image_bytes = await file.read()
    ok = await asyncio.to_thread(vision_engine.register_face_from_upload, student_id, image_bytes)
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
        "message": f"บันทึกรูปลงโฟลเดอร์ faces/{student_id}/ และนำเข้าเป็น Face-Shape Array เรียบร้อย",
        "vision": vision_engine.get_status(),
    }


@app.delete("/api/face-data/{student_id}")
async def api_delete_face_data(student_id: str):
    ok = vision_engine.delete_face_data(student_id)
    if not ok:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": f"ไม่พบข้อมูลของ {student_id}"},
        )
    return {
        "success": True,
        "student_id": student_id,
        "message": f"ลบรูปและ Face-Shape Array ของ {student_id} เรียบร้อย",
        "vision": vision_engine.get_status(),
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
    """Force an immediate import of all saved images in faces/ into face_data.json."""
    await asyncio.to_thread(vision_engine.sync_folder_to_array, True)
    attendance_store.sync_from_excel()
    vision_engine.load_known_faces()
    return {
        "success": True,
        "metrics": attendance_store.get_public_metrics(),
        "vision": vision_engine.get_status(),
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
