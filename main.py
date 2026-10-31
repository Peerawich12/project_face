import argparse
import logging
import threading
import webbrowser

import uvicorn

from app import app
from data_sync import attendance_store
from vision import vision_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("school_attendance.main")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified launcher for Smooth Face-Shape Array Detection (No Excel Required)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host address for the web dashboard (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web dashboard (default: 8000)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera device index (default: 1 for USB Webcam, 0 for PC Camera)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=300,
        help="Webcam debounce cooldown in seconds per person (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--hide-camera",
        action="store_true",
        help="Hide the local OpenCV camera_pc window (stream only in web browser)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web browser on startup",
    )

    args = parser.parse_args()

    # Configure camera_pc settings
    vision_engine.camera_index = args.camera
    vision_engine.cooldown_seconds = args.cooldown
    vision_engine.show_window = not args.hide_camera

    # Load any existing face-shape arrays from face_data.json (No Excel required)
    vision_engine.load_known_faces()
    attendance_store.sync_from_face_data()
    metrics = attendance_store.get_public_metrics()

    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print("  SMOOTH FACE-SHAPE ARRAY DETECTOR (NO EXCEL REQUIRED FOR TEST)")
    print("=" * 72)
    print(f"  • Live Web Dashboard     : {url}")
    print(f"  • PC Webcam (camera_pc)  : Device Index {args.camera} (Press 'R' to register label)")
    print(f"  • Face-Shape Array Store : {vision_engine.face_data_file.resolve()} (128-d Arrays, No Images)")
    print(f"  • Cooldown Debounce      : {args.cooldown} seconds ({args.cooldown // 60} mins)")
    print("-" * 72)
    print(
        f"  • Registered Labels      : {metrics['total_students']} | "
        f"Detected Today: {metrics['present_students']}"
    )
    print("=" * 72)

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
