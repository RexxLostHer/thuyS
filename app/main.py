"""
Wink HD Enhancer — FastAPI Backend Server
High-performance REST API and Server-Sent Events (SSE) server for AI image and video super-resolution.
"""

import os
import sys
import time
import uuid
import json
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.enhancer.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path
from app.enhancer.image_enhancer import enhance_image
from app.enhancer.video_enhancer import enhance_video, probe_video

# Directories
STATIC_DIR = PROJECT_ROOT / "app" / "static"
SAMPLES_DIR = PROJECT_ROOT / "samples"
MEDIA_DIR = PROJECT_ROOT / "media"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SERVER_START_TIME = time.time()

# In-memory storage for active tasks
# task_id -> {status, stage, percent, fps, message, input_url, output_url, result, error, queue}
TASKS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="Wink HD Enhancer API",
    description="Local AI Super-Resolution and Portrait Retouching Engine inspired by Wink",
    version="1.0.0",
)

# CORS middleware for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Mount static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/")
async def serve_index():
    """Serve the Wink HD Enhancer web application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "UI not found. Please ensure app/static/index.html is created."},
        )
    return FileResponse(str(index_file))


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    Returns server status, uptime, and confirms FFmpeg binary discovery.
    """
    ffmpeg_bin = get_ffmpeg_path()
    ffmpeg_found = (
        (ffmpeg_bin and Path(ffmpeg_bin).exists())
        or bool(shutil.which("ffmpeg"))
    )
    ffprobe_bin = get_ffprobe_path()
    ffprobe_found = (
        (ffprobe_bin and Path(ffprobe_bin).exists())
        or bool(shutil.which("ffprobe"))
    )

    return {
        "status": "ok",
        "uptime": round(time.time() - SERVER_START_TIME, 2),
        "ffmpeg": {
            "path": ffmpeg_bin,
            "available": bool(ffmpeg_found),
        },
        "ffprobe": {
            "path": ffprobe_bin,
            "available": bool(ffprobe_found),
        },
        "version": "1.0.0",
        "app": "Wink HD Enhancer",
    }


@app.get("/api/samples")
async def get_samples():
    """
    Retrieve pre-generated demonstration media for instant one-click UI evaluation.
    """
    samples = {}

    img_lowres = SAMPLES_DIR / "sample_image_lowres.png"
    img_hd = SAMPLES_DIR / "sample_image_enhanced_hd.png"
    if img_lowres.exists() and img_hd.exists():
        samples["image"] = {
            "id": "sample_portrait",
            "title": "Beauty Portrait (Low-Res)",
            "type": "image",
            "lowres_url": "/samples/sample_image_lowres.png",
            "enhanced_url": "/samples/sample_image_enhanced_hd.png",
            "lowres_res": [128, 128],
            "enhanced_res": [512, 512],
            "upscale_factor": 4,
            "description": "Pre-generated portrait with facial retouching and 4x Lanczos-4 super-resolution.",
        }

    vid_lowres = SAMPLES_DIR / "sample_video_lowres.mp4"
    vid_hd = SAMPLES_DIR / "sample_video_enhanced_hd.mp4"
    if vid_lowres.exists() and vid_hd.exists():
        samples["video"] = {
            "id": "sample_video",
            "title": "Action Motion Video (Low-Res)",
            "type": "video",
            "lowres_url": "/samples/sample_video_lowres.mp4",
            "enhanced_url": "/samples/sample_video_enhanced_hd.mp4",
            "lowres_res": [320, 240],
            "enhanced_res": [640, 480],
            "upscale_factor": 2,
            "fps": 15.0,
            "duration": 3.0,
            "description": "Synchronized video sample with bit-exact 440Hz audio preservation.",
        }

    return {
        "status": "success",
        "count": len(samples),
        "samples": samples,
    }


@app.get("/api/media/{filename}")
async def get_media_file(filename: str):
    """Serve media files directly through the API."""
    target = MEDIA_DIR / filename
    if not target.exists() or not target.is_file():
        # Check samples as fallback
        sample_target = SAMPLES_DIR / filename
        if sample_target.exists() and sample_target.is_file():
            return FileResponse(str(sample_target))
        raise HTTPException(status_code=404, detail=f"Media file '{filename}' not found")
    return FileResponse(str(target))


@app.post("/api/enhance/image")
async def api_enhance_image(
    file: UploadFile = File(...),
    upscale_factor: int = Form(2),
    face_retouch: float = Form(0.5),
    sharpen: float = Form(0.5),
    vibrance: float = Form(0.3),
    denoise: float = Form(0.3),
):
    """
    Enhance an uploaded image using super-resolution, portrait retouching, and CAS sharpening.
    """
    # Validate parameters
    if upscale_factor not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="upscale_factor must be 1, 2, or 4")

    face_retouch = max(0.0, min(1.0, float(face_retouch)))
    sharpen = max(0.0, min(1.0, float(sharpen)))
    vibrance = max(0.0, min(1.0, float(vibrance)))
    denoise = max(0.0, min(1.0, float(denoise)))

    # Read uploaded bytes
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes)")

    # Validate image decoding
    nparr = np.frombuffer(content, np.uint8)
    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if decoded_img is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image format")

    # If image has alpha channel, convert to 3-channel BGR
    if len(decoded_img.shape) == 3 and decoded_img.shape[2] == 4:
        decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGRA2BGR)
    elif len(decoded_img.shape) == 2:
        decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_GRAY2BGR)

    task_id = uuid.uuid4().hex[:12]
    orig_ext = Path(file.filename).suffix.lower() if file.filename else ".png"
    if orig_ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        orig_ext = ".png"

    in_filename = f"upload_img_{task_id}{orig_ext}"
    out_filename = f"enhanced_img_{task_id}.png"

    in_path = MEDIA_DIR / in_filename
    out_path = MEDIA_DIR / out_filename

    # Save input image
    cv2.imwrite(str(in_path), decoded_img)

    try:
        # Run enhancement
        result = enhance_image(
            input_path=str(in_path),
            output_path=str(out_path),
            upscale_factor=upscale_factor,
            face_retouch=face_retouch,
            sharpen=sharpen,
            vibrance=vibrance,
            denoise=denoise,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(exc)}")

    return {
        "status": "success",
        "task_id": task_id,
        "input_url": f"/media/{in_filename}",
        "output_url": f"/media/{out_filename}",
        "input_resolution": result["input_resolution"],
        "output_resolution": result["output_resolution"],
        "processing_time_ms": result["processing_time_ms"],
        "metrics": result["metrics"],
    }


async def _run_video_enhancer_task(
    task_id: str,
    in_path: Path,
    out_path: Path,
    upscale_factor: int,
    face_retouch: float,
    sharpen: float,
    vibrance: float,
    denoise: float,
):
    """Asynchronous background worker executing video enhancement pipeline."""
    task = TASKS[task_id]
    input_url = f"/media/{in_path.name}"
    output_url = f"/media/{out_path.name}"

    async def progress_callback(stage: str, percent: float, fps: float, message: str = ""):
        # Avoid race condition: do not overwrite terminal state if task already finished
        if task.get("status") in ("completed", "failed"):
            return
        # Avoid duplicate completion event without result payload
        if stage == "completed":
            return
        event = {
            "task_id": task_id,
            "status": "processing",
            "stage": stage,
            "percent": round(percent, 1),
            "fps": round(fps, 1),
            "message": message,
            "input_url": input_url,
            "output_url": output_url,
        }
        task["status"] = "processing"
        task["stage"] = stage
        task["percent"] = round(percent, 1)
        task["fps"] = round(fps, 1)
        task["message"] = message
        task["history"].append(event)
        try:
            task["queue"].put_nowait(event)
        except Exception:
            pass

    try:
        task["status"] = "processing"
        result = await enhance_video(
            input_path=str(in_path),
            output_path=str(out_path),
            upscale_factor=upscale_factor,
            face_retouch=face_retouch,
            sharpen=sharpen,
            vibrance=vibrance,
            denoise=denoise,
            progress_callback=progress_callback,
        )

        # WAJIB: Embed input_url and output_url into result so frontend data.result.input_url is never undefined
        result["input_url"] = input_url
        result["output_url"] = output_url

        final_event = {
            "task_id": task_id,
            "status": "completed",
            "stage": "completed",
            "percent": 100.0,
            "fps": result.get("fps", 0.0),
            "message": "Video enhancement completed successfully",
            "input_url": input_url,
            "output_url": output_url,
            "result": result,
        }
        task["status"] = "completed"
        task["stage"] = "completed"
        task["percent"] = 100.0
        task["input_url"] = input_url
        task["output_url"] = output_url
        task["result"] = result
        task["final_event"] = final_event
        task["history"].append(final_event)
        try:
            task["queue"].put_nowait(final_event)
        except Exception:
            pass

    except Exception as exc:
        err_msg = str(exc)
        err_event = {
            "task_id": task_id,
            "status": "failed",
            "stage": "failed",
            "percent": task.get("percent", 0.0),
            "fps": 0.0,
            "message": f"Video enhancement error: {err_msg}",
            "error": err_msg,
            "input_url": input_url,
            "output_url": None,
        }
        task["status"] = "failed"
        task["stage"] = "failed"
        task["error"] = err_msg
        task["final_event"] = err_event
        task["history"].append(err_event)
        try:
            task["queue"].put_nowait(err_event)
        except Exception:
            pass


@app.post("/api/enhance/video")
async def api_enhance_video(
    file: UploadFile = File(...),
    upscale_factor: int = Form(2),
    face_retouch: float = Form(0.5),
    sharpen: float = Form(0.5),
    vibrance: float = Form(0.3),
    denoise: float = Form(0.3),
):
    """
    Start an asynchronous video enhancement task.
    Returns immediately with a unique task_id for SSE progress tracking.
    """
    if upscale_factor not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="upscale_factor must be 1, 2, or 4")

    face_retouch = max(0.0, min(1.0, float(face_retouch)))
    sharpen = max(0.0, min(1.0, float(sharpen)))
    vibrance = max(0.0, min(1.0, float(vibrance)))
    denoise = max(0.0, min(1.0, float(denoise)))

    # Read uploaded bytes
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes)")

    task_id = uuid.uuid4().hex[:12]
    orig_ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    if orig_ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        orig_ext = ".mp4"

    in_filename = f"upload_vid_{task_id}{orig_ext}"
    out_filename = f"enhanced_vid_{task_id}.mp4"

    in_path = MEDIA_DIR / in_filename
    out_path = MEDIA_DIR / out_filename

    # Write video to disk
    with open(in_path, "wb") as f:
        f.write(content)

    # Validate that ffprobe can probe the video file
    try:
        probe_meta = probe_video(str(in_path))
        if probe_meta["width"] <= 0 or probe_meta["height"] <= 0:
            if in_path.exists():
                in_path.unlink()
            raise HTTPException(status_code=400, detail="Invalid video dimensions in file")
    except Exception as exc:
        if in_path.exists():
            in_path.unlink()
        raise HTTPException(status_code=400, detail=f"Corrupt or invalid video format: {str(exc)}")

    # Initialize task record
    TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "percent": 0.0,
        "fps": 0.0,
        "message": "Task queued for processing",
        "input_url": f"/media/{in_filename}",
        "output_url": None,
        "result": None,
        "error": None,
        "history": [],
        "final_event": None,
        "queue": asyncio.Queue(),
    }

    # Start background task in a dedicated daemon thread to ensure it survives
    # TestClient per-request cancel scopes and never blocks or aborts prematurely
    def _run_worker_thread():
        asyncio.run(
            _run_video_enhancer_task(
                task_id=task_id,
                in_path=in_path,
                out_path=out_path,
                upscale_factor=upscale_factor,
                face_retouch=face_retouch,
                sharpen=sharpen,
                vibrance=vibrance,
                denoise=denoise,
            )
        )

    t = threading.Thread(target=_run_worker_thread, daemon=True)
    t.start()

    return {
        "status": "queued",
        "task_id": task_id,
        "input_url": f"/media/{in_filename}",
        "message": "Video enhancement task initiated",
    }


@app.get("/api/tasks/{task_id}/events")
@app.get("/api/progress/{task_id}")
async def stream_task_events(request: Request, task_id: str):
    """SSE (Server-Sent Events) endpoint streaming real-time video processing updates."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task = TASKS[task_id]

    async def event_generator():
        sent = 0
        last_heartbeat = time.time()
        while True:
            if await request.is_disconnected():
                break

            events = list(task.get("history", []))
            while sent < len(events):
                event = events[sent]
                sent += 1
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ("completed", "failed", "error"):
                    return

            if task.get("status") in ("completed", "failed", "error") and sent >= len(events):
                if task.get("final_event") and sent == 0:
                    yield f"data: {json.dumps(task['final_event'])}\n\n"
                return

            now = time.time()
            if now - last_heartbeat >= 1.0:
                last_heartbeat = now
                yield ": heartbeat\n\n"

            await asyncio.sleep(0.05)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/tasks/{task_id}")
@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """JSON poll fallback endpoint to check status of async task."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task = TASKS[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "stage": task["stage"],
        "percent": task["percent"],
        "fps": task["fps"],
        "message": task["message"],
        "input_url": task.get("input_url"),
        "output_url": task.get("output_url"),
        "result": task.get("result"),
        "error": task.get("error"),
    }
