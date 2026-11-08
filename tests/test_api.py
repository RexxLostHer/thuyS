"""
Unit and Integration Tests for Wink HD Enhancer FastAPI Server
Covers:
- /api/health
- /api/samples
- Static file serving (root /, /static/css, /static/js)
- Media serving (/api/media/{filename}, /media/{filename})
- POST /api/enhance/image (success, 1x, 2x, 4x, invalid factors, empty/corrupt files)
- POST /api/enhance/video (task initialization, async progress SSE streaming, status polling)
- Boundary and error conditions (404, 400)
"""

import os
import io
import time
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app, TASKS, MEDIA_DIR, SAMPLES_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_path():
    """Return path to existing lowres sample image."""
    img_path = SAMPLES_DIR / "sample_image_lowres.png"
    assert img_path.exists(), f"Sample image missing at {img_path}"
    return img_path


@pytest.fixture
def sample_video_path():
    """Return path to existing lowres sample video."""
    vid_path = SAMPLES_DIR / "sample_video_lowres.mp4"
    assert vid_path.exists(), f"Sample video missing at {vid_path}"
    return vid_path


def test_health_endpoint(client):
    """GET /api/health returns server status, uptime, and confirms FFmpeg."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "uptime" in data and data["uptime"] >= 0
    assert "ffmpeg" in data
    assert data["ffmpeg"]["available"] is True
    assert "ffprobe" in data
    assert data["ffprobe"]["available"] is True
    assert data["app"] == "Wink HD Enhancer"


def test_samples_endpoint(client):
    """GET /api/samples returns demonstration media metadata for 1-click UI evaluation."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["count"] >= 2
    assert "image" in data["samples"]
    assert "video" in data["samples"]

    img_meta = data["samples"]["image"]
    assert img_meta["lowres_res"] == [128, 128]
    assert img_meta["enhanced_res"] == [512, 512]
    assert img_meta["upscale_factor"] == 4

    vid_meta = data["samples"]["video"]
    assert vid_meta["lowres_res"] == [320, 240]
    assert vid_meta["enhanced_res"] == [640, 480]
    assert vid_meta["upscale_factor"] == 2


def test_root_index_serving(client):
    """GET / serves index.html."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "WINK HD" in res.text or "Wink HD" in res.text
    assert "comparisonContainer" in res.text


def test_static_asset_serving(client):
    """GET /static/... serves CSS and JavaScript files correctly."""
    css_res = client.get("/static/css/style.css")
    assert css_res.status_code == 200
    assert "comparison-container" in css_res.text or "clip-path" in css_res.text

    slider_res = client.get("/static/js/slider.js")
    assert slider_res.status_code == 200
    assert "ComparisonSlider" in slider_res.text

    app_js_res = client.get("/static/js/app.js")
    assert app_js_res.status_code == 200
    assert "DOMContentLoaded" in app_js_res.text


def test_enhance_image_valid_2x(client, sample_image_path):
    """POST /api/enhance/image processes lowres image with 2x upscaling."""
    with open(sample_image_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/enhance/image",
        files={"file": ("test_portrait.png", file_bytes, "image/png")},
        data={
            "upscale_factor": "2",
            "face_retouch": "0.5",
            "sharpen": "0.5",
            "vibrance": "0.3",
            "denoise": "0.2",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "task_id" in data
    assert data["input_resolution"] == [128, 128]
    assert data["output_resolution"] == [256, 256]
    assert data["processing_time_ms"] > 0
    assert "metrics" in data
    assert data["metrics"]["sharpness_gain"] > 1.0
    assert data["metrics"]["psnr_baseline"] > 20.0

    # Verify output URL is accessible
    out_url = data["output_url"]
    out_res = client.get(out_url)
    assert out_res.status_code == 200
    assert len(out_res.content) > 1000


def test_enhance_image_valid_4x(client, sample_image_path):
    """POST /api/enhance/image processes image with 4x upscaling."""
    with open(sample_image_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/enhance/image",
        files={"file": ("test_portrait.png", file_bytes, "image/png")},
        data={
            "upscale_factor": "4",
            "face_retouch": "0.6",
            "sharpen": "0.4",
            "vibrance": "0.2",
            "denoise": "0.2",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["output_resolution"] == [512, 512]


def test_enhance_image_invalid_upscale_factor(client, sample_image_path):
    """POST /api/enhance/image rejects unsupported upscale factors."""
    with open(sample_image_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/enhance/image",
        files={"file": ("test.png", file_bytes, "image/png")},
        data={"upscale_factor": "3"},
    )
    assert response.status_code == 400
    assert "upscale_factor must be 1, 2, or 4" in response.json()["detail"]


def test_enhance_image_empty_file(client):
    """POST /api/enhance/image returns 400 when uploaded file is empty."""
    response = client.post(
        "/api/enhance/image",
        files={"file": ("empty.png", b"", "image/png")},
        data={"upscale_factor": "2"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_enhance_image_corrupt_file(client):
    """POST /api/enhance/image returns 400 when uploaded file cannot be decoded."""
    response = client.post(
        "/api/enhance/image",
        files={"file": ("corrupt.png", b"NOT_A_VALID_IMAGE_HEADER_12345", "image/png")},
        data={"upscale_factor": "2"},
    )
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_enhance_video_task_and_sse_lifecycle(client, sample_video_path):
    """
    POST /api/enhance/video queues a background task, returns task_id,
    and streams progress via SSE until completion.
    """
    with open(sample_video_path, "rb") as f:
        file_bytes = f.read()

    # 1. Queue task
    response = client.post(
        "/api/enhance/video",
        files={"file": ("sample_action.mp4", file_bytes, "video/mp4")},
        data={
            "upscale_factor": "1",  # 1x for speedy test execution
            "face_retouch": "0.2",
            "sharpen": "0.3",
            "vibrance": "0.2",
            "denoise": "0.1",
        },
    )

    assert response.status_code == 200
    init_data = response.json()
    assert init_data["status"] == "queued"
    task_id = init_data["task_id"]
    assert len(task_id) > 0

    # 2. Check status polling endpoint
    status_res = client.get(f"/api/status/{task_id}")
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert st_data["task_id"] == task_id
    assert st_data["status"] in ("queued", "processing", "completed")

    # 3. Stream SSE events
    events_received = []
    with client.stream("GET", f"/api/progress/{task_id}") as stream:
        assert stream.status_code == 200
        assert "text/event-stream" in stream.headers["content-type"]

        for line in stream.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                events_received.append(payload)
                if payload.get("status") in ("completed", "error"):
                    break

    assert len(events_received) > 0
    final_event = events_received[-1]
    assert final_event["status"] == "completed"
    assert final_event["percent"] == 100.0
    assert "output_url" in final_event

    # 4. Verify output video file is accessible via media route
    out_url = final_event["output_url"]
    media_res = client.get(out_url)
    assert media_res.status_code == 200
    assert len(media_res.content) > 1000


def test_video_progress_non_existent_task(client):
    """GET /api/progress/{invalid_id} returns 404."""
    response = client.get("/api/progress/non_existent_random_id_999999")
    assert response.status_code == 404


def test_video_status_non_existent_task(client):
    """GET /api/status/{invalid_id} returns 404."""
    response = client.get("/api/status/non_existent_random_id_999999")
    assert response.status_code == 404


def test_enhance_video_empty_file(client):
    """POST /api/enhance/video returns 400 when video file is empty."""
    response = client.post(
        "/api/enhance/video",
        files={"file": ("empty.mp4", b"", "video/mp4")},
        data={"upscale_factor": "2"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_enhance_video_corrupt_file(client):
    """POST /api/enhance/video returns 400 when video cannot be probed."""
    response = client.post(
        "/api/enhance/video",
        files={"file": ("corrupt.mp4", b"CORRUPTED_VIDEO_HEADER_DATA_12345", "video/mp4")},
        data={"upscale_factor": "2"},
    )
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_media_file_serving(client, sample_image_path):
    """GET /api/media/{filename} serves files from MEDIA_DIR and SAMPLES_DIR."""
    # Test sample serving
    res = client.get(f"/api/media/{sample_image_path.name}")
    assert res.status_code == 200
    assert len(res.content) > 0

    # Test 404 for missing media
    missing_res = client.get("/api/media/definitely_missing_media_file_12345.png")
    assert missing_res.status_code == 404
