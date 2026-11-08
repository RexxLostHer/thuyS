"""
Unit Tests for Wink Video Enhancer Module
"""

import os
import pytest
from pathlib import Path
from app.enhancer.video_enhancer import enhance_video, probe_video

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_probe_video():
    sample_vid = str(SAMPLES_DIR / "sample_video_lowres.mp4")
    assert os.path.exists(sample_vid), "Sample video must exist for tests"
    
    meta = probe_video(sample_vid)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 15.0
    assert meta["frame_count"] == 45
    assert meta["has_audio"] is True
    assert meta["audio_codec"] == "aac"
    assert abs(meta["duration"] - 3.0) < 0.1


import asyncio


def test_enhance_video_e2e(tmp_path):
    sample_vid = str(SAMPLES_DIR / "sample_video_lowres.mp4")
    out_vid = str(tmp_path / "enhanced_test_video.mp4")
    
    stages_logged = []
    
    def on_progress(stage, percent, fps, message=""):
        stages_logged.append(stage)
        
    res = asyncio.run(enhance_video(
        input_path=sample_vid,
        output_path=out_vid,
        upscale_factor=2,
        face_retouch=0.4,
        sharpen=0.5,
        vibrance=0.3,
        denoise=0.2,
        progress_callback=on_progress,
    ))
    
    assert res["status"] == "success"
    assert res["frame_count"] == 45
    assert res["audio_preserved"] is True
    assert abs(res["output_duration"] - 3.0) < 0.1
    assert os.path.exists(out_vid)
    
    out_meta = probe_video(out_vid)
    assert out_meta["width"] == 640
    assert out_meta["height"] == 480
    assert out_meta["has_audio"] is True
    assert "enhancing" in stages_logged
    assert "completed" in stages_logged


def test_enhance_video_missing_input():
    with pytest.raises(FileNotFoundError):
        asyncio.run(enhance_video(
            input_path="missing_video_9999.mp4",
            output_path="out.mp4"
        ))


def test_enhance_video_odd_dimensions_upscale_1x(tmp_path):
    """Test odd dimensions (127x95) with upscale_factor=1 does not crash libx264 yuv420p."""
    from app.enhancer.ffmpeg_utils import run_ffmpeg
    odd_vid = str(tmp_path / "odd_127x95.avi")
    out_vid = str(tmp_path / "odd_1x_out.mp4")
    
    run_ffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=blue:size=127x95:rate=15:duration=0.5",
        "-c:v", "mjpeg", "-pix_fmt", "yuvj444p",
        odd_vid
    ], check=True)
    
    res = asyncio.run(enhance_video(
        input_path=odd_vid,
        output_path=out_vid,
        upscale_factor=1,
    ))
    
    assert res["status"] == "success"
    assert os.path.exists(out_vid)
    out_meta = probe_video(out_vid)
    assert out_meta["width"] % 2 == 0
    assert out_meta["height"] % 2 == 0
    assert out_meta["width"] >= 127
    assert out_meta["height"] >= 95


def test_enhance_video_odd_dimensions_upscale_2x(tmp_path):
    """Test odd dimensions (127x95) with upscale_factor=2 yields exact 254x190 dimensions."""
    from app.enhancer.ffmpeg_utils import run_ffmpeg
    odd_vid = str(tmp_path / "odd_127x95_2x.avi")
    out_vid = str(tmp_path / "odd_2x_out.mp4")
    
    run_ffmpeg([
        "-y", "-f", "lavfi", "-i", "color=c=red:size=127x95:rate=15:duration=0.5",
        "-c:v", "mjpeg", "-pix_fmt", "yuvj444p",
        odd_vid
    ], check=True)
    
    res = asyncio.run(enhance_video(
        input_path=odd_vid,
        output_path=out_vid,
        upscale_factor=2,
    ))
    
    assert res["status"] == "success"
    assert os.path.exists(out_vid)
    out_meta = probe_video(out_vid)
    assert out_meta["width"] == 254
    assert out_meta["height"] == 190

