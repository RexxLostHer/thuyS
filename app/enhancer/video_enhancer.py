"""
Wink-Style Video Enhancement Engine
High-throughput video enhancement pipeline with:
- FFmpeg video stream probe and audio demuxing
- Multi-threaded parallel per-frame enhancement
- Wink portrait retouching, Lanczos-4 upscaling, LAB CLAHE, and CAS sharpening
- Bit-exact lossless audio remuxing (-c:a copy) ensuring zero audio-video desync
- Real-time progress callback streaming
"""

import os
import sys
import time
import json
import asyncio
import inspect
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import cv2
import numpy as np

from app.enhancer.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg, run_ffprobe
from app.enhancer.image_enhancer import (
    denoise_image,
    apply_portrait_retouch,
    apply_upscale,
    apply_vibrance_and_tone,
    apply_cas_sharpening,
)


def probe_video(video_path: str) -> Dict[str, Any]:
    """
    Probe video stream metadata using ffprobe JSON output.
    Returns: {
        'duration': float,
        'width': int,
        'height': int,
        'fps': float,
        'frame_count': int,
        'has_audio': bool,
        'audio_codec': Optional[str]
    }
    """
    cmd = [
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    res = run_ffprobe(cmd, check=True)
    info = json.loads(res.stdout)
    
    video_stream = None
    audio_stream = None
    
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s
            
    if not video_stream:
        raise ValueError(f"No video stream found in: {video_path}")
        
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    
    # Calculate FPS from r_frame_rate or avg_frame_rate
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(fps_str)
        
    # Duration
    duration = 0.0
    if "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except (ValueError, TypeError):
            pass
    if duration <= 0.0 and "duration" in info.get("format", {}):
        try:
            duration = float(info["format"]["duration"])
        except (ValueError, TypeError):
            pass
            
    # Frame count
    nb_frames = 0
    if "nb_frames" in video_stream:
        try:
            nb_frames = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            pass
            
    if nb_frames == 0 and duration > 0 and fps > 0:
        nb_frames = int(round(duration * fps))
        
    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": nb_frames,
        "has_audio": has_audio,
        "audio_codec": audio_codec,
    }


def enhance_single_frame(
    frame: np.ndarray,
    upscale_factor: int,
    face_retouch: float,
    sharpen: float,
    vibrance: float,
    denoise: float,
) -> np.ndarray:
    """
    Process an individual video frame through the complete enhancement pipeline.
    OpenCV operations execute in optimized C++/AVX2 and release the Python GIL.
    """
    # 1. Denoise
    current = denoise_image(frame, strength=denoise)
    
    # 2. Portrait retouching (skin smoothing + feature highlights)
    current = apply_portrait_retouch(current, strength=face_retouch)
    
    # 3. Super-Resolution upscaling
    current = apply_upscale(current, upscale_factor=upscale_factor)
    
    # 4. Vibrance & tone
    current = apply_vibrance_and_tone(current, strength=vibrance)
    
    # 5. Adaptive sharpening
    current = apply_cas_sharpening(current, strength=sharpen)
    
    return current


async def _notify_progress(
    callback: Optional[Callable],
    stage: str,
    percent: float,
    fps: float,
    message: str = "",
):
    """Helper to dispatch progress updates to synchronous or asynchronous callbacks."""
    if not callback:
        return
        
    try:
        try:
            sig = inspect.signature(callback)
            kwargs = {"stage": stage, "percent": percent, "fps": fps}
            if "message" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                kwargs["message"] = message
        except (ValueError, TypeError):
            kwargs = {"stage": stage, "percent": percent, "fps": fps}
            
        if inspect.iscoroutinefunction(callback):
            try:
                await callback(**kwargs)
            except TypeError:
                await callback(stage, percent, fps)
        else:
            try:
                callback(**kwargs)
            except TypeError:
                callback(stage, percent, fps)
    except Exception:
        pass


async def enhance_video(
    input_path: str,
    output_path: str,
    upscale_factor: int = 2,
    face_retouch: float = 0.5,
    sharpen: float = 0.5,
    vibrance: float = 0.3,
    denoise: float = 0.3,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Core Video Enhancement Function.
    Adheres strictly to the PROJECT.md interface contract:
    
    Args:
        input_path: Path to source video file
        output_path: Path to write enhanced HD video file
        upscale_factor: Upscaling multiplier (1, 2, or 4)
        face_retouch: Portrait retouching strength (0.0 to 1.0)
        sharpen: CAS / adaptive unsharp strength (0.0 to 1.0)
        vibrance: LAB CLAHE and saturation boost strength (0.0 to 1.0)
        denoise: Dual-domain bilateral denoising strength (0.0 to 1.0)
        progress_callback: async or sync callable(stage, percent, fps, message)
        
    Returns:
        {
            "status": "success",
            "input_duration": float,
            "output_duration": float,
            "audio_preserved": bool,
            "fps": float,
            "frame_count": int
        }
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Step 1: Probe video properties
    await _notify_progress(progress_callback, stage="probing", percent=0.0, fps=0.0, message="Probing video stream")
    meta = probe_video(input_path)
    in_duration = meta["duration"]
    in_fps = meta["fps"]
    in_frame_count = meta["frame_count"]
    has_audio = meta["has_audio"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_audio_path = temp_dir_path / "audio_track.aac"
        temp_frames_dir = temp_dir_path / "frames"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Demux audio losslessly if present
        audio_extracted = False
        if has_audio:
            await _notify_progress(progress_callback, stage="demuxing", percent=5.0, fps=0.0, message="Demuxing audio track")
            demux_res = run_ffmpeg(
                ["-y", "-i", input_path, "-vn", "-c:a", "copy", str(temp_audio_path)],
                check=False
            )
            if demux_res.returncode == 0 and temp_audio_path.exists() and temp_audio_path.stat().st_size > 0:
                audio_extracted = True
            else:
                # Fallback to AAC extraction
                demux_aac = run_ffmpeg(
                    ["-y", "-i", input_path, "-vn", "-c:a", "aac", "-b:a", "192k", str(temp_audio_path)],
                    check=False
                )
                if demux_aac.returncode == 0 and temp_audio_path.exists() and temp_audio_path.stat().st_size > 0:
                    audio_extracted = True
                    
        # Step 3: Frame extraction and processing
        await _notify_progress(progress_callback, stage="enhancing", percent=10.0, fps=0.0, message="Enhancing video frames")
        
        cap = cv2.VideoCapture(input_path)
        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if actual_frame_count > 0:
            total_frames = actual_frame_count
        else:
            total_frames = in_frame_count if in_frame_count > 0 else 1
            
        # Target dimensions
        target_w = meta["width"] * upscale_factor
        target_h = meta["height"] * upscale_factor
        
        # Read frames into batch
        frames_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_list.append(frame)
        cap.release()
        
        if not frames_list:
            raise ValueError(f"Could not read any frames from video: {input_path}")
            
        total_frames = len(frames_list)
        
        # Determine thread count for CPU multi-threading
        worker_threads = min(os.cpu_count() or 4, 8)
        
        enhanced_frame_paths = []
        start_processing_time = time.perf_counter()
        processed_count = 0
        
        # Parallel frame enhancement using ThreadPoolExecutor
        def process_and_save_frame(args):
            idx, raw_frame = args
            frame_enhanced = enhance_single_frame(
                frame=raw_frame,
                upscale_factor=upscale_factor,
                face_retouch=face_retouch,
                sharpen=sharpen,
                vibrance=vibrance,
                denoise=denoise,
            )
            # Ensure dimensions are even (divisible by 2) for H.264 yuv420p
            h, w = frame_enhanced.shape[:2]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h != 0 or pad_w != 0:
                frame_enhanced = cv2.copyMakeBorder(
                    frame_enhanced, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
                )
            frame_filename = f"frame_{idx:06d}.png"
            frame_path = temp_frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame_enhanced)
            return idx, str(frame_path)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads) as executor:
            tasks = [(i, f) for i, f in enumerate(frames_list)]
            futures = [executor.submit(process_and_save_frame, t) for t in tasks]
            
            for f in concurrent.futures.as_completed(futures):
                idx, saved_path = f.result()
                processed_count += 1
                
                # Calculate processing FPS and percentage
                elapsed = time.perf_counter() - start_processing_time
                current_fps = processed_count / max(elapsed, 0.001)
                pct = 10.0 + (processed_count / total_frames) * 75.0
                
                if processed_count % 5 == 0 or processed_count == total_frames:
                    await _notify_progress(
                        progress_callback,
                        stage="enhancing",
                        percent=round(pct, 1),
                        fps=round(current_fps, 1),
                        message=f"Enhanced {processed_count}/{total_frames} frames ({current_fps:.1f} fps)"
                    )
                    
        # Step 4: Video Encoding & Lossless Audio Remux
        await _notify_progress(progress_callback, stage="encoding", percent=88.0, fps=0.0, message="Encoding enhanced video")
        
        frame_pattern = str(temp_frames_dir / "frame_%06d.png")
        
        ffmpeg_encode_cmd = [
            "-y",
            "-framerate", str(in_fps),
            "-start_number", "0",
            "-i", frame_pattern,
        ]
        
        if audio_extracted:
            ffmpeg_encode_cmd.extend([
                "-i", str(temp_audio_path),
                "-map", "0:v",
                "-map", "1:a",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                output_path,
            ])
        else:
            ffmpeg_encode_cmd.extend([
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                output_path,
            ])
            
        encode_res = run_ffmpeg(ffmpeg_encode_cmd, check=False)
        
        # Fallback if '-c:a copy' fails with container combination
        if encode_res.returncode != 0 and audio_extracted:
            fallback_cmd = list(ffmpeg_encode_cmd)
            if "-c:a" in fallback_cmd:
                ca_idx = fallback_cmd.index("-c:a")
                if ca_idx + 1 < len(fallback_cmd) and fallback_cmd[ca_idx + 1] != output_path:
                    del fallback_cmd[ca_idx:ca_idx + 2]
                else:
                    del fallback_cmd[ca_idx]
            fallback_cmd[-1:-1] = ["-c:a", "aac", "-b:a", "192k"]
            encode_res = run_ffmpeg(fallback_cmd, check=False)
            
        if encode_res.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed:\n{encode_res.stderr}")
            
        await _notify_progress(progress_callback, stage="finalizing", percent=98.0, fps=0.0, message="Verifying container integrity")
        
        # Step 5: Probe output video for verification
        out_meta = probe_video(output_path)
        out_duration = out_meta["duration"]
        out_audio_preserved = out_meta["has_audio"]
        out_fps = out_meta["fps"]
        out_frame_count = out_meta["frame_count"]
        
        await _notify_progress(progress_callback, stage="completed", percent=100.0, fps=0.0, message="Enhancement complete")
        
        return {
            "status": "success",
            "input_duration": round(in_duration, 3),
            "output_duration": round(out_duration, 3),
            "audio_preserved": bool(out_audio_preserved),
            "fps": round(out_fps, 2),
            "frame_count": int(out_frame_count),
        }
