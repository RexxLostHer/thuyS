"""
Wink-Style Video Enhancement Engine (High-Performance Streaming Pipe)
Optimized for zero-disk-I/O frame streaming, low RAM consumption, faststart MP4 headers, and sub-millisecond audio sync.
"""

import os
import sys
import time
import json
import asyncio
import inspect
import tempfile
import subprocess
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


def ensure_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Ensure the frame array is strictly uint8 in [0, 255] BGR to prevent blank/black video outputs.
    Adheres to PRD Remediasi Section 5.
    """
    if frame is None:
        return frame
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size > 0 else 1.0
        if max_val <= 1.05:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


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
    Process an individual video frame through the enhancement pipeline.
    Optimized: Denoise and portrait retouch are performed before upscale to minimize computation.
    """
    current = frame
    
    # 1. Denoise (on original resolution)
    if denoise > 0.0:
        current = denoise_image(current, strength=denoise)
    
    # 2. Portrait retouching (skin smoothing + feature highlights)
    if face_retouch > 0.0:
        current = apply_portrait_retouch(current, strength=face_retouch)
    
    # 3. Super-Resolution upscaling
    if upscale_factor > 1:
        current = apply_upscale(current, upscale_factor=upscale_factor)
    
    # 4. Vibrance & tone
    if vibrance > 0.0:
        current = apply_vibrance_and_tone(current, strength=vibrance)
    
    # 5. Adaptive sharpening
    if sharpen > 0.0:
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
    High-Performance Video Enhancement Function using Direct FFmpeg Stdin Pipe.
    Eliminates all temporary image disk writes, streaming raw BGR frames directly
    to FFmpeg with faststart web streaming headers and zero-deadlock pipe management.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Step 1: Probe video properties
    await _notify_progress(progress_callback, stage="probing", percent=0.0, fps=0.0, message="Probing video stream")
    meta = probe_video(input_path)
    in_duration = meta["duration"]
    in_fps = meta["fps"] if meta["fps"] > 0 else 30.0
    in_frame_count = meta["frame_count"]
    has_audio = meta["has_audio"]
    
    # Compute target dimensions (must be even for libx264 yuv420p)
    raw_target_w = meta["width"] * upscale_factor
    raw_target_h = meta["height"] * upscale_factor
    
    # Clamp to max standard 4K (3840x2160) for universal browser and hardware decoder compatibility
    if raw_target_w > 3840 or raw_target_h > 2160:
        scale_fit = min(3840.0 / raw_target_w, 2160.0 / raw_target_h)
        raw_target_w = int(round(raw_target_w * scale_fit))
        raw_target_h = int(round(raw_target_h * scale_fit))
        
    target_w = raw_target_w + (raw_target_w % 2)
    target_h = raw_target_h + (raw_target_h % 2)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_audio_path = temp_dir_path / "audio_track.aac"
        
        # Step 2: Extract and transcode audio to universal AAC-LC for web playback
        audio_extracted = False
        if has_audio:
            await _notify_progress(progress_callback, stage="demuxing", percent=5.0, fps=0.0, message="Transcoding audio to universal AAC-LC")
            demux_aac = run_ffmpeg(
                ["-y", "-i", input_path, "-vn", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(temp_audio_path)],
                check=False
            )
            if demux_aac.returncode == 0 and temp_audio_path.exists() and temp_audio_path.stat().st_size > 0:
                audio_extracted = True
                    
        # Step 3: Spawn FFmpeg encoder process with stdin rawvideo pipe & faststart MP4 header
        await _notify_progress(progress_callback, stage="enhancing", percent=8.0, fps=0.0, message="Starting video stream pipe")
        
        ffmpeg_bin = get_ffmpeg_path()
        ffmpeg_encode_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{target_w}x{target_h}",
            "-pix_fmt", "bgr24",
            "-framerate", str(in_fps),
            "-i", "-",  # pipe input
        ]
        
        if audio_extracted:
            ffmpeg_encode_cmd.extend([
                "-i", str(temp_audio_path),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",  # Move moov atom to start for instant browser playback
                "-c:a", "copy",
                output_path,
            ])
        else:
            ffmpeg_encode_cmd.extend([
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",  # Move moov atom to start for instant browser playback
                output_path,
            ])
            
        # Use temporary file for stderr to completely avoid OS pipe buffer deadlock
        temp_err_file = tempfile.TemporaryFile()
        proc = subprocess.Popen(
            ffmpeg_encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=temp_err_file,
        )
        
        cap = cv2.VideoCapture(input_path)
        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = actual_frame_count if actual_frame_count > 0 else (in_frame_count if in_frame_count > 0 else 1)
        
        # Parallel frame enhancement in micro-batches
        cpu_cores = os.cpu_count() or 4
        worker_threads = min(cpu_cores, 8)
        batch_size = max(4, worker_threads * 2)
        
        def process_frame(frame: np.ndarray) -> bytes:
            enhanced = enhance_single_frame(
                frame=frame,
                upscale_factor=upscale_factor,
                face_retouch=face_retouch,
                sharpen=sharpen,
                vibrance=vibrance,
                denoise=denoise,
            )
            enhanced = ensure_uint8_bgr(enhanced)
            # Ensure dimensions match target_w, target_h
            h, w = enhanced.shape[:2]
            if h != target_h or w != target_w:
                pad_h = max(0, target_h - h)
                pad_w = max(0, target_w - w)
                if pad_h > 0 or pad_w > 0:
                    enhanced = cv2.copyMakeBorder(
                        enhanced, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
                    )
                if enhanced.shape[0] != target_h or enhanced.shape[1] != target_w:
                    enhanced = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return enhanced.tobytes()

        processed_count = 0
        start_processing_time = time.perf_counter()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads) as executor:
                while cap.isOpened():
                    batch_frames = []
                    for _ in range(batch_size):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        batch_frames.append(frame)
                        
                    if not batch_frames:
                        break
                        
                    # Process current batch in parallel while preserving exact order
                    futures = [executor.submit(process_frame, f) for f in batch_frames]
                    for fut in futures:
                        raw_bytes = fut.result()
                        proc.stdin.write(raw_bytes)
                        processed_count += 1
                        
                    # Update progress and allow event loop to breathe
                    elapsed = time.perf_counter() - start_processing_time
                    current_fps = processed_count / max(elapsed, 0.001)
                    pct = 8.0 + (processed_count / max(total_frames, processed_count)) * 82.0
                    
                    await _notify_progress(
                        progress_callback,
                        stage="enhancing",
                        percent=min(92.0, round(pct, 1)),
                        fps=round(current_fps, 1),
                        message=f"Enhanced {processed_count} frames ({current_fps:.1f} fps)",
                    )
                    await asyncio.sleep(0.001)  # Flush SSE events without blocking
                    
        finally:
            cap.release()
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                    
        # Step 4: Await FFmpeg finalization
        await _notify_progress(progress_callback, stage="encoding", percent=93.0, fps=0.0, message="Finalizing video container")
        proc.wait()
        
        if proc.returncode != 0:
            temp_err_file.seek(0)
            err_text = temp_err_file.read().decode(errors="replace")
            temp_err_file.close()
            raise RuntimeError(f"FFmpeg streaming encoding failed (exit code {proc.returncode}):\n{err_text}")
            
        temp_err_file.close()
            
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
            "frame_count": int(out_frame_count if out_frame_count > 0 else processed_count),
        }
