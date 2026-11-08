"""
Wink-Style Video Enhancement Engine (High-Performance Non-Blocking Streaming Pipe)
Refactored to eliminate FastAPI event-loop blocking and Windows OS pipe deadlocks.
"""

import os
import sys
import time
import json
import asyncio
import inspect
import tempfile
import threading
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
    Ensure the frame array is strictly contiguous uint8 in [0, 255] BGR.
    """
    if frame is None:
        return frame
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size > 0 else 1.0
        if max_val <= 1.05:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def probe_video(video_path: str) -> Dict[str, Any]:
    """
    Probe video stream metadata using ffprobe JSON output.
    Returns union of PRD keys and legacy test keys.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffprobe_bin = get_ffprobe_path()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    meta = json.loads(res.stdout)

    video_stream = None
    audio_stream = None
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video" and not video_stream:
            video_stream = s
        elif s.get("codec_type") == "audio" and not audio_stream:
            audio_stream = s

    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    fps_raw = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate", "30/1")
    if "/" in fps_raw:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_raw)
    if fps <= 0 or np.isnan(fps) or fps > 240:
        fps = 30.0

    width = int(video_stream.get("width", 1920))
    height = int(video_stream.get("height", 1080))
    duration = 0.0
    if "duration" in video_stream:
        try:
            duration = float(video_stream["duration"])
        except (ValueError, TypeError):
            pass
    if duration <= 0.0 and "duration" in meta.get("format", {}):
        try:
            duration = float(meta["format"]["duration"])
        except (ValueError, TypeError):
            pass

    total_frames = int(video_stream.get("nb_frames", 0) or 0)
    if total_frames == 0 and duration > 0 and fps > 0:
        total_frames = max(1, int(round(duration * fps)))

    has_audio = audio_stream is not None
    audio_codec = audio_stream.get("codec_name") if audio_stream else None

    return {
        "duration": duration,
        "input_duration": round(duration, 3),
        "output_duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "frame_count": total_frames,
        "has_audio": has_audio,
        "audio_codec": audio_codec,
        "codec": video_stream.get("codec_name", "h264"),
        "pix_fmt": video_stream.get("pix_fmt", "yuv420p"),
    }


def process_video_frame(
    frame: np.ndarray,
    upscale_factor: int = 2,
    face_retouch: float = 0.5,
    sharpen: float = 0.5,
    vibrance: float = 0.3,
    denoise: float = 0.3,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Process a single video frame array through OpenCV / NumPy filters.
    """
    out = ensure_uint8_bgr(frame)

    if denoise > 0.0:
        out = denoise_image(out, strength=denoise)

    if face_retouch > 0.0:
        out = apply_portrait_retouch(out, strength=face_retouch)

    if upscale_factor > 1:
        out = apply_upscale(out, scale_factor=upscale_factor)

    if target_size is not None and (out.shape[1] != target_size[0] or out.shape[0] != target_size[1]):
        out = cv2.resize(out, target_size, interpolation=cv2.INTER_LANCZOS4)

    if vibrance > 0.0:
        out = apply_vibrance_and_tone(out, vibrance=vibrance)

    if sharpen > 0.0:
        out = apply_cas_sharpening(out, amount=sharpen)

    return ensure_uint8_bgr(out)


# Alias for backwards compatibility
enhance_single_frame = process_video_frame


def _drain_pipe_background(pipe, sink_list: list):
    """
    Continuously drains a subprocess stderr pipe in a daemon thread so OS pipe buffers never deadlock.
    """
    try:
        for line in iter(pipe.readline, b""):
            sink_list.append(line.decode("utf-8", errors="ignore"))
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _enhance_video_sync(
    input_path: str,
    output_path: str,
    upscale_factor: int,
    face_retouch: float,
    sharpen: float,
    vibrance: float,
    denoise: float,
    emit_progress: Callable[[str, float, float, str], None],
) -> Dict[str, Any]:
    """
    Synchronous worker executed in a background thread via asyncio.to_thread.
    Keeps the FastAPI event loop 100% free so SSE progress streams in real time without freezing.
    """
    start_time = time.time()
    emit_progress("probing", 2.0, 0.0, "Reading video stream metadata...")
    meta = probe_video(input_path)

    in_w, in_h = meta["width"], meta["height"]
    fps = meta["fps"]
    total_frames = meta["total_frames"]
    has_audio = meta["has_audio"]
    in_duration = meta["duration"]

    # libx264 yuv420p strictly requires even width and height
    raw_target_w = int(in_w * upscale_factor)
    raw_target_h = int(in_h * upscale_factor)
    if raw_target_w > 3840 or raw_target_h > 2160:
        scale_fit = min(3840.0 / raw_target_w, 2160.0 / raw_target_h)
        raw_target_w = int(round(raw_target_w * scale_fit))
        raw_target_h = int(round(raw_target_h * scale_fit))

    out_w = raw_target_w + (raw_target_w % 2)
    out_h = raw_target_h + (raw_target_h % 2)

    ffmpeg_bin = get_ffmpeg_path()
    raw_frame_size = in_w * in_h * 3

    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(temp_video_fd)

    decoder_cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-nostdin",
        "-i", str(input_path),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    encoder_cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temp_video_path),
    ]

    decoder_proc = subprocess.Popen(
        decoder_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=raw_frame_size * 2,
    )
    encoder_proc = subprocess.Popen(
        encoder_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=out_w * out_h * 3 * 2,
    )

    decoder_err_lines: list = []
    encoder_err_lines: list = []
    t_dec_err = threading.Thread(target=_drain_pipe_background, args=(decoder_proc.stderr, decoder_err_lines), daemon=True)
    t_enc_err = threading.Thread(target=_drain_pipe_background, args=(encoder_proc.stderr, encoder_err_lines), daemon=True)
    t_dec_err.start()
    t_enc_err.start()

    max_workers = min(os.cpu_count() or 4, 6)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    frame_count = 0
    written_count = 0
    last_callback_time = time.time()
    emit_progress("enhancing", 5.0, 0.0, f"Enhancing frames (0/{total_frames or '?'})...")

    try:
        futures = []

        while True:
            raw_bytes = decoder_proc.stdout.read(raw_frame_size) if decoder_proc.stdout else b""
            if not raw_bytes or len(raw_bytes) < raw_frame_size:
                break

            frame_count += 1
            # .copy() ensures the array is writeable for OpenCV filters
            frame_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((in_h, in_w, 3)).copy()

            fut = executor.submit(
                process_video_frame,
                frame_arr,
                upscale_factor,
                face_retouch,
                sharpen,
                vibrance,
                denoise,
                (out_w, out_h),
            )
            futures.append(fut)

            if len(futures) >= max_workers * 2:
                processed_frame = futures.pop(0).result()
                if encoder_proc.stdin:
                    encoder_proc.stdin.write(processed_frame.tobytes())
                written_count += 1

                now = time.time()
                if now - last_callback_time >= 0.2:
                    last_callback_time = now
                    elapsed = max(now - start_time, 1e-3)
                    curr_fps = written_count / elapsed
                    pct = (written_count / total_frames * 90.0) if total_frames > 0 else 50.0
                    pct = min(92.0, max(5.0, pct))
                    emit_progress(
                        "enhancing",
                        pct,
                        curr_fps,
                        f"Enhanced frame {written_count}/{total_frames or frame_count}",
                    )

        # Drain remaining futures in order
        for fut in futures:
            processed_frame = fut.result()
            if encoder_proc.stdin:
                encoder_proc.stdin.write(processed_frame.tobytes())
            written_count += 1

            now = time.time()
            if now - last_callback_time >= 0.2 or written_count == frame_count:
                last_callback_time = now
                elapsed = max(now - start_time, 1e-3)
                curr_fps = written_count / elapsed
                pct = (written_count / max(total_frames, frame_count, 1)) * 92.0
                pct = min(94.0, max(5.0, pct))
                emit_progress(
                    "enhancing",
                    pct,
                    curr_fps,
                    f"Enhanced frame {written_count}/{max(total_frames, frame_count)}",
                )

    finally:
        executor.shutdown(wait=True)
        try:
            if decoder_proc.stdout:
                decoder_proc.stdout.close()
        except Exception:
            pass
        try:
            if decoder_proc.poll() is None:
                decoder_proc.terminate()
            decoder_proc.wait(timeout=5)
        except Exception:
            try:
                decoder_proc.kill()
            except Exception:
                pass

        try:
            if encoder_proc.stdin:
                encoder_proc.stdin.close()
        except Exception:
            pass
        try:
            encoder_proc.wait(timeout=30)
        except Exception:
            try:
                encoder_proc.kill()
            except Exception:
                pass

        t_dec_err.join(timeout=2)
        t_enc_err.join(timeout=2)

    if written_count == 0:
        raise RuntimeError(f"No video frames were decoded. Decoder details: {''.join(decoder_err_lines)}")

    if encoder_proc.returncode not in (0, None):
        raise RuntimeError(f"FFmpeg video encoder failed (code {encoder_proc.returncode}): {''.join(encoder_err_lines)}")

    emit_progress("muxing", 95.0, 0.0, "Merging Audio Track & Finalizing Faststart MP4...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if has_audio:
        mux_cmd = [
            ffmpeg_bin,
            "-v", "error",
            "-nostdin",
            "-y",
            "-i", str(temp_video_path),
            "-i", str(input_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-movflags", "+faststart",
            str(output_path),
        ]
        res_mux = subprocess.run(
            mux_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res_mux.returncode != 0:
            mux_fallback = [
                ffmpeg_bin,
                "-v", "error",
                "-nostdin",
                "-y",
                "-i", str(temp_video_path),
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]
            subprocess.run(
                mux_fallback,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
    else:
        mux_cmd = [
            ffmpeg_bin,
            "-v", "error",
            "-nostdin",
            "-y",
            "-i", str(temp_video_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(
            mux_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    if os.path.exists(temp_video_path):
        try:
            os.remove(temp_video_path)
        except OSError:
            pass

    total_time_ms = float((time.time() - start_time) * 1000.0)

    try:
        out_meta = probe_video(output_path)
        out_dur = out_meta.get("duration", in_duration)
        out_audio = out_meta.get("has_audio", has_audio)
    except Exception:
        out_dur = in_duration
        out_audio = has_audio

    return {
        "status": "success",
        "input_resolution": [in_w, in_h],
        "output_resolution": [out_w, out_h],
        "fps": round(fps, 2),
        "total_frames": written_count,
        "frame_count": written_count,
        "duration": round(out_dur, 3),
        "input_duration": round(in_duration, 3),
        "output_duration": round(out_dur, 3),
        "processing_time_ms": round(total_time_ms, 2),
        "has_audio": bool(out_audio),
        "audio_preserved": bool(out_audio),
    }


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
    Asynchronous wrapper that offloads heavy synchronous FFmpeg/OpenCV processing to a background thread
    while forwarding real-time progress events safely back to the FastAPI asyncio event loop.
    Adapts safely to both 3-argument and 4-argument progress callbacks.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    loop = asyncio.get_running_loop()

    def thread_safe_emit(stage: str, percent: float, fps: float, msg: str):
        if not progress_callback:
            return
        if inspect.iscoroutinefunction(progress_callback):
            async def _coro():
                try:
                    await progress_callback(stage, percent, fps, msg)
                except TypeError:
                    await progress_callback(stage, percent, fps)
            asyncio.run_coroutine_threadsafe(_coro(), loop)
        else:
            def _fn():
                try:
                    progress_callback(stage, percent, fps, msg)
                except TypeError:
                    progress_callback(stage, percent, fps)
            loop.call_soon_threadsafe(_fn)

    result = await asyncio.to_thread(
        _enhance_video_sync,
        str(input_path),
        str(output_path),
        int(upscale_factor),
        float(face_retouch),
        float(sharpen),
        float(vibrance),
        float(denoise),
        thread_safe_emit,
    )
    # Drain any in-flight progress callbacks scheduled on event loop
    await asyncio.sleep(0)

    if progress_callback:
        msg = "Video Enhancement Completed Successfully!"
        if inspect.iscoroutinefunction(progress_callback):
            try:
                await progress_callback("completed", 100.0, result.get("fps", 0.0), msg)
            except TypeError:
                await progress_callback("completed", 100.0, result.get("fps", 0.0))
        else:
            try:
                progress_callback("completed", 100.0, result.get("fps", 0.0), msg)
            except TypeError:
                progress_callback("completed", 100.0, result.get("fps", 0.0))

    return result
