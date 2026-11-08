"""
Synthetic Test Media Generator for Wink HD Enhancer
Programmatically creates high-fidelity reference media and systematically degraded low-resolution inputs:
- 512x512 structured reference image (face portrait, geometric patterns, typography, gradients)
- 128x128 degraded low-resolution image (downsampled, Gaussian noise, blur, JPEG compression)
- 320x240 15fps reference video (moving elements, timecodes, face target, 440Hz synchronized audio tone)
- 320x240 15fps degraded low-resolution video with identical synchronized audio
"""

import os
import math
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

from app.enhancer.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg

DEFAULT_SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


def draw_stylized_face(canvas: np.ndarray, center: Tuple[int, int], scale: float = 1.0):
    """
    Draw a stylized face portrait on canvas with natural skin tones, eyes, eyebrows, and lips.
    Ensures skin color falls strictly inside YCbCr/HSV detection boundaries for retouch evaluation.
    """
    cx, cy = center
    
    # Face dimensions
    face_w = int(120 * scale)
    face_h = int(150 * scale)
    
    # 1. Hair background (dark brown/black)
    hair_color = (30, 25, 20)
    cv2.ellipse(canvas, (cx, cy - int(10 * scale)), (int(face_w * 1.15), int(face_h * 1.15)), 0, 0, 360, hair_color, -1)
    
    # 2. Skin oval: BGR (175, 200, 240) -> RGB (240, 200, 175)
    # Cb ~ 96, Cr ~ 152 (well within skin range Cb: 77-127, Cr: 133-173)
    skin_color = (175, 200, 240)
    cv2.ellipse(canvas, (cx, cy), (face_w, face_h), 0, 0, 360, skin_color, -1)
    
    # Subtle skin texture (fine pore simulation)
    rng = np.random.RandomState(42)
    face_roi_y1 = max(0, cy - face_h)
    face_roi_y2 = min(canvas.shape[0], cy + face_h)
    face_roi_x1 = max(0, cx - face_w)
    face_roi_x2 = min(canvas.shape[1], cx + face_w)
    
    noise = rng.normal(0, 3, (face_roi_y2 - face_roi_y1, face_roi_x2 - face_roi_x1, 3))
    canvas_roi = canvas[face_roi_y1:face_roi_y2, face_roi_x1:face_roi_x2].astype(np.float32)
    canvas[face_roi_y1:face_roi_y2, face_roi_x1:face_roi_x2] = np.clip(canvas_roi + noise, 0, 255).astype(np.uint8)
    
    # 3. Eyebrows
    brow_color = (40, 35, 30)
    brow_offset_x = int(45 * scale)
    brow_y = cy - int(45 * scale)
    t_brow = max(1, int(4 * scale))
    cv2.line(canvas, (cx - brow_offset_x - int(25 * scale), brow_y + int(3 * scale)), (cx - brow_offset_x + int(25 * scale), brow_y - int(3 * scale)), brow_color, t_brow, cv2.LINE_AA)
    cv2.line(canvas, (cx + brow_offset_x - int(25 * scale), brow_y - int(3 * scale)), (cx + brow_offset_x + int(25 * scale), brow_y + int(3 * scale)), brow_color, t_brow, cv2.LINE_AA)
    
    # 4. Eyes (whites, iris, pupil, highlight)
    eye_offset_x = int(45 * scale)
    eye_y = cy - int(25 * scale)
    eye_rx = max(3, int(22 * scale))
    eye_ry = max(2, int(14 * scale))
    t_outline = max(1, int(1.5 * scale))
    r_iris = max(2, int(10 * scale))
    r_pupil = max(1, int(5 * scale))
    r_shine = max(1, int(2.5 * scale))
    
    for ex in [cx - eye_offset_x, cx + eye_offset_x]:
        # Eye white
        cv2.ellipse(canvas, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (245, 245, 250), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (120, 100, 90), t_outline, cv2.LINE_AA)
        # Iris (hazel brown/green)
        cv2.circle(canvas, (ex, eye_y), r_iris, (60, 110, 80), -1, cv2.LINE_AA)
        # Pupil (black)
        cv2.circle(canvas, (ex, eye_y), r_pupil, (15, 15, 15), -1, cv2.LINE_AA)
        # Specular light reflection
        cv2.circle(canvas, (ex - max(1, int(3 * scale)), eye_y - max(1, int(3 * scale))), r_shine, (255, 255, 255), -1, cv2.LINE_AA)
        
    # 5. Nose contour
    nose_color = (150, 175, 215)
    nose_pt1 = (cx, cy - int(10 * scale))
    nose_pt2 = (cx - max(1, int(7 * scale)), cy + int(25 * scale))
    nose_pt3 = (cx + max(1, int(7 * scale)), cy + int(25 * scale))
    t_nose = max(1, int(2 * scale))
    cv2.line(canvas, nose_pt1, nose_pt2, nose_color, t_nose, cv2.LINE_AA)
    cv2.line(canvas, nose_pt2, nose_pt3, nose_color, t_nose, cv2.LINE_AA)
    
    # 6. Lips: BGR (80, 85, 215) -> rosy pink/red
    lip_color = (80, 85, 215)
    lip_y = cy + int(60 * scale)
    lip_rx = max(4, int(36 * scale))
    lip_ry = max(2, int(16 * scale))
    cv2.ellipse(canvas, (cx, lip_y), (lip_rx, lip_ry), 0, 0, 360, lip_color, -1, cv2.LINE_AA)
    cv2.line(canvas, (cx - max(2, int(34 * scale)), lip_y), (cx + max(2, int(34 * scale)), lip_y), (50, 50, 160), max(1, int(2 * scale)), cv2.LINE_AA)
    # Lip gloss shine
    cv2.ellipse(canvas, (cx, lip_y + max(1, int(4 * scale))), (max(2, int(16 * scale)), max(1, int(4 * scale))), 0, 0, 360, (150, 150, 245), -1, cv2.LINE_AA)


def generate_reference_image(output_path: str, width: int = 512, height: int = 512) -> str:
    """
    Generate high-resolution structured reference image.
    Contains:
    - Gradient backdrop
    - Stylized face portrait
    - Concentric circles and high-frequency Siemens-star textures
    - Fine typography and resolution markers
    - Color calibration wedges
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Obsidian/violet gradient background
    for y in range(height):
        ratio = y / height
        b = int(24 + 40 * ratio)
        g = int(14 + 18 * ratio)
        r = int(32 + 55 * ratio)
        canvas[y, :] = (b, g, r)
        
    # 2. Geometric grid / texture pattern in corners
    # Concentric circles in top-left
    cx_tl, cy_tl = int(width * 0.15), int(height * 0.18)
    for rad in range(5, 65, 6):
        cv2.circle(canvas, (cx_tl, cy_tl), rad, (180, 180, 200), 1, cv2.LINE_AA)
        
    # High-frequency checkerboard in top-right
    cb_size = 6
    cb_x1, cb_y1 = int(width * 0.72), int(height * 0.08)
    for row in range(12):
        for col in range(12):
            if (row + col) % 2 == 0:
                cv2.rectangle(
                    canvas,
                    (cb_x1 + col * cb_size, cb_y1 + row * cb_size),
                    (cb_x1 + (col + 1) * cb_size, cb_y1 + (row + 1) * cb_size),
                    (230, 230, 230),
                    -1
                )
                
    # 3. Stylized Portrait in Center
    draw_stylized_face(canvas, center=(width // 2, int(height * 0.48)), scale=0.88)
    
    # 4. Color bar wedges at bottom
    bar_y = int(height * 0.82)
    bar_h = int(height * 0.06)
    colors = [
        (40, 40, 240),   # Red
        (40, 220, 40),   # Green
        (240, 40, 40),   # Blue
        (40, 230, 240),  # Yellow
        (220, 40, 220),  # Magenta
        (240, 220, 40),  # Cyan
        (255, 255, 255), # White
        (128, 128, 128), # 50% Gray
    ]
    bar_w = width // len(colors)
    for idx, c in enumerate(colors):
        cv2.rectangle(canvas, (idx * bar_w, bar_y), ((idx + 1) * bar_w, bar_y + bar_h), c, -1)
        
    # 5. Crisp typography
    cv2.putText(canvas, "WINK HD ENHANCER - 4K ULTRA REFERENCE", (int(width * 0.08), int(height * 0.94)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CALIBRATION CHART 512x512", (int(width * 0.22), int(height * 0.98)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 255), 1, cv2.LINE_AA)
    
    # Header title
    cv2.putText(canvas, "PORTRAIT & DETAIL TEST SUITE", (int(width * 0.18), int(height * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 240), 1, cv2.LINE_AA)
    
    cv2.imwrite(output_path, canvas)
    return output_path


def degrade_image(
    ref_image_path: str,
    output_path: str,
    target_size: Tuple[int, int] = (128, 128),
    noise_sigma: float = 10.0,
    blur_ksize: int = 5,
    jpeg_quality: int = 35,
) -> str:
    """
    Generate degraded low-resolution image from reference:
    - Downsample to target_size (e.g. 128x128)
    - Add Gaussian noise
    - Apply Gaussian blur
    - Apply JPEG lossy compression artifacts
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ref_img = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {ref_image_path}")
        
    # 1. Downscale
    low_res = cv2.resize(ref_img, target_size, interpolation=cv2.INTER_AREA)
    
    # 2. Gaussian blur
    if blur_ksize > 1:
        low_res = cv2.GaussianBlur(low_res, (blur_ksize, blur_ksize), 1.2)
        
    # 3. Add Gaussian noise
    if noise_sigma > 0:
        rng = np.random.RandomState(1337)
        noise = rng.normal(0, noise_sigma, low_res.shape).astype(np.float32)
        low_res = np.clip(low_res.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    # 4. JPEG compression artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encimg = cv2.imencode(".jpg", low_res, encode_param)
    degraded = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    
    cv2.imwrite(output_path, degraded)
    return output_path


def generate_reference_video(
    output_path: str,
    width: int = 320,
    height: int = 240,
    fps: int = 15,
    duration_sec: float = 3.0,
) -> str:
    """
    Generate reference test video (320x240, 15fps, 3 seconds) with synchronized 440Hz reference audio tone.
    Uses FFmpeg to synthesize audio and encode clean H.264 + AAC container.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    total_frames = int(fps * duration_sec)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        raw_video_path = temp_dir_path / "raw_video.mp4"
        
        # 1. Render frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_video_path), fourcc, fps, (width, height))
        
        try:
            for frame_idx in range(total_frames):
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                time_sec = frame_idx / float(fps)
                
                # Dynamic dark background gradient
                for y in range(height):
                    val = int(25 + 30 * (y / height))
                    canvas[y, :] = (val, int(val * 0.7), int(val * 1.2))
                    
                # Animated face portrait moving smoothly in a figure-8
                face_cx = int(width * 0.5 + width * 0.22 * math.sin(time_sec * 2.0 * math.pi / duration_sec))
                face_cy = int(height * 0.45 + height * 0.12 * math.cos(time_sec * 4.0 * math.pi / duration_sec))
                draw_stylized_face(canvas, center=(face_cx, face_cy), scale=0.45)
                
                # Animated bouncing indicator
                ball_x = int(width * 0.1 + (width * 0.8) * (frame_idx / total_frames))
                ball_y = int(height * 0.82 + 15 * math.sin(frame_idx * 0.6))
                cv2.circle(canvas, (ball_x, ball_y), 8, (0, 220, 255), -1, cv2.LINE_AA)
                
                # Timecode and frame counter
                timecode_str = f"FRAME: {frame_idx:02d}/{total_frames} | T: {time_sec:.2f}s"
                cv2.putText(canvas, timecode_str, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "WINK HD TEST VIDEO (REF)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
                
                writer.write(canvas)
        finally:
            writer.release()
        
        # 2. Multiplex synchronized 440Hz reference audio tone using FFmpeg
        # Generates a pure 440Hz sine wave tone matching exact video duration
        cmd = [
            "-y",
            "-i", str(raw_video_path),
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration_sec}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            output_path,
        ]
        res = run_ffmpeg(cmd, check=True)
        
    return output_path


def generate_degraded_video(
    ref_video_path: str,
    output_path: str,
    fps: int = 15,
) -> str:
    """
    Generate degraded low-resolution test video from reference:
    - Downsample & blur video frames, add compression noise
    - Preserve exact synchronized 440Hz audio stream lossless via FFmpeg (-c:a copy)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_audio_path = temp_dir_path / "audio.aac"
        temp_degraded_frames_path = temp_dir_path / "degraded_video.mp4"
        
        # 1. Demux audio lossless
        run_ffmpeg(["-y", "-i", ref_video_path, "-vn", "-c:a", "copy", str(temp_audio_path)], check=True)
        
        # 2. Extract and degrade frames
        cap = cv2.VideoCapture(ref_video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_degraded_frames_path), fourcc, fps, (w, h))
        
        rng = np.random.RandomState(777)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                # Downsample to half and upsample with nearest neighbor to simulate severe pixelation
                down = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                
                # Blur
                blurred = cv2.GaussianBlur(down, (5, 5), 1.5)
                
                # Noise
                noise = rng.normal(0, 12, blurred.shape).astype(np.float32)
                noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
                # Resize back to original video dimensions
                degraded_frame = cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)
                writer.write(degraded_frame)
        finally:
            cap.release()
            writer.release()
        
        # 3. Remux degraded video with original lossless audio track
        cmd = [
            "-y",
            "-i", str(temp_degraded_frames_path),
            "-i", str(temp_audio_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            output_path,
        ]
        run_ffmpeg(cmd, check=True)
        
    return output_path


def generate_all_samples(samples_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Generate all sample test media:
    - sample_image_ref.png (512x512)
    - sample_image_lowres.png (128x128 degraded)
    - sample_video_ref.mp4 (320x240, 15fps, 3s, 440Hz audio)
    - sample_video_lowres.mp4 (320x240 degraded, 440Hz audio)
    """
    if samples_dir is None:
        samples_dir = DEFAULT_SAMPLES_DIR
        
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    img_ref = str(samples_dir / "sample_image_ref.png")
    img_lowres = str(samples_dir / "sample_image_lowres.png")
    vid_ref = str(samples_dir / "sample_video_ref.mp4")
    vid_lowres = str(samples_dir / "sample_video_lowres.mp4")
    
    print("[*] Generating reference image (512x512)...")
    generate_reference_image(img_ref, width=512, height=512)
    
    print("[*] Generating degraded low-res image (128x128)...")
    degrade_image(img_ref, img_lowres, target_size=(128, 128))
    
    print("[*] Generating reference video with 440Hz audio (320x240, 15fps, 3s)...")
    generate_reference_video(vid_ref, width=320, height=240, fps=15, duration_sec=3.0)
    
    print("[*] Generating degraded low-res video with audio...")
    generate_degraded_video(vid_ref, vid_lowres, fps=15)
    
    results = {
        "sample_image_ref": img_ref,
        "sample_image_lowres": img_lowres,
        "sample_video_ref": vid_ref,
        "sample_video_lowres": vid_lowres,
    }
    
    print("[OK] All synthetic test samples successfully generated:")
    for k, v in results.items():
        size_kb = os.path.getsize(v) / 1024.0
        print(f"  - {k}: {v} ({size_kb:.1f} KB)")
        
    return results


if __name__ == "__main__":
    generate_all_samples()
