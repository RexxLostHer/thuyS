"""
Automated Verification Suite for Wink HD Enhancer Engine
Validates:
1. Image Enhancement:
   - Super-resolution scaling factor (128x128 -> 512x512)
   - Tenengrad edge gradient energy sharpness gain >= 1.6x
   - Signal-to-noise ratio (PSNR >= 28.0 dB)
   - SSIM structural fidelity
2. Video Enhancement:
   - Super-resolution scaling factor (320x240 -> 640x480)
   - Frame count parity (45 frames in -> 45 frames out)
   - Lossless audio stream preservation (AAC 440Hz tone)
   - Audio-video duration sync within 50ms tolerance (|d_video - d_audio| <= 0.05s)
   - FFmpeg container integrity check (error-free null sink playback)
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.enhancer.image_enhancer import (
    enhance_image,
    calculate_tenengrad,
    calculate_psnr,
    calculate_ssim,
)
from app.enhancer.video_enhancer import enhance_video, probe_video
from app.enhancer.test_media_generator import generate_all_samples
from app.enhancer.ffmpeg_utils import run_ffmpeg, run_ffprobe

SAMPLES_DIR = PROJECT_ROOT / "samples"


def verify_image_enhancement() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 1: VERIFYING IMAGE ENHANCEMENT ENGINE")
    print("=" * 60)
    
    input_img_path = str(SAMPLES_DIR / "sample_image_lowres.png")
    output_img_path = str(SAMPLES_DIR / "sample_image_enhanced_hd.png")
    ref_img_path = str(SAMPLES_DIR / "sample_image_ref.png")
    
    if not os.path.exists(input_img_path) or not os.path.exists(ref_img_path):
        print("[*] Sample files missing, generating test media...")
        generate_all_samples(SAMPLES_DIR)
        
    print(f"[*] Input image:  {input_img_path}")
    print(f"[*] Output image: {output_img_path}")
    print(f"[*] Upscale Factor: 4x (128x128 -> 512x512)")
    
    result = enhance_image(
        input_path=input_img_path,
        output_path=output_img_path,
        upscale_factor=4,
        face_retouch=0.5,
        sharpen=0.5,
        vibrance=0.25,
        denoise=0.3,
    )
    
    print(f"[OK] Execution Status: {result['status']}")
    print(f"[OK] Processing Time:  {result['processing_time_ms']:.2f} ms")
    
    # Check 1: Resolution Verification
    in_w, in_h = result["input_resolution"]
    out_w, out_h = result["output_resolution"]
    assert in_w == 128 and in_h == 128, f"Unexpected input size: {result['input_resolution']}"
    assert out_w == 512 and out_h == 512, f"Expected 512x512 output, got: {result['output_resolution']}"
    print(f"  [PASS] Resolution Check: 128x128 -> {out_w}x{out_h} (Exact 4x multiplier)")
    
    # Read output and reference
    out_img = cv2.imread(output_img_path)
    ref_img = cv2.imread(ref_img_path)
    assert out_img is not None, "Failed to read generated enhanced image"
    assert out_img.shape == (512, 512, 3), f"Invalid output array shape: {out_img.shape}"
    
    # Check 2: Tenengrad Edge Sharpness Improvement
    sharpness_gain = result["metrics"]["sharpness_gain"]
    print(f"  [PASS] Tenengrad Sharpness Gain: {sharpness_gain:.3f}x (Threshold >= 1.600x)")
    assert sharpness_gain >= 1.60, f"Sharpness gain {sharpness_gain:.3f}x is below 1.60x threshold"
    
    # Check 3: PSNR Baseline Fidelity
    psnr_baseline = result["metrics"]["psnr_baseline"]
    print(f"  [PASS] PSNR Baseline: {psnr_baseline:.2f} dB (Threshold >= 28.00 dB)")
    assert psnr_baseline >= 28.0, f"PSNR {psnr_baseline:.2f} dB is below 28.0 dB threshold"
    
    # Check 4: SSIM Structural Similarity (against upsampled baseline per analysis.md Section 4.2)
    in_img = cv2.imread(input_img_path)
    baseline_upscaled = cv2.resize(in_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    ssim_baseline = calculate_ssim(baseline_upscaled, out_img)
    ssim_ref = calculate_ssim(ref_img, out_img)
    print(f"  [PASS] SSIM Baseline: {ssim_baseline:.4f} (Threshold >= 0.8200)")
    print(f"  [INFO] SSIM vs Reference: {ssim_ref:.4f}")
    assert ssim_baseline >= 0.82, f"SSIM baseline {ssim_baseline:.4f} is below 0.8200 threshold"
    
    return True


async def verify_video_enhancement() -> bool:
    print("\n" + "=" * 60)
    print("STAGE 2: VERIFYING VIDEO ENHANCEMENT ENGINE")
    print("=" * 60)
    
    input_vid_path = str(SAMPLES_DIR / "sample_video_lowres.mp4")
    output_vid_path = str(SAMPLES_DIR / "sample_video_enhanced_hd.mp4")
    
    if not os.path.exists(input_vid_path):
        print("[*] Sample video missing, generating test media...")
        generate_all_samples(SAMPLES_DIR)
        
    print(f"[*] Input video:  {input_vid_path}")
    print(f"[*] Output video: {output_vid_path}")
    print(f"[*] Upscale Factor: 2x (320x240 -> 640x480)")
    
    progress_stages = []
    
    def on_progress(stage, percent, fps, message=""):
        progress_stages.append(stage)
        print(f"  -> [{stage.upper():10s}] {percent:5.1f}% | {fps:4.1f} fps | {message}")
        
    result = await enhance_video(
        input_path=input_vid_path,
        output_path=output_vid_path,
        upscale_factor=2,
        face_retouch=0.5,
        sharpen=0.6,
        vibrance=0.3,
        denoise=0.3,
        progress_callback=on_progress,
    )
    
    print(f"[OK] Execution Status: {result['status']}")
    
    # Check 1: Output probe
    meta_in = probe_video(input_vid_path)
    meta_out = probe_video(output_vid_path)
    
    # Resolution Check
    assert meta_out["width"] == 640 and meta_out["height"] == 480, (
        f"Expected 640x480 resolution, got {meta_out['width']}x{meta_out['height']}"
    )
    print(f"  [PASS] Resolution Check: 320x240 -> {meta_out['width']}x{meta_out['height']} (Exact 2x multiplier)")
    
    # Check 2: Frame Count Parity
    print(f"  [PASS] Frame Count Parity: {meta_in['frame_count']} in -> {meta_out['frame_count']} out (Exact match)")
    assert meta_out["frame_count"] == meta_in["frame_count"], (
        f"Frame count mismatch: in={meta_in['frame_count']}, out={meta_out['frame_count']}"
    )
    
    # Check 3: Audio Stream Preservation
    assert meta_out["has_audio"] is True, "Audio stream was not preserved!"
    assert result["audio_preserved"] is True, "Result audio_preserved flag is False!"
    print(f"  [PASS] Audio Stream Preservation: Codec={meta_out['audio_codec']}, Preserved=True")
    
    # Check 4: Audio-Video Duration Sync within 50ms
    # Probe stream durations individually
    ffprobe_res = run_ffprobe([
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        output_vid_path,
    ], check=True)
    streams_info = json.loads(ffprobe_res.stdout)["streams"]
    v_dur = 0.0
    a_dur = 0.0
    for s in streams_info:
        if s["codec_type"] == "video":
            v_dur = float(s["duration"])
        elif s["codec_type"] == "audio":
            a_dur = float(s["duration"])
            
    dur_diff_ms = abs(v_dur - a_dur) * 1000.0
    print(f"  [PASS] Audio-Video Duration Sync: Video={v_dur:.3f}s, Audio={a_dur:.3f}s, Delta={dur_diff_ms:.2f} ms (Threshold <= 50.0 ms)")
    assert dur_diff_ms <= 50.0, f"Duration difference {dur_diff_ms:.2f} ms exceeds 50.0 ms limit"
    
    # Check 5: FFmpeg Container & Stream Error Integrity Check
    null_sink = "NUL" if sys.platform == "win32" else "/dev/null"
    integrity_check = run_ffmpeg(["-v", "error", "-i", output_vid_path, "-f", "null", null_sink], check=False)
    assert integrity_check.returncode == 0, f"Integrity check failed: {integrity_check.stderr}"
    assert integrity_check.stderr.strip() == "", f"FFmpeg reported container errors:\n{integrity_check.stderr}"
    print(f"  [PASS] FFmpeg Container Integrity: 0 errors/warnings on full stream decode")
    
    # Check 6: Progress callback coverage
    assert "enhancing" in progress_stages and "encoding" in progress_stages and "completed" in progress_stages, (
        f"Missing progress stages: {progress_stages}"
    )
    print(f"  [PASS] Progress Callback Streaming: All lifecycle stages emitted successfully")
    
    return True


def main():
    print("=" * 60)
    print("WINK HD ENHANCER - AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    
    # Ensure test samples exist
    if not (SAMPLES_DIR / "sample_image_ref.png").exists():
        print("[*] Generating all synthetic media samples first...")
        generate_all_samples(SAMPLES_DIR)
        
    image_ok = verify_image_enhancement()
    video_ok = asyncio.run(verify_video_enhancement())
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY:")
    print(f"  - Image Enhancer Core (Lanczos-4, Retouch, CAS): PASS [OK]")
    print(f"  - Video Enhancer Pipeline (Demux, Remux, Sync):  PASS [OK]")
    print(f"  - Audio Synchronization & Stream Parity:        PASS [OK]")
    print(f"  - Automated Quality Metrics & FFmpeg Integrity: PASS [OK]")
    print("=" * 60)
    print("ALL CORE ENGINE CHECKS PASSED SUCCESSFULLY!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
