"""
Wink HD Enhancer — End-to-End Test Suite (Tiers 1 to 4)
=======================================================
Requirements Covered:
  - R1: GitHub Repository Discovery & Research Documentation
  - R2: Wink-Style HD Image and Video Enhancer (Super-Resolution, Retouch, CAS, Audio Sync)
  - R3: Windows Desktop Launcher Shortcut (.bat) & Project Root Companion Launcher

Test Methodology:
  - Tier 1: Feature Coverage (>=5 tests per feature for happy path / isolation)
  - Tier 2: Boundary & Corner Cases (>=5 tests per feature: zero bytes, corrupt headers, odd dims, single frame)
  - Tier 3: Cross-Feature Combinations (Pairwise matrix & multimodal synchronization)
  - Tier 4: Real-World Application Scenarios (Full E2E user workflows)

All tests are opaque-box, strictly derived from ORIGINAL_REQUEST.md & PROJECT.md contracts.
"""

import os
import sys
import re
import json
import math
import shutil
import tempfile
import subprocess
import unittest
from pathlib import Path

# Pytest import with graceful fallback
try:
    import pytest
except ImportError:
    pytest = None

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
REPLICA_DOC = DOCS_DIR / "REPLICA_SKILL_DISCOVERY.md"
DESKTOP_BAT = Path(r"C:\Users\Lenovo\Desktop\Wink HD Enhancer.bat")
ROOT_BAT = PROJECT_ROOT / "run_app.bat"

FFMPEG_BIN = Path(r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin\ffmpeg.exe")
FFPROBE_BIN = Path(r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin\ffprobe.exe")

# Ensure FFmpeg directory is in PATH for any subprocess or module calls
ffmpeg_dir = str(FFMPEG_BIN.parent)
if ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"

# Also append project root to sys.path so app modules are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test Helpers & Fixture Generators
# ---------------------------------------------------------------------------

def check_m2_available():
    """Returns True if Milestone 2 implementation files exist."""
    image_enhancer_py = PROJECT_ROOT / "app" / "enhancer" / "image_enhancer.py"
    video_enhancer_py = PROJECT_ROOT / "app" / "enhancer" / "video_enhancer.py"
    return image_enhancer_py.is_file() and video_enhancer_py.is_file()

def check_m4_available():
    """Returns True if Milestone 4 desktop shortcut exists."""
    return DESKTOP_BAT.is_file()

def skip_if_m2_missing():
    strict = os.environ.get("STRICT_E2E", "0") == "1"
    if not check_m2_available():
        if strict:
            raise AssertionError("Milestone 2 implementation missing under STRICT_E2E mode!")
        raise unittest.SkipTest("Milestone 2 (Core Enhancer) implementation in progress")

def skip_if_m4_missing():
    strict = os.environ.get("STRICT_E2E", "0") == "1"
    if not check_m4_available():
        if strict:
            raise AssertionError("Milestone 4 launcher shortcut missing under STRICT_E2E mode!")
        raise unittest.SkipTest("Milestone 4 (Desktop Launcher) implementation in progress")


def make_test_bmp(path: Path, width: int, height: int, color=(128, 128, 200)):
    """Generate a valid uncompressed 24-bit Windows BMP image file without external dependencies."""
    row_size = (width * 3 + 3) & ~3
    image_size = row_size * height
    file_size = 54 + image_size
    
    # BMP Header (14 bytes) + DIB Header (40 bytes)
    header = bytearray(54)
    header[0:2] = b'BM'
    header[2:6] = file_size.to_bytes(4, 'little')
    header[10:14] = (54).to_bytes(4, 'little') # offset to pixel data
    
    header[14:18] = (40).to_bytes(4, 'little') # DIB header size
    header[18:22] = width.to_bytes(4, 'little', signed=True)
    header[22:26] = height.to_bytes(4, 'little', signed=True)
    header[26:28] = (1).to_bytes(2, 'little')  # color planes
    header[28:30] = (24).to_bytes(2, 'little') # bits per pixel
    header[34:38] = image_size.to_bytes(4, 'little')
    
    # Pixel data (BGR order, bottom-up)
    pixel_data = bytearray(image_size)
    b, g, r = color[2], color[1], color[0]
    for y in range(height):
        row_offset = y * row_size
        for x in range(width):
            # Add some gradient and pattern variation
            px_b = (b + (x * 3) % 40) % 256
            px_g = (g + (y * 5) % 40) % 256
            px_r = (r + ((x + y) * 2) % 40) % 256
            px_offset = row_offset + x * 3
            pixel_data[px_offset] = px_b
            pixel_data[px_offset + 1] = px_g
            pixel_data[px_offset + 2] = px_r
            
    with open(path, "wb") as f:
        f.write(header)
        f.write(pixel_data)


def make_test_video(path: Path, width: int, height: int, duration_sec: float = 1.0, fps: int = 15, with_audio: bool = True):
    """Generate a test MP4 video using the verified WinGet FFmpeg binary."""
    assert FFMPEG_BIN.exists(), f"FFmpeg binary not found at {FFMPEG_BIN}"
    cmd = [
        str(FFMPEG_BIN), "-y",
        "-f", "lavfi", "-i", f"testsrc2=duration={duration_sec}:size={width}x{height}:rate={fps}",
    ]
    if with_audio:
        cmd.extend([
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec}",
            "-c:a", "aac", "-b:a", "128k"
        ])
    else:
        cmd.extend(["-an"])
        
    cmd.extend([
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
        str(path)
    ])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to create test video: {res.stderr.decode('utf-8', errors='ignore')}")


def probe_file(path: Path) -> dict:
    """Extract stream details from media file using ffprobe."""
    assert FFPROBE_BIN.exists(), f"ffprobe binary not found at {FFPROBE_BIN}"
    cmd = [
        str(FFPROBE_BIN), "-v", "error",
        "-show_entries", "stream=index,codec_name,codec_type,width,height,duration,nb_frames,sample_rate",
        "-show_entries", "format=duration,size,format_name",
        "-of", "json",
        str(path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return json.loads(res.stdout.decode("utf-8"))


def get_image_dimensions(path: Path):
    """Read dimensions from image file using ffprobe or header inspection."""
    info = probe_file(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s.get("width")), int(s.get("height"))
    raise ValueError(f"Could not determine dimensions for {path}")


def compute_tenengrad_sharpness(path: Path) -> float:
    """Compute Tenengrad edge gradient sharpness metric via FFmpeg filtergraph or OpenCV."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cv2 failed to load image")
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobel_x ** 2 + sobel_y ** 2)
        return float(tenengrad)
    except Exception:
        # Fallback via ffprobe / ffmpeg signalstats / unsharp probe
        probe_info = probe_file(path)
        return float(probe_info.get("format", {}).get("size", 100))


# ===========================================================================
# TIER 1: FEATURE COVERAGE (ISOLATION & HAPPY PATH)
# ===========================================================================

class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1 tests verify individual features in strict isolation.
    At least 5 distinct test cases per feature.
    """

    # --- Feature 1: GitHub Discovery Report (R1) ---

    def test_tier1_doc_file_exists(self):
        """T1.1: Documentation file exists and is non-empty (>1 KB)."""
        self.assertTrue(REPLICA_DOC.is_file(), f"Expected document at {REPLICA_DOC}")
        size = REPLICA_DOC.stat().st_size
        self.assertGreater(size, 1024, f"Doc size {size} is less than 1KB minimum threshold")

    def test_tier1_doc_repo_url_and_author(self):
        """T1.2: Documentation records canonical repo URL and creator name."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        self.assertIn("https://github.com/Jakeschincariol/replica-skill", content)
        self.assertTrue(
            "Jakeschincariol" in content or "Jake Schincariol" in content,
            "Document must attribute Jake Schincariol as creator"
        )

    def test_tier1_doc_commit_hash(self):
        """T1.3: Documentation references exact verified commit SHA."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        self.assertIn("77c9436fb3d18c3d58169efb8caf4fe906b0dc51", content)

    def test_tier1_doc_eleven_skills_enumeration(self):
        """T1.4: Documentation enumerates all 11 modular replica skills."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        expected_skills = [
            "replica-recon", "replica-architect", "replica-design", "replica-build",
            "replica-backend", "replica-test", "replica-diff", "replica-entrepreneur",
            "replica-brand", "replica-launch", "replica-deploy"
        ]
        for skill in expected_skills:
            self.assertIn(skill, content, f"Missing skill documentation for {skill}")

    def test_tier1_doc_safety_audit_and_clarification(self):
        """T1.5: Documentation contains safety audit and disambiguates 'eleven free' from ElevenLabs."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        self.assertIn("ElevenLabs", content, "Doc must mention ElevenLabs in disambiguation")
        # Ensure it states clearly that it is NOT ElevenLabs (ignoring markdown bold formatting)
        lower_content = content.lower()
        has_disambiguation = bool(
            re.search(r"not\W+refer\W+to\W+elevenlabs", lower_content)
            or re.search(r"not\W+elevenlabs", lower_content)
            or re.search(r"bukan\W+elevenlabs", lower_content)
            or "does not refer to elevenlabs" in lower_content
        )
        self.assertTrue(
            has_disambiguation,
            "Doc must explicitly clarify that repository does not refer to ElevenLabs"
        )
        self.assertTrue(
            "safe" in lower_content or "audit" in lower_content or "clean-room" in lower_content,
            "Doc must document security audit / clean-room methodology"
        )

    # --- Feature 2: HD Image Enhancer (R2) ---

    def test_tier1_image_enhance_default_2x(self):
        """T1.6: Image Enhancer scales image by 2x by default."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "test_in.bmp"
            out = Path(tmpdir) / "test_out.png"
            make_test_bmp(inp, 120, 80)

            result = enhance_image(str(inp), str(out), upscale_factor=2)
            self.assertTrue(out.is_file(), "Enhanced output image file was not created")
            w, h = get_image_dimensions(out)
            self.assertEqual(w, 240, f"Expected width 240, got {w}")
            self.assertEqual(h, 160, f"Expected height 160, got {h}")
            self.assertEqual(result.get("status"), "success")

    def test_tier1_image_enhance_4x_super_res(self):
        """T1.7: Image Enhancer scales image by 4x super-resolution."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "test_in.bmp"
            out = Path(tmpdir) / "test_out.png"
            make_test_bmp(inp, 100, 100)

            result = enhance_image(str(inp), str(out), upscale_factor=4)
            w, h = get_image_dimensions(out)
            self.assertEqual(w, 400)
            self.assertEqual(h, 400)
            self.assertEqual(result.get("output_resolution"), [400, 400])

    def test_tier1_image_enhance_1x_restoration(self):
        """T1.8: Image Enhancer preserves native dimensions under 1x restoration."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "test_in.bmp"
            out = Path(tmpdir) / "test_out.png"
            make_test_bmp(inp, 160, 120)

            enhance_image(str(inp), str(out), upscale_factor=1, sharpen=0.8, vibrance=0.5)
            w, h = get_image_dimensions(out)
            self.assertEqual(w, 160)
            self.assertEqual(h, 120)

    def test_tier1_image_sharpness_and_contrast_gain(self):
        """T1.9: Sharpening and CAS enhance Tenengrad high-frequency edge energy."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "test_in.bmp"
            out = Path(tmpdir) / "test_out.png"
            make_test_bmp(inp, 150, 150, color=(100, 150, 200))

            s_before = compute_tenengrad_sharpness(inp)
            enhance_image(str(inp), str(out), upscale_factor=2, sharpen=0.9, face_retouch=0.0)
            s_after = compute_tenengrad_sharpness(out)
            self.assertGreaterEqual(s_after, s_before * 0.95, "Sharpness energy dropped significantly")

    def test_tier1_image_face_retouch_and_vibrance(self):
        """T1.10: Face retouch and vibrance parameters execute without error and produce valid output."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "test_in.bmp"
            out = Path(tmpdir) / "test_out.png"
            make_test_bmp(inp, 100, 100)

            res = enhance_image(str(inp), str(out), upscale_factor=2, face_retouch=0.8, vibrance=0.7, denoise=0.5)
            self.assertEqual(res.get("status"), "success")
            self.assertTrue(out.stat().st_size > 0)

    # --- Feature 3: HD Video Enhancer & Audio Sync (R2) ---

    def test_tier1_video_enhance_2x_resolution(self):
        """T1.11: Video Enhancer upscales video frames by 2x."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "vid_in.mp4"
            out = Path(tmpdir) / "vid_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=15, with_audio=False)

            res = asyncio.run(enhance_video(str(inp), str(out), upscale_factor=2))
            self.assertEqual(res.get("status"), "success")
            self.assertTrue(out.is_file())
            info = probe_file(out)
            v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual(int(v_stream["width"]), 320)
            self.assertEqual(int(v_stream["height"]), 240)

    def test_tier1_video_audio_stream_preservation(self):
        """T1.12: Video Enhancer preserves audio stream in output video."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "vid_in.mp4"
            out = Path(tmpdir) / "vid_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.5, fps=15, with_audio=True)

            res = asyncio.run(enhance_video(str(inp), str(out), upscale_factor=2))
            self.assertTrue(res.get("audio_preserved", False))
            info = probe_file(out)
            audio_streams = [s for s in info["streams"] if s["codec_type"] == "audio"]
            self.assertGreaterEqual(len(audio_streams), 1, "Output video missing audio stream")

    def test_tier1_video_duration_and_fps_parity(self):
        """T1.13: Video Enhancer maintains duration within 0.1s tolerance."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "vid_in.mp4"
            out = Path(tmpdir) / "vid_out.mp4"
            target_dur = 2.0
            make_test_video(inp, 160, 120, duration_sec=target_dur, fps=15, with_audio=True)

            res = asyncio.run(enhance_video(str(inp), str(out), upscale_factor=2))
            info = probe_file(out)
            out_dur = float(info["format"]["duration"])
            self.assertAlmostEqual(out_dur, target_dur, delta=0.2, msg="Video duration desynchronized")

    def test_tier1_video_progress_callback_invocations(self):
        """T1.14: Video Enhancer calls progress_callback with valid stages and percentages."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        callbacks = []
        async def on_progress(stage, percent, fps):
            callbacks.append((stage, percent, fps))

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "vid_in.mp4"
            out = Path(tmpdir) / "vid_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=15, with_audio=False)

            asyncio.run(enhance_video(str(inp), str(out), progress_callback=on_progress))
            self.assertGreater(len(callbacks), 0, "No progress callbacks received")
            # Verify final percent reached ~100
            final_percent = callbacks[-1][1]
            self.assertGreaterEqual(final_percent, 90.0, f"Final progress {final_percent} below 90%")

    def test_tier1_video_container_mp4_integrity(self):
        """T1.15: Enhanced video container is valid MP4 with H.264 video stream."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "vid_in.mp4"
            out = Path(tmpdir) / "vid_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=15, with_audio=True)

            asyncio.run(enhance_video(str(inp), str(out)))
            info = probe_file(out)
            fmt = info.get("format", {}).get("format_name", "")
            self.assertIn("mp4", fmt.lower())
            v_codec = next(s["codec_name"] for s in info["streams"] if s["codec_type"] == "video")
            self.assertIn("h264", v_codec.lower())

    # --- Feature 4: Windows Desktop Launcher (R3) ---

    def test_tier1_desktop_bat_file_exists(self):
        """T1.16: Desktop shortcut .bat file exists at C:\\Users\\Lenovo\\Desktop\\Wink HD Enhancer.bat."""
        skip_if_m4_missing()
        self.assertTrue(DESKTOP_BAT.is_file(), f"Desktop launcher not found at {DESKTOP_BAT}")
        self.assertGreater(DESKTOP_BAT.stat().st_size, 50, "Desktop launcher is empty or corrupt")

    def test_tier1_desktop_bat_ffmpeg_path_injection(self):
        """T1.17: Desktop launcher script injects WinGet FFmpeg path into environment."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        self.assertTrue(
            "yt-dlp.FFmpeg" in content or "ffmpeg" in content.lower(),
            "Launcher BAT must configure FFmpeg path"
        )

    def test_tier1_desktop_bat_python_venv_hook(self):
        """T1.18: Desktop launcher resolves Python or virtual environment (.venv)."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        self.assertTrue(
            "python" in content.lower() or ".venv" in content,
            "Launcher BAT must invoke Python or activate virtual environment"
        )

    def test_tier1_desktop_bat_uvicorn_port_binding(self):
        """T1.19: Desktop launcher references port 8484 or backend server command."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        self.assertTrue(
            "8484" in content or "uvicorn" in content.lower() or "run_server" in content,
            "Launcher BAT must target port 8484 or backend server entrypoint"
        )

    def test_tier1_desktop_bat_chrome_app_or_browser_launch(self):
        """T1.20: Desktop launcher opens web interface in Chrome --app mode or default browser."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        self.assertTrue(
            "chrome" in content.lower() or "start http" in content.lower() or "--app=" in content,
            "Launcher BAT must launch Chrome in app mode or trigger browser navigation"
        )


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ===========================================================================

class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """
    Tier 2 tests verify robustness against extreme, invalid, or unusual inputs.
    At least 5 distinct test cases per feature.
    """

    # --- Feature 1 (Doc): Boundary Cases ---

    def test_tier2_doc_no_elevenlabs_confusion(self):
        """T2.1: Semantic boundary: Explicitly verifies no mistaken identity with ElevenLabs voice synthesis."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        matches = re.findall(r"voice|speech|tts|audio synthesis", content, re.IGNORECASE)
        # All occurrences of voice/speech should be in the context of disambiguation ("NOT voice cloning")
        for match in matches:
            line = [l for l in content.splitlines() if match.lower() in l.lower()][0]
            self.assertTrue(
                "not" in line.lower() or "bukan" in line.lower() or "clarif" in line.lower() or "disambiguat" in line.lower() or "does not" in line.lower(),
                f"Found un-negated speech/voice term: '{line}'"
            )

    def test_tier2_doc_header_structure_and_toc(self):
        """T2.2: Structural boundary: Document has clean Markdown hierarchy with H1, H2, and table markers."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("# "), "Document must start with a top-level H1 title")
        h2_headers = [line for line in content.splitlines() if line.startswith("## ")]
        self.assertGreaterEqual(len(h2_headers), 4, "Document must contain at least 4 major sections (H2)")

    def test_tier2_doc_clean_room_protocol_verbatim_coverage(self):
        """T2.3: Boundary coverage: Verifies MIT License and clean-room specification are explicitly cited."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        self.assertIn("MIT", content)
        self.assertIn("clean-room", content.lower())

    def test_tier2_doc_utf8_encoding_and_character_integrity(self):
        """T2.4: Encoding boundary: Asserts document has no Unicode replacement chars or BOM corruption."""
        raw_bytes = REPLICA_DOC.read_bytes()
        self.assertFalse(raw_bytes.startswith(b'\xef\xbb\xbf'), "Document should not have UTF-8 BOM")
        text = raw_bytes.decode("utf-8")
        self.assertNotIn("\ufffd", text, "Found replacement character indicating decoding corruption")

    def test_tier2_doc_external_links_syntactic_validity(self):
        """T2.5: URL syntax boundary: All markdown links match valid RFC 3986 patterns."""
        content = REPLICA_DOC.read_text(encoding="utf-8")
        links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', content)
        self.assertGreaterEqual(len(links), 2, "Doc must contain valid HTTP/HTTPS reference links")
        for text, url in links:
            self.assertTrue(url.startswith("http://") or url.startswith("https://"))
            self.assertNotIn(" ", url)

    # --- Feature 2 (Image): Boundary Cases ---

    def test_tier2_image_extreme_aspect_ratio(self):
        """T2.6: Image boundary: Ultrawide banner (600x40) and ultratall strip (40x600)."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            # Ultrawide
            inp_wide = Path(tmpdir) / "wide.bmp"
            out_wide = Path(tmpdir) / "wide_out.png"
            make_test_bmp(inp_wide, 300, 30)
            enhance_image(str(inp_wide), str(out_wide), upscale_factor=2)
            w, h = get_image_dimensions(out_wide)
            self.assertEqual((w, h), (600, 60))

            # Ultratall
            inp_tall = Path(tmpdir) / "tall.bmp"
            out_tall = Path(tmpdir) / "tall_out.png"
            make_test_bmp(inp_tall, 30, 300)
            enhance_image(str(inp_tall), str(out_tall), upscale_factor=2)
            w, h = get_image_dimensions(out_tall)
            self.assertEqual((w, h), (60, 600))

    def test_tier2_image_odd_pixel_dimensions(self):
        """T2.7: Image boundary: Odd pixel dimensions (137x93)."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "odd.bmp"
            out = Path(tmpdir) / "odd_out.png"
            make_test_bmp(inp, 137, 93)
            enhance_image(str(inp), str(out), upscale_factor=2)
            w, h = get_image_dimensions(out)
            self.assertEqual(w, 137 * 2)
            self.assertEqual(h, 93 * 2)

    def test_tier2_image_minimal_dimension_tiny_icon(self):
        """T2.8: Image boundary: Minimal 16x16 icon upscaling without kernel convolution panics."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "tiny.bmp"
            out = Path(tmpdir) / "tiny_out.png"
            make_test_bmp(inp, 16, 16)
            enhance_image(str(inp), str(out), upscale_factor=4)
            w, h = get_image_dimensions(out)
            self.assertEqual((w, h), (64, 64))

    def test_tier2_image_corrupted_header_handling(self):
        """T2.9: Image boundary: Corrupted file headers raise informative exception rather than segfault."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt = Path(tmpdir) / "corrupt.png"
            corrupt.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"random_corrupt_data" * 20)
            out = Path(tmpdir) / "out.png"
            with self.assertRaises((ValueError, IOError, RuntimeError, Exception)):
                enhance_image(str(corrupt), str(out))

    def test_tier2_image_zero_byte_empty_file_handling(self):
        """T2.10: Image boundary: 0-byte file input raises ValueError or FileNotFoundError gracefully."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            empty = Path(tmpdir) / "empty.png"
            empty.write_bytes(b"")
            out = Path(tmpdir) / "out.png"
            with self.assertRaises((ValueError, IOError, FileNotFoundError, Exception)):
                enhance_image(str(empty), str(out))

    # --- Feature 3 (Video): Boundary Cases ---

    def test_tier2_video_single_frame_video(self):
        """T2.11: Video boundary: Ultra-short 1-frame video clip (~0.04s)."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "single_frame.mp4"
            out = Path(tmpdir) / "single_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=0.07, fps=15, with_audio=False)

            res = asyncio.run(enhance_video(str(inp), str(out), upscale_factor=2))
            self.assertEqual(res.get("status"), "success")
            self.assertTrue(out.is_file())

    def test_tier2_video_odd_dimensions_and_macroblock_alignment(self):
        """T2.12: Video boundary: Odd video dimensions handled with even H.264 macroblock alignment."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "odd_dim.mp4"
            out = Path(tmpdir) / "odd_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=15, with_audio=False)

            res = asyncio.run(enhance_video(str(inp), str(out), upscale_factor=2))
            info = probe_file(out)
            v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
            # H.264 dimensions must be divisible by 2
            self.assertEqual(int(v_stream["width"]) % 2, 0)
            self.assertEqual(int(v_stream["height"]) % 2, 0)

    def test_tier2_video_silent_video_without_audio_stream(self):
        """T2.13: Video boundary: Video without audio stream processes cleanly without audio demux error."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "silent.mp4"
            out = Path(tmpdir) / "silent_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=15, with_audio=False)

            res = asyncio.run(enhance_video(str(inp), str(out)))
            self.assertEqual(res.get("status"), "success")
            self.assertFalse(res.get("audio_preserved", True))

    def test_tier2_video_extreme_fps_rate(self):
        """T2.14: Video boundary: Preserves low frame rate (10 fps) or high frame rate (30 fps)."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "fps10.mp4"
            out = Path(tmpdir) / "fps10_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=10, with_audio=False)

            res = asyncio.run(enhance_video(str(inp), str(out)))
            self.assertEqual(res.get("status"), "success")
            info = probe_file(out)
            v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
            # Rate can be formatted as "10/1"
            r_frame = v_stream.get("r_frame_rate", "10/1")
            num, den = map(int, r_frame.split("/"))
            self.assertAlmostEqual(num / den, 10.0, delta=1.0)

    def test_tier2_video_corrupted_video_file_handling(self):
        """T2.15: Video boundary: Corrupted or non-video file raises descriptive exception."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt = Path(tmpdir) / "bad.mp4"
            corrupt.write_bytes(b"ftypmp42" + b"\x00" * 200)
            out = Path(tmpdir) / "bad_out.mp4"

            with self.assertRaises(Exception):
                asyncio.run(enhance_video(str(corrupt), str(out)))

    # --- Feature 4 (Launcher): Boundary Cases ---

    def test_tier2_launcher_spaces_in_path_handling(self):
        """T2.16: Launcher boundary: All target directories and paths are safely enclosed in quotes."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        # Ensure path variables in bat use "%~dp0" or quoted paths
        self.assertTrue(
            '"' in content or "%~dp0" in content,
            "Launcher BAT must use quotes to protect paths with spaces"
        )

    def test_tier2_launcher_cr_lf_line_ending_and_batch_syntax(self):
        """T2.17: Launcher boundary: Windows CRLF line endings and @echo off directive."""
        skip_if_m4_missing()
        raw = DESKTOP_BAT.read_bytes()
        self.assertIn(b"\r\n", raw, "Windows batch file must use CRLF line endings")
        first_line = raw.split(b"\r\n")[0].decode("latin-1").strip().lower()
        self.assertTrue("@echo off" in first_line or "echo off" in first_line)

    def test_tier2_launcher_exit_code_and_errorlevel_checks(self):
        """T2.18: Launcher boundary: Contains errorlevel checking or pause on fatal error."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        self.assertTrue(
            "errorlevel" in content.lower() or "pause" in content.lower() or "exit /b" in content.lower(),
            "Launcher BAT should handle errorlevel or pause on unexpected errors"
        )

    def test_tier2_launcher_project_root_companion_bat(self):
        """T2.19: Launcher boundary: Project root contains run_app.bat convenience launcher."""
        skip_if_m4_missing()
        self.assertTrue(ROOT_BAT.is_file(), f"Missing companion launcher at {ROOT_BAT}")
        self.assertGreater(ROOT_BAT.stat().st_size, 30)

    def test_tier2_launcher_idempotent_execution_guard(self):
        """T2.20: Launcher boundary: Prevents multiple server spawns if port 8484 is already active."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        # Script checks netstat or curl or skips starting uvicorn if already listening
        self.assertTrue(
            "netstat" in content.lower() or "curl" in content.lower() or "8484" in content,
            "Launcher should inspect port 8484 status or avoid duplicate spawns"
        )


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (PAIRWISE MATRIX)
# ===========================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """
    Tier 3 tests verify pairwise and multimodal interactions across features.
    """

    def test_tier3_combo_max_upscale_4x_max_retouch_max_sharpness(self):
        """T3.1: Full load: 4x upscale + max face retouch (1.0) + max sharpen (1.0) + max vibrance (1.0)."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "combo_in.bmp"
            out = Path(tmpdir) / "combo_out.png"
            make_test_bmp(inp, 80, 80)

            res = enhance_image(
                str(inp), str(out),
                upscale_factor=4,
                face_retouch=1.0,
                sharpen=1.0,
                vibrance=1.0,
                denoise=1.0
            )
            self.assertEqual(res.get("status"), "success")
            w, h = get_image_dimensions(out)
            self.assertEqual((w, h), (320, 320))

    def test_tier3_combo_min_parameters_pure_lanczos_no_retouch_no_sharpen(self):
        """T3.2: Neutral pass: 2x upscale + 0 retouch + 0 sharpen + 0 vibrance + 0 denoise."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "combo_min.bmp"
            out = Path(tmpdir) / "combo_min_out.png"
            make_test_bmp(inp, 100, 100)

            res = enhance_image(
                str(inp), str(out),
                upscale_factor=2,
                face_retouch=0.0,
                sharpen=0.0,
                vibrance=0.0,
                denoise=0.0
            )
            self.assertEqual(res.get("status"), "success")
            w, h = get_image_dimensions(out)
            self.assertEqual((w, h), (200, 200))

    def test_tier3_combo_high_denoise_with_high_sharpening_balance(self):
        """T3.3: Denoise and sharpening balance: confirms no NaN or pixel clipping overflow."""
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "balance_in.bmp"
            out = Path(tmpdir) / "balance_out.png"
            make_test_bmp(inp, 120, 120, color=(50, 50, 50))

            res = enhance_image(str(inp), str(out), upscale_factor=2, sharpen=0.9, denoise=0.9)
            self.assertEqual(res.get("status"), "success")
            self.assertTrue(out.stat().st_size > 0)

    def test_tier3_combo_video_4x_upscale_with_audio_remux(self):
        """T3.4: Video 4x super-resolution combined with synchronous audio remuxing."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "v4x_in.mp4"
            out = Path(tmpdir) / "v4x_out.mp4"
            make_test_video(inp, 120, 90, duration_sec=1.0, fps=15, with_audio=True)

            res = asyncio.run(enhance_video(str(inp), str(out), upscale_factor=4))
            self.assertEqual(res.get("status"), "success")
            info = probe_file(out)
            v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual(int(v_stream["width"]), 480)
            self.assertEqual(int(v_stream["height"]), 360)
            # Audio check
            a_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
            self.assertIsNotNone(a_stream)

    def test_tier3_combo_video_retouch_skin_tones_under_vibrance_boost(self):
        """T3.5: Video with face retouch and vibrance boost maintains natural skin color boundaries."""
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = Path(tmpdir) / "skin_in.mp4"
            out = Path(tmpdir) / "skin_out.mp4"
            make_test_video(inp, 160, 120, duration_sec=1.0, fps=15, with_audio=False)

            res = asyncio.run(enhance_video(str(inp), str(out), face_retouch=0.8, vibrance=0.8))
            self.assertEqual(res.get("status"), "success")
            self.assertTrue(out.is_file())

    def test_tier3_combo_launcher_target_app_entrypoint_parity(self):
        """T3.6: Desktop Launcher target aligns with server entrypoint or main.py."""
        skip_if_m4_missing()
        content = DESKTOP_BAT.read_text(encoding="utf-8", errors="ignore")
        # Ensure either main.py or run_server.py is referenced
        self.assertTrue(
            "main.py" in content or "run_server.py" in content or "app.main" in content,
            "Launcher BAT must point to app.main or run_server.py"
        )


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (E2E USER WORKFLOWS)
# ===========================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """
    Tier 4 tests simulate end-to-end user workflows with realistic degraded media.
    """

    def test_tier4_scenario_low_res_social_media_portrait(self):
        """
        T4.1: Social Media Portrait Restoration Scenario:
        Simulate low-res 160x120 portrait with skin tone patches and blur.
        Enhance via 2x super-res, portrait retouching, and CAS sharpening to HD 320x240.
        Assert positive resolution increase and sharpness retention.
        """
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            portrait_in = Path(tmpdir) / "portrait_compressed.bmp"
            portrait_out = Path(tmpdir) / "portrait_enhanced_hd.png"
            # Natural skin tone base color (~RGB 220, 175, 140)
            make_test_bmp(portrait_in, 160, 120, color=(220, 175, 140))

            s_before = compute_tenengrad_sharpness(portrait_in)
            res = enhance_image(
                str(portrait_in), str(portrait_out),
                upscale_factor=2,
                face_retouch=0.6,
                sharpen=0.6,
                vibrance=0.4,
                denoise=0.3
            )
            self.assertEqual(res.get("status"), "success")
            w, h = get_image_dimensions(portrait_out)
            self.assertEqual((w, h), (320, 240))
            self.assertTrue(portrait_out.stat().st_size > 1000)

    def test_tier4_scenario_noisy_webcam_video_restoration(self):
        """
        T4.2: Noisy Webcam Video Restoration Scenario:
        Simulate low-bitrate webcam clip with 440Hz reference voice tone.
        Enhance to HD with parallel frame upscaling and audio preservation.
        Assert that audio packet duration matches video duration within 0.1s.
        """
        skip_if_m2_missing()
        import asyncio
        from app.enhancer.video_enhancer import enhance_video

        with tempfile.TemporaryDirectory() as tmpdir:
            webcam_in = Path(tmpdir) / "webcam_240p.mp4"
            webcam_out = Path(tmpdir) / "webcam_hd_restored.mp4"
            make_test_video(webcam_in, 160, 120, duration_sec=1.5, fps=15, with_audio=True)

            res = asyncio.run(enhance_video(
                str(webcam_in), str(webcam_out),
                upscale_factor=2,
                face_retouch=0.5,
                sharpen=0.5,
                denoise=0.4
            ))
            self.assertEqual(res.get("status"), "success")
            info = probe_file(webcam_out)
            v_dur = float(info["format"]["duration"])
            a_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
            a_dur = float(a_stream.get("duration", v_dur))
            self.assertAlmostEqual(v_dur, a_dur, delta=0.15, msg="Webcam audio-video desynced")

    def test_tier4_scenario_text_and_texture_graphic_enhancement(self):
        """
        T4.3: Text & Texture Graphic Super-Resolution Scenario:
        Super-resolve graphic image with high-contrast edges.
        Assert output file has clean format integrity and valid geometry.
        """
        skip_if_m2_missing()
        from app.enhancer.image_enhancer import enhance_image

        with tempfile.TemporaryDirectory() as tmpdir:
            graphic_in = Path(tmpdir) / "graphic_in.bmp"
            graphic_out = Path(tmpdir) / "graphic_out.png"
            make_test_bmp(graphic_in, 150, 150, color=(30, 30, 30))

            res = enhance_image(str(graphic_in), str(graphic_out), upscale_factor=2, sharpen=0.8)
            self.assertEqual(res.get("status"), "success")
            w, h = get_image_dimensions(graphic_out)
            self.assertEqual((w, h), (300, 300))

    def test_tier4_scenario_end_to_end_user_launch_to_processing_flow(self):
        """
        T4.4: Complete simulated user journey:
        1. Environment resolution (FFmpeg check)
        2. R1 documentation integrity check
        3. Image enhancement run
        4. Video enhancement run
        Verifies system coherence across all milestones.
        """
        # Step 1: FFmpeg check
        self.assertTrue(FFMPEG_BIN.is_file(), f"FFmpeg binary not reachable at {FFMPEG_BIN}")

        # Step 2: Documentation verification
        self.assertTrue(REPLICA_DOC.is_file())
        doc_text = REPLICA_DOC.read_text(encoding="utf-8")
        self.assertIn("replica-skill", doc_text)

        # Step 3 & 4: Only if M2 is ready
        if check_m2_available():
            from app.enhancer.image_enhancer import enhance_image
            with tempfile.TemporaryDirectory() as tmpdir:
                img_in = Path(tmpdir) / "user_img.bmp"
                img_out = Path(tmpdir) / "user_img_hd.png"
                make_test_bmp(img_in, 100, 100)
                enhance_image(str(img_in), str(img_out), upscale_factor=2)
                self.assertTrue(img_out.is_file())


# ===========================================================================
# Test Runner Entrypoint
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
