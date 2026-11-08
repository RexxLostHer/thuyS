"""
FFmpeg Utilities and Binary Path Resolver
Ensures FFmpeg and FFprobe are properly discovered and accessible across all modules.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

KNOWN_FFMPEG_DIR = Path(
    r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin"
)


def ensure_ffmpeg_env():
    """Ensure KNOWN_FFMPEG_DIR is in os.environ['PATH']."""
    curr = os.environ.get("PATH", "")
    if KNOWN_FFMPEG_DIR.exists() and str(KNOWN_FFMPEG_DIR) not in curr:
        os.environ["PATH"] = f"{KNOWN_FFMPEG_DIR};{curr}"


def get_ffmpeg_path() -> str:
    """Return path to ffmpeg executable."""
    ensure_ffmpeg_env()
    # Check known path first
    direct = KNOWN_FFMPEG_DIR / "ffmpeg.exe"
    if direct.exists():
        return str(direct)
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Return path to ffprobe executable."""
    ensure_ffmpeg_env()
    direct = KNOWN_FFMPEG_DIR / "ffprobe.exe"
    if direct.exists():
        return str(direct)
    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path
    return "ffprobe"


def run_ffmpeg(args: list, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run an FFmpeg command with the resolved binary."""
    ffmpeg_bin = get_ffmpeg_path()
    cmd = [ffmpeg_bin] + args
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)


def run_ffprobe(args: list, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run an FFprobe command with the resolved binary."""
    ffprobe_bin = get_ffprobe_path()
    cmd = [ffprobe_bin] + args
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
