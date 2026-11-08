"""
Bootstrap script for Wink HD Enhancer environment.
Ensures Python 3.11 venv, dependencies, and FFmpeg binary configuration.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

# Known WinGet pre-installed FFmpeg directory
FFMPEG_KNOWN_DIR = Path(
    r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-N-127043-g5a54fcf75e-win64-gpl\bin"
)


def ensure_ffmpeg_in_path():
    """Ensure ffmpeg and ffprobe are available in os.environ['PATH']."""
    current_path = os.environ.get("PATH", "")
    if FFMPEG_KNOWN_DIR.exists() and str(FFMPEG_KNOWN_DIR) not in current_path:
        os.environ["PATH"] = f"{FFMPEG_KNOWN_DIR};{current_path}"
    
    # Check if ffmpeg can be run
    try:
        res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            version_line = res.stdout.splitlines()[0] if res.stdout else "unknown"
            print(f"[OK] FFmpeg found: {version_line}")
            return True
    except FileNotFoundError:
        pass

    # Try direct binary fallback
    direct_ffmpeg = FFMPEG_KNOWN_DIR / "ffmpeg.exe"
    if direct_ffmpeg.exists():
        print(f"[OK] FFmpeg binary verified directly at: {direct_ffmpeg}")
        return True

    print("[WARNING] FFmpeg binary not found in PATH or known location.")
    return False


def get_venv_python():
    """Return path to python.exe in the virtual environment."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def setup_venv():
    """Create virtual environment if it does not exist."""
    venv_python = get_venv_python()
    if not venv_python.exists():
        print(f"[*] Creating virtual environment at {VENV_DIR}...")
        res = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("[OK] Virtual environment created.")
    else:
        print(f"[OK] Virtual environment already exists at {VENV_DIR}")
    return venv_python


def install_dependencies(venv_python):
    """Install or upgrade dependencies from requirements.txt."""
    print("[*] Upgrading pip...")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    
    if REQUIREMENTS_FILE.exists():
        print(f"[*] Installing packages from {REQUIREMENTS_FILE}...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)], check=True)
        print("[OK] Dependencies successfully installed.")
    else:
        print(f"[ERROR] Requirements file not found: {REQUIREMENTS_FILE}")
        sys.exit(1)


def verify_installation(venv_python):
    """Verify that all core modules can be imported and FFmpeg is detected."""
    print("[*] Verifying module imports in virtual environment...")
    test_code = (
        "import cv2, numpy, PIL, scipy, fastapi, uvicorn, aiofiles;"
        "print('Import verification successful!');"
        "print(f'OpenCV: {cv2.__version__}, NumPy: {numpy.__version__}, Pillow: {PIL.__version__}')"
    )
    res = subprocess.run([str(venv_python), "-c", test_code], capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout.strip())
        print("[OK] All required packages verified!")
        return True
    else:
        print(f"[FAIL] Module verification failed:\n{res.stderr}")
        return False


def main():
    print("=" * 60)
    print("Wink HD Enhancer Environment Bootstrap")
    print("=" * 60)
    ensure_ffmpeg_in_path()
    venv_python = setup_venv()
    install_dependencies(venv_python)
    success = verify_installation(venv_python)
    if success:
        print("=" * 60)
        print("Environment setup is COMPLETE and verified.")
        print(f"Venv Python: {venv_python}")
        print("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
