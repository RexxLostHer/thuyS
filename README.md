# Wink HD Enhancer (`thuyS`)

A high-performance media enhancement and restoration application inspired by the core capabilities of **Meitu Wink**. It enhances images and videos to crisp HD quality with advanced super-resolution, skin retouching, adaptive sharpening, and vibrance optimization.

---

## 🌟 Key Features

- **Super-Resolution Upscaling (2x / 4x):** High-fidelity bicubic and deep detail upscaling for low-resolution media.
- **Bilateral Skin Retouching:** Natural, smooth facial texture refinement while preserving pores and realistic skin features (70/30 edge-preserving blend).
- **Adaptive Contrast Sharpening (CAS):** Enhances edges, facial contours, eyes, and hair without over-sharpening halos or artifacts.
- **Color & Vibrance Dynamics:** LAB CLAHE dynamic range expansion and subtle saturation boost.
- **Synchronized Video Enhancement Engine:**
  - Multi-threaded per-frame processing.
  - Automatic even-dimension padding for strict H.264/FFmpeg compatibility.
  - Lossless audio remuxing with sub-millisecond audio-video synchronization (delta < 5 ms).
- **Interactive Obsidian Dark Web UI:**
  - Glassmorphism interface (`#0B0B10`).
  - Interactive Before/After Split Comparison Slider (with real-time 60 FPS dual-video playback synchronization).
  - Real-time progress streaming via Server-Sent Events (SSE).
  - One-click "Try Sample" demo mode.
- **One-Click Windows Desktop Launcher:** Native App-Mode startup via `Wink HD Enhancer.bat`.

---

## 📁 Project Structure

```
wink_hd_enhancer/
├── app/
│   ├── enhancer/
│   │   ├── image_enhancer.py    # Core image upscaling & retouching algorithms
│   │   └── video_enhancer.py    # Frame-by-frame video processing & audio remuxing
│   ├── static/
│   │   ├── css/style.css        # Obsidian dark theme & glassmorphic styling
│   │   ├── js/
│   │   │   ├── app.js           # UI logic, uploads & SSE progress handler
│   │   │   └── slider.js        # Synchronized before/after split slider engine
│   │   └── index.html           # Main user interface
│   └── main.py                  # FastAPI server & REST API endpoints
├── docs/
│   └── REPLICA_SKILL_DISCOVERY.md # In-depth research & clean-room architecture report
├── media/                       # Working directory for media processing
├── samples/                     # Test media inputs & verified HD outputs
├── scripts/
│   └── verify_enhancer.py       # Standalone verification script
├── tests/                       # Complete automated pytest test suite (77 tests)
├── requirements.txt             # Python dependencies
├── run_app.bat                  # Local launcher script
├── .gitignore                   # Clean repository ignore rules
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### 1. Using the Desktop Launcher (Recommended on Windows)
Double-click `run_app.bat` in the project root (or `Wink HD Enhancer.bat` on your Desktop). The launcher will:
1. Verify FFmpeg and the Python virtual environment.
2. Start the FastAPI backend on port 8484.
3. Automatically launch your browser in native App Mode.

### 2. Manual Startup
```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
uvicorn app.main:app --host 127.0.0.1 --port 8484
```
Then navigate to `http://127.0.0.1:8484` in your browser.

---

## 🧪 Running Tests

To run the complete test suite (77 tests):
```bash
pytest tests/ -v
```

---

## 📜 License
MIT License.
