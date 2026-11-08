"""
Lumina HD Studio — Cloudflare Tunnel & Mobile Access Service
Creates free, zero-config Cloudflare public HTTPS tunnels and generates QR codes
for instant smartphone access over local Wi-Fi or cellular networks.
"""

import socket
import threading
import io
import base64
import time
from typing import Dict, Any, Optional

try:
    import qrcode
except ImportError:
    qrcode = None

try:
    from pycloudflared import try_cloudflare
except ImportError:
    try_cloudflare = None

# Global Tunnel State
_TUNNEL_STATE: Dict[str, Any] = {
    "status": "idle",  # "idle", "starting", "running", "error"
    "tunnel_url": None,
    "local_url": None,
    "local_ip": None,
    "qr_tunnel_b64": None,
    "qr_local_b64": None,
    "error_message": None,
    "started_at": None,
}
_LOCK = threading.Lock()


def get_local_lan_ip() -> str:
    """Detect LAN IP address on current active network interface."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def generate_qr_base64(url: str) -> Optional[str]:
    """Generate base64 encoded PNG data URI for a given URL."""
    if not qrcode or not url:
        return None
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f"[TUNNEL] Failed to generate QR code: {exc}")
        return None


def print_ascii_qr(url: str):
    """Print ASCII representation of QR code to terminal."""
    if not qrcode or not url:
        return
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        print("\n" + "=" * 50)
        print(f"📱 SCAN DENGAN KAMERA HP UNTUK MEMBUKA LUMINA HD:")
        print(f"🔗 URL: {url}")
        print("=" * 50)
        qr.print_ascii(invert=True)
        print("=" * 50 + "\n")
    except Exception:
        pass


def start_tunnel_background(port: int = 8484):
    """Start Cloudflare Quick Tunnel in background worker thread."""
    with _LOCK:
        if _TUNNEL_STATE["status"] in ("starting", "running"):
            return _TUNNEL_STATE.copy()
        _TUNNEL_STATE["status"] = "starting"
        _TUNNEL_STATE["error_message"] = None

    lan_ip = get_local_lan_ip()
    local_url = f"http://{lan_ip}:{port}"
    qr_local = generate_qr_base64(local_url)

    with _LOCK:
        _TUNNEL_STATE["local_ip"] = lan_ip
        _TUNNEL_STATE["local_url"] = local_url
        _TUNNEL_STATE["qr_local_b64"] = qr_local

    def _worker():
        if not try_cloudflare:
            with _LOCK:
                _TUNNEL_STATE["status"] = "error"
                _TUNNEL_STATE["error_message"] = "pycloudflared not installed"
            return

        try:
            print(f"[TUNNEL] Starting Cloudflare Quick Tunnel on port {port}...")
            tunnel_res = try_cloudflare(port=port, verbose=False)
            tunnel_url = str(tunnel_res.tunnel).strip()

            if not tunnel_url.startswith("http"):
                tunnel_url = f"https://{tunnel_url}"

            qr_tunnel = generate_qr_base64(tunnel_url)

            with _LOCK:
                _TUNNEL_STATE["status"] = "running"
                _TUNNEL_STATE["tunnel_url"] = tunnel_url
                _TUNNEL_STATE["qr_tunnel_b64"] = qr_tunnel
                _TUNNEL_STATE["started_at"] = time.time()

            print(f"[TUNNEL] Public HTTPS Tunnel Live: {tunnel_url}")
            print_ascii_qr(tunnel_url)

        except Exception as exc:
            err = str(exc)
            print(f"[TUNNEL] Cloudflare Tunnel failed: {err}")
            with _LOCK:
                _TUNNEL_STATE["status"] = "error"
                _TUNNEL_STATE["error_message"] = err

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return _TUNNEL_STATE.copy()


def get_tunnel_status(port: int = 8484) -> Dict[str, Any]:
    """Retrieve current tunnel status and URLs."""
    with _LOCK:
        if _TUNNEL_STATE["local_url"] is None:
            lan_ip = get_local_lan_ip()
            local_url = f"http://{lan_ip}:{port}"
            _TUNNEL_STATE["local_ip"] = lan_ip
            _TUNNEL_STATE["local_url"] = local_url
            _TUNNEL_STATE["qr_local_b64"] = generate_qr_base64(local_url)
        return _TUNNEL_STATE.copy()


if __name__ == "__main__":
    print("Initializing Lumina HD Studio Tunnel Service...")
    start_tunnel_background(8484)
    while True:
        st = get_tunnel_status(8484)
        if st["status"] == "running":
            print(f"Active Tunnel: {st['tunnel_url']}")
            print(f"Local Wi-Fi:   {st['local_url']}")
            break
        elif st["status"] == "error":
            print(f"Tunnel Error: {st['error_message']}")
            break
        time.sleep(1)
