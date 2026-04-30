"""
parking_server.py
=================
Receives MJPEG frames from the XIAO ESP32-S3 over TCP, runs two parking-space
occupancy detectors (Laplacian and Std-Dev methods translated from MATLAB),
and serves results to the dashboard via plain HTTP polling.

Usage
-----
    pip install numpy Pillow
    python parking_server.py

Expects:
    parking_spaces.json  — produced by parking_space_definer.py

Exposes:
    TCP  :5001   — raw frame stream from ESP32
    HTTP :8080   — serves index.html  +  GET /status  (JSON)

Open http://parkingCuFinal.gleeze.com:8080 in your browser.
Only port 8080 needs to be forwarded on your router.
"""

import io
import json
import logging
import math
import socket
import struct
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Pure-NumPy replacements for scipy.ndimage ────────────────────────────────

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()

def gaussian_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur — drop-in for scipy.ndimage.gaussian_filter."""
    radius = int(4 * sigma + 0.5)
    k = _gaussian_kernel_1d(sigma, radius)
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1, img)
    out = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 0, out)
    return out

def laplace(img: np.ndarray) -> np.ndarray:
    """Discrete Laplacian — drop-in for scipy.ndimage.laplace."""
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (
        img[:-2, 1:-1] + img[2:,  1:-1] +
        img[1:-1, :-2] + img[1:-1, 2:] -
        4 * img[1:-1, 1:-1]
    )
    return out

# ─── PIL-based helpers (no cv2, no AVX required) ──────────────────────────────

def _jpeg_decode(data: bytes) -> np.ndarray | None:
    """Decode a JPEG byte string to an RGB numpy array."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None

def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 array to float64 grayscale using BT.601 weights."""
    return (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2]).astype(np.float64)

def _poly_mask(h: int, w: int, poly: np.ndarray) -> np.ndarray:
    """
    Rasterise a polygon into a boolean mask of shape (h, w).
    Uses a scanline even-odd fill — no cv2 required.
    poly: float array of shape (N, 2) with columns [x, y], local coords.
    """
    mask = np.zeros((h, w), dtype=bool)
    pts  = [(float(p[0]), float(p[1])) for p in poly]
    n    = len(pts)
    for y in range(h):
        intersections = []
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersections.append(x_int)
        intersections.sort()
        for k in range(0, len(intersections) - 1, 2):
            xa = max(0, int(math.ceil(intersections[k])))
            xb = min(w, int(math.floor(intersections[k + 1])) + 1)
            if xa < xb:
                mask[y, xa:xb] = True
    return mask

# ─── Configuration ─────────────────────────────────────────────────────────────
TCP_HOST            = "0.0.0.0"
TCP_PORT            = 5001
HTTP_PORT           = 8080
SPACES_FILE         = "parking_spaces.json"

LAPLACIAN_THRESHOLD = 0.9
STD_THRESHOLD       = 0.27
MEAN_DROP_THRESHOLD = 0.27
DETECT_DELAY_SEC    = 0.0   # 0 = flip immediately (suitable for 1-frame-per-minute)
GAUSS_SIGMA         = 3

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("parking")

# ─── Shared state ───────────────────────────────────────────────────────────────
_latest_status: dict = {}
_status_lock         = threading.Lock()

# ─── Space loading ─────────────────────────────────────────────────────────────

def load_spaces(path: str):
    data   = json.loads(Path(path).read_text())
    spaces = []
    for sp in data["spaces"]:
        poly = np.array(sp["polygon"], dtype=np.float32)
        xs, ys = poly[:, 0], poly[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        w, h   = max(x2 - x1, 1), max(y2 - y1, 1)

        local_poly = poly.copy()
        local_poly[:, 0] -= x1
        local_poly[:, 1] -= y1
        mask = _poly_mask(h, w, local_poly)

        spaces.append({
            "id":      sp["id"],
            "polygon": sp["polygon"],
            "rect":    (x1, y1, w, h),
            "mask":    mask,
        })
    log.info("Loaded %d parking spaces from %s", len(spaces), path)
    return spaces, data.get("frame_width", 640), data.get("frame_height", 480)

# ─── Image helpers ─────────────────────────────────────────────────────────────

def gen_roi(blurred: np.ndarray, rect, mask: np.ndarray):
    x1, y1, w, h = rect
    H, W = blurred.shape
    r1, r2 = max(0, y1), min(H, y1 + h)
    c1, c2 = max(0, x1), min(W, x1 + w)
    roi = blurred[r1:r2, c1:c2]
    mh, mw = mask.shape
    if roi.shape != (mh, mw):
        # nearest-neighbour resize via PIL (no cv2 needed)
        pil = Image.fromarray(roi)
        roi = np.array(pil.resize((mw, mh), Image.Resampling.BILINEAR))
    return roi

def get_std(data_flat: np.ndarray, mean_val: float) -> float:
    return math.sqrt(np.sum((data_flat - mean_val) ** 2) / len(data_flat))

# ─── Detection methods ─────────────────────────────────────────────────────────

def apply_laplacian(blurred, rect, mask, threshold):
    """True = FREE.  Low edge energy → smooth → empty space."""
    roi   = gen_roi(blurred, rect, mask)
    lap   = laplace(roi)
    score = float(np.mean(np.abs(lap[mask])))
    return score < threshold

def apply_std(blurred, rect, mask, threshold_std, threshold_mean_drop, mean_init):
    """True = FREE.  Low std-ratio AND low mean-drop → empty space."""
    roi  = gen_roi(blurred, rect, mask)
    flat = roi[mask].astype(np.float64)
    if flat.size == 0 or mean_init == 0:
        return True
    current_mean = float(flat.mean())
    std_ratio    = get_std(flat, mean_init) / mean_init
    mean_drop    = (mean_init - current_mean) / mean_init
    return std_ratio < threshold_std and mean_drop < threshold_mean_drop

def debounce(status, candidate, pending, pos_sec, delay):
    if pending is not None and candidate == status:
        return status, None
    if pending is not None and candidate != status:
        if pos_sec - pending >= delay:
            return candidate, None
        return status, pending
    if pending is None and candidate != status:
        return status, pos_sec
    return status, None

# ─── Detector ──────────────────────────────────────────────────────────────────

class ParkingDetector:
    def __init__(self, spaces):
        self.spaces      = spaces
        n                = len(spaces)
        self.mean_init   = np.zeros(n)
        self.mean_ref    = np.zeros(n)
        self.status_lap  = [False] * n
        self.status_std  = [False] * n
        self.pending_lap = [None]  * n
        self.pending_std = [None]  * n
        self.frame_count = 0
        self.initialised = False
        self.start_time  = time.time()

    def initialise(self, frame):
        gray    = _to_gray(frame)
        blurred = gaussian_filter(gray, sigma=GAUSS_SIGMA)
        for i, sp in enumerate(self.spaces):
            roi  = gen_roi(blurred, sp["rect"], sp["mask"])
            flat = roi[sp["mask"]].astype(np.float64)
            if flat.size:
                self.mean_init[i] = float(flat.mean())
                self.mean_ref[i]  = self.mean_init[i]
        self.initialised = True
        log.info("Detector initialised on first frame.")

    def process(self, frame):
        if not self.initialised:
            self.initialise(frame)

        gray    = _to_gray(frame)
        blurred = gaussian_filter(gray, sigma=GAUSS_SIGMA)
        self.frame_count += 1
        pos_sec = time.time() - self.start_time

        new_lap, new_std = [], []
        for i, sp in enumerate(self.spaces):
            fl = apply_laplacian(blurred, sp["rect"], sp["mask"], LAPLACIAN_THRESHOLD)
            fs = apply_std(blurred, sp["rect"], sp["mask"],
                           STD_THRESHOLD, MEAN_DROP_THRESHOLD, self.mean_init[i])
            new_lap.append(fl)
            new_std.append(fs)
            if fs:
                roi  = gen_roi(blurred, sp["rect"], sp["mask"])
                flat = roi[sp["mask"]].astype(np.float64)
                if flat.size:
                    self.mean_init[i] = float(flat.mean())

        for i in range(len(self.spaces)):
            self.status_lap[i], self.pending_lap[i] = debounce(
                self.status_lap[i], new_lap[i], self.pending_lap[i], pos_sec, DETECT_DELAY_SEC)
            self.status_std[i], self.pending_std[i] = debounce(
                self.status_std[i], new_std[i], self.pending_std[i], pos_sec, DETECT_DELAY_SEC)

        return self._build_status()

    def _build_status(self):
        spaces_out = [
            {
                "id":       sp["id"],
                "polygon":  sp["polygon"],
                "free_lap": self.status_lap[i],
                "free_std": self.status_std[i],
            }
            for i, sp in enumerate(self.spaces)
        ]
        return {
            "timestamp": time.time(),
            "frame":     self.frame_count,
            "spaces":    spaces_out,
            "free_lap":  sum(s["free_lap"] for s in spaces_out),
            "free_std":  sum(s["free_std"] for s in spaces_out),
            "total":     len(self.spaces),
        }

# ─── TCP frame receiver ────────────────────────────────────────────────────────

def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ESP32 disconnected")
        buf += chunk
    return buf

def camera_thread(detector: ParkingDetector):
    while True:
        log.info("Waiting for ESP32 on TCP %s:%d …", TCP_HOST, TCP_PORT)
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((TCP_HOST, TCP_PORT))
        srv.listen(1)
        srv.settimeout(1.0)
        conn = None
        try:
            while True:
                try:
                    conn, addr = srv.accept()
                    log.info("ESP32 connected from %s", addr)
                    conn.settimeout(90.0)   # must exceed the 60 s send interval
                    break
                except socket.timeout:
                    continue

            while True:
                try:
                    header = recv_exact(conn, 4)
                    length = struct.unpack(">I", header)[0]
                    if length == 0 or length > 500_000:
                        continue
                    data = recv_exact(conn, length)
                    img  = _jpeg_decode(data)
                    if img is not None:
                        status = detector.process(img)
                        with _status_lock:
                            _latest_status.clear()
                            _latest_status.update(status)
                        log.info("Frame %d — LAP free: %d  STD free: %d / %d",
                                 status["frame"], status["free_lap"],
                                 status["free_std"], status["total"])
                except (ConnectionError, socket.timeout) as e:
                    log.warning("Stream error: %s", e)
                    break
        except Exception as e:
            log.error("TCP server error: %s", e)
        finally:
            if conn:
                conn.close()
            srv.close()
            log.info("ESP32 disconnected — ready for next wakeup cycle")

# ─── HTTP server ───────────────────────────────────────────────────────────────

class ParkingHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass   # silence per-request noise

    def do_GET(self):
        if self.path == "/status":
            with _status_lock:
                snap = dict(_latest_status)
            body = json.dumps(snap).encode()
            self.send_response(200)
            self.send_header("Content-Type",   "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()   # serve index.html and static files normally

# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not Path(SPACES_FILE).exists():
        log.error("'%s' not found — run parking_space_definer.py first.", SPACES_FILE)
        return

    spaces, fw, fh = load_spaces(SPACES_FILE)
    detector       = ParkingDetector(spaces)

    with _status_lock:
        _latest_status.update(detector._build_status())

    t_cam = threading.Thread(target=camera_thread, args=(detector,), daemon=True)
    t_cam.start()

    log.info("=== Parking Detection Server ===")
    log.info("Dashboard → http://parkingCuFinal.gleeze.com:%d", HTTP_PORT)
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), ParkingHandler)
    httpd.serve_forever()   # blocks — no asyncio needed

if __name__ == "__main__":
    main()
