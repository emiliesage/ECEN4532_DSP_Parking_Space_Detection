#!/usr/bin/env python3
"""
OpenCV + NumPy optimized parking detection server
With live Laplacian dashboard web UI
Hosted at parkingCuFinal.gleeze.com
"""

import base64
import datetime
import io
import json
import logging
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import imageio
import numpy as np

# ─── Configuration ─────────────────────────────────────────
TCP_HOST = "0.0.0.0"
TCP_PORT = 5001
HTTP_PORT = 8080
SPACES_FILE = "parking_spaces.json"

LAPLACIAN_THRESHOLD = 2.0
STD_THRESHOLD = 25.0
GAUSS_RADIUS = 0

RECORD_VIDEO = True
VIDEO_DIR = "recordings"
VIDEO_FPS = 10.0
VIDEO_CODEC = "libx264"

SITE_HOST = "parkingCuFinal.gleeze.com"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("parking")

_latest_status = {}
_status_lock = threading.Lock()
_latest_frame = b""
_frame_lock = threading.Lock()

# ─── Dashboard HTML ───────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ParkVision — Laplacian Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;500;700&display=swap');

  :root {
    --bg: #0a0c10;
    --surface: #111318;
    --border: #1e2330;
    --accent-free: #00e5a0;
    --accent-occ: #ff4060;
    --accent-dim-free: rgba(0,229,160,0.15);
    --accent-dim-occ: rgba(255,64,96,0.15);
    --text: #c8cdd8;
    --muted: #4a5068;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Barlow Condensed', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    background: rgba(10,12,16,0.95);
    position: sticky; top: 0; z-index: 10;
  }
  .logo {
    font-family: var(--mono);
    font-size: 1.1rem;
    letter-spacing: 0.12em;
    color: var(--accent-free);
  }
  .logo span { color: var(--muted); }
  .live-pill {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 0.75rem;
    color: var(--muted); letter-spacing: 0.08em;
  }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-free);
    animation: pulse 1.4s ease-in-out infinite;
  }
  .live-dot.disconnected { background: var(--accent-occ); animation: none; }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }
  #clock { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }

  /* ── Main layout ── */
  main {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 0;
    height: calc(100vh - 60px - 44px);
    overflow: hidden;
  }

  /* ── Camera panel ── */
  .camera-panel {
    position: relative;
    background: #050608;
    border-right: 1px solid var(--border);
    overflow: hidden;
    display: flex; align-items: center; justify-content: center;
  }
  #overlay-canvas {
    max-width: 100%; max-height: 100%;
    display: block;
    image-rendering: auto;
  }
  .no-signal {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    font-family: var(--mono); color: var(--muted); font-size: 0.8rem;
  }
  .no-signal svg { opacity: 0.3; }

  /* ── Stats panel ── */
  .stats-panel {
    display: flex; flex-direction: column;
    overflow-y: auto;
    background: var(--surface);
  }
  .stats-header {
    padding: 20px 20px 12px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--surface); z-index: 1;
  }
  .stats-header h2 {
    font-family: var(--sans); font-weight: 700;
    font-size: 0.7rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--muted);
  }
  .summary-row {
    display: flex; gap: 16px; margin-top: 14px;
  }
  .summary-card {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    text-align: center;
  }
  .summary-card .num {
    font-family: var(--mono); font-size: 2rem; line-height: 1;
    font-weight: 400;
  }
  .summary-card .label {
    font-size: 0.65rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin-top: 4px;
  }
  .free .num { color: var(--accent-free); }
  .occ .num { color: var(--accent-occ); }

  /* ── Space list ── */
  .space-list { padding: 16px; display: flex; flex-direction: column; gap: 8px; }
  .space-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 8px;
    transition: border-color 0.3s, background 0.3s;
  }
  .space-card.free {
    border-color: rgba(0,229,160,0.35);
    background: var(--accent-dim-free);
  }
  .space-card.occ {
    border-color: rgba(255,64,96,0.35);
    background: var(--accent-dim-occ);
  }
  .space-id {
    font-family: var(--mono); font-size: 0.85rem;
    color: var(--text);
  }
  .space-metrics {
    font-family: var(--mono); font-size: 0.68rem;
    color: var(--muted); margin-top: 3px;
  }
  .status-badge {
    font-family: var(--mono); font-size: 0.65rem;
    letter-spacing: 0.1em; padding: 4px 9px;
    border-radius: 3px; font-weight: 400;
  }
  .free .status-badge { background: rgba(0,229,160,0.2); color: var(--accent-free); }
  .occ .status-badge { background: rgba(255,64,96,0.2); color: var(--accent-occ); }

  /* ── Laplacian bar ── */
  .lap-bar-wrap {
    margin-top: 5px;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .lap-bar {
    height: 100%; border-radius: 2px;
    transition: width 0.4s ease, background 0.4s;
  }

  /* ── Footer ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 10px 32px;
    display: flex; justify-content: space-between; align-items: center;
    font-family: var(--mono); font-size: 0.68rem; color: var(--muted);
  }

  /* scrollbar */
  .stats-panel::-webkit-scrollbar { width: 4px; }
  .stats-panel::-webkit-scrollbar-track { background: transparent; }
  .stats-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="logo">PARK<span>//</span>VISION</div>
  <div style="display:flex;align-items:center;gap:24px">
    <div id="clock"></div>
    <div class="live-pill">
      <div class="live-dot disconnected" id="live-dot"></div>
      <span id="live-label">CONNECTING</span>
    </div>
  </div>
</header>

<main>
  <div class="camera-panel">
    <div class="no-signal" id="no-signal">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
      </svg>
      <div>AWAITING CAMERA FEED</div>
    </div>
    <canvas id="overlay-canvas" style="display:none"></canvas>
  </div>

  <div class="stats-panel">
    <div class="stats-header">
      <h2>Parking Spaces</h2>
      <div class="summary-row">
        <div class="summary-card free"><div class="num" id="count-free">—</div><div class="label">Free</div></div>
        <div class="summary-card occ"><div class="num" id="count-occ">—</div><div class="label">Occupied</div></div>
      </div>
    </div>
    <div class="space-list" id="space-list">
      <div style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);padding:12px 0">
        Waiting for data…
      </div>
    </div>
  </div>
</main>

<footer>
  <span>LAPLACIAN THR <strong style="color:var(--text)">LAPLACIAN_THRESHOLD_VALUE</strong> &nbsp;|&nbsp; STD THR <strong style="color:var(--text)">STD_THRESHOLD_VALUE</strong></span>
  <span id="frame-num">FRAME —</span>
</footer>

<script>
const LAP_THR = LAPLACIAN_THRESHOLD_VALUE;
const STD_THR = STD_THRESHOLD_VALUE;

// Always use the canonical hostname so the dashboard works whether accessed
// via localhost, IP, or the public hostname.
const API_BASE = 'http://parkingCuFinal.gleeze.com:8080';

let spaces = [];
let connected = false;

const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');
const noSignal = document.getElementById('no-signal');
const liveDot = document.getElementById('live-dot');
const liveLabel = document.getElementById('live-label');

// ── Clock ──
function updateClock() {
  const d = new Date();
  document.getElementById('clock').textContent =
    d.toLocaleTimeString('en-US', {hour12: false});
}
setInterval(updateClock, 1000); updateClock();

// ── Set connection state ──
function setConnected(v) {
  connected = v;
  liveDot.className = 'live-dot' + (v ? '' : ' disconnected');
  liveLabel.textContent = v ? 'LIVE' : 'DISCONNECTED';
}

// ── Draw overlays on canvas ──
function drawOverlays() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const sp of spaces) {
    if (!sp.polygon || sp.polygon.length < 3) continue;
    const free = sp.free_lap;
    ctx.beginPath();
    ctx.moveTo(sp.polygon[0][0], sp.polygon[0][1]);
    for (let i = 1; i < sp.polygon.length; i++)
      ctx.lineTo(sp.polygon[i][0], sp.polygon[i][1]);
    ctx.closePath();
    ctx.fillStyle = free ? 'rgba(0,229,160,0.18)' : 'rgba(255,64,96,0.22)';
    ctx.fill();
    ctx.strokeStyle = free ? '#00e5a0' : '#ff4060';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    const cx = sp.polygon.reduce((s,p)=>s+p[0],0)/sp.polygon.length;
    const cy = sp.polygon.reduce((s,p)=>s+p[1],0)/sp.polygon.length;
    ctx.font = 'bold 12px "Share Tech Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = free ? '#00e5a0' : '#ff4060';
    ctx.fillText(sp.id, cx, cy);
  }
}

// ── Update space list sidebar ──
function updateList() {
  const freeCount = spaces.filter(s=>s.free_lap).length;
  const occCount = spaces.length - freeCount;
  document.getElementById('count-free').textContent = freeCount;
  document.getElementById('count-occ').textContent = occCount;

  const list = document.getElementById('space-list');
  list.innerHTML = '';
  for (const sp of spaces) {
    const free = sp.free_lap;
    const lapPct = Math.min(100, ((sp.lap ?? 0) / (LAP_THR * 3)) * 100);
    const barColor = free ? '#00e5a0' : '#ff4060';

    const card = document.createElement('div');
    card.className = 'space-card ' + (free ? 'free' : 'occ');
    card.innerHTML = `
      <div>
        <div class="space-id">Space ${sp.id}</div>
        <div class="space-metrics">
          LAP ${(sp.lap ?? 0).toFixed(3)} &nbsp;|&nbsp; STD ${(sp.std ?? 0).toFixed(1)}
        </div>
        <div class="lap-bar-wrap">
          <div class="lap-bar" style="width:${lapPct}%;background:${barColor}"></div>
        </div>
      </div>
      <div class="status-badge">${free ? 'FREE' : 'OCC'}</div>
    `;
    list.appendChild(card);
  }
}

// ── Polygon lookup ──
let polygonMap = {};

// ── Poll /status ──
async function pollStatus() {
  try {
    const r = await fetch(API_BASE + '/status');
    if (!r.ok) throw new Error();
    const data = await r.json();
    if (!data.spaces) return;
    setConnected(true);

    document.getElementById('frame-num').textContent = 'FRAME ' + (data.frame ?? '—');

    spaces = data.spaces.map(s => ({
      ...s,
      polygon: polygonMap[s.id] || [],
    }));

    updateList();
    drawOverlays();
  } catch {
    setConnected(false);
  }
}

// ── Poll /frame (JPEG) ──
let frameUrl = null;
const img = new Image();
img.onload = () => {
  if (canvas.width !== img.width || canvas.height !== img.height) {
    canvas.width = img.width;
    canvas.height = img.height;
  }
  ctx.drawImage(img, 0, 0);
  drawOverlays();
  noSignal.style.display = 'none';
  canvas.style.display = 'block';
  if (frameUrl) URL.revokeObjectURL(frameUrl);
};

async function pollFrame() {
  try {
    const r = await fetch(API_BASE + '/frame?t=' + Date.now());
    if (!r.ok) throw new Error();
    const blob = await r.blob();
    frameUrl = URL.createObjectURL(blob);
    img.src = frameUrl;
  } catch {}
}

// ── Load polygon data from /spaces ──
async function loadSpaces() {
  try {
    const r = await fetch(API_BASE + '/spaces');
    if (!r.ok) return;
    const data = await r.json();
    for (const sp of data.spaces) polygonMap[sp.id] = sp.polygon;
  } catch {}
}

loadSpaces();
setInterval(pollStatus, 500);
setInterval(pollFrame, 120);
</script>
</body>
</html>
"""

# ─── Video Recorder ───────────────────────────────────────
class VideoRecorder:
    def __init__(self):
        self.writer = None
        self.path = None
        if RECORD_VIDEO:
            Path(VIDEO_DIR).mkdir(exist_ok=True)

    def _open(self, w, h):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = str(Path(VIDEO_DIR) / f"parking_{ts}.mp4")
        self.writer = imageio.get_writer(self.path, fps=VIDEO_FPS, codec=VIDEO_CODEC)
        log.info(f"Recording → {self.path}")

    def write(self, jpeg):
        if not RECORD_VIDEO:
            return
        arr = np.frombuffer(jpeg, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        if self.writer is None:
            self._open(frame.shape[1], frame.shape[0])
        self.writer.append_data(frame[:, :, ::-1])

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None

# ─── OpenCV Detection ─────────────────────────────────────

def jpeg_to_gray(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if GAUSS_RADIUS > 0:
        k = int(GAUSS_RADIUS * 2 + 1)
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def polygon_to_mask(poly, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def laplacian_score(gray, mask):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = (1/4)*np.abs(lap)[mask > 0]
    return float(vals.mean()) if vals.size else 0.0


def std_score(gray, mask):
    vals = gray[mask > 0]
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())

# ─── Load spaces ──────────────────────────────────────────

def load_spaces(path):
    data = json.loads(Path(path).read_text())
    return data["spaces"]

# ─── Detector ─────────────────────────────────────────────

class Detector:
    def __init__(self, spaces):
        self.spaces = spaces
        self.masks = []
        self.initialised = False
        self.frame = 0

    def init(self, w, h):
        self.masks = [polygon_to_mask(sp["polygon"], w, h) for sp in self.spaces]
        self.initialised = True

    def process(self, jpeg):
        gray = jpeg_to_gray(jpeg)
        if gray is None:
            return None

        if not self.initialised:
            self.init(gray.shape[1], gray.shape[0])

        results = []
        for i, sp in enumerate(self.spaces):
            mask = self.masks[i]
            lap = laplacian_score(gray, mask)
            mean, std = std_score(gray, mask)

            results.append({
                "id": sp["id"],
                "lap": round(lap, 4),
                "std": round(std, 2),
                "free_lap": lap < LAPLACIAN_THRESHOLD,
                "free_std": std < STD_THRESHOLD,
            })

        self.frame += 1
        return {"frame": self.frame, "spaces": results}

# ─── Networking ───────────────────────────────────────────

def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return buf


def camera_thread(det, rec):
    """
    FIX: The listening socket (srv) is created ONCE outside the accept loop.
    Previously srv was closed and re-created on every disconnect, creating a
    brief window where the ESP32's reconnect attempt would be refused.
    Now the server socket stays open permanently and immediately accepts the
    next connection after a disconnect.
    """
    global _latest_frame

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((TCP_HOST, TCP_PORT))
    srv.listen(1)
    log.info(f"TCP listener ready on :{TCP_PORT}")

    while True:
        conn, addr = srv.accept()
        log.info(f"ESP32 connected from {addr}")

        try:
            while True:
                size = struct.unpack(">I", recv_exact(conn, 4))[0]
                data = recv_exact(conn, size)

                rec.write(data)

                with _frame_lock:
                    _latest_frame = data

                status = det.process(data)
                if status:
                    with _status_lock:
                        _latest_status.clear()
                        _latest_status.update(status)

        except Exception as e:
            log.info(f"ESP32 disconnected: {e}")
            rec.close()
            conn.close()
            # srv stays open — next loop iteration calls accept() immediately

# ─── HTTP ─────────────────────────────────────────────────

_spaces_json = b""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send_cors(self):
        """Allow the public hostname and localhost to both reach the API."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_cors()
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/spaces":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_cors()
            self.end_headers()
            self.wfile.write(_spaces_json)

        elif self.path == "/status":
            with _status_lock:
                body = json.dumps(_latest_status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_cors()
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/frame"):
            with _frame_lock:
                frame = _latest_frame
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.send_cors()
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(204)
                self.send_cors()
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

# ─── Main ─────────────────────────────────────────────────

def main():
    global _spaces_json, DASHBOARD_HTML

    spaces = load_spaces(SPACES_FILE)

    # Inject thresholds into HTML
    DASHBOARD_HTML = DASHBOARD_HTML.replace(
        "LAPLACIAN_THRESHOLD_VALUE", str(LAPLACIAN_THRESHOLD)
    ).replace(
        "STD_THRESHOLD_VALUE", str(STD_THRESHOLD)
    )

    # Pre-build spaces JSON (polygons needed by the browser for overlays)
    _spaces_json = json.dumps({"spaces": spaces}).encode()

    det = Detector(spaces)
    rec = VideoRecorder()

    threading.Thread(target=camera_thread, args=(det, rec), daemon=True).start()

    log.info(f"Dashboard → http://{SITE_HOST}:{HTTP_PORT}/")
    log.info(f"TCP listener on :{TCP_PORT}")
    HTTPServer(("0.0.0.0", HTTP_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
