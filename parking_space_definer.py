"""
parking_space_definer.py
========================
1. Listens for the XIAO ESP32-S3 TCP stream (frame format: 4-byte big-endian
   length followed by raw JPEG bytes).
2. Displays the live feed in an OpenCV window.
3. Press  F  to freeze a frame, then click to draw parking-space polygons:
     - Left-click  : add a vertex to the current polygon
     - Right-click : close & save the current polygon (min 3 points)
     - Z           : undo last vertex of current polygon
     - D           : delete the most recently completed space
     - C           : cancel / clear the polygon you're currently drawing
4. Press  S  to save all spaces to  parking_spaces.json
5. Press  Q  to quit (auto-saves first).

JSON format
-----------
{
  "spaces": [
    {
      "id": 1,
      "polygon": [[x1,y1], [x2,y2], ...]   // pixel coords on the frozen frame
    },
    ...
  ],
  "frame_width":  640,
  "frame_height": 480
}
"""

import socket
import struct
import threading
import json
import time
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
HOST        = "0.0.0.0"   # listen on all interfaces
PORT        = 5001
OUTPUT_FILE = "parking_spaces.json"

# Colours (BGR)
COLOR_COMPLETE   = (0,   200,   0)   # finished polygon
COLOR_CURRENT    = (0,   165, 255)   # polygon being drawn
COLOR_POINT      = (255,   0,   0)   # vertex dot
COLOR_LABEL      = (255, 255, 255)
COLOR_FROZEN_BG  = (0,    0,  180)   # status bar in frozen mode

# ── Shared state ──────────────────────────────────────────────────────────────
latest_frame_lock = threading.Lock()
latest_frame      = None          # most-recent decoded JPEG as numpy array
running           = True

# ── TCP receiver thread ───────────────────────────────────────────────────────

def recv_exact(sock, n):
    """Read exactly n bytes from sock, or raise ConnectionError."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ESP32 disconnected")
        buf += chunk
    return buf


def camera_thread(host, port):
    global latest_frame, running

    while running:
        print(f"[server] Waiting for ESP32 on {host}:{port} …")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        srv.settimeout(1.0)

        conn = None
        try:
            while running:
                try:
                    conn, addr = srv.accept()
                    print(f"[server] ESP32 connected from {addr}")
                    conn.settimeout(5.0)
                    break
                except socket.timeout:
                    continue

            if conn is None:
                srv.close()
                continue

            while running:
                try:
                    header = recv_exact(conn, 4)
                    length = struct.unpack(">I", header)[0]
                    if length == 0 or length > 500_000:
                        print(f"[server] Suspicious frame length {length}, skipping")
                        continue
                    data = recv_exact(conn, length)
                    arr  = np.frombuffer(data, dtype=np.uint8)
                    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        with latest_frame_lock:
                            latest_frame = img
                except (ConnectionError, socket.timeout) as e:
                    print(f"[server] Stream error: {e}")
                    break

        except Exception as e:
            print(f"[server] Unexpected error: {e}")
        finally:
            if conn:
                conn.close()
            srv.close()
            if running:
                print("[server] Retrying in 2 s …")
                time.sleep(2)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_polygon(img, pts, color, label=None, closed=True):
    if len(pts) == 0:
        return
    poly = np.array(pts, dtype=np.int32)
    if closed and len(pts) >= 3:
        cv2.fillPoly(img, [poly], (*color[:3], 40))   # slight fill
        cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
    elif len(pts) >= 2:
        cv2.polylines(img, [poly], False, color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 5, COLOR_POINT, -1)
    if label and len(pts) >= 1:
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(img, label, (cx - 10, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LABEL, 2, cv2.LINE_AA)


def overlay_status(img, frozen, spaces, current_pts):
    h, w = img.shape[:2]
    bar_h = 34
    bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)
    if frozen:
        bar[:] = COLOR_FROZEN_BG
        mode_txt = "FROZEN"
    else:
        bar[:] = (40, 40, 40)
        mode_txt = "LIVE"

    n = len(spaces)
    pts_txt = f"  |  drawing: {len(current_pts)} pts" if current_pts else ""
    txt = (f"[{mode_txt}]  spaces: {n}{pts_txt}  |  "
           "F=freeze  LClick=add pt  RClick=close  Z=undo  D=del last  C=cancel  S=save  Q=quit")
    cv2.putText(bar, txt, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


# ── Mouse callback ────────────────────────────────────────────────────────────

class ClickState:
    def __init__(self):
        self.current_pts: list[list[int]] = []
        self.spaces:      list[dict]      = []
        self.frozen       = False

    def mouse_cb(self, event, x, y, flags, param):
        if not self.frozen:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_pts) >= 3:
                sid = len(self.spaces) + 1
                self.spaces.append({
                    "id":      sid,
                    "polygon": copy.deepcopy(self.current_pts)
                })
                print(f"[ui] Space {sid} saved ({len(self.current_pts)} vertices)")
                self.current_pts = []
            else:
                print("[ui] Need at least 3 points to close a polygon")


# ── Save / load ───────────────────────────────────────────────────────────────

def save_spaces(spaces, frame_shape, path=OUTPUT_FILE):
    h, w = frame_shape[:2]
    data = {
        "frame_width":  w,
        "frame_height": h,
        "spaces":       spaces
    }
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"[save] {len(spaces)} space(s) written to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global running

    # Start background receiver
    t = threading.Thread(target=camera_thread, args=(HOST, PORT), daemon=True)
    t.start()

    state         = ClickState()
    frozen_frame  = None
    display_frame = None

    cv2.namedWindow("Parking Space Definer", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Parking Space Definer", state.mouse_cb)

    print("\n=== Parking Space Definer ===")
    print(f"Listening for ESP32 on port {PORT}.")
    print("Press F to freeze a frame, then click to define spaces.\n")

    while True:
        # ── grab latest frame ──────────────────────────────────────────────
        if not state.frozen:
            with latest_frame_lock:
                if latest_frame is not None:
                    frozen_frame  = latest_frame.copy()
                    display_frame = frozen_frame.copy()

        if display_frame is None:
            # Show a waiting screen
            placeholder = np.zeros((240, 480, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for ESP32 stream…",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            cv2.imshow("Parking Space Definer", placeholder)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            continue

        # ── build display image ────────────────────────────────────────────
        vis = display_frame.copy()

        # Draw completed spaces
        for sp in state.spaces:
            draw_polygon(vis, sp["polygon"], COLOR_COMPLETE,
                         label=str(sp["id"]), closed=True)

        # Draw current (in-progress) polygon
        if state.current_pts:
            draw_polygon(vis, state.current_pts, COLOR_CURRENT, closed=False)
            # Preview line to mouse position (next click preview via hover)
            # — OpenCV doesn't expose hover natively; skip for simplicity

        vis = overlay_status(vis, state.frozen, state.spaces, state.current_pts)
        cv2.imshow("Parking Space Definer", vis)

        # ── key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            if state.spaces:
                save_spaces(state.spaces, display_frame.shape)
            break

        elif key == ord('f') or key == ord('F'):
            if not state.frozen:
                with latest_frame_lock:
                    if latest_frame is not None:
                        frozen_frame  = latest_frame.copy()
                        display_frame = frozen_frame.copy()
                        state.frozen  = True
                        state.current_pts = []
                        print("[ui] Frame frozen — start clicking to define spaces")
            else:
                state.frozen      = False
                state.current_pts = []
                print("[ui] Resumed live feed")

        elif key == ord('s') or key == ord('S'):
            if state.spaces and display_frame is not None:
                save_spaces(state.spaces, display_frame.shape)
            else:
                print("[ui] No spaces to save yet")

        elif key == ord('z') or key == ord('Z'):
            if state.current_pts:
                state.current_pts.pop()
                print(f"[ui] Undo — {len(state.current_pts)} pts remaining")

        elif key == ord('c') or key == ord('C'):
            state.current_pts = []
            print("[ui] Current polygon cleared")

        elif key == ord('d') or key == ord('D'):
            if state.spaces:
                removed = state.spaces.pop()
                print(f"[ui] Deleted space {removed['id']}")
            else:
                print("[ui] No spaces to delete")

    running = False
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
