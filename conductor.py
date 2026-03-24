import os
import sys
import time
import threading
import subprocess
import tempfile
import urllib.request

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import mediapipe
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

RUBBERBAND = r"H:\Downloads\rubberband-4.0.0-gpl-executable-windows\rubberband-4.0.0-gpl-executable-windows\rubberband.exe"

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.\n")

# ── Audio load ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100

def load_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        n_out = int(len(data) * SAMPLE_RATE / sr)
        data  = np.array([
            np.interp(np.linspace(0, len(data)-1, n_out), np.arange(len(data)), data[:, ch])
            for ch in range(data.shape[1])
        ]).T.astype(np.float32)
    if data.shape[1] == 1:
        data = np.column_stack([data, data])
    return data.astype(np.float32)

# ── Rubberband stretch (calls the .exe directly via temp WAV files) ───────────

def rubberband_stretch(src, speed):
    """
    Write src to a temp WAV, run rubberband.exe on it, read back the result.
    This is exactly what pyrubberband does internally — we just call the exe directly.
    """
    if abs(speed - 1.0) < 0.02:
        return src

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        in_path  = f_in.name
        out_path = f_out.name

    try:
        sf.write(in_path, src, SAMPLE_RATE)
        subprocess.check_call(
            [RUBBERBAND, "--tempo", str(speed), "--fine", in_path, out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result, _ = sf.read(out_path, dtype="float32", always_2d=True)
        if result.shape[1] == 1:
            result = np.column_stack([result, result])
        return result.astype(np.float32)
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except Exception:
                pass

# ── Player ────────────────────────────────────────────────────────────────────

SEGMENT   = SAMPLE_RATE * 4   # stretch 4s at a time
LOOKAHEAD = SAMPLE_RATE * 2   # refill when < 2s left in buffer

class Player:
    def __init__(self, audio):
        self.audio       = audio
        self._src_pos    = 0
        self._play_pos   = 0
        self._buf        = np.zeros((0, 2), dtype=np.float32)
        self._speed      = 1.0
        self._last_speed = None
        self._lock       = threading.Lock()
        self._stream     = None
        self._running    = False
        self._thread     = threading.Thread(target=self._refill_loop, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()
        print("Buffering first segment (rubberband)...")
        while True:
            with self._lock:
                if len(self._buf) - self._play_pos > SAMPLE_RATE:
                    break
            time.sleep(0.05)

        def callback(outdata, frames, time_info, status):
            with self._lock:
                avail = len(self._buf) - self._play_pos
                take  = min(frames, avail)
                if take > 0:
                    outdata[:take] = self._buf[self._play_pos:self._play_pos + take]
                    self._play_pos += take
                if take < frames:
                    outdata[take:] = 0

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            blocksize=512,
            callback=callback,
        )
        self._stream.start()

    def set_speed(self, s):
        with self._lock:
            self._speed = float(np.clip(s, 0.25, 3.0))

    def get_speed(self):
        with self._lock:
            return self._speed

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _refill_loop(self):
        while self._running:
            with self._lock:
                speed      = self._speed
                last_speed = self._last_speed
                remaining  = len(self._buf) - self._play_pos
                src_pos    = self._src_pos

            speed_changed = last_speed is None or abs(speed - last_speed) > 0.02

            if speed_changed or remaining < LOOKAHEAD:
                end = min(src_pos + SEGMENT, len(self.audio))
                seg = self.audio[src_pos:end]

                if len(seg) == 0:
                    with self._lock:
                        self._src_pos = 0
                    continue

                stretched = rubberband_stretch(seg, speed)

                with self._lock:
                    if speed_changed:
                        kept           = self._buf[:self._play_pos]
                        self._buf      = np.vstack([kept, stretched]) if len(kept) else stretched
                        self._play_pos = len(kept)
                    else:
                        self._buf = np.vstack([self._buf, stretched]) if len(self._buf) else stretched

                    self._src_pos    = 0 if end >= len(self.audio) else end
                    self._last_speed = speed

                    # trim consumed audio to keep memory bounded
                    trim = self._play_pos - SAMPLE_RATE
                    if trim > 0:
                        self._buf      = self._buf[trim:]
                        self._play_pos -= trim
            else:
                time.sleep(0.1)

# ── Speed mapping ─────────────────────────────────────────────────────────────

NEUTRAL_LO = 0.30
NEUTRAL_HI = 0.45
MIN_SPEED  = 0.5
MAX_SPEED  = 2.0

def dist_to_speed(d):
    if d < NEUTRAL_LO:
        return MIN_SPEED + (d / NEUTRAL_LO) * (1.0 - MIN_SPEED)
    elif d <= NEUTRAL_HI:
        return 1.0
    else:
        return 1.0 + min((d - NEUTRAL_HI) / (1.0 - NEUTRAL_HI), 1.0) * (MAX_SPEED - 1.0)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python conductor.py <song.mp3>")
        sys.exit(1)

    ensure_model()

    print(f"Loading: {sys.argv[1]}")
    audio  = load_audio(sys.argv[1])
    player = Player(audio)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoothed = 1.0
    frame_i  = 0
    started  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        frame_i += 1

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(frame_i * 1000 / 30))

        left_tip  = None
        right_tip = None

        if result.hand_landmarks:
            for lm, hand in zip(result.hand_landmarks, result.handedness):
                side  = "Left" if hand[0].category_name == "Right" else "Right"
                tip   = lm[8]
                px    = (int(tip.x * w), int(tip.y * h))
                color = (0, 255, 0) if side == "Left" else (100, 100, 255)
                cv2.circle(frame, px, 12, color, -1)
                if side == "Left":
                    left_tip  = (tip.x, tip.y, px)
                else:
                    right_tip = (tip.x, tip.y, px)

        if left_tip and right_tip:
            dx   = left_tip[0] - right_tip[0]
            dy   = left_tip[1] - right_tip[1]
            dist = (dx**2 + dy**2) ** 0.5
            smoothed = 0.08 * dist_to_speed(dist) + 0.92 * smoothed
            player.set_speed(smoothed)
            cv2.line(frame, left_tip[2], right_tip[2], (255, 220, 0), 2)

        spd = player.get_speed()
        cv2.putText(frame, f"Speed: {spd:.2f}x", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (100, 255, 100) if 0.98 < spd < 1.02 else (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, "Q = quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow("Gesture Tempo", frame)

        if not started:
            player.start()
            started = True

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    player.stop()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
    