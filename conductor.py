import os
import sys
import time
import threading
import urllib.request
import collections

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import mediapipe
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from pedalboard import time_stretch

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

# ── Pedalboard stretch ────────────────────────────────────────────────────────
#
#  pedalboard.time_stretch expects (audio, sr, stretch_factor)
#  where audio is (channels, samples) float32.
#  stretch_factor > 1 = slower (more output samples), < 1 = faster.
#  We want speed > 1 = faster, so stretch_factor = 1/speed.

GRAIN_FRAMES  = SAMPLE_RATE // 4     # 250 ms of source per grain — small enough
                                     # for snappy response, big enough for quality
PREFILL_GRAINS = 6                   # ~1.5 s of output buffered before playback starts
MAX_BUF_FRAMES = SAMPLE_RATE * 8    # cap buffer memory (~8 s)

def stretch_grain(grain_audio, speed):
    """
    grain_audio : (N, 2) float32  interleaved stereo
    returns     : (M, 2) float32  time-stretched
    """
    speed = float(np.clip(speed, 0.25, 3.0))
    if abs(speed - 1.0) < 0.01:
        return grain_audio

    # pedalboard wants (channels, samples)
    pcm = grain_audio.T  # (2, N)
    out = time_stretch(pcm, SAMPLE_RATE, stretch_factor=1.0 / speed)
    return out.T.astype(np.float32)   # back to (M, 2)

# ── Player ────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self, audio):
        self.audio      = audio          # (N, 2) float32
        self._src_pos   = 0              # next source frame to consume
        self._buf       = collections.deque()   # deque of np arrays (chunks)
        self._buf_len   = 0              # total frames in deque
        self._speed     = 1.0
        self._lock      = threading.Lock()
        self._stream    = None
        self._running   = False
        self._refill_event = threading.Event()
        self._thread    = threading.Thread(target=self._refill_loop, daemon=True)

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread.start()

        print("Buffering initial audio...")
        while True:
            with self._lock:
                ready = self._buf_len >= GRAIN_FRAMES * PREFILL_GRAINS
            if ready:
                break
            time.sleep(0.05)
        print("Ready — showing camera.\n")

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            blocksize=512,
            callback=self._audio_callback,
        )
        self._stream.start()

    def set_speed(self, s):
        with self._lock:
            self._speed = float(np.clip(s, 0.25, 3.0))
        self._refill_event.set()   # wake refill thread immediately

    def get_speed(self):
        with self._lock:
            return self._speed

    def stop(self):
        self._running = False
        self._refill_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()

    # ── audio callback (runs on sounddevice thread) ───────────────────────────

    def _audio_callback(self, outdata, frames, time_info, status):
        with self._lock:
            needed = frames
            filled = 0
            while needed > 0 and self._buf:
                chunk = self._buf[0]
                take  = min(len(chunk), needed)
                outdata[filled:filled + take] = chunk[:take]
                filled  += take
                needed  -= take
                self._buf_len -= take
                if take == len(chunk):
                    self._buf.popleft()
                else:
                    self._buf[0] = chunk[take:]
            if filled < frames:
                outdata[filled:] = 0   # underrun — refill thread is behind

        self._refill_event.set()   # wake refill thread to top up

    # ── refill loop (background thread) ──────────────────────────────────────

    REFILL_TARGET = SAMPLE_RATE * 3   # keep ~3 s of stretched audio buffered

    def _refill_loop(self):
        while self._running:
            with self._lock:
                buf_len = self._buf_len
                speed   = self._speed
                src_pos = self._src_pos

            if buf_len < self.REFILL_TARGET:
                end  = min(src_pos + GRAIN_FRAMES, len(self.audio))
                grain = self.audio[src_pos:end]

                if len(grain) == 0:
                    with self._lock:
                        self._src_pos = 0
                    continue

                stretched = stretch_grain(grain, speed)

                with self._lock:
                    # Trim buffer if it somehow grew huge (e.g. speed < 1 for a long time)
                    if self._buf_len > MAX_BUF_FRAMES:
                        excess = self._buf_len - MAX_BUF_FRAMES
                        while excess > 0 and self._buf:
                            c = self._buf.popleft()
                            if len(c) <= excess:
                                excess -= len(c)
                                self._buf_len -= len(c)
                            else:
                                self._buf.appendleft(c[excess:])
                                self._buf_len -= excess
                                break

                    self._buf.append(stretched)
                    self._buf_len += len(stretched)
                    self._src_pos  = 0 if end >= len(self.audio) else end

                # Don't sleep — immediately check if more needed
                continue

            # Buffer is full enough; wait until callback signals we need more
            self._refill_event.wait(timeout=0.05)
            self._refill_event.clear()

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
            # Slightly faster smoothing than before so gestures feel more responsive
            smoothed = 0.15 * dist_to_speed(dist) + 0.85 * smoothed
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