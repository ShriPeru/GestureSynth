"""
video_sync.py  –  Gesture + keyboard tempo control for video+audio.

Key fixes vs previous version:
  - Wall-clock master clock: src_time advances via time.perf_counter() × speed,
    no buffer-math guessing. Video seeks to the exact matching frame every tick.
  - No buffer discard on speed change: we keep whatever is already queued and
    just start appending new stretched audio from the current src position.
    This eliminates the skip/gap on speed transitions.
  - Rubberband is called in a background thread so the main loop never stalls.
  - Gesture smoothing alpha raised so finger control feels less jittery.

Usage:
    python video_sync.py <video.mp4> [--rubberband PATH]

Controls:
    Two index fingertips  →  distance = speed (close=slow, far=fast)
    + / =                 →  +0.1x
    -                     →  -0.1x
    r                     →  reset 1.0x
    q / ESC               →  quit
"""

import os, sys, time, threading, subprocess, tempfile, argparse, urllib.request

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    import mediapipe
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] mediapipe not installed – gesture control disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────
SR          = 44100
CHUNK       = SR * 3        # source samples per rubberband call (3 s)
PREBUFFER   = SR * 2        # output samples to keep ahead of playhead
MIN_SPEED   = 0.30
MAX_SPEED   = 2.50
NEUTRAL_LO  = 0.30
NEUTRAL_HI  = 0.45

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~5 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.\n")

def load_audio(video_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    try:
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-ar", str(SR), "-ac", "2", "-f", "wav", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        sys.exit("[ERROR] ffmpeg not found – install it and add to PATH.")
    data, _ = sf.read(tmp.name, dtype="float32", always_2d=True)
    os.unlink(tmp.name)
    if data.shape[1] == 1:
        data = np.column_stack([data, data])
    return data.astype(np.float32)

def dist_to_speed(d):
    if d < NEUTRAL_LO:
        return MIN_SPEED + (d / NEUTRAL_LO) * (1.0 - MIN_SPEED)
    elif d <= NEUTRAL_HI:
        return 1.0
    else:
        return 1.0 + min((d - NEUTRAL_HI) / (1.0 - NEUTRAL_HI), 1.0) * (MAX_SPEED - 1.0)

def rb_stretch(src, speed, rb_exe):
    """Tempo-stretch src by speed. Uses rubberband if available, else linear resample."""
    if abs(speed - 1.0) < 0.015:
        return src.copy()
    if rb_exe and os.path.isfile(rb_exe):
        fi = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fo = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fi.close(); fo.close()
        try:
            sf.write(fi.name, src, SR)
            subprocess.check_call(
                [rb_exe, "--tempo", str(speed), "--fine", fi.name, fo.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out, _ = sf.read(fo.name, dtype="float32", always_2d=True)
            if out.shape[1] == 1:
                out = np.column_stack([out, out])
            return out.astype(np.float32)
        finally:
            for p in (fi.name, fo.name):
                try: os.unlink(p)
                except: pass
    # Fallback – pitch shifts but at least stays in sync
    n = max(1, int(len(src) / speed))
    t = np.linspace(0, len(src)-1, n)
    L = np.interp(t, np.arange(len(src)), src[:,0])
    R = np.interp(t, np.arange(len(src)), src[:,1])
    return np.column_stack([L, R]).astype(np.float32)

# ── Audio engine ──────────────────────────────────────────────────────────────
class AudioEngine:
    """
    Streams tempo-stretched audio.  A wall-clock master clock tracks how far
    into the *original* source we are so the video can seek to the right frame.

    Design:
      • _buf  : ring of already-stretched output samples waiting to be played
      • _play_pos : index into _buf of next sample the callback will consume
      • _src_pos  : next source sample to be processed by the refill thread
      • src_time  : seconds into original audio (wall-clock based, very stable)
    """
    def __init__(self, audio, rb_exe):
        self.audio    = audio
        self.rb_exe   = rb_exe
        self.duration = len(audio) / SR

        self._buf      = np.zeros((0,2), dtype=np.float32)
        self._play_pos = 0
        self._src_pos  = 0          # source samples already enqueued

        self._speed    = 1.0
        self._lock     = threading.Lock()

        # Wall-clock master clock
        self._clock_src_time = 0.0  # src_time at _clock_ref
        self._clock_ref      = None # time.perf_counter() when clock was set
        self._clock_speed    = 1.0  # speed active at _clock_ref

        self._stream   = None
        self._running  = False
        self._ready    = threading.Event()
        self._refill_t = threading.Thread(target=self._refill_loop, daemon=True)

    # ── public API ────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._refill_t.start()
        print("Buffering…")
        self._ready.wait()

        def _cb(outdata, frames, tinfo, status):
            with self._lock:
                avail = len(self._buf) - self._play_pos
                take  = min(frames, avail)
                if take:
                    outdata[:take] = self._buf[self._play_pos:self._play_pos+take]
                    self._play_pos += take
                if take < frames:
                    outdata[take:] = 0
                # trim consumed head to keep memory bounded
                trim = self._play_pos - SR
                if trim > 0:
                    self._buf      = self._buf[trim:]
                    self._play_pos -= trim

        self._stream = sd.OutputStream(
            samplerate=SR, channels=2, dtype="float32",
            blocksize=512, callback=_cb)
        self._stream.start()

        # Anchor the wall clock exactly when audio starts
        with self._lock:
            self._clock_ref      = time.perf_counter()
            self._clock_src_time = 0.0
            self._clock_speed    = self._speed

    def set_speed(self, s):
        s = float(np.clip(s, MIN_SPEED, MAX_SPEED))
        with self._lock:
            if abs(s - self._speed) < 0.005:
                return
            # Re-anchor the clock at the current src_time before changing speed
            now = time.perf_counter()
            elapsed = now - (self._clock_ref or now)
            self._clock_src_time = min(
                self._clock_src_time + elapsed * self._clock_speed,
                self.duration)
            self._clock_ref   = now
            self._clock_speed = s
            self._speed       = s

    def get_speed(self):
        with self._lock: return self._speed

    def get_src_time(self):
        """Stable wall-clock position in original source (seconds)."""
        with self._lock:
            if self._clock_ref is None:
                return 0.0
            elapsed = time.perf_counter() - self._clock_ref
            t = self._clock_src_time + elapsed * self._clock_speed
            return min(t, self.duration)

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop(); self._stream.close()

    # ── refill thread ─────────────────────────────────────────────────────────
    def _refill_loop(self):
        while self._running:
            with self._lock:
                speed     = self._speed
                remaining = len(self._buf) - self._play_pos
                src_pos   = self._src_pos

            if remaining >= PREBUFFER:
                time.sleep(0.05)
                continue

            end = min(src_pos + CHUNK, len(self.audio))
            seg = self.audio[src_pos:end]

            if len(seg) == 0:
                # Loop: reset source position and clock
                with self._lock:
                    self._src_pos        = 0
                    self._clock_src_time = 0.0
                    self._clock_ref      = time.perf_counter()
                continue

            stretched = rb_stretch(seg, speed, self.rb_exe)

            with self._lock:
                # Just append – never discard queued audio on speed change.
                # This is the key fix: no gap, no skip.
                self._buf     = (np.vstack([self._buf, stretched])
                                 if len(self._buf) else stretched)
                self._src_pos = end

            if not self._ready.is_set():
                self._ready.set()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--rubberband", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"[ERROR] Not found: {args.video}")

    print(f"Loading audio from {args.video}…")
    audio  = load_audio(args.video)
    rb_exe = args.rubberband
    if rb_exe and not os.path.isfile(rb_exe):
        print(f"[WARN] rubberband not found at '{rb_exe}', using fallback.")
        rb_exe = None
    print(f"[INFO] {'rubberband ✓' if rb_exe else 'fallback stretch (no pitch correction)'}")

    engine = AudioEngine(audio, rb_exe)

    cap          = cv2.VideoCapture(args.video)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.2f} fps")

    # MediaPipe
    landmarker = None
    if HAS_MEDIAPIPE:
        ensure_model()
        landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        ))

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoothed     = 1.0
    speed_val    = 1.0
    cam_fi       = 0
    last_vid_fi  = -1
    vid_buf      = None

    engine.start()
    print("Playing!  +/= speed up | - slow down | r reset | q/ESC quit\n")

    while True:
        # ── Video: seek to frame matching audio clock ──────────────────────
        src_t     = engine.get_src_time()
        target_fi = int(src_t * fps) % max(total_frames, 1)

        if target_fi != last_vid_fi:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_fi)
            ret, raw = cap.read()
            if ret:
                h, w = raw.shape[:2]
                if max(h, w) > 960:
                    sc  = 960 / max(h, w)
                    raw = cv2.resize(raw, (int(w*sc), int(h*sc)))
                vid_buf     = raw
                last_vid_fi = target_fi

        # ── Webcam + gesture ───────────────────────────────────────────────
        ret, cam_frame = cam.read()
        if ret:
            cam_frame = cv2.flip(cam_frame, 1)
            ch, cw    = cam_frame.shape[:2]
            cam_fi   += 1

            left_tip = right_tip = None

            if landmarker:
                rgb    = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                mp_img = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, int(cam_fi * 1000 / 30))

                if result.hand_landmarks:
                    for lm, hand in zip(result.hand_landmarks, result.handedness):
                        side  = "Left" if hand[0].category_name == "Right" else "Right"
                        tip   = lm[8]
                        px    = (int(tip.x * cw), int(tip.y * ch))
                        col   = (0,220,80) if side=="Left" else (80,100,255)
                        cv2.circle(cam_frame, px, 14, col, -1)
                        cv2.circle(cam_frame, px, 14, (255,255,255), 2)
                        cv2.putText(cam_frame, "L" if side=="Left" else "R",
                                    (px[0]-7, px[1]+6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                        if side=="Left":  left_tip  = (tip.x, tip.y, px)
                        else:             right_tip = (tip.x, tip.y, px)

                if left_tip and right_tip:
                    dx   = left_tip[0]  - right_tip[0]
                    dy   = left_tip[1]  - right_tip[1]
                    dist = (dx**2 + dy**2)**0.5
                    # Smoother: alpha=0.12 (was 0.08) for more responsive feel
                    smoothed  = 0.12 * dist_to_speed(dist) + 0.88 * smoothed
                    speed_val = smoothed
                    cv2.line(cam_frame, left_tip[2], right_tip[2], (255,220,50), 2)

                    # Distance bar
                    bx,by,bw,bh = 20, ch-60, int(cw*0.4), 12
                    filled = int(np.clip(dist/0.8, 0, 1)*bw)
                    cv2.rectangle(cam_frame,(bx,by),(bx+bw,by+bh),(40,40,40),-1)
                    cv2.rectangle(cam_frame,(bx,by),(bx+filled,by+bh),(255,220,50),-1)
                    cv2.putText(cam_frame,"Finger dist",(bx,by-6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,150),1)

            engine.set_speed(speed_val)

            spd   = speed_val
            col   = (80,230,80) if 0.97<spd<1.03 else (255,255,255)
            cv2.rectangle(cam_frame,(0,0),(260,58),(0,0,0),-1)
            cv2.putText(cam_frame,f"Speed: {spd:.2f}x",(10,36),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,col,2,cv2.LINE_AA)
            cv2.putText(cam_frame,f"t={src_t:.1f}s  f={target_fi}",
                        (10,56),cv2.FONT_HERSHEY_SIMPLEX,0.4,(100,100,100),1)
            cv2.putText(cam_frame,"+/- | r=reset | q=quit",
                        (10,ch-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(110,110,110),1)
            cv2.imshow("Gesture Cam", cam_frame)

        # ── Video display ──────────────────────────────────────────────────
        if vid_buf is not None:
            disp = vid_buf.copy()
            spd  = speed_val
            cv2.putText(disp,f"Speed: {spd:.2f}x",(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,
                        (80,230,80) if 0.97<spd<1.03 else (255,255,255),
                        2,cv2.LINE_AA)
            cv2.imshow("Video", disp)

        # ── Keyboard ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('+'), ord('=')):
            speed_val = min(MAX_SPEED, round(speed_val+0.1, 2)); smoothed = speed_val
        elif key == ord('-'):
            speed_val = max(MIN_SPEED, round(speed_val-0.1, 2)); smoothed = speed_val
        elif key == ord('r'):
            speed_val = 1.0; smoothed = 1.0

    engine.stop()
    cap.release(); cam.release()
    if landmarker: landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()