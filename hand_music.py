import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import pygame
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)



MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model (~5 MB) — one-time only...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}\n")
    except Exception as e:
        print(f"\nERROR: Could not download model: {e}")
        print("Check your internet connection, then try again.")
        sys.exit(1)



pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
SAMPLE_RATE = 44100


def _to_sound(wave: np.ndarray) -> pygame.mixer.Sound:
    """Convert a normalised float64 wave array → stereo pygame Sound."""
    mono   = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((mono, mono))
    return pygame.sndarray.make_sound(stereo)


def karplus_strong(freq: float, duration: float = 3.0,
                   volume: float = 0.7, damping: float = 0.996) -> pygame.mixer.Sound:
    """
    Karplus-Strong plucked string — sounds like guitar/banjo.
    Seeds a ring buffer with noise, then averages+damps each cycle.
    The periodic ring buffer naturally generates the harmonic series.
    damping 0.99–0.999: lower = more muted, higher = brighter longer ring.
    """
    n_samples = int(SAMPLE_RATE * duration)
    buf_size  = max(2, int(SAMPLE_RATE / freq))
    buf = np.random.uniform(-1.0, 1.0, buf_size)
    out = np.zeros(n_samples)
    for i in range(n_samples):
        out[i] = buf[0]
        buf = np.roll(buf, -1)
        buf[-1] = damping * 0.5 * (out[i] + buf[0])
    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * volume
    return _to_sound(out)


def piano_tone(freq: float, duration: float = 3.0, volume: float = 0.6) -> pygame.mixer.Sound:
    """
    Piano simulation using additive harmonic synthesis + ADSR envelope.
    Harmonics weighted to give the bright, slightly metallic piano timbre.
    Envelope: near-instant attack → fast decay → sustain plateau → slow release.
    """
    frames = int(SAMPLE_RATE * duration)
    t      = np.linspace(0, duration, frames, endpoint=False)

    # Harmonic series with piano-like weights (strong fundamental + bright overtones)
    harmonics = [
        (1.0, 1.00),
        (2.0, 0.60),
        (3.0, 0.30),
        (4.0, 0.15),
        (5.0, 0.08),
        (6.0, 0.04),
        (7.0, 0.02),
    ]
    wave = np.zeros(frames)
    for mult, strength in harmonics:
        wave += strength * np.sin(2 * np.pi * freq * mult * t)
    wave /= np.max(np.abs(wave))   # normalise before envelope

    # ADSR envelope
    attack       = int(SAMPLE_RATE * 0.003)   # 3 ms  — hammer hit
    decay        = int(SAMPLE_RATE * 0.15)    # 150 ms — string settling
    release      = int(SAMPLE_RATE * 0.5)     # 500 ms — pedal release tail
    sustain_lvl  = 0.35
    sustain_end  = max(attack + decay, frames - release)

    env = np.ones(frames)
    env[:attack]                   = np.linspace(0, 1, attack)
    env[attack:attack + decay]     = np.linspace(1, sustain_lvl, decay)
    env[attack + decay:sustain_end]= sustain_lvl
    env[sustain_end:]              = np.linspace(sustain_lvl, 0, frames - sustain_end)

    wave = wave * env * volume
    return _to_sound(wave)


# ── GUITAR notes (left hand) — E minor pentatonic ────────────────────────────
G_SWIPE       = karplus_strong(330.0, duration=3.0, damping=0.997)  # E4
G_HOLD        = karplus_strong(247.0, duration=4.0, damping=0.998)  # B3
G_PINCH_PINKY = karplus_strong(196.0, duration=3.0, damping=0.996)  # G3  — pinky+thumb
G_PINCH_INDEX = karplus_strong(294.0, duration=3.0, damping=0.997)  # D4  — index+thumb

# ── PIANO notes (right hand) — same scale, one octave up ─────────────────────
P_SWIPE       = piano_tone(330.0, duration=3.0)   # E4
P_HOLD        = piano_tone(261.63, duration=4.0)  # C4 (middle C)
P_PINCH_PINKY = piano_tone(392.0, duration=3.0)   # G4  — pinky+thumb
P_PINCH_INDEX = piano_tone(523.25, duration=3.0)  # C5  — index+thumb

# Channels: 0-1 guitar swipe/hold, 2-3 guitar pinches, 4-5 piano swipe/hold, 6-7 piano pinches
ch_g_swipe        = pygame.mixer.Channel(0)
ch_g_hold         = pygame.mixer.Channel(1)
ch_g_pinch_pinky  = pygame.mixer.Channel(2)
ch_g_pinch_index  = pygame.mixer.Channel(3)
ch_p_swipe        = pygame.mixer.Channel(4)
ch_p_hold         = pygame.mixer.Channel(5)
ch_p_pinch_pinky  = pygame.mixer.Channel(6)
ch_p_pinch_index  = pygame.mixer.Channel(7)

def fade_all(ms: int = 300):
    for ch in [ch_g_swipe, ch_g_hold, ch_g_pinch_pinky, ch_g_pinch_index,
               ch_p_swipe, ch_p_hold, ch_p_pinch_pinky, ch_p_pinch_index]:
        ch.fadeout(ms)


# ──────────────────────────────────────────────────────────────────────────────
# HAND CONNECTIONS (for cv2 drawing)
# ──────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Guitar = green skeleton, Piano = blue skeleton
COLOR_GUITAR = (0, 220, 100)
COLOR_PIANO  = (100, 180, 255)


def draw_hand(frame, landmarks, width, height, color):
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# GESTURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def fingers_up(lm) -> list:
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    up = [lm[tips[0]].x < lm[pips[0]].x]   # thumb uses x-axis
    for tip, pip in zip(tips[1:], pips[1:]):
        up.append(lm[tip].y < lm[pip].y)
    return up  # [thumb, index, middle, ring, pinky]

def is_two_fingers(lm) -> bool:
    up = fingers_up(lm)
    return up[1] and up[2] and not up[3] and not up[4]

def is_open_palm(lm) -> bool:
    up = fingers_up(lm)
    return up[1] and up[2] and up[3] and up[4]

def is_pinch(lm, tip_a: int, tip_b: int, threshold: float = 0.05) -> bool:
    dx = lm[tip_a].x - lm[tip_b].x
    dy = lm[tip_a].y - lm[tip_b].y
    return (dx**2 + dy**2) ** 0.5 < threshold

def palm_center_x(lm) -> float:
    return (lm[0].x + lm[5].x + lm[17].x) / 3.0


# ──────────────────────────────────────────────────────────────────────────────
# SWIPE DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class SwipeDetector:
    def __init__(self, threshold: float = 0.10, cooldown: float = 0.7):
        self.prev_x       : dict  = {}
        self.last_trigger : float = 0.0
        self.threshold            = threshold
        self.cooldown             = cooldown

    def update(self, hand_id: int, x: float) -> bool:
        now = time.time()
        triggered = False
        if hand_id in self.prev_x:
            delta = x - self.prev_x[hand_id]
            if delta > self.threshold and (now - self.last_trigger) > self.cooldown:
                triggered         = True
                self.last_trigger = now
        self.prev_x[hand_id] = x
        return triggered

    def remove(self, hand_id: int):
        self.prev_x.pop(hand_id, None)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ensure_model()

    options = HandLandmarkerOptions(
        base_options    = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode    = RunningMode.VIDEO,
        num_hands       = 2,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence  = 0.6,
        min_tracking_confidence       = 0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open webcam on index 0 or 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    swipe             = SwipeDetector(threshold=0.10, cooldown=0.7)
    guitar_hold_on    = False   # is guitar two-finger sustain active?
    piano_hold_on     = False   # is piano two-finger sustain active?
    palm_was_up       = False
    last_pinch_g_pinky = 0.0
    last_pinch_g_index = 0.0
    last_pinch_p_pinky = 0.0
    last_pinch_p_index = 0.0
    PINCH_COOLDOWN     = 0.7
    status_text       = ""
    status_until      = 0.0
    frame_idx         = 0

    # print("\nHand Gesture Music")
    # print("  LEFT  hand = Guitar (green skeleton)")
    # print("  RIGHT hand = Piano  (blue skeleton)")
    # print()
    # print("  Swipe Right      ->  pluck note")
    # print("  Two Fingers held ->  sustain note")
    # print("  Pinky+Thumb      ->  pinch note (G)")
    # print("  Index+Thumb      ->  pinch note (D/C5)")
    # print("  Open Palm        ->  fade everything out")
    print("  Q                ->  quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        frame_idx += 1

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = __import__("mediapipe").Image(
            image_format=__import__("mediapipe").ImageFormat.SRGB,
            data=rgb
        )
        timestamp_ms = int(frame_idx * (1000 / 30))
        result: HandLandmarkerResult = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Per-frame state
        guitar_two_now = False
        piano_two_now  = False
        open_palm_now  = False

        if result.hand_landmarks:
            seen_ids = set()

            for i, (hand_lm, handedness_list) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                # MediaPipe returns "Left"/"Right" from its perspective.
                # Because we flip the frame, we swap them so they match
                # the user's actual left/right hand.
                raw_side = handedness_list[0].category_name
                side     = "Right" if raw_side == "Left" else "Left"
                is_left  = (side == "Left")
                seen_ids.add(i)

                color = COLOR_GUITAR if is_left else COLOR_PIANO
                draw_hand(frame, hand_lm, w, h, color)

                cx = palm_center_x(hand_lm)

                # ── SWIPE RIGHT ───────────────────────────────────────────────
                if swipe.update(i, cx):
                    if is_left:
                        ch_g_swipe.play(G_SWIPE)
                        status_text = "Guitar: Swipe -> E4"
                    else:
                        ch_p_swipe.play(P_SWIPE)
                        status_text = "Piano:  Swipe -> E4"
                    status_until = time.time() + 1.5
                    print(f"[{side}] Swipe -> {'Guitar' if is_left else 'Piano'} E4")

                # ── TWO FINGERS ───────────────────────────────────────────────
                if is_two_fingers(hand_lm):
                    if is_left:
                        guitar_two_now = True
                    else:
                        piano_two_now = True

                # ── OPEN PALM ─────────────────────────────────────────────────
                if is_open_palm(hand_lm):
                    open_palm_now = True

                # ── PINKY-THUMB PINCH (landmarks 4 + 20) ─────────────────────
                if is_pinch(hand_lm, 4, 20):
                    now = time.time()
                    if is_left and (now - last_pinch_g_pinky) > PINCH_COOLDOWN:
                        ch_g_pinch_pinky.play(G_PINCH_PINKY)
                        last_pinch_g_pinky = now
                        status_text  = "Guitar: Pinky+Thumb -> G3"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Pinky+Thumb -> Guitar G3")
                    elif not is_left and (now - last_pinch_p_pinky) > PINCH_COOLDOWN:
                        ch_p_pinch_pinky.play(P_PINCH_PINKY)
                        last_pinch_p_pinky = now
                        status_text  = "Piano:  Pinky+Thumb -> G4"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Pinky+Thumb -> Piano G4")

                # ── INDEX-THUMB PINCH (landmarks 4 + 8) ──────────────────────
                if is_pinch(hand_lm, 4, 8):
                    now = time.time()
                    if is_left and (now - last_pinch_g_index) > PINCH_COOLDOWN:
                        ch_g_pinch_index.play(G_PINCH_INDEX)
                        last_pinch_g_index = now
                        status_text  = "Guitar: Index+Thumb -> D4"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Index+Thumb -> Guitar D4")
                    elif not is_left and (now - last_pinch_p_index) > PINCH_COOLDOWN:
                        ch_p_pinch_index.play(P_PINCH_INDEX)
                        last_pinch_p_index = now
                        status_text  = "Piano:  Index+Thumb -> C5"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Index+Thumb -> Piano C5")

            for gone in list(swipe.prev_x.keys()):
                if gone not in seen_ids:
                    swipe.remove(gone)
        else:
            for k in list(swipe.prev_x.keys()):
                swipe.remove(k)

        if guitar_two_now and not guitar_hold_on:
            ch_g_hold.play(G_HOLD, loops=-1)
            guitar_hold_on = True
            status_text    = "Guitar: Two fingers -> B3 sustaining"
            status_until   = time.time() + 999
            print("[Left] Two fingers -> Guitar B3 sustaining")
        elif not guitar_two_now and guitar_hold_on:
            ch_g_hold.fadeout(250)
            guitar_hold_on = False
            status_text    = "Guitar hold released"
            status_until   = time.time() + 0.8
            print("[Left] Guitar B3 released")

        if piano_two_now and not piano_hold_on:
            ch_p_hold.play(P_HOLD, loops=-1)
            piano_hold_on = True
            status_text   = "Piano:  Two fingers -> C4 sustaining"
            status_until  = time.time() + 999
            print("[Right] Two fingers -> Piano C4 sustaining")
        elif not piano_two_now and piano_hold_on:
            ch_p_hold.fadeout(250)
            piano_hold_on = False
            status_text   = "Piano hold released"
            status_until  = time.time() + 0.8
            print("[Right] Piano C4 released")

        if open_palm_now and not palm_was_up:
            fade_all(ms=300)
            guitar_hold_on = False
            piano_hold_on  = False
            status_text    = "Open Palm -> Fade Out"
            status_until   = time.time() + 1.0
            print("Open Palm -> Fading all sound")
        palm_was_up = open_palm_now

        # ── HUD ───────────────────────────────────────────────────────────────
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 60), (10, 10, 10), -1)
        cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

        # Left label (guitar)
        cv2.putText(frame, "LEFT = Guitar", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GUITAR, 1, cv2.LINE_AA)
        # Right label (piano)
        cv2.putText(frame, "RIGHT = Piano", (w - 155, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PIANO, 1, cv2.LINE_AA)
        cv2.putText(frame, "Swipe | Two Fingers | Pinky+Thumb | Palm = stop | Q = quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1, cv2.LINE_AA)

        # Status text at bottom
        if time.time() < status_until and status_text:
            cv2.putText(frame, status_text, (10, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 180), 2, cv2.LINE_AA)

        # Sustain indicator dots (top corners)
        g_dot = COLOR_GUITAR if guitar_hold_on else (40, 40, 40)
        p_dot = COLOR_PIANO  if piano_hold_on  else (40, 40, 40)
        cv2.circle(frame, (18, 50), 8, g_dot, -1, cv2.LINE_AA)
        cv2.circle(frame, (w - 18, 50), 8, p_dot, -1, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Music", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("\nShutting down...")
    fade_all(500)
    pygame.time.wait(600)
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    pygame.mixer.quit()
    print("Done.")


if __name__ == "__main__":
    main()