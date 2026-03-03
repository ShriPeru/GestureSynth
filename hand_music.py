"""
hand_music.py — GestureSynth
==============================
Real-time hand gesture music player using MediaPipe + Pygame.

LEFT  hand (green skeleton) = Guitar sounds (Karplus-Strong synthesis)
RIGHT hand (blue skeleton)  = Piano sounds  (Additive harmonic synthesis)

Gesture Controls:
  Swipe Right        → Pluck/play a note
  Two Fingers held   → Sustain a note (loops until fingers drop)
  Pinky + Thumb      → Pinch note (G3 guitar / G4 piano)
  Index + Thumb      → Pinch note (D4 guitar / C5 piano)
  Open Palm          → Fade out all sounds
  Q key              → Quit
"""

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


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# The MediaPipe hand landmark model file is required to detect hand positions.
# If it doesn't exist locally, we download it automatically (~5 MB, one-time).
# ──────────────────────────────────────────────────────────────────────────────

# Build the path to the model file (same folder as this script)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Official Google MediaPipe model download URL
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    """Download the hand landmark model if it's not already present locally."""
    if os.path.exists(MODEL_PATH):
        return  # Already downloaded, nothing to do
    print("Downloading hand landmark model (~5 MB) — one-time only...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}\n")
    except Exception as e:
        print(f"\nERROR: Could not download model: {e}")
        print("Check your internet connection, then try again.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO SETUP
# Initialize pygame's mixer for low-latency audio playback.
# frequency=44100 → CD-quality sample rate (44,100 samples per second)
# size=-16        → 16-bit signed audio (standard for most audio)
# channels=2      → Stereo output (left + right speaker)
# buffer=512      → Small buffer = lower latency (less delay between gesture and sound)
# ──────────────────────────────────────────────────────────────────────────────

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
SAMPLE_RATE = 44100  # Must match the mixer frequency above


def _to_sound(wave: np.ndarray) -> pygame.mixer.Sound:
    """
    Convert a float64 numpy wave array into a pygame Sound object.

    Steps:
      1. Scale the float values (-1.0 to 1.0) to 16-bit integers (-32767 to 32767)
      2. Duplicate mono channel into stereo (left + right are identical)
      3. Wrap in pygame's Sound object for playback
    """
    mono   = (wave * 32767).astype(np.int16)         # Convert float → 16-bit int
    stereo = np.column_stack((mono, mono))            # Stack into [L, R] stereo pairs
    return pygame.sndarray.make_sound(stereo)         # Create pygame Sound


def karplus_strong(freq: float, duration: float = 3.0,
                   volume: float = 0.7, damping: float = 0.996) -> pygame.mixer.Sound:
    """
    Karplus-Strong plucked string synthesis — sounds like a guitar or banjo.

    How it works:
      1. Fill a short ring buffer (length = samples per one period at `freq`) with random noise.
         This simulates the chaotic initial pluck of a string.
      2. Each output sample is taken from the front of the buffer.
      3. The buffer is updated by averaging adjacent samples and multiplying by a damping factor.
         This averaging smooths out the noise over time, leaving only the fundamental frequency
         and its harmonics — just like how a real string vibrates.
      4. Damping < 1.0 causes the sound to decay naturally (like a real string losing energy).

    Parameters:
      freq     - Desired pitch in Hz (e.g. 330.0 = E4)
      duration - How many seconds of audio to generate
      volume   - Peak amplitude (0.0 to 1.0)
      damping  - How quickly the note fades (0.99 = fast decay, 0.999 = long ring)
    """
    n_samples = int(SAMPLE_RATE * duration)           # Total number of audio samples to generate
    buf_size  = max(2, int(SAMPLE_RATE / freq))       # Ring buffer size = samples per one wave cycle
    buf = np.random.uniform(-1.0, 1.0, buf_size)      # Seed buffer with white noise (the "pluck")
    out = np.zeros(n_samples)                         # Output array (will be filled sample by sample)

    for i in range(n_samples):
        out[i] = buf[0]                               # Take the front sample as output
        buf = np.roll(buf, -1)                        # Shift buffer left by 1
        # Update last element: average of current + next sample, scaled by damping
        # This is the core of Karplus-Strong — it's what creates the natural pitch + decay
        buf[-1] = damping * 0.5 * (out[i] + buf[0])

    # Normalize to prevent clipping, then apply volume
    peak = np.max(np.abs(out))
    if peak > 0:
        out = (out / peak) * volume

    return _to_sound(out)


def piano_tone(freq: float, duration: float = 3.0, volume: float = 0.6) -> pygame.mixer.Sound:
    """
    Piano simulation using additive harmonic synthesis + ADSR envelope.

    How it works:
      1. ADDITIVE SYNTHESIS: Layer multiple sine waves at harmonic multiples of the
         fundamental frequency. Real piano strings vibrate at the fundamental + overtones.
         We weight them to mimic the bright, slightly metallic piano timbre.

      2. ADSR ENVELOPE: Shape the volume over time to simulate a real piano key press:
           Attack  (3ms)   — Near-instant rise, like a hammer hitting a string
           Decay   (150ms) — Quick drop from peak, string settling into vibration
           Sustain (flat)  — Held note plateau (key held down)
           Release (500ms) — Gradual fade-out (key released, damper touching string)

    Parameters:
      freq     - Desired pitch in Hz (e.g. 261.63 = Middle C / C4)
      duration - How many seconds of audio to generate
      volume   - Peak amplitude (0.0 to 1.0)
    """
    frames = int(SAMPLE_RATE * duration)               # Total audio samples
    t      = np.linspace(0, duration, frames, endpoint=False)  # Time axis in seconds

    # Harmonic series: (frequency_multiplier, relative_strength)
    # Higher harmonics are progressively quieter — mimics real piano string physics
    harmonics = [
        (1.0, 1.00),   # Fundamental — loudest
        (2.0, 0.60),   # 2nd harmonic (octave above)
        (3.0, 0.30),   # 3rd harmonic
        (4.0, 0.15),   # 4th harmonic
        (5.0, 0.08),   # 5th harmonic
        (6.0, 0.04),   # 6th harmonic
        (7.0, 0.02),   # 7th harmonic — barely audible, adds "air"
    ]

    # Sum all harmonic sine waves together
    wave = np.zeros(frames)
    for mult, strength in harmonics:
        wave += strength * np.sin(2 * np.pi * freq * mult * t)

    wave /= np.max(np.abs(wave))   # Normalize before applying envelope

    # ── ADSR Envelope ─────────────────────────────────────────────────────────
    attack       = int(SAMPLE_RATE * 0.003)   # 3 ms  — hammer hit
    decay        = int(SAMPLE_RATE * 0.15)    # 150 ms — string settling into sustain
    release      = int(SAMPLE_RATE * 0.5)     # 500 ms — pedal release / key lift tail
    sustain_lvl  = 0.35                        # Sustain volume level (35% of peak)
    sustain_end  = max(attack + decay, frames - release)  # Where release phase begins

    env = np.ones(frames)
    env[:attack]                    = np.linspace(0, 1, attack)           # Attack ramp up
    env[attack:attack + decay]      = np.linspace(1, sustain_lvl, decay)  # Decay ramp down
    env[attack + decay:sustain_end] = sustain_lvl                         # Flat sustain
    env[sustain_end:]               = np.linspace(sustain_lvl, 0, frames - sustain_end)  # Release fade

    wave = wave * env * volume     # Apply envelope and volume to wave
    return _to_sound(wave)


# ──────────────────────────────────────────────────────────────────────────────
# PRE-GENERATE ALL SOUND OBJECTS
# We generate these once at startup (not during the loop) to avoid audio lag.
# Guitar uses Karplus-Strong (plucked string feel).
# Piano uses additive synthesis + ADSR (keyboard feel).
# Notes are from the E minor pentatonic scale: E, G, B, D (great for jamming!)
# ──────────────────────────────────────────────────────────────────────────────

# ── GUITAR notes (left hand) — E minor pentatonic ────────────────────────────
G_SWIPE       = karplus_strong(330.0, duration=3.0, damping=0.997)  # E4 — swipe trigger
G_HOLD        = karplus_strong(247.0, duration=4.0, damping=0.998)  # B3 — two-finger sustain
G_PINCH_PINKY = karplus_strong(196.0, duration=3.0, damping=0.996)  # G3 — pinky+thumb pinch
G_PINCH_INDEX = karplus_strong(294.0, duration=3.0, damping=0.997)  # D4 — index+thumb pinch

# ── PIANO notes (right hand) — same scale, one octave up ─────────────────────
P_SWIPE       = piano_tone(330.0,   duration=3.0)   # E4 — swipe trigger
P_HOLD        = piano_tone(261.63,  duration=4.0)   # C4 (middle C) — two-finger sustain
P_PINCH_PINKY = piano_tone(392.0,   duration=3.0)   # G4 — pinky+thumb pinch
P_PINCH_INDEX = piano_tone(523.25,  duration=3.0)   # C5 — index+thumb pinch

# ──────────────────────────────────────────────────────────────────────────────
# MIXER CHANNELS
# Each sound gets its own dedicated channel so multiple sounds can play
# simultaneously without cutting each other off.
# Channels 0-3 = Guitar, Channels 4-7 = Piano
# ──────────────────────────────────────────────────────────────────────────────
ch_g_swipe        = pygame.mixer.Channel(0)   # Guitar swipe note
ch_g_hold         = pygame.mixer.Channel(1)   # Guitar sustain (loops)
ch_g_pinch_pinky  = pygame.mixer.Channel(2)   # Guitar pinky+thumb note
ch_g_pinch_index  = pygame.mixer.Channel(3)   # Guitar index+thumb note
ch_p_swipe        = pygame.mixer.Channel(4)   # Piano swipe note
ch_p_hold         = pygame.mixer.Channel(5)   # Piano sustain (loops)
ch_p_pinch_pinky  = pygame.mixer.Channel(6)   # Piano pinky+thumb note
ch_p_pinch_index  = pygame.mixer.Channel(7)   # Piano index+thumb note


def fade_all(ms: int = 300):
    """Fade out all 8 channels simultaneously over `ms` milliseconds."""
    for ch in [ch_g_swipe, ch_g_hold, ch_g_pinch_pinky, ch_g_pinch_index,
               ch_p_swipe, ch_p_hold, ch_p_pinch_pinky, ch_p_pinch_index]:
        ch.fadeout(ms)


# ──────────────────────────────────────────────────────────────────────────────
# HAND DRAWING
# MediaPipe gives us 21 landmark points per hand (wrist + finger joints).
# We connect them with lines to draw a skeleton overlay on the webcam feed.
# ──────────────────────────────────────────────────────────────────────────────

# Pairs of landmark indices that should be connected by a line.
# Each tuple (A, B) means: draw a line from landmark A to landmark B.
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index finger
    (0,9),(9,10),(10,11),(11,12),   # Middle finger
    (0,13),(13,14),(14,15),(15,16), # Ring finger
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17),           # Palm knuckle line (connects finger bases)
]

COLOR_GUITAR = (0, 220, 100)    # Green — left hand (guitar)
COLOR_PIANO  = (100, 180, 255)  # Blue  — right hand (piano)


def draw_hand(frame, landmarks, width, height, color):
    """
    Draw the hand skeleton (bones + joint dots) onto the video frame.

    Landmarks are in normalized coordinates (0.0–1.0), so we multiply
    by frame width/height to get pixel positions.
    """
    # Convert normalized landmark coords → pixel coordinates
    pts = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    # Draw bone lines between connected landmarks
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # Draw a small white dot at each joint
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# GESTURE DETECTION HELPERS
# These functions analyze landmark positions to detect specific hand shapes.
# MediaPipe hand landmark indices:
#   4 = Thumb tip,  8 = Index tip,  12 = Middle tip,  16 = Ring tip,  20 = Pinky tip
#   3 = Thumb PIP,  6 = Index PIP, 10 = Middle PIP,  14 = Ring PIP,  18 = Pinky PIP
# (PIP = Proximal Interphalangeal joint — the middle knuckle)
# ──────────────────────────────────────────────────────────────────────────────

def fingers_up(lm) -> list:
    """
    Determine which fingers are currently extended (pointing up).

    For fingers (index–pinky): if the fingertip Y is ABOVE (smaller Y value) the
    middle knuckle (PIP), the finger is considered "up".

    For the thumb: we use the X axis instead, because the thumb extends sideways.
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    """
    tips = [4, 8, 12, 16, 20]   # Fingertip landmark indices
    pips = [3, 6, 10, 14, 18]   # Middle knuckle landmark indices

    # Thumb: extended if tip X is to the LEFT of its knuckle (for right hand, mirrored frame)
    up = [lm[tips[0]].x < lm[pips[0]].x]

    # Other fingers: extended if tip Y is ABOVE (less than) their middle knuckle Y
    for tip, pip in zip(tips[1:], pips[1:]):
        up.append(lm[tip].y < lm[pip].y)

    return up  # [thumb, index, middle, ring, pinky]


def is_two_fingers(lm) -> bool:
    """
    Detect the "two finger" / peace sign gesture.
    Index + middle fingers are up, ring + pinky are curled down.
    Used to trigger a sustaining held note.
    """
    up = fingers_up(lm)
    return up[1] and up[2] and not up[3] and not up[4]


def is_open_palm(lm) -> bool:
    """
    Detect an open flat palm (all four fingers extended).
    Used as a "stop all sounds" gesture — fade everything out.
    Note: thumb state is ignored since it's often ambiguous sideways.
    """
    up = fingers_up(lm)
    return up[1] and up[2] and up[3] and up[4]


def is_pinch(lm, tip_a: int, tip_b: int, threshold: float = 0.05) -> bool:
    """
    Detect whether two fingertips are close enough to be considered "pinching".

    Calculates the Euclidean distance between two landmark points in normalized
    coordinates (0.0–1.0 scale). If the distance is below the threshold, it's a pinch.

    Parameters:
      tip_a, tip_b - Landmark indices of the two fingertips to check
      threshold    - Max distance to count as a pinch (0.05 = ~5% of frame width)
    """
    dx = lm[tip_a].x - lm[tip_b].x
    dy = lm[tip_a].y - lm[tip_b].y
    return (dx**2 + dy**2) ** 0.5 < threshold


def palm_center_x(lm) -> float:
    """
    Calculate the horizontal center of the palm using 3 anchor landmarks:
      0  = Wrist base
      5  = Index finger base knuckle
      17 = Pinky base knuckle
    Averaging these 3 points gives a stable palm center for swipe detection.
    """
    return (lm[0].x + lm[5].x + lm[17].x) / 3.0


# ──────────────────────────────────────────────────────────────────────────────
# SWIPE DETECTOR
# Tracks horizontal palm movement across frames to detect a rightward swipe.
# Uses a cooldown timer to prevent rapid repeated triggers from a single gesture.
# ──────────────────────────────────────────────────────────────────────────────

class SwipeDetector:
    def __init__(self, threshold: float = 0.10, cooldown: float = 0.7):
        """
        Parameters:
          threshold - Minimum rightward movement (in normalized coords) to count as a swipe
                      0.10 = must move 10% of frame width in one frame interval
          cooldown  - Minimum seconds between swipe triggers (prevents double-triggers)
        """
        self.prev_x       : dict  = {}    # Last known X position per hand ID
        self.last_trigger : float = 0.0  # Timestamp of the most recent swipe trigger
        self.threshold            = threshold
        self.cooldown             = cooldown

    def update(self, hand_id: int, x: float) -> bool:
        """
        Update the detector with the current palm X position for a given hand.
        Returns True if a valid rightward swipe was just detected.

        hand_id - Unique index assigned by MediaPipe (0 for first hand, 1 for second)
        x       - Current palm center X in normalized coordinates (0.0 = left, 1.0 = right)
        """
        now = time.time()
        triggered = False

        if hand_id in self.prev_x:
            delta = x - self.prev_x[hand_id]   # Positive = moved right
            # Trigger if movement exceeds threshold AND cooldown has passed
            if delta > self.threshold and (now - self.last_trigger) > self.cooldown:
                triggered         = True
                self.last_trigger = now

        self.prev_x[hand_id] = x   # Update stored position for next frame
        return triggered

    def remove(self, hand_id: int):
        """Remove a hand's tracking data when that hand disappears from the frame."""
        self.prev_x.pop(hand_id, None)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# Initializes the webcam + MediaPipe landmarker, then processes frames in a loop.
# Each frame: detect hands → classify gestures → trigger sounds → draw HUD → display.
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Ensure the MediaPipe model file is present (downloads if missing)
    ensure_model()

    # Configure the MediaPipe Hand Landmarker
    # VIDEO mode processes frames sequentially with timestamps (vs LIVE_STREAM which uses callbacks)
    options = HandLandmarkerOptions(
        base_options    = BaseOptions(model_asset_path=MODEL_PATH),
        running_mode    = RunningMode.VIDEO,    # Synchronous frame-by-frame processing
        num_hands       = 2,                    # Track up to 2 hands simultaneously
        min_hand_detection_confidence = 0.6,   # Minimum confidence to detect a hand initially
        min_hand_presence_confidence  = 0.6,   # Minimum confidence to keep tracking a hand
        min_tracking_confidence       = 0.5,   # Minimum confidence for landmark tracking
    )
    landmarker = HandLandmarker.create_from_options(options)

    # Open webcam (try index 0 first, fall back to 1 for external webcams)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Could not open webcam on index 0 or 1.")
        return

    # Set webcam resolution (640x480 is a good balance of quality and speed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize swipe detector
    swipe = SwipeDetector(threshold=0.10, cooldown=0.7)

    # Track whether hold (sustain) notes are currently active
    guitar_hold_on = False   # True when guitar two-finger sustain is playing
    piano_hold_on  = False   # True when piano two-finger sustain is playing
    palm_was_up    = False   # Used for edge detection: only trigger fade on palm raise (not hold)

    # Timestamps of last pinch triggers (used for per-gesture cooldowns)
    last_pinch_g_pinky = 0.0
    last_pinch_g_index = 0.0
    last_pinch_p_pinky = 0.0
    last_pinch_p_index = 0.0
    PINCH_COOLDOWN     = 0.7  # Seconds between allowed pinch triggers

    # HUD status message (shown at bottom of screen)
    status_text  = ""
    status_until = 0.0   # Time until status message disappears

    # Frame counter used to generate timestamps for MediaPipe (assumes ~30 FPS)
    frame_idx = 0

    print("  Q  →  quit\n")

    # ── MAIN PROCESSING LOOP ──────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            continue   # Skip frame if webcam read failed

        # Mirror the frame horizontally so it acts like a mirror (selfie-view)
        # This is important: we also swap Left/Right labels below to match
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        frame_idx += 1

        # Convert BGR (OpenCV default) → RGB (MediaPipe requirement)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the numpy array in a MediaPipe Image object
        mp_image = __import__("mediapipe").Image(
            image_format=__import__("mediapipe").ImageFormat.SRGB,
            data=rgb
        )

        # Create a timestamp in milliseconds (MediaPipe VIDEO mode requires monotonically increasing timestamps)
        timestamp_ms = int(frame_idx * (1000 / 30))

        # Run hand landmark detection on this frame
        result: HandLandmarkerResult = landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── Per-frame gesture flags (reset each frame) ─────────────────────────
        guitar_two_now = False   # Is the guitar two-finger gesture active THIS frame?
        piano_two_now  = False   # Is the piano two-finger gesture active THIS frame?
        open_palm_now  = False   # Is an open palm detected THIS frame?

        # ── Process detected hands ─────────────────────────────────────────────
        if result.hand_landmarks:
            seen_ids = set()  # Track which hand IDs are visible this frame

            for i, (hand_lm, handedness_list) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                # MediaPipe labels hands from its own perspective (camera-facing).
                # Since we flipped the frame, we swap Left/Right to match the user's view.
                raw_side = handedness_list[0].category_name          # MediaPipe's label
                side     = "Right" if raw_side == "Left" else "Left" # Corrected label
                is_left  = (side == "Left")
                seen_ids.add(i)

                # Draw the hand skeleton in the appropriate color
                color = COLOR_GUITAR if is_left else COLOR_PIANO
                draw_hand(frame, hand_lm, w, h, color)

                # Get the palm's horizontal center position (used for swipe detection)
                cx = palm_center_x(hand_lm)

                # ── GESTURE: SWIPE RIGHT ───────────────────────────────────────
                # Detected when palm moves quickly to the right in one frame interval
                if swipe.update(i, cx):
                    if is_left:
                        ch_g_swipe.play(G_SWIPE)
                        status_text = "Guitar: Swipe → E4"
                    else:
                        ch_p_swipe.play(P_SWIPE)
                        status_text = "Piano:  Swipe → E4"
                    status_until = time.time() + 1.5
                    print(f"[{side}] Swipe → {'Guitar' if is_left else 'Piano'} E4")

                # ── GESTURE: TWO FINGERS (sustain) ─────────────────────────────
                # Set flag if two-finger pose detected — sound logic handled after loop
                if is_two_fingers(hand_lm):
                    if is_left:
                        guitar_two_now = True
                    else:
                        piano_two_now = True

                # ── GESTURE: OPEN PALM (stop all) ──────────────────────────────
                # Set flag — actual fadeout handled after loop (edge detection)
                if is_open_palm(hand_lm):
                    open_palm_now = True

                # ── GESTURE: PINKY + THUMB PINCH (landmarks 4=thumb tip, 20=pinky tip) ──
                if is_pinch(hand_lm, 4, 20):
                    now = time.time()
                    if is_left and (now - last_pinch_g_pinky) > PINCH_COOLDOWN:
                        ch_g_pinch_pinky.play(G_PINCH_PINKY)
                        last_pinch_g_pinky = now
                        status_text  = "Guitar: Pinky+Thumb → G3"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Pinky+Thumb → Guitar G3")
                    elif not is_left and (now - last_pinch_p_pinky) > PINCH_COOLDOWN:
                        ch_p_pinch_pinky.play(P_PINCH_PINKY)
                        last_pinch_p_pinky = now
                        status_text  = "Piano:  Pinky+Thumb → G4"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Pinky+Thumb → Piano G4")

                # ── GESTURE: INDEX + THUMB PINCH (landmarks 4=thumb tip, 8=index tip) ──
                if is_pinch(hand_lm, 4, 8):
                    now = time.time()
                    if is_left and (now - last_pinch_g_index) > PINCH_COOLDOWN:
                        ch_g_pinch_index.play(G_PINCH_INDEX)
                        last_pinch_g_index = now
                        status_text  = "Guitar: Index+Thumb → D4"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Index+Thumb → Guitar D4")
                    elif not is_left and (now - last_pinch_p_index) > PINCH_COOLDOWN:
                        ch_p_pinch_index.play(P_PINCH_INDEX)
                        last_pinch_p_index = now
                        status_text  = "Piano:  Index+Thumb → C5"
                        status_until = time.time() + 1.5
                        print(f"[{side}] Index+Thumb → Piano C5")

            # Clean up tracking data for hands that left the frame
            for gone in list(swipe.prev_x.keys()):
                if gone not in seen_ids:
                    swipe.remove(gone)
        else:
            # No hands detected — clear all swipe tracking
            for k in list(swipe.prev_x.keys()):
                swipe.remove(k)

        # ── SUSTAIN LOGIC (edge detection: only trigger on state CHANGE) ───────
        # Guitar: start sustain when two-finger gesture begins, stop when it ends
        if guitar_two_now and not guitar_hold_on:
            ch_g_hold.play(G_HOLD, loops=-1)   # loops=-1 means loop forever
            guitar_hold_on = True
            status_text    = "Guitar: Two fingers → B3 sustaining"
            status_until   = time.time() + 999  # Show until manually cleared
            print("[Left] Two fingers → Guitar B3 sustaining")
        elif not guitar_two_now and guitar_hold_on:
            ch_g_hold.fadeout(250)             # Smooth 250ms fade instead of hard stop
            guitar_hold_on = False
            status_text    = "Guitar hold released"
            status_until   = time.time() + 0.8
            print("[Left] Guitar B3 released")

        # Piano: same sustain logic as guitar
        if piano_two_now and not piano_hold_on:
            ch_p_hold.play(P_HOLD, loops=-1)
            piano_hold_on = True
            status_text   = "Piano:  Two fingers → C4 sustaining"
            status_until  = time.time() + 999
            print("[Right] Two fingers → Piano C4 sustaining")
        elif not piano_two_now and piano_hold_on:
            ch_p_hold.fadeout(250)
            piano_hold_on = False
            status_text   = "Piano hold released"
            status_until  = time.time() + 0.8
            print("[Right] Piano C4 released")

        # ── OPEN PALM: FADE ALL (only triggers on initial raise, not while held) ──
        if open_palm_now and not palm_was_up:
            fade_all(ms=300)          # Fade all 8 channels over 300ms
            guitar_hold_on = False    # Reset hold states since all sounds are stopping
            piano_hold_on  = False
            status_text    = "Open Palm → Fade Out"
            status_until   = time.time() + 1.0
            print("Open Palm → Fading all sound")
        palm_was_up = open_palm_now   # Store current state for next frame edge detection

        # ── HUD (Heads-Up Display) ─────────────────────────────────────────────
        # Draw a semi-transparent dark bar at the top for labels
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 60), (10, 10, 10), -1)
        cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)  # Blend bar onto frame

        # Hand role labels (top-left = guitar, top-right = piano)
        cv2.putText(frame, "LEFT = Guitar", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GUITAR, 1, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT = Piano", (w - 155, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PIANO, 1, cv2.LINE_AA)

        # Gesture cheat-sheet (small text below labels)
        cv2.putText(frame, "Swipe | Two Fingers | Pinky+Thumb | Palm = stop | Q = quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1, cv2.LINE_AA)

        # Status message at the bottom (shows last triggered gesture)
        if time.time() < status_until and status_text:
            cv2.putText(frame, status_text, (10, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 180), 2, cv2.LINE_AA)

        # Sustain indicator dots in the top corners (lit = sustain active, dark = off)
        g_dot = COLOR_GUITAR if guitar_hold_on else (40, 40, 40)
        p_dot = COLOR_PIANO  if piano_hold_on  else (40, 40, 40)
        cv2.circle(frame, (18, 50), 8, g_dot, -1, cv2.LINE_AA)      # Guitar dot (top-left)
        cv2.circle(frame, (w - 18, 50), 8, p_dot, -1, cv2.LINE_AA)  # Piano dot (top-right)

        # Show the frame in the window
        cv2.imshow("Hand Gesture Music", frame)

        # Exit the loop if 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    print("\nShutting down...")
    fade_all(500)                  # Gracefully fade out all audio
    pygame.time.wait(600)          # Wait for fade to complete before closing mixer
    cap.release()                  # Release webcam
    cv2.destroyAllWindows()        # Close OpenCV windows
    landmarker.close()             # Release MediaPipe resources
    pygame.mixer.quit()            # Shut down audio mixer
    print("Done.")


# Only run main() if this script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()