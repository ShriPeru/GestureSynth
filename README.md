# 🎵 GestureSynth — Hand Gesture Music

Play guitar and piano sounds in real-time using just your hands and a webcam.  
Your **left hand** controls a guitar (green skeleton), your **right hand** controls a piano (blue skeleton).

---

##  Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed
- [VS Code](https://code.visualstudio.com/) installed
- A working webcam

---

##  Step 1 — Set Up VS Code

1. Open **VS Code**
2. Install the **Python extension**:
   - Go to Extensions (`Ctrl+Shift+X`)
   - Search **"Python"** → Install the one by Microsoft
3. Install the **Pylance extension** (optional but recommended for autocomplete)

---

##  Step 2 — Create the Conda Environment

Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
conda create -n gesture python=3.10
conda activate gesture
```

---

##  Step 3 — Install Dependencies

With the `gesture` environment active, install all required libraries:

```bash
pip install mediapipe opencv-python pygame numpy
```

>  If you're on Windows and get audio errors, also run:
> ```bash
> pip install sounddevice
> ```

---

##  Step 4 — Set Up the Project Folder

1. Clone or download this repo into a folder, e.g. `H:\College\Junior\GradClassGesture`
2. Make sure these two files are in the same folder:
   - `hand_music.py`
   - `hand_landmarker.task` *(will auto-download on first run if missing)*

---

##  Step 5 — Open the Project in VS Code

1. Open VS Code → **File > Open Folder** → select your project folder
2. Open the **Command Palette** (`Ctrl+Shift+P`)
3. Type `Python: Select Interpreter`
4. Choose the `gesture` conda environment (should show `Python 3.10 (gesture)`)

---

##  Step 6 — Run the Program

In the VS Code terminal (make sure `gesture` env is active):

```bash
conda activate gesture
python hand_music.py
```

A webcam window will open titled **"Hand Gesture Music"**.  
> On first run, the model (`hand_landmarker.task`) will auto-download (~5 MB).

---

## Gesture Controls

| Gesture | Hand | Sound |
|---|---|---|
| **Swipe Right** | Left (Guitar) | Pluck — E4 |
| **Swipe Right** | Right (Piano) | Piano — E4 |
| **Two Fingers held** | Left (Guitar) | Sustain — B3 |
| **Two Fingers held** | Right (Piano) | Sustain — C4 (Middle C) |
| **Pinky + Thumb pinch** | Left (Guitar) | Pluck — G3 |
| **Pinky + Thumb pinch** | Right (Piano) | Piano — G4 |
| **Index + Thumb pinch** | Left (Guitar) | Pluck — D4 |
| **Index + Thumb pinch** | Right (Piano) | Piano — C5 |
| **Open Palm** | Either | Fade all sounds out |
| **Q key** | — | Quit the program |

---

##  Troubleshooting

**Webcam not detected?**  
The program tries index `0` then `1`. Make sure no other app is using your webcam.

**Model not downloading?**  
Manually download it from:  
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`  
and place it in the same folder as `hand_music.py`.

**Wrong hand labeled?**  
The frame is mirrored (like a selfie), so your physical left hand appears on the left side of the screen and is labeled Guitar (green).

---

##  Full Dependency List

| Package | Version |
|---|---|
| Python | 3.10 |
| mediapipe | 0.10.32 |
| opencv-python | 4.13.0.92 |
| pygame | 2.6.1 |
| numpy | 2.2.6 |
| sounddevice | 0.5.5 |

---

## 👤 Author

**ShriPeru** — GradClassGesture Project  
[github.com/ShriPeru/GestureSynth](https://github.com/ShriPeru/GestureSynth)
