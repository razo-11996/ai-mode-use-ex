# Camera + Audio AI Demos (Beginner-Friendly)

This folder contains two small beginner-friendly demos:

- `whisper-math.py`: **Microphone audio → Whisper transcription → optional simple math**
- `fingers-count.py`: **Webcam video → MediaPipe hand landmarks → finger counting**

## Setup (for both scripts)

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- Whisper requires **PyTorch**. On macOS it will typically be installed automatically via `pip`.
- If installation fails, install `torch` first (Apple Silicon often works best with the official PyTorch macOS wheels).

## Demo 1: `whisper-math.py` (Voice → Text → Math)

### What it does

- **Records** audio from your microphone (default: 5 seconds)
- **Transcribes** speech to text with Whisper
- **Optionally** parses simple expressions like `2 + 2` and prints the result

### Run

Transcribe continuously (Ctrl+C to stop):

```bash
python3 whisper-math.py
```

Record/transcribe once:

```bash
python3 whisper-math.py --once
```

Change recording length:

```bash
python3 whisper-math.py --seconds 3
```

List audio devices and select an input device:

```bash
python3 whisper-math.py --list-devices
python3 whisper-math.py --device 1
```

Disable math evaluation (transcribe only):

```bash
python3 whisper-math.py --no-math
```

### Math support (intentionally simple)

The math parser looks for one pattern like:

```
<number> <operator> <number>
```

Supported operators:
- `plus` → `+`
- `minus` → `-`
- `times` / `multiply` → `*`
- `divided by` / `divide` → `/`
- `to the power of` / `power` → `**`

Tip: Saying digits (e.g. **"2 plus 2"**) is usually more reliable than number words (e.g. "two plus two").

## Demo 2: `fingers-count.py` (Webcam → Hand Landmarks → Finger Count)

### What it does

- Opens your **webcam** (OpenCV)
- Detects one hand using **MediaPipe**
- Draws the hand landmarks on the frame
- Shows an on-screen counter like `Fingers: 3`

### Run

```bash
python3 fingers-count.py
```

Controls:
- Press **q** to quit.

macOS note: the first time you run it, macOS may ask for **camera permission**.

### How finger counting works (simple heuristic)

MediaPipe gives you 21 hand landmark points (wrist, joints, fingertips).
The script counts “open” fingers like this:

- **Thumb**: checks if the thumb tip is to the right of its previous joint (X-axis comparison).
- **Other fingers**: checks if each fingertip is above its middle joint (Y-axis comparison).

This is a quick heuristic for a front-facing camera view (it won’t be perfect for all hand rotations).

### MediaPipe version support

The script tries to work with both:

- **Solutions API** (`mp.solutions.hands`) if available
- **Tasks API** (`mp.tasks...`) otherwise (it may download `hand_landmarker.task` into `~/.mediapipe/models`)

