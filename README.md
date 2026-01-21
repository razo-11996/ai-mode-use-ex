# Whisper Voice Math (Beginner-Friendly)

Record a short audio clip from your microphone, transcribe it with **OpenAI Whisper**, and (optionally) evaluate **simple math** you said out loud.

## What this project does

- **Records** audio from your microphone (default: 5 seconds)
- **Transcribes** speech to text with Whisper
- **Optionally** parses simple expressions like `2 + 2` and prints the result

## Setup

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

## Run

Transcribe continuously (Ctrl+C to stop):

```bash
python whisper-math.py
```

Record/transcribe once:

```bash
python whisper-math.py --once
```

Change recording length:

```bash
python whisper-math.py --seconds 3
```

List audio devices and select an input device:

```bash
python whisper-math.py --list-devices
python whisper-math.py --device 1
```

Disable math evaluation (transcribe only):

```bash
python whisper-math.py --no-math
```

## Math support (intentionally simple)

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

