"""
Voice -> Text -> (Optional) Math.

This script records audio from your microphone, transcribes it with OpenAI Whisper,
and tries to evaluate *simple* math expressions you say out loud.

Examples that usually work well:
- "2 plus 2"
- "10 divided by 4"
- "3 to the power of 2"

Tip: Whisper understands digits ("2") more reliably than number words ("two").
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import numpy as np


Number = Union[int, float]


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16000
    record_seconds: float = 5.0
    whisper_model: str = "base"
    device: Optional[int] = None
    sleep_seconds: float = 0.2
    fp16: bool = False  # set True only if you have a supported GPU


def list_audio_devices() -> None:
    """Print available audio input/output devices."""
    import sounddevice as sd

    print(sd.query_devices())


def record_microphone_audio(settings: Settings) -> "np.ndarray":
    """
    Record audio from the default (or chosen) microphone.

    Returns a 1D numpy array of float32 samples in range [-1, 1].
    """
    import numpy as np
    import sounddevice as sd

    frames = int(settings.record_seconds * settings.sample_rate)
    audio = sd.rec(
        frames,
        samplerate=settings.sample_rate,
        channels=1,
        dtype="float32",
        device=settings.device,
    )
    sd.wait()
    return np.squeeze(audio)


def save_wav_tempfile(audio: "np.ndarray", sample_rate: int) -> str:
    """
    Save audio to a temporary .wav file and return its path.

    Whisper's high-level API (`model.transcribe`) expects a file path,
    so we write a short WAV clip to disk.
    """
    import numpy as np
    from scipy.io.wavfile import write

    # scipy.io.wavfile.write supports float32, but int16 WAV is the most compatible.
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)  # close the OS handle; scipy will write to the path
    write(path, sample_rate, audio_int16)
    return path


def transcribe_wav(model: Any, wav_path: str, fp16: bool) -> str:
    """Transcribe a WAV file with Whisper and return plain text."""
    result = model.transcribe(wav_path, fp16=fp16)
    return (result.get("text") or "").strip()


def normalize_math_text(text: str) -> str:
    """
    Convert some common spoken operator words into symbols.

    This is intentionally simple and beginner-friendly.
    """
    text = text.lower().strip()

    # Replace longer phrases first so they don't get partially replaced.
    replacements = [
        ("to the power of", "**"),
        ("multiplied by", "*"),
        ("divided by", "/"),
        ("plus", "+"),
        ("minus", "-"),
        ("times", "*"),
        ("multiply", "*"),
        ("divide", "/"),
        ("power", "**"),
    ]

    for spoken, symbol in replacements:
        text = re.sub(rf"\b{re.escape(spoken)}\b", symbol, text)

    return text


def evaluate_simple_math_expression(text: str) -> Optional[Union[Number, str]]:
    """
    Extract and evaluate one simple binary expression, like:
      "<number> <operator> <number>"

    Supported operators: +, -, *, /, **

    Returns:
    - int or float for a valid expression
    - "Error: ..." string for a known math error (e.g., division by zero)
    - None if no expression was found
    """
    normalized = normalize_math_text(text)

    # Capture: number (op) number, e.g. "3.5 * -2"
    pattern = r"(-?\d+\.?\d*)\s*([+\-*/]|\*\*)\s*(-?\d+\.?\d*)"
    match = re.search(pattern, normalized)
    if not match:
        return None

    try:
        left = float(match.group(1))
        op = match.group(2)
        right = float(match.group(3))

        if op == "+":
            result = left + right
        elif op == "-":
            result = left - right
        elif op == "*":
            result = left * right
        elif op == "/":
            if right == 0:
                return "Error: Division by zero"
            result = left / right
        elif op == "**":
            result = left**right
        else:
            return None

        # Print clean integers as ints (e.g. 4.0 -> 4)
        return int(result) if result.is_integer() else result
    except (ValueError, ZeroDivisionError) as exc:
        return f"Error: {exc}"


def run(settings: Settings, once: bool, no_math: bool) -> None:
    """Main loop: record -> transcribe -> (optional) evaluate math."""
    import whisper

    model = whisper.load_model(settings.whisper_model)

    print("Speak now. Press Ctrl+C to stop.")
    while True:
        print(f"\nRecording {settings.record_seconds:.1f}s...")
        audio = record_microphone_audio(settings)
        wav_path = save_wav_tempfile(audio, settings.sample_rate)

        try:
            text = transcribe_wav(model, wav_path, fp16=settings.fp16)
        finally:
            # Ensure temp files don't pile up.
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if text:
            print("Text:", text)

            if not no_math:
                math_result = evaluate_simple_math_expression(text)
                if math_result is not None:
                    print("Math:", "=", math_result)

        if once:
            return

        time.sleep(settings.sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record mic audio, transcribe with Whisper, and optionally evaluate simple math."
    )
    parser.add_argument("--model", default="base", help="Whisper model name (tiny, base, small, medium, large).")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording duration per chunk.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate.")
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices).")
    parser.add_argument("--once", action="store_true", help="Record/transcribe one chunk and exit.")
    parser.add_argument("--no-math", action="store_true", help="Disable math evaluation (transcribe only).")
    parser.add_argument("--list-devices", action="store_true", help="Print audio devices and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        raise SystemExit(0)

    settings = Settings(
        sample_rate=args.sample_rate,
        record_seconds=args.seconds,
        whisper_model=args.model,
        device=args.device,
    )

    try:
        run(settings=settings, once=args.once, no_math=args.no_math)
    except KeyboardInterrupt:
        print("\nStopped.")
