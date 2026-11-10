"""Audio conversion helpers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

FFMPEG_MISSING_MESSAGE = (
    "ffmpeg was not found. Install ffmpeg and make sure it is on PATH, "
    "or place ffmpeg.exe in a bin folder next to the app."
)


def ffmpeg_path(bundled_dir: Optional[Path] = None) -> str:
    """Locate ffmpeg on PATH, then fall back to a bundled copy."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    if bundled_dir is not None:
        candidates = [
            bundled_dir / "bin" / "ffmpeg.exe",
            bundled_dir / "ffmpeg.exe",
            bundled_dir / "bin" / "ffmpeg",
            bundled_dir / "ffmpeg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    raise RuntimeError(FFMPEG_MISSING_MESSAGE)


def convert_to_wav_16k(
    input_path: Path,
    out_dir: Path,
    bundled_dir: Optional[Path] = None,
    executable: Optional[str] = None,
) -> Path:
    """Convert media to a mono 16 kHz WAV in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio.wav"
    ffmpeg = executable or ffmpeg_path(bundled_dir)

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    LOGGER.info("ffmpeg convert -> %s", wav_path.name)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(FFMPEG_MISSING_MESSAGE) from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip().splitlines()
        tail = " | ".join(details[-3:]) if details else "no ffmpeg output"
        raise RuntimeError(f"ffmpeg failed for {input_path.name}: {tail}") from exc

    return wav_path


def load_audio_16k(input_path: Path, bundled_dir: Optional[Path] = None):
    """Load audio for WhisperX using a temporary WAV that is always cleaned up."""
    import whisperx  # imported here so tests do not need torch/whisperx

    tmp_dir = Path(tempfile.mkdtemp(prefix="transcripto-"))
    try:
        wav_path = convert_to_wav_16k(input_path, tmp_dir, bundled_dir=bundled_dir)
        return whisperx.load_audio(str(wav_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
