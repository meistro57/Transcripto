# transcripto_batch.py
import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import torch

# PyTorch 2.6+ safe loading allowlist for trusted HF/Pyannote checkpoints
try:
    allowlist = []
    try:
        from torch.torch_version import TorchVersion  # type: ignore
        allowlist.append(TorchVersion)
    except Exception:
        pass
    try:
        from omegaconf import DictConfig, ListConfig  # type: ignore
        allowlist.extend([DictConfig, ListConfig])
    except Exception:
        pass
    try:
        from pyannote.audio.core.task import Specifications, Problem  # type: ignore
        allowlist.extend([Specifications, Problem])
    except Exception:
        pass
    if allowlist and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(allowlist)
except Exception:
    pass

import whisperx

# Avoid torchvision import chain
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

SUPPORTED_EXTS = {
    # audio
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
    # video
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
}

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
MIN_SPEAKERS = int(os.getenv("MIN_SPEAKERS", "2"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "2"))

LOG_FILE = "transcripto.log"
LEDGER_FILE = "processed.json"


def app_dir() -> Path:
    # When frozen by PyInstaller, sys.executable points to the EXE
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def setup_logging(folder: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(folder / LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def ffmpeg_path(folder: Path) -> str:
    # We will bundle ffmpeg.exe into ./bin/ffmpeg.exe in the final app
    local = folder / "bin" / "ffmpeg.exe"
    return str(local) if local.exists() else "ffmpeg"


def to_wav_16k_mono(input_path: Path, folder: Path) -> Path:
    wav_path = input_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path

    cmd = [
        ffmpeg_path(folder),
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    logging.info(f"FFmpeg convert → {wav_path.name}")
    subprocess.run(cmd, check=True)
    return wav_path


def pick_device():
    device = "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "float32"
    return device, compute_type


def import_diarization_pipeline():
    try:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
        return DiarizationPipeline
    except Exception:
        pass
    try:
        from whisperx.diarization import DiarizationPipeline  # type: ignore
        return DiarizationPipeline
    except Exception as e:
        raise RuntimeError("Could not import DiarizationPipeline from whisperx.") from e


def map_speakers_to_letters(segments):
    mapping = {}
    next_letter = ord("A")
    for seg in segments:
        raw = seg.get("speaker", "Unknown")
        if raw not in mapping:
            mapping[raw] = f"Speaker {chr(next_letter)}"
            next_letter += 1
        seg["speaker_label"] = mapping[raw]
    return segments


def load_ledger(folder: Path) -> dict:
    path = folder / LEDGER_FILE
    if not path.exists():
        return {"processed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(folder: Path, ledger: dict) -> None:
    (folder / LEDGER_FILE).write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def already_done(media_path: Path, folder: Path, ledger: dict) -> bool:
    # Primary skip check: output exists
    out_json = media_path.with_suffix(".json")
    if out_json.exists():
        return True
    # Secondary: ledger
    key = str(media_path.name)
    return key in ledger.get("processed", {})


def write_outputs(media_path: Path, segments):
    txt_path = media_path.with_suffix(".txt")
    json_path = media_path.with_suffix(".json")

    lines = []
    structured = []
    for seg in segments:
        speaker = seg.get("speaker_label", "Speaker ?")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()

        lines.append(f"[{start:0.1f}s - {end:0.1f}s] {speaker}: {text}")
        structured.append({"speaker": speaker, "start": start, "end": end, "text": text})

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")


def transcribe_one(media_path: Path, folder: Path) -> None:
    device, compute_type = pick_device()
    logging.info(f"Device={device} compute_type={compute_type} model={MODEL_SIZE}")
    logging.info(f"Processing: {media_path.name}")

    wav_path = to_wav_16k_mono(media_path, folder)
    audio = whisperx.load_audio(str(wav_path))

    model = whisperx.load_model(MODEL_SIZE, device=device, compute_type=compute_type, vad_method="silero")
    result = model.transcribe(audio)
    language = result.get("language", "unknown")
    logging.info(f"Detected language: {language}")

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    if HF_TOKEN:
        logging.info("Running diarization...")
        DiarizationPipeline = import_diarization_pipeline()
        diar = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        diar_segs = diar(audio, min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
        result = whisperx.assign_word_speakers(diar_segs, result)
    else:
        logging.warning("HF_TOKEN missing: diarization skipped (no Speaker A/B).")

    segments = map_speakers_to_letters(result["segments"])
    write_outputs(media_path, segments)

    logging.info(f"Done: {media_path.name}")


def main():
    folder = app_dir()
    setup_logging(folder)

    ledger = load_ledger(folder)

    candidates = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not candidates:
        logging.info("No compatible media files found in this folder.")
        return

    logging.info(f"Found {len(candidates)} candidate file(s).")

    for p in candidates:
        if already_done(p, folder, ledger):
            logging.info(f"Skipping (already done): {p.name}")
            continue

        try:
            transcribe_one(p, folder)
            ledger["processed"][p.name] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_SIZE,
                "min_speakers": MIN_SPEAKERS,
                "max_speakers": MAX_SPEAKERS,
            }
            save_ledger(folder, ledger)

        except Exception as e:
            logging.exception(f"FAILED: {p.name} ({e})")
            # continue to next file

    logging.info("All done.")


if __name__ == "__main__":
    main()
