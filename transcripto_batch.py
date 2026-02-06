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
DEVICE = os.getenv("DEVICE", "auto").lower()
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "").strip()
ENABLE_TF32 = os.getenv("ENABLE_TF32", "0") == "1"
MIN_SPEAKERS = int(os.getenv("MIN_SPEAKERS", "2"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "2"))

LOG_FILE = "transcripto.log"
LEDGER_FILE = "processed.json"
SUPPORTED_CUDA = ["12.8", "12.6", "11.8"]


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


def is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        for path in ("/proc/version", "/proc/sys/kernel/osrelease"):
            text = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
            if "microsoft" in text or "wsl" in text:
                return True
    except Exception:
        pass
    return False


def detect_cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version", 1)[-1]
                    version = parts.split(":", 1)[-1].strip().split()[0]
                    return version
    except Exception:
        pass

    return ""


def pick_supported_cuda(cuda_version: str) -> str:
    def as_tuple(v: str) -> tuple[int, int]:
        major, minor = v.split(".", 1)
        return int(major), int(minor)

    if not cuda_version:
        return ""
    try:
        target = as_tuple(cuda_version)
    except Exception:
        return ""

    supported = []
    for v in SUPPORTED_CUDA:
        try:
            supported.append((as_tuple(v), v))
        except Exception:
            continue
    supported.sort(reverse=True)

    for (maj_min, v) in supported:
        if maj_min <= target:
            return v
    return ""


def cuda_index_url(cuda_version: str) -> str:
    return f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"


def warn_if_cuda_missing() -> None:
    if torch.cuda.is_available():
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return
    if result.returncode != 0 or not result.stdout.strip():
        return

    cuda_version = detect_cuda_version()
    supported = pick_supported_cuda(cuda_version)
    if supported:
        index_url = cuda_index_url(supported)
        pip_cmd = f"python -m pip install torch torchvision torchaudio --index-url {index_url}"
    else:
        pip_cmd = "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    if is_wsl():
        platform_hint = "WSL2"
    elif os.name == "nt":
        platform_hint = "Windows"
    else:
        platform_hint = "Linux"

    msg = [
        "NVIDIA GPU detected but CUDA is not available in PyTorch.",
        f"Platform: {platform_hint}",
    ]
    if cuda_version:
        msg.append(f"Detected CUDA version (driver): {cuda_version}")
    msg.append(f"Suggested pip install: {pip_cmd}")
    msg.append("If this fails, use the PyTorch selector to pick a matching CUDA build.")
    msg.append("Tip: run `python setup_gpu.py` for a full GPU diagnostic.")
    logging.warning(" ".join(msg))


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
    if FORCE_CPU:
        device = "cpu"
    else:
        if DEVICE in {"", "auto"}:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        elif DEVICE in {"cuda", "mps", "cpu"}:
            device = DEVICE
        else:
            logging.warning(f"Unknown DEVICE '{DEVICE}', falling back to CPU.")
            device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("DEVICE=cuda requested but CUDA is not available; falling back to CPU.")
        device = "cpu"
    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            logging.warning("DEVICE=mps requested but MPS is not available; falling back to CPU.")
            device = "cpu"

    if ENABLE_TF32 and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    compute_type = COMPUTE_TYPE or ("float16" if device == "cuda" else "float32")
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
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            gb = props.total_memory / (1024 ** 3)
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)} ({gb:0.1f} GB)")
        except Exception:
            pass
    elif device == "mps":
        logging.info("Using Apple Metal (MPS) backend.")
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
    warn_if_cuda_missing()

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
