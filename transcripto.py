# whisper_to_text_diarized.py
"""
WhisperX transcription + alignment + 2-speaker diarization (Speaker A / Speaker B)
Windows / WSL friendly. Uses .env for HF_TOKEN.

Outputs:
- <input>.txt  (human readable)
- <input>.json (structured segments)
- transcription.log
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

# Load .env early
load_dotenv()
import torch

# --- PyTorch 2.6+ safe loading allowlist for trusted HF/Pyannote checkpoints ---
try:
    allowlist = []

    # Torch internal version object sometimes present in checkpoints
    try:
        from torch.torch_version import TorchVersion  # type: ignore
        allowlist.append(TorchVersion)
    except Exception:
        pass

    # OmegaConf objects sometimes present in checkpoints
    try:
        from omegaconf import DictConfig, ListConfig  # type: ignore
        allowlist.extend([DictConfig, ListConfig])
    except Exception:
        pass

    # Pyannote task objects sometimes present in checkpoints
    try:
        from pyannote.audio.core.task import Specifications, Problem, Resolution  # type: ignore
        allowlist.extend([Specifications, Problem, Resolution])
    except Exception:
        pass


    if allowlist and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(allowlist)

except Exception:
    pass




import whisperx


# -------------------------
# Config via env
# -------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
DEVICE = os.getenv("DEVICE", "auto").lower()
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "").strip()
ENABLE_TF32 = os.getenv("ENABLE_TF32", "0") == "1"

MIN_SPEAKERS = int(os.getenv("MIN_SPEAKERS", "2"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "2"))

# Helps avoid Transformers importing torchvision (which can explode with torch/torchvision mismatches)
# You can also set this in your shell: set TRANSFORMERS_NO_TORCHVISION=1
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

LOG_FILE = os.getenv("LOG_FILE", "transcription.log")
SUPPORTED_CUDA = ["12.8", "12.6", "11.8"]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=LOG_FILE,
        filemode="a",
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


def to_wav_16k_mono(input_path: str) -> str:
    """
    Convert any media file to mono 16k WAV (best for whisper/diarization stability).
    If the WAV already exists, reuse it.
    """
    in_path = Path(input_path)
    wav_path = in_path.with_suffix(".wav")

    if wav_path.exists():
        logging.info(f"Using existing WAV: {wav_path}")
        return str(wav_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    logging.info("Converting input to WAV (mono, 16k) via ffmpeg...")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it's on PATH, then retry."
        ) from e

    return str(wav_path)


def pick_device() -> Dict[str, str]:
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
    return {"device": device, "compute_type": compute_type}


def import_diarization_pipeline():
    """
    WhisperX has moved diarization classes around across versions.
    Try the common locations.
    """
    try:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
        return DiarizationPipeline
    except Exception:
        pass

    try:
        from whisperx.diarization import DiarizationPipeline  # type: ignore
        return DiarizationPipeline
    except Exception as e:
        raise RuntimeError(
            "Could not import DiarizationPipeline from whisperx. "
            "Your whisperx version may not include diarization helpers."
        ) from e


def map_speakers_to_letters(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Map SPEAKER_00/SPEAKER_01/etc to Speaker A/Speaker B based on first appearance.
    """
    mapping: Dict[str, str] = {}
    next_letter = ord("A")

    for seg in segments:
        raw = seg.get("speaker", "Unknown")
        if raw not in mapping:
            mapping[raw] = f"Speaker {chr(next_letter)}"
            next_letter += 1
        seg["speaker_label"] = mapping[raw]

    return segments


def write_outputs(input_file: str, segments: List[Dict[str, Any]]) -> None:
    txt_path = Path(input_file).with_suffix(".txt")
    json_path = Path(input_file).with_suffix(".json")

    lines: List[str] = []
    structured: List[Dict[str, Any]] = []

    for seg in segments:
        speaker = seg.get("speaker_label", "Speaker ?")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()

        line = f"[{start:0.1f}s - {end:0.1f}s] {speaker}: {text}"
        print(line)
        lines.append(line)

        structured.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": text,
        })

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info(f"Wrote: {txt_path}")
    logging.info(f"Wrote: {json_path}")


def main() -> None:
    setup_logging()
    warn_if_cuda_missing()

    if len(sys.argv) < 2:
        print("Usage: python whisper_to_text_diarized.py <audio_or_video_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    logging.info(f"Input: {input_file}")

    # Device config
    dc = pick_device()
    device = dc["device"]
    compute_type = dc["compute_type"]
    logging.info(f"Device: {device} | compute_type: {compute_type} | model: {MODEL_SIZE}")
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            gb = props.total_memory / (1024 ** 3)
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)} ({gb:0.1f} GB)")
        except Exception:
            pass
    elif device == "mps":
        logging.info("Using Apple Metal (MPS) backend.")

    # Convert and load audio
    wav_file = to_wav_16k_mono(input_file)
    audio = whisperx.load_audio(wav_file)

    # Load ASR with silero VAD (more robust on Windows and avoids pyannote VAD checkpoint issues)
    logging.info("Loading WhisperX ASR...")
    model = whisperx.load_model(
        MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        vad_method="silero",
    )

    # Transcribe
    logging.info("Transcribing...")
    result = model.transcribe(audio)
    language = result.get("language", "unknown")
    logging.info(f"Detected language: {language}")

    # Align
    logging.info("Loading align model...")
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)

    logging.info("Aligning...")
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
    )

    # Diarize (optional if token provided)
    if HF_TOKEN:
        # Ensure omegaconf exists if needed
        try:
            import omegaconf  # noqa: F401
        except Exception:
            logging.warning("omegaconf not installed; install with: pip install -U omegaconf")

        logging.info("Running diarization...")
        DiarizationPipeline = import_diarization_pipeline()
        diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

        diarize_segments = diarize_model(
            audio,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS,
        )

        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        logging.warning("HF_TOKEN missing: diarization skipped (no Speaker A/B).")

    segments = map_speakers_to_letters(result["segments"])

    logging.info("Writing outputs...")
    write_outputs(input_file, segments)

    logging.info("Done.")


if __name__ == "__main__":
    main()
