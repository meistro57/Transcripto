"""Model loading, device selection, and transcription."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .audio import load_audio_16k

LOGGER = logging.getLogger(__name__)

SUPPORTED_CUDA = ["12.8", "12.6", "11.8"]
_SAFE_GLOBALS_ADDED = False


class TranscriptionCancelled(RuntimeError):
    """Raised when a caller asks an in-flight transcription to stop."""


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
    def as_tuple(value: str) -> Tuple[int, int]:
        major, minor = value.split(".", 1)
        return int(major), int(minor)

    if not cuda_version:
        return ""
    try:
        target = as_tuple(cuda_version)
    except Exception:
        return ""

    supported = []
    for version in SUPPORTED_CUDA:
        try:
            supported.append((as_tuple(version), version))
        except Exception:
            continue
    supported.sort(reverse=True)

    for minimum, version in supported:
        if minimum <= target:
            return version
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
        pip_cmd = (
            "python -m pip install torch torchvision torchaudio "
            f"--index-url {cuda_index_url(supported)}"
        )
    else:
        pip_cmd = (
            "python -m pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu"
        )

    if is_wsl():
        platform_hint = "WSL2"
    elif os.name == "nt":
        platform_hint = "Windows"
    else:
        platform_hint = "Linux"

    message = [
        "NVIDIA GPU detected but CUDA is not available in PyTorch.",
        f"Platform: {platform_hint}",
    ]
    if cuda_version:
        message.append(f"Detected CUDA version (driver): {cuda_version}")
    message.append(f"Suggested pip install: {pip_cmd}")
    message.append("If this fails, use the PyTorch selector to pick a matching CUDA build.")
    message.append("Tip: run `python setup_gpu.py` for a full GPU diagnostic.")
    LOGGER.warning(" ".join(message))


def pick_device(
    force_cpu: bool = False,
    device: Optional[str] = None,
    compute_type: str = "",
    enable_tf32: bool = False,
) -> Tuple[str, str]:
    requested = (device or "auto").lower()

    if force_cpu:
        chosen = "cpu"
    elif requested in {"", "auto"}:
        if torch.cuda.is_available():
            chosen = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            chosen = "mps"
        else:
            chosen = "cpu"
    elif requested in {"cuda", "mps", "cpu"}:
        chosen = requested
    else:
        LOGGER.warning("Unknown DEVICE '%s', falling back to CPU.", requested)
        chosen = "cpu"

    if chosen == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("DEVICE=cuda requested but unavailable; falling back to CPU.")
        chosen = "cpu"
    if chosen == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        LOGGER.warning("DEVICE=mps requested but unavailable; falling back to CPU.")
        chosen = "cpu"

    if enable_tf32 and chosen == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    chosen_compute = compute_type or ("float16" if chosen == "cuda" else "float32")
    return chosen, chosen_compute


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_safe_globals() -> None:
    """Allow trusted PyTorch 2.6+ checkpoints used by WhisperX and Pyannote."""
    global _SAFE_GLOBALS_ADDED
    if _SAFE_GLOBALS_ADDED:
        return

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
        from pyannote.audio.core.task import Specifications, Problem, Resolution  # type: ignore
        allowlist.extend([Specifications, Problem, Resolution])
    except Exception:
        pass

    if allowlist and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(allowlist)
    _SAFE_GLOBALS_ADDED = True


def safe_globals_hint(exc: Exception) -> Optional[str]:
    message = str(exc)
    if "Unsupported global" in message or "safe globals" in message:
        return (
            "PyTorch safe loading blocked a checkpoint class. If this persists, "
            "add the missing class to transcripto_core.models.add_safe_globals."
        )
    return None


def import_diarization_pipeline():
    """WhisperX has moved diarization classes around across versions."""
    try:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
        return DiarizationPipeline
    except Exception:
        pass

    try:
        from whisperx.diarization import DiarizationPipeline  # type: ignore
        return DiarizationPipeline
    except Exception as exc:
        raise RuntimeError(
            "Could not import DiarizationPipeline from whisperx. "
            "Your whisperx version may not include diarization helpers."
        ) from exc


def _check_stop(should_stop: Optional[Callable[[], bool]]) -> None:
    if should_stop is not None and should_stop():
        raise TranscriptionCancelled("Transcription cancelled by caller")


class Transcriber:
    """Loads WhisperX once and reuses alignment and diarization models."""

    def __init__(
        self,
        model_size: str = "base",
        *,
        hf_token: Optional[str] = None,
        min_speakers: int = 1,
        max_speakers: int = 2,
        language: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "",
        force_cpu: bool = False,
        enable_tf32: bool = False,
    ) -> None:
        if min_speakers > max_speakers:
            raise ValueError("min_speakers must be <= max_speakers")

        self.model_size = model_size
        self.hf_token = hf_token or None
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.language = language or None
        self.device, self.compute_type = pick_device(
            force_cpu=force_cpu,
            device=device,
            compute_type=compute_type,
            enable_tf32=enable_tf32,
        )
        self.align_models: Dict[str, Any] = {}
        self._diarization = None

        LOGGER.info(
            "Device=%s compute_type=%s model=%s",
            self.device,
            self.compute_type,
            self.model_size,
        )
        if self.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                gb = props.total_memory / (1024 ** 3)
                LOGGER.info("CUDA: %s (%.1f GB)", torch.cuda.get_device_name(0), gb)
            except Exception:
                pass
        elif self.device == "mps":
            LOGGER.info("Using Apple Metal (MPS)")

        import whisperx

        LOGGER.info("Loading ASR model...")
        self.model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            vad_method="silero",
        )

    def __enter__(self) -> "Transcriber":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.model = None
        self._diarization = None
        self.align_models.clear()
        empty_cuda_cache()

    def warm_up(self, language: str = "en") -> None:
        self._align_model(language)
        if self.hf_token:
            self._diarization_pipeline()

    def _align_model(self, language: str) -> Any:
        import whisperx

        if language not in self.align_models:
            LOGGER.info("Loading alignment model (%s)...", language)
            self.align_models[language] = whisperx.load_align_model(
                language_code=language, device=self.device
            )
        return self.align_models[language]

    def _diarization_pipeline(self) -> Any:
        if self._diarization is None:
            LOGGER.info("Loading diarization pipeline...")
            pipeline_class = import_diarization_pipeline()
            try:
                self._diarization = pipeline_class(
                    use_auth_token=self.hf_token, device=self.device
                )
            except TypeError:
                self._diarization = pipeline_class(token=self.hf_token, device=self.device)
        return self._diarization

    def align(self, segments: List[dict], language: str, audio) -> Any:
        import whisperx

        align_model, metadata = self._align_model(language)
        return whisperx.align(segments, align_model, metadata, audio, self.device)

    def diarize(self, audio) -> Any:
        pipeline = self._diarization_pipeline()
        return pipeline(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)

    def transcribe_audio(
        self, audio, should_stop: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        import whisperx

        _check_stop(should_stop)

        kwargs = {"language": self.language} if self.language else {}
        result = self.model.transcribe(audio, **kwargs)
        language = result.get("language", "unknown")
        LOGGER.info("Detected language: %s", language)

        _check_stop(should_stop)
        LOGGER.info("Aligning...")
        result = self.align(result["segments"], language, audio)

        _check_stop(should_stop)
        if self.hf_token:
            LOGGER.info("Diarizing (speaker identification)...")
            try:
                diar_segments = self.diarize(audio)
                result = whisperx.assign_word_speakers(diar_segments, result)
            except TranscriptionCancelled:
                raise
            except Exception as exc:
                LOGGER.warning("Diarization failed, continuing without speakers: %s", exc)
        else:
            LOGGER.warning("HF_TOKEN missing: skipping diarization")

        _check_stop(should_stop)
        return result

    def transcribe_file(
        self,
        media_path: Path,
        bundled_dir: Optional[Path] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        LOGGER.info("Processing: %s", media_path)
        audio = load_audio_16k(media_path, bundled_dir=bundled_dir)
        return self.transcribe_audio(audio, should_stop=should_stop)
