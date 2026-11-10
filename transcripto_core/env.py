"""Environment setup and configuration shared by all entry points."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

TRUE_VALUES = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def detect_app_dir() -> Path:
    """Return the folder the app runs from (EXE folder when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    return detect_app_dir()


def prepare_environment(app_folder: Optional[Path] = None) -> Path:
    """Set cache variables and load .env before torch/whisperx are imported."""
    folder = (app_folder or detect_app_dir()).resolve()

    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("HF_HOME", str(folder / "_hf_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(folder / "_hf_cache" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(folder / "_torch_cache"))

    (folder / "_hf_cache" / "hub").mkdir(parents=True, exist_ok=True)
    (folder / "_torch_cache").mkdir(parents=True, exist_ok=True)

    load_dotenv(folder / ".env")
    return folder


@dataclass(frozen=True)
class Settings:
    """Transcription settings resolved from the environment."""

    model_size: str = "base"
    device: str = "auto"
    compute_type: str = ""
    force_cpu: bool = False
    enable_tf32: bool = False
    min_speakers: int = 1
    max_speakers: int = 2
    hf_token: Optional[str] = None
    language: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        min_speakers = env_int("MIN_SPEAKERS", 1)
        max_speakers = env_int("MAX_SPEAKERS", max(2, min_speakers))
        return cls(
            model_size=os.getenv("MODEL_SIZE", "base").strip() or "base",
            device=os.getenv("DEVICE", "auto").strip().lower() or "auto",
            compute_type=os.getenv("COMPUTE_TYPE", "").strip(),
            force_cpu=env_bool("FORCE_CPU"),
            enable_tf32=env_bool("ENABLE_TF32"),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=os.getenv("HF_TOKEN") or None,
            language=os.getenv("LANGUAGE") or None,
        )
