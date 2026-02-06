#!/usr/bin/env python3
"""Transcripto GUI: Windows-first WhisperX batch transcriber with portable caches."""
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as e:
    print("Tkinter is not available. On Linux/WSL2, install it with:")
    print("  sudo apt-get update && sudo apt-get install -y python3-tk")
    raise

from dotenv import load_dotenv

# -----------------------------
# Environment and portable caches
# -----------------------------
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
HF_CACHE = APP_DIR / "_hf_cache"
TORCH_CACHE = APP_DIR / "_torch_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Load .env if present
load_dotenv(APP_DIR / ".env")

# Late imports after env setup
import torch  # noqa: E402
import whisperx  # noqa: E402

# -----------------------------
# Safe loading allowlist for PyTorch 2.6+
# -----------------------------

def setup_safe_globals() -> None:
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
            from pyannote.audio.core.task import Specifications, Problem, Resolution  # type: ignore
            allowlist.extend([Specifications, Problem, Resolution])
        except Exception:
            pass

        if allowlist and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(allowlist)
    except Exception:
        pass


def safe_globals_hint(exc: Exception) -> Optional[str]:
    msg = str(exc)
    if "Unsupported global" in msg or "safe globals" in msg:
        return (
            "PyTorch safe loading blocked a checkpoint class. "
            "If this persists, add the missing class to the allowlist in transcripto_gui.py."
        )
    return None


# -----------------------------
# App config
# -----------------------------
SUPPORTED_EXTS = {
    ".mp4", ".mp3", ".m4a", ".wav", ".flac", ".mov", ".mkv", ".avi", ".webm"
}
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
LOG_FILE = "transcripto_gui.log"
LEDGER_FILE = "processed.json"
SUMMARY_FILE = "summary.txt"


@dataclass
class Options:
    folder: Path
    model_size: str
    min_speakers: int
    max_speakers: int
    force_cpu: bool
    move_done: bool
    move_failed: bool
    overwrite_outputs: bool


# -----------------------------
# Utility functions
# -----------------------------

def app_dir() -> Path:
    return APP_DIR


def setup_logging(log_queue: queue.Queue) -> None:
    class QueueHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = self.format(record)
            log_queue.put(msg)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(app_dir() / LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)

    queue_handler = QueueHandler()
    queue_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(queue_handler)


def ffmpeg_path() -> str:
    local = app_dir() / "bin" / "ffmpeg.exe"
    return str(local) if local.exists() else "ffmpeg"


def to_wav_16k_mono(input_path: Path) -> Path:
    wav_path = input_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path

    cmd = [
        ffmpeg_path(),
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    logging.info(f"FFmpeg convert -> {wav_path.name}")
    subprocess.run(cmd, check=True)
    return wav_path


def pick_device(force_cpu: bool) -> Tuple[str, str]:
    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
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


def map_speakers_to_letters(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapping: Dict[str, str] = {}
    next_letter = ord("A")
    for seg in segments:
        raw = seg.get("speaker", "Unknown")
        if raw not in mapping:
            mapping[raw] = f"Speaker {chr(next_letter)}"
            next_letter += 1
        seg["speaker_label"] = mapping[raw]
    return segments


def write_outputs(media_path: Path, segments: List[Dict[str, Any]]) -> None:
    txt_path = media_path.with_suffix(".txt")
    json_path = media_path.with_suffix(".json")

    lines: List[str] = []
    structured: List[Dict[str, Any]] = []
    for seg in segments:
        speaker = seg.get("speaker_label", "Speaker ?")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"[{start:0.1f}s - {end:0.1f}s] {speaker}: {text}")
        structured.append({"speaker": speaker, "start": start, "end": end, "text": text})

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")


def load_ledger(folder: Path) -> Dict[str, Any]:
    path = folder / LEDGER_FILE
    if not path.exists():
        return {"processed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_ledger(folder: Path, ledger: Dict[str, Any]) -> None:
    (folder / LEDGER_FILE).write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def already_done(media_path: Path, folder: Path, ledger: Dict[str, Any]) -> bool:
    out_txt = media_path.with_suffix(".txt")
    out_json = media_path.with_suffix(".json")
    if out_txt.exists() and out_json.exists():
        return True
    key = str(media_path.name)
    return key in ledger.get("processed", {})


def ensure_cache_dirs() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    (HF_CACHE / "hub").mkdir(parents=True, exist_ok=True)
    TORCH_CACHE.mkdir(parents=True, exist_ok=True)


def prepare_offline_models(model_size: str, hf_token: Optional[str]) -> None:
    ensure_cache_dirs()

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    device = "cpu"
    compute_type = "float32"

    # WhisperX ASR (Whisper model)
    logging.info(f"Preparing WhisperX model: {model_size}")
    model = whisperx.load_model(model_size, device=device, compute_type=compute_type, vad_method="silero")

    # Prepare an alignment model for a common language (English). This avoids first-run network later.
    logging.info("Preparing alignment model (en)")
    whisperx.load_align_model(language_code="en", device=device)

    # Diarization models require HF token on first download
    if hf_token:
        logging.info("Preparing diarization pipeline")
        DiarizationPipeline = import_diarization_pipeline()
        _ = DiarizationPipeline(use_auth_token=hf_token, device=device)
    else:
        raise RuntimeError(
            "HF_TOKEN is required on first run to download diarization models. "
            "Add HF_TOKEN to .env or paste it in the app."
        )

    # Free memory
    try:
        del model
    except Exception:
        pass


# -----------------------------
# Transcription worker
# -----------------------------

def transcribe_batch(options: Options, ui_queue: queue.Queue, stop_event: threading.Event) -> None:
    setup_safe_globals()
    ensure_cache_dirs()

    folder = options.folder
    ledger = load_ledger(folder)

    candidates = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
    if not candidates:
        ui_queue.put(("status", "No compatible media files found."))
        return

    ui_queue.put(("status", f"Found {len(candidates)} file(s)."))
    summary_lines: List[str] = []

    device, compute_type = pick_device(options.force_cpu)
    logging.info(f"Device={device} compute_type={compute_type} model={options.model_size}")

    model = whisperx.load_model(options.model_size, device=device, compute_type=compute_type, vad_method="silero")

    for idx, media_path in enumerate(candidates, start=1):
        if stop_event.is_set():
            ui_queue.put(("status", "Stopped by user."))
            break

        if not options.overwrite_outputs and already_done(media_path, folder, ledger):
            logging.info(f"Skipping (already done): {media_path.name}")
            continue

        ui_queue.put(("status", f"Processing {idx}/{len(candidates)}: {media_path.name}"))

        try:
            wav_path = to_wav_16k_mono(media_path)
            audio = whisperx.load_audio(str(wav_path))

            result = model.transcribe(audio)
            language = result.get("language", "unknown")
            logging.info(f"Detected language: {language}")

            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device)

            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                DiarizationPipeline = import_diarization_pipeline()
                diar = DiarizationPipeline(use_auth_token=hf_token, device=device)
                diar_segs = diar(audio, min_speakers=options.min_speakers, max_speakers=options.max_speakers)
                result = whisperx.assign_word_speakers(diar_segs, result)
            else:
                logging.warning("HF_TOKEN missing: diarization skipped.")

            segments = map_speakers_to_letters(result["segments"])
            write_outputs(media_path, segments)

            summary_lines.append(f"=== {media_path.name} ===")
            summary_lines.append("\n".join([seg["text"].strip() for seg in segments if seg.get("text")]))
            summary_lines.append("")

            ledger["processed"][media_path.name] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": options.model_size,
                "min_speakers": options.min_speakers,
                "max_speakers": options.max_speakers,
            }
            save_ledger(folder, ledger)

            if options.move_done:
                done_dir = folder / "done"
                done_dir.mkdir(exist_ok=True)
                shutil.move(str(media_path), str(done_dir / media_path.name))

        except Exception as e:
            logging.exception(f"FAILED: {media_path.name} ({e})")
            hint = safe_globals_hint(e)
            if hint:
                logging.error(hint)
            if options.move_failed:
                failed_dir = folder / "failed"
                failed_dir.mkdir(exist_ok=True)
                try:
                    shutil.move(str(media_path), str(failed_dir / media_path.name))
                except Exception:
                    pass
            continue

    if summary_lines:
        (folder / SUMMARY_FILE).write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")

    ui_queue.put(("status", "Done."))


# -----------------------------
# GUI
# -----------------------------

class TranscriptoGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Transcripto")
        self.geometry("860x640")

        self.log_queue: queue.Queue = queue.Queue()
        self.ui_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self._build_ui()
        setup_logging(self.log_queue)
        self._poll_queues()

    def _build_ui(self) -> None:
        # Folder
        folder_frame = ttk.Frame(self)
        folder_frame.pack(fill="x", padx=12, pady=8)

        ttk.Label(folder_frame, text="Folder").pack(side="left")
        self.folder_var = tk.StringVar(value=str(app_dir()))
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=70)
        self.folder_entry.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).pack(side="left")

        # Options
        opts_frame = ttk.Labelframe(self, text="Options")
        opts_frame.pack(fill="x", padx=12, pady=8)

        self.model_var = tk.StringVar(value="base")
        ttk.Label(opts_frame, text="Model size").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(opts_frame, textvariable=self.model_var, values=MODEL_SIZES, width=12, state="readonly").grid(
            row=0, column=1, sticky="w", padx=8, pady=6
        )

        self.min_speakers_var = tk.IntVar(value=2)
        self.max_speakers_var = tk.IntVar(value=2)
        ttk.Label(opts_frame, text="Min speakers").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(opts_frame, textvariable=self.min_speakers_var, width=5).grid(row=0, column=3, sticky="w")
        ttk.Label(opts_frame, text="Max speakers").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(opts_frame, textvariable=self.max_speakers_var, width=5).grid(row=0, column=5, sticky="w")

        self.force_cpu_var = tk.BooleanVar(value=False)
        self.move_done_var = tk.BooleanVar(value=False)
        self.move_failed_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(opts_frame, text="Force CPU", variable=self.force_cpu_var).grid(row=1, column=0, sticky="w", padx=8)
        ttk.Checkbutton(opts_frame, text="Move processed to done", variable=self.move_done_var).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Checkbutton(opts_frame, text="Move failed to failed", variable=self.move_failed_var).grid(row=1, column=2, sticky="w", padx=8)
        ttk.Checkbutton(opts_frame, text="Overwrite outputs", variable=self.overwrite_var).grid(row=1, column=3, sticky="w", padx=8)

        # Token
        token_frame = ttk.Labelframe(self, text="Hugging Face Token (first run only for diarization model download)")
        token_frame.pack(fill="x", padx=12, pady=8)
        self.token_var = tk.StringVar(value=os.getenv("HF_TOKEN", ""))
        ttk.Entry(token_frame, textvariable=self.token_var, width=80, show="*").pack(side="left", padx=8, pady=6, fill="x", expand=True)
        ttk.Button(token_frame, text="Save to .env", command=self._save_token).pack(side="left", padx=8)
        ttk.Button(token_frame, text="Prepare Offline Models", command=self._prepare_offline).pack(side="left")

        # Controls
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=12, pady=8)
        self.start_btn = ttk.Button(control_frame, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side="left")

        # Log output
        log_frame = ttk.Labelframe(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=12, pady=8)
        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self.folder_var.get())
        if path:
            self.folder_var.set(path)

    def _save_token(self) -> None:
        token = self.token_var.get().strip()
        if not token:
            messagebox.showwarning("HF_TOKEN", "Please enter a token to save.")
            return
        env_path = app_dir() / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        lines = [line for line in existing.splitlines() if not line.startswith("HF_TOKEN=")]
        lines.append(f"HF_TOKEN={token}")
        env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        messagebox.showinfo("HF_TOKEN", "Token saved to .env")

    def _prepare_offline(self) -> None:
        token = self.token_var.get().strip()
        model_size = self.model_var.get()

        def run() -> None:
            try:
                self.ui_queue.put(("status", "Preparing offline models..."))
                prepare_offline_models(model_size, token or None)
                self.ui_queue.put(("status", "Offline models ready."))
            except Exception as e:
                logging.exception(str(e))
                self.ui_queue.put(("error", str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return

        folder = Path(self.folder_var.get())
        if not folder.exists():
            messagebox.showerror("Folder", "Selected folder does not exist.")
            return

        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(10)

        token = self.token_var.get().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        options = Options(
            folder=folder,
            model_size=self.model_var.get(),
            min_speakers=self.min_speakers_var.get(),
            max_speakers=self.max_speakers_var.get(),
            force_cpu=self.force_cpu_var.get(),
            move_done=self.move_done_var.get(),
            move_failed=self.move_failed_var.get(),
            overwrite_outputs=self.overwrite_var.get(),
        )

        def run() -> None:
            try:
                transcribe_batch(options, self.ui_queue, self.stop_event)
            except Exception as e:
                logging.exception(str(e))
                self.ui_queue.put(("error", str(e)))

        self.worker_thread = threading.Thread(target=run, daemon=True)
        self.worker_thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def _poll_queues(self) -> None:
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")

        while not self.ui_queue.empty():
            kind, payload = self.ui_queue.get_nowait()
            if kind == "status":
                self.status_var.set(payload)
                if payload in {"Done.", "Stopped by user."}:
                    self.progress.stop()
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
            elif kind == "error":
                self.progress.stop()
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                messagebox.showerror("Error", payload)

        self.after(200, self._poll_queues)


def main() -> int:
    setup_safe_globals()
    ensure_cache_dirs()

    app = TranscriptoGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
