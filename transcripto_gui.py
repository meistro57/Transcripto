#!/usr/bin/env python3
"""Transcripto GUI: Windows-first WhisperX batch transcriber with portable caches."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:
    print("Tkinter is not available. On Linux/WSL2, install it with:")
    print("  sudo apt-get update && sudo apt-get install -y python3-tk")
    raise

from transcripto_core.env import prepare_environment  # noqa: E402

prepare_environment(APP_DIR)

from transcripto_core.ledger import (  # noqa: E402
    already_done,
    load_ledger,
    mark_processed,
    save_ledger,
)
from transcripto_core.logging_utils import setup_logging  # noqa: E402
from transcripto_core.models import (  # noqa: E402
    Transcriber,
    TranscriptionCancelled,
    add_safe_globals,
    safe_globals_hint,
)
from transcripto_core.outputs import (  # noqa: E402
    SUPPORTED_EXTS,
    OutputLayout,
    collect_folder_media,
    find_collision_stems,
    map_speakers_to_letters,
    write_outputs,
)
from transcripto_core.summary import SummaryWriter  # noqa: E402

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
LOG_FILE = "transcripto_gui.log"
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
    selected_files: Optional[List[Path]] = None
    write_srt: bool = False
    write_vtt: bool = False
    merged: bool = False


def _queue_log_handler(log_queue: queue.Queue) -> logging.Handler:
    class QueueHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_queue.put(self.format(record))

    return QueueHandler()


def summary_path_for(options: Options, candidates: List[Path]) -> Path:
    if options.selected_files:
        parents = {p.parent for p in candidates}
        return parents.pop() / SUMMARY_FILE if len(parents) == 1 else APP_DIR / SUMMARY_FILE
    return options.folder / SUMMARY_FILE


def transcribe_batch(
    options: Options, ui_queue: queue.Queue, stop_event: threading.Event
) -> Dict[str, Any]:
    """Run a batch, reporting progress and a final summary through ui_queue."""
    add_safe_globals()

    if options.selected_files:
        raw_candidates = [
            p for p in options.selected_files if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
    else:
        raw_candidates = [
            p for p in options.folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]

    candidates = sorted({p.resolve() for p in raw_candidates})
    if not candidates:
        ui_queue.put(("status", "No compatible media files found."))
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "failures": [],
            "cancelled": False,
        }

    layout = OutputLayout()
    collision_stems = find_collision_stems(collect_folder_media(candidates), layout)
    total = len(candidates)
    ui_queue.put(("status", f"Found {total} file(s)."))
    ui_queue.put(("progress", (0, total)))

    ledger_by_folder: Dict[Path, Dict[str, Any]] = {}
    pending: List[Tuple[int, Path, Dict[str, Any]]] = []
    skipped = 0

    for index, media_path in enumerate(candidates, start=1):
        parent_folder = media_path.parent
        if parent_folder not in ledger_by_folder:
            ledger_by_folder[parent_folder] = load_ledger(parent_folder)
        ledger = ledger_by_folder[parent_folder]

        if not options.overwrite_outputs and already_done(
            media_path, ledger, layout, collision_stems
        ):
            logging.info("Skipping (already done): %s", media_path.name)
            skipped += 1
            ui_queue.put(("progress", (index, total)))
            continue

        pending.append((index, media_path, ledger))

    summary_path = summary_path_for(options, candidates)
    formats = ["txt", "json"]
    if options.write_srt:
        formats.append("srt")
    if options.write_vtt:
        formats.append("vtt")

    stats = {"total": total, "success": 0, "failed": 0, "skipped": skipped}
    failures: List[Dict[str, str]] = []
    cancelled = False

    summary = SummaryWriter(summary_path)
    try:
        transcription = (
            Transcriber(
                model_size=options.model_size,
                hf_token=os.getenv("HF_TOKEN") or None,
                min_speakers=options.min_speakers,
                max_speakers=options.max_speakers,
                device="auto",
                force_cpu=options.force_cpu,
                language=os.getenv("LANGUAGE") or None,
            )
            if pending
            else nullcontext(None)
        )

        with transcription as transcriber:
            for index, media_path, ledger in pending:
                if stop_event.is_set():
                    cancelled = True
                    break

                ui_queue.put(
                    ("status", f"Processing {index}/{total}: {media_path.name}")
                )
                try:
                    result = transcriber.transcribe_file(
                        media_path,
                        bundled_dir=APP_DIR,
                        should_stop=stop_event.is_set,
                    )
                    segments = map_speakers_to_letters(result["segments"])
                    write_outputs(
                        media_path,
                        segments,
                        layout,
                        collision_stems,
                        formats=formats,
                        merged=options.merged,
                    )
                    summary.add(media_path.name, segments)

                    mark_processed(
                        ledger,
                        media_path,
                        options.model_size,
                        options.min_speakers,
                        options.max_speakers,
                    )
                    save_ledger(media_path.parent, ledger)

                    if options.move_done:
                        done_dir = media_path.parent / "done"
                        done_dir.mkdir(exist_ok=True)
                        shutil.move(str(media_path), str(done_dir / media_path.name))

                    stats["success"] += 1
                except TranscriptionCancelled:
                    logging.info("Stopped during %s", media_path.name)
                    cancelled = True
                    break
                except Exception as exc:
                    logging.exception("FAILED: %s (%s)", media_path.name, exc)
                    hint = safe_globals_hint(exc)
                    if hint:
                        logging.error(hint)

                    failures.append(
                        {"file": str(media_path), "error": f"{type(exc).__name__}: {exc}"}
                    )
                    stats["failed"] += 1

                    if options.move_failed:
                        failed_dir = media_path.parent / "failed"
                        failed_dir.mkdir(exist_ok=True)
                        try:
                            shutil.move(str(media_path), str(failed_dir / media_path.name))
                        except Exception:
                            pass
                finally:
                    ui_queue.put(("progress", (index, total)))
    finally:
        summary.close()

    payload = {
        **stats,
        "failures": failures,
        "cancelled": cancelled,
        "summary_path": str(summary.path) if summary.written else None,
    }

    if cancelled:
        ui_queue.put(("status", "Stopped by user."))
    else:
        ui_queue.put(("status", "Done."))
    ui_queue.put(("finished", payload))
    return payload


class TranscriptoGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Transcripto")
        self.geometry("880x720")

        self.log_queue: queue.Queue = queue.Queue()
        self.ui_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.preparing = False

        self._build_ui()
        setup_logging(
            APP_DIR / LOG_FILE,
            console=False,
            queue_handler=_queue_log_handler(self.log_queue),
        )
        self._load_files_from_folder()
        self._poll_queues()

    def _build_ui(self) -> None:
        folder_frame = ttk.Frame(self)
        folder_frame.pack(fill="x", padx=12, pady=8)

        ttk.Label(folder_frame, text="Folder").pack(side="left")
        self.folder_var = tk.StringVar(value=str(APP_DIR))
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=70)
        self.folder_entry.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).pack(side="left")

        opts_frame = ttk.Labelframe(self, text="Options")
        opts_frame.pack(fill="x", padx=12, pady=8)

        self.model_var = tk.StringVar(value="base")
        ttk.Label(opts_frame, text="Model size").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            opts_frame,
            textvariable=self.model_var,
            values=MODEL_SIZES,
            width=12,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        self.min_speakers_var = tk.IntVar(value=1)
        self.max_speakers_var = tk.IntVar(value=2)
        ttk.Label(opts_frame, text="Min speakers").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(opts_frame, textvariable=self.min_speakers_var, width=5).grid(
            row=0, column=3, sticky="w"
        )
        ttk.Label(opts_frame, text="Max speakers").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Entry(opts_frame, textvariable=self.max_speakers_var, width=5).grid(
            row=0, column=5, sticky="w"
        )

        self.force_cpu_var = tk.BooleanVar(value=False)
        self.move_done_var = tk.BooleanVar(value=False)
        self.move_failed_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.srt_var = tk.BooleanVar(value=False)
        self.vtt_var = tk.BooleanVar(value=False)
        self.merged_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(opts_frame, text="Force CPU", variable=self.force_cpu_var).grid(
            row=1, column=0, sticky="w", padx=8
        )
        ttk.Checkbutton(
            opts_frame, text="Move processed to done", variable=self.move_done_var
        ).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Checkbutton(
            opts_frame, text="Move failed to failed", variable=self.move_failed_var
        ).grid(row=1, column=2, sticky="w", padx=8)
        ttk.Checkbutton(
            opts_frame, text="Overwrite outputs", variable=self.overwrite_var
        ).grid(row=1, column=3, sticky="w", padx=8)

        ttk.Checkbutton(opts_frame, text="Write SRT", variable=self.srt_var).grid(
            row=2, column=0, sticky="w", padx=8, pady=6
        )
        ttk.Checkbutton(opts_frame, text="Write VTT", variable=self.vtt_var).grid(
            row=2, column=1, sticky="w", padx=8, pady=6
        )
        ttk.Checkbutton(
            opts_frame, text="Speaker-merged transcript", variable=self.merged_var
        ).grid(row=2, column=2, sticky="w", padx=8, pady=6)

        self.selection_mode = tk.StringVar(value="folder")
        ttk.Label(opts_frame, text="Selection").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(
            opts_frame, text="Folder (all)", variable=self.selection_mode, value="folder"
        ).grid(row=3, column=1, sticky="w", padx=8)
        ttk.Radiobutton(
            opts_frame, text="Selected files", variable=self.selection_mode, value="files"
        ).grid(row=3, column=2, sticky="w", padx=8)

        files_frame = ttk.Labelframe(self, text="Files (for Selected files mode)")
        files_frame.pack(fill="both", expand=False, padx=12, pady=8)

        btns = ttk.Frame(files_frame)
        btns.pack(fill="x", padx=8, pady=6)
        ttk.Button(btns, text="Load from folder", command=self._load_files_from_folder).pack(
            side="left"
        )
        ttk.Button(btns, text="Pick files...", command=self._pick_files).pack(side="left", padx=6)
        ttk.Button(btns, text="Select all", command=lambda: self._select_all_files(True)).pack(
            side="left", padx=6
        )
        ttk.Button(btns, text="Select none", command=lambda: self._select_all_files(False)).pack(
            side="left"
        )

        self.files_canvas = tk.Canvas(files_frame, height=130)
        self.files_canvas.pack(side="left", fill="both", expand=True, padx=8, pady=6)
        self.files_scroll = ttk.Scrollbar(
            files_frame, orient="vertical", command=self.files_canvas.yview
        )
        self.files_scroll.pack(side="right", fill="y")
        self.files_canvas.configure(yscrollcommand=self.files_scroll.set)

        self.files_inner = ttk.Frame(self.files_canvas)
        self.files_canvas.create_window((0, 0), window=self.files_inner, anchor="nw")
        self.files_inner.bind(
            "<Configure>",
            lambda e: self.files_canvas.configure(scrollregion=self.files_canvas.bbox("all")),
        )
        self.file_items: List[Tuple[Path, tk.BooleanVar]] = []

        token_frame = ttk.Labelframe(
            self, text="Hugging Face Token (first run only for diarization model download)"
        )
        token_frame.pack(fill="x", padx=12, pady=8)
        self.token_var = tk.StringVar(value=os.getenv("HF_TOKEN", ""))
        ttk.Entry(
            token_frame, textvariable=self.token_var, width=80, show="*"
        ).pack(side="left", padx=8, pady=6, fill="x", expand=True)
        ttk.Button(token_frame, text="Save to .env", command=self._save_token).pack(
            side="left", padx=8
        )
        self.prepare_btn = ttk.Button(
            token_frame, text="Prepare Offline Models", command=self._prepare_offline
        )
        self.prepare_btn.pack(side="left")

        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=12, pady=8)
        self.start_btn = ttk.Button(control_frame, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(control_frame, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side="left")

        log_frame = ttk.Labelframe(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=12, pady=8)
        self.log_text = tk.Text(log_frame, wrap="word", height=16)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _set_idle_controls(self) -> None:
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.prepare_btn.configure(state="normal")

    def _set_running_controls(self) -> None:
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.prepare_btn.configure(state="disabled")

    def _set_preparing_controls(self) -> None:
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        self.prepare_btn.configure(state="disabled")

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self.folder_var.get())
        if path:
            self.folder_var.set(path)
            self._load_files_from_folder()

    def _clear_file_list(self) -> None:
        for child in self.files_inner.winfo_children():
            child.destroy()
        self.file_items.clear()

    def _load_files_from_folder(self) -> None:
        folder = Path(self.folder_var.get())
        if not folder.exists():
            return
        files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        self._populate_file_list(files)

    def _pick_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTS))
        filetypes = [("Media", patterns), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(
            initialdir=self.folder_var.get(), filetypes=filetypes
        )
        if paths:
            self._populate_file_list([Path(p) for p in paths])

    def _populate_file_list(self, files: List[Path]) -> None:
        self._clear_file_list()
        for path in files:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.files_inner, text=path.name, variable=var).pack(anchor="w")
            self.file_items.append((path, var))

    def _select_all_files(self, value: bool) -> None:
        for _, var in self.file_items:
            var.set(value)

    def _save_token(self) -> None:
        token = self.token_var.get().strip()
        if not token:
            messagebox.showwarning("HF_TOKEN", "Please enter a token to save.")
            return
        env_path = APP_DIR / ".env"
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        lines = [line for line in existing.splitlines() if not line.startswith("HF_TOKEN=")]
        lines.append(f"HF_TOKEN={token}")
        env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        messagebox.showinfo("HF_TOKEN", "Token saved to .env")

    def _prepare_offline(self) -> None:
        if self.preparing or (self.worker_thread and self.worker_thread.is_alive()):
            return

        token = self.token_var.get().strip()
        if not token:
            messagebox.showwarning(
                "HF_TOKEN", "HF_TOKEN is required to download diarization models."
            )
            return

        self.preparing = True
        self._set_preparing_controls()
        self.status_var.set("Preparing offline models...")

        model_size = self.model_var.get()
        min_speakers = self.min_speakers_var.get()
        max_speakers = self.max_speakers_var.get()

        def run() -> None:
            try:
                add_safe_globals()
                with Transcriber(
                    model_size=model_size,
                    hf_token=token,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    force_cpu=True,
                ) as transcriber:
                    transcriber.warm_up("en")
                self.ui_queue.put(("prepared", None))
            except Exception as exc:
                logging.exception("Offline preparation failed: %s", exc)
                self.ui_queue.put(("prepare_error", str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def _start(self) -> None:
        if self.preparing or (self.worker_thread and self.worker_thread.is_alive()):
            return

        folder = Path(self.folder_var.get())
        if not folder.exists():
            messagebox.showerror("Folder", "Selected folder does not exist.")
            return

        min_speakers = self.min_speakers_var.get()
        max_speakers = self.max_speakers_var.get()
        if min_speakers < 1 or max_speakers < min_speakers:
            messagebox.showerror(
                "Speakers", "Min speakers must be at least 1 and no larger than max speakers."
            )
            return

        self.stop_event.clear()
        self._set_running_controls()
        self.status_var.set("Starting...")
        self.progress.configure(value=0)

        token = self.token_var.get().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        options = Options(
            folder=folder,
            model_size=self.model_var.get(),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            force_cpu=self.force_cpu_var.get(),
            move_done=self.move_done_var.get(),
            move_failed=self.move_failed_var.get(),
            overwrite_outputs=self.overwrite_var.get(),
            selected_files=None,
            write_srt=self.srt_var.get(),
            write_vtt=self.vtt_var.get(),
            merged=self.merged_var.get(),
        )

        if self.selection_mode.get() == "files":
            selected = [p for p, var in self.file_items if var.get()]
            if not selected:
                messagebox.showerror("Files", "No files selected.")
                self._set_idle_controls()
                return
            options.selected_files = selected

        def run() -> None:
            try:
                transcribe_batch(options, self.ui_queue, self.stop_event)
            except Exception as exc:
                logging.exception("Batch failed: %s", exc)
                self.ui_queue.put(("error", str(exc)))

        self.worker_thread = threading.Thread(target=run, daemon=True)
        self.worker_thread.start()

    def _stop(self) -> None:
        if not (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Finishing current file...")

    def _poll_queues(self) -> None:
        while not self.log_queue.empty():
            self.log_text.insert("end", self.log_queue.get_nowait() + "\n")
            self.log_text.see("end")

        while not self.ui_queue.empty():
            kind, payload = self.ui_queue.get_nowait()
            if kind == "status":
                self.status_var.set(payload)
            elif kind == "progress":
                current, total = payload
                self.progress.configure(maximum=max(total, 1), value=current)
            elif kind == "finished":
                self._show_finished(payload)
            elif kind == "prepared":
                self.preparing = False
                self._set_idle_controls()
                self.status_var.set("Offline models ready.")
                messagebox.showinfo("Offline models", "Offline models are ready.")
            elif kind == "prepare_error":
                self.preparing = False
                self._set_idle_controls()
                self.status_var.set("Offline preparation failed.")
                messagebox.showerror("Offline models", payload)
            elif kind == "error":
                self._set_idle_controls()
                self.status_var.set("Error.")
                messagebox.showerror("Error", payload)

        self.after(200, self._poll_queues)

    def _show_finished(self, payload: Dict[str, Any]) -> None:
        self._set_idle_controls()
        self.preparing = False
        if payload.get("cancelled"):
            self.status_var.set("Stopped by user.")
            title = "Stopped"
        else:
            self.status_var.set("Done.")
            title = "Finished"

        lines = [
            f"Processed: {payload.get('success', 0)}",
            f"Skipped: {payload.get('skipped', 0)}",
            f"Failed: {payload.get('failed', 0)}",
        ]
        failures = payload.get("failures") or []
        if failures:
            lines.append("")
            lines.append("Failures:")
            lines.extend(
                f"- {Path(item['file']).name}: {item['error'][:160]}" for item in failures[:5]
            )
            if len(failures) > 5:
                lines.append(f"... and {len(failures) - 5} more (see log)")
        if payload.get("summary_path"):
            lines.append("")
            lines.append(f"Summary: {payload['summary_path']}")

        message = "\n".join(lines)
        if failures:
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)


def main() -> int:
    add_safe_globals()
    app = TranscriptoGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
