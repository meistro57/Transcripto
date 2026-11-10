# Transcripto

Windows-first transcription app built on [WhisperX](https://github.com/m-bain/whisperX) (ASR + alignment) and Pyannote (speaker diarization). It ships with a Tkinter GUI, a single-file CLI, and a recursive batch CLI, all backed by one shared core.

## Features

- Tkinter GUI with folder or checkbox file selection, live log, determinate progress, and a per-run summary
- WhisperX transcription with alignment and optional speaker diarization
- Models are loaded once per run and reused for every file
- Outputs: timestamped TXT and structured JSON, plus optional SRT, WebVTT, and speaker-merged transcripts
- Portable offline caches (`_hf_cache`, `_torch_cache`) next to the app
- Recursive CLI with multiple roots, depth limits, output folders, and JSON run summaries
- Skip logic backed by an absolute-path ledger, so same-named files never collide
- Rotating logs (5 MB x 3) and clear per-file error reporting
- Pure helpers covered by pytest and GitHub Actions

## Requirements

- Python 3.10+ (3.12 is tested)
- FFmpeg
- A Hugging Face token for diarization model downloads (first run only)
- Optional: an NVIDIA GPU with a matching CUDA build of PyTorch, or Apple Metal (MPS)

## Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA, install the torch wheels for your driver first, then the rest:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install FFmpeg:

- Windows: `winget install ffmpeg`
- Debian/Ubuntu/WSL2: `sudo apt-get update && sudo apt-get install -y ffmpeg`

On Linux/WSL2 also install Tkinter:

```bash
sudo apt-get install -y python3-tk
```

Transcripto looks for `ffmpeg` on `PATH` first, then for `bin/ffmpeg.exe` or `bin/ffmpeg` next to the app.

## Hugging Face token

Create a token at <https://huggingface.co/settings/tokens>, then either paste it into the app and click **Save to .env**, or create `.env` in the app folder:

```env
HF_TOKEN=your_token_here
```

The token only authenticates model downloads. Compute stays on your machine. If you use gated or private model repositories, your account needs access to them.

## First run and offline mode

Models are cached in portable folders next to the app:

- `_hf_cache`
- `_torch_cache`

After the first run the app works offline. Click **Prepare Offline Models** to download the ASR, alignment, and diarization models ahead of time. The token is required for the diarization download.

## Usage

### GUI

```bash
python transcripto_gui.py
```

1. Choose a folder or switch to **Selected files** and load/pick files.
2. Set the model, speaker range, and output options.
3. Click **Start**. Stop finishes the current file and does not begin the next one.
4. When the run ends, a summary shows processed, skipped, and failed counts, with the first failures listed.

Output options:

- **Write SRT** and **Write VTT** add subtitle files next to each transcript.
- **Speaker-merged transcript** writes one paragraph per speaker turn.
- **Overwrite outputs** re-transcribes files that already have outputs.
- **Move processed to done** / **Move failed to failed** move media into `done/` or `failed/`.

### Single file

```bash
python transcripto.py interview.mp3
python transcripto.py talk.mp4 --srt --vtt --merged --output-dir transcripts/
python transcripto.py talk.mp4 --model small --min-speakers 1 --max-speakers 4
```

### Recursive batch

```bash
python transcripto_recursive.py /path/to/library
python transcripto_recursive.py /path/to/library --dry-run
python transcripto_recursive.py /path/to/library --output-dir /path/to/transcripts
```

See [README_RECURSIVE.md](README_RECURSIVE.md) for all options.

`transcripto_batch.py` remains as a deprecated wrapper around the recursive CLI. It processes the app folder (top level only).

## Outputs

For `example.mp4`, Transcripto writes:

- `example.txt` with timestamped lines
- `example.json` with structured segments
- optional `example.srt`, `example.vtt`, and `example.merged.txt`

If two media files in the same folder share a name (`talk.mp4` and `talk.mp3`), the extension is kept in the output names (`talk.mp4.txt`, `talk.mp3.txt`, and matching files) so they cannot overwrite each other.

The GUI also streams a `summary.txt` with NotebookLM-ready text as files finish. The ledger (`processed.json`) is keyed by the resolved file path. Entries written by older versions used bare file names and are ignored once; those files are re-transcribed a single time if their outputs are missing.

Logs rotate at 5 MB with three backups (`transcription.log`, `transcripto_gui.log`, `transcripto_recursive.log`).

## Configuration

`.env` values are defaults for the CLIs; GUI controls override them per run.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token for diarization downloads |
| `MODEL_SIZE` | `base` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `COMPUTE_TYPE` | auto | e.g. `float16` on CUDA, `float32` on CPU |
| `FORCE_CPU` | `0` | Force CPU even when a GPU is available |
| `ENABLE_TF32` | `0` | Enable TF32 matmuls on CUDA |
| `MIN_SPEAKERS` | `1` | Minimum speakers for diarization |
| `MAX_SPEAKERS` | `2` | Maximum speakers for diarization |
| `LANGUAGE` | unset | Force a language instead of auto-detecting |

## Tests

Unit tests do not need Torch or WhisperX:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

CI runs `compileall` and pytest on every push and pull request (`.github/workflows/ci.yml`).

## Build a Windows EXE

Build on Windows so PyInstaller produces a `.exe`:

```powershell
pip install -U pyinstaller
pyinstaller Transcripto.spec
```

The spec bundles WhisperX, Pyannote, Torch, Torchaudio, and Transformers data and hidden imports with `collect_all`, and disables UPX for CUDA DLL compatibility. The build is large and slow; test it on a clean Windows machine. Model weights are still downloaded on first run into `_hf_cache` (or bundled manually next to the EXE).

Portable layout for the built app:

```text
Transcripto/
  Transcripto.exe
  bin/ffmpeg.exe        (optional, recommended)
  _hf_cache/            (created on first run)
  _torch_cache/
```

## Troubleshooting

### PyTorch safe loading errors

If a checkpoint class is blocked, the app logs a hint. Add the missing class to `transcripto_core/models.py` (`add_safe_globals`).

### Gated models

Confirm `HF_TOKEN` is set and that your account accepted the model terms on Hugging Face.

### FFmpeg not found

Install FFmpeg or place `ffmpeg.exe` in `bin/` next to the app. The error message reports the ffmpeg output when a conversion fails.

### CUDA mismatch

Run `python setup_gpu.py` for a diagnostic, or reinstall the torch build that matches your driver as shown in the install section.

### Symlink warnings on Windows

Enable Developer Mode or ignore the warning; it only affects cache symlinks.

## Project layout

```text
transcripto_core/        Shared implementation
  env.py                 .env, cache variables, Settings
  audio.py               FFmpeg discovery and temp WAV loading
  models.py              Device selection and Transcriber
  outputs.py             Naming, collisions, TXT/JSON/SRT/VTT/merged
  ledger.py              Path-keyed processed ledger
  logging_utils.py       Rotating log setup
  summary.py             Streaming summary writer
transcripto_gui.py       Tkinter GUI
transcripto_recursive.py Recursive CLI
transcripto.py           Single-file CLI
transcripto_batch.py     Deprecated batch wrapper
tests/                   Pytest suite
```

---

👉 ⚡Created by a neurodivergent mind — building tools that respect different brains. 🧠
