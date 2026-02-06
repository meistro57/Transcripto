# Transcripto

Windows-first, double-click transcription app using WhisperX for ASR + alignment and Pyannote (via WhisperX diarize utilities) for speaker diarization.

## Features

- Simple Tkinter GUI with folder picker, Start/Stop, progress, and live log
- WhisperX transcription with alignment and diarization
- Portable offline cache in the app folder
- Batch mode with skip logic, done/failed moves, and summary output

## Install (Windows)

1. Install Python 3.10+
2. Open PowerShell and create a virtual environment:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -U whisperx python-dotenv omegaconf pyannote.audio
```

4. Install FFmpeg:

```powershell
winget install ffmpeg
```

## Install (WSL2)

1. Create and activate a venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -U whisperx python-dotenv omegaconf pyannote.audio
```

3. Install FFmpeg:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

4. Install Tkinter (GUI dependency):

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

## Hugging Face Token

Create a token at:

https://huggingface.co/settings/tokens

Create a `.env` file in the app folder:

```
HF_TOKEN=your_token_here
```

### About the token and cost

The token is only used to authenticate and download the diarization model weights from HuggingFace. It does not incur usage-based charges by itself. Any compute cost comes from running the models on your own machine. If you use private or paid HuggingFace model repositories, you may still need an appropriate subscription or access plan.

## First Run and Offline Mode

Transcripto uses a **portable cache** in the app folder:

- `._hf_cache`
- `._torch_cache`

On first run, it downloads and stores all required models. After that, it can run fully offline with the cached models.

To force a clean offline-ready cache, open the app and click **Prepare Offline Models**.

If the diarization models are not present and `HF_TOKEN` is missing, you will see a clear error in the app. The token is only required to download gated models the first time.

## Usage

Double-click `transcripto_gui.py` (or run it from a terminal):

```powershell
python transcripto_gui.py
```

### Selecting Files

By default, Transcripto processes **all supported files** in the selected folder.  
If you want to process a subset, switch the selection mode to **Selected files** and use the checkbox list.

You can populate the list in two ways:
- **Load from folder** (scans the current folder)
- **Pick files...** (multi-select individual files)

If you select files from multiple folders, `summary.txt` is written to the app folder.

## Outputs

For each input file `example.mp4`, Transcripto writes:

- `example.txt` with timestamped transcript lines
- `example.json` with structured segments

After each batch run it also writes:

- `summary.txt` with NotebookLM-ready concatenated transcripts

## Troubleshooting

### PyTorch Safe Loading Errors

If you see an error mentioning “Unsupported global” or “safe globals”, the app logs a hint about extending the allowlist in `transcripto_gui.py`.

### HuggingFace Gated Models

If diarization fails, confirm your token has access to the required models and that it is set in `.env` or pasted into the app.

### Symlink Warning on Windows

If you see HuggingFace cache symlink warnings, you can ignore them or run the app with Developer Mode enabled in Windows.

## Notes

- The app uses `TRANSFORMERS_NO_TORCHVISION=1` to avoid torchvision import issues.
- On GPU systems, PyTorch must be installed with CUDA support.

## Build an EXE (Windows)

Install PyInstaller:

```powershell
pip install -U pyinstaller
```

Build a Windows GUI EXE:

```powershell
pyinstaller --onefile --windowed --name Transcripto transcripto_gui.py
```

The EXE will be created in `dist\\Transcripto.exe`.

When the EXE runs, it creates portable caches next to the EXE:
- `_hf_cache`
- `_torch_cache`

If you want to bundle `ffmpeg.exe`, place it at `bin\\ffmpeg.exe` in the same folder as the EXE.

---

👉 ⚡Created by a neurodivergent mind — building tools that respect different brains. 🧠
