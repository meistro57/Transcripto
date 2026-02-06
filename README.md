<p align="center">
  <img src="https://github.com/user-attachments/assets/80faa50f-9c7b-48e1-9735-278c972a9fbd" width="300"/>
</p>
# 🎙️ WhisperX Speaker Transcription Pipeline

> Transcribes audio and video into timestamped, speaker-labeled transcripts using WhisperX, Voice Activity Detection, alignment, and speaker diarization.

---

## ✨ Features

* 🎧 Supports audio **and** video input
* 🧠 Automatic speech recognition using WhisperX
* 🗣️ Multi-speaker diarization (Speaker A / Speaker B)
* ⏱️ Precise timestamp alignment
* 📄 Generates both human-readable and structured outputs
* 📦 Works with most media formats via FFmpeg
* 🪟 Windows, WSL, and Linux compatible
* 🔐 Uses HuggingFace authentication for diarization models

---

## 📁 Output

For each input file:

```
example.mp4
```

The pipeline generates:

```
example.txt              → readable transcript
example.json             → structured speaker segments
transcription.log        → processing log
```

---

## 📜 Example Output

### TXT

```
[12.4s - 18.1s] Speaker A: ADHD is not a deficit...
[18.2s - 23.6s] Speaker B: That’s actually a huge misunderstanding...
```

### JSON

```json
{
  "speaker": "Speaker A",
  "start": 12.4,
  "end": 18.1,
  "text": "ADHD is not a deficit..."
}
```

---

## 🎬 Supported Input Formats

### Audio

* mp3
* wav
* m4a
* flac
* ogg
* opus
* aac

### Video

* mp4
* mkv
* mov
* avi
* webm

---

## ⚙️ Requirements

* Python 3.10+
* FFmpeg
* HuggingFace account (for diarization models)

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/meistro57/Transcripto.git
cd Transcripto
```

---

### 2️⃣ Create Virtual Environment

#### Windows PowerShell

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
venv\Scripts\activate.bat
```

#### Linux / WSL

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -U whisperx python-dotenv omegaconf
```

---

### 4️⃣ Install FFmpeg

#### Windows (Recommended)

```powershell
winget install ffmpeg
```

Verify installation:

```bash
ffmpeg -version
```

---

### 5️⃣ Create `.env` File

```
HF_TOKEN=your_huggingface_token_here
MODEL_SIZE=base
```

Get a HuggingFace token here:

[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Usage

```bash
python Transcripto.py "input_file.mp4"
```

### Quick GPU Check

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("mps:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

---

## 🧠 Pipeline Overview

```
Media Input
   ↓
FFmpeg Conversion
   ↓
Voice Activity Detection (Silero)
   ↓
WhisperX Transcription
   ↓
Timestamp Alignment
   ↓
Speaker Diarization (Pyannote)
   ↓
TXT + JSON Output
```

---

## ⚡ Performance Notes

| Hardware | 30 Minute File |
| -------- | -------------- |
| CPU      | ~15-30 minutes |
| GPU      | ~5-10 minutes  |

---

## 🧩 Configuration Options

Inside `.env`:

```
MODEL_SIZE=base
FORCE_CPU=0
MIN_SPEAKERS=2
MAX_SPEAKERS=2
DEVICE=auto
COMPUTE_TYPE=
ENABLE_TF32=0
```

Notes:
* `DEVICE=auto` will prefer CUDA, then Apple MPS, then CPU. You can force `cpu`, `cuda`, or `mps`.
* `COMPUTE_TYPE` lets you override precision (e.g., `float16` on CUDA). Leave blank for defaults.
* `ENABLE_TF32=1` can speed up CUDA on Ampere+ GPUs with a small precision tradeoff.

---

## 🖥️ GPU Setup (Windows / WSL2)

When a GPU is present but PyTorch lacks CUDA support, the app will:
* Detect platform (Windows vs WSL2)
* Detect CUDA driver version (via `nvidia-smi`)
* Print a pip install command for the closest supported CUDA build

Example log output:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If the suggested command fails, use the PyTorch “Get Started” selector and choose your OS + Pip + CUDA version.

### GPU Setup Assistant (Windows / WSL2)

Run:

```bash
python setup_gpu.py
```

This script checks:
* NVIDIA GPU presence
* CUDA driver version
* PyTorch CUDA availability

It prints the best matching pip command for your system.

---

## 🧪 Troubleshooting

### FFmpeg Not Found

Install FFmpeg and ensure it is available in your system PATH.

---

### HuggingFace Token Missing

Diarization requires authentication. Add your token to `.env`.

**About the HuggingFace token and cost**

The token is only used to authenticate and download the diarization model weights from HuggingFace. It does not incur usage-based charges by itself. Any compute cost comes from running the models on your own machine.
If you use private or paid HuggingFace model repositories, you may still need an appropriate subscription or access plan.

---

### PyTorch Safe Loading Errors

The script automatically allowlists trusted model checkpoint classes.

---

## 📦 Dependencies

* WhisperX
* Pyannote Audio
* PyTorch
* FFmpeg
* Python-Dotenv

---

## 🗺️ Roadmap

* Batch folder transcription
* Speaker name training
* Transcript summarization
* Semantic transcript search
* NotebookLM formatting
* Real-time folder watcher

---

## 🤝 Contributing

Pull requests are welcome. Feature ideas and optimizations are encouraged.

---

## 📜 License

MIT License

---

## ⭐ Acknowledgements

* OpenAI Whisper
* WhisperX
* Pyannote Audio
* Silero VAD
* FFmpeg

---

## 💡 Future Vision

This project aims to provide a reliable foundation for:

* Podcast processing
* Research transcription
* AI knowledge ingestion
* Meeting automation
* Content indexing

---

👉 ⚡Created by a neurodivergent mind — building tools that respect different brains. 🧠
