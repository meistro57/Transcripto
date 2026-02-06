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
git clone https://github.com/YOUR_USERNAME/whisperx-transcriber.git
cd whisperx-transcriber
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

👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Usage

```bash
python whisper_to_text_diarized.py "input_file.mp4"
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
```

---

## 🧪 Troubleshooting

### FFmpeg Not Found

Install FFmpeg and ensure it is available in your system PATH.

---

### HuggingFace Token Missing

Diarization requires authentication. Add your token to `.env`.

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
