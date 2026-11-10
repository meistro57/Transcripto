# Transcripto Recursive

Recursively transcribe audio and video files, writing transcripts next to the media or into an output tree.

Built on the shared `transcripto_core` package, so it uses the same model caching, output formats, ledger, and FFmpeg discovery as the GUI.

## Features

- Recursive search across one or more roots
- Models, alignment models, and the diarization pipeline load once per run
- In-place outputs by default; `--output-dir` mirrors the input tree
- Skip logic based on existing outputs plus an absolute-path ledger
- Optional SRT, WebVTT, and speaker-merged transcripts
- Machine-readable JSON run summary
- Depth limits, dry-run, and force flags
- Rotating logs and safe directory filtering

## Usage

```bash
# Activate the venv first
source venv/bin/activate

# Transcribe a library in place
python transcripto_recursive.py /path/to/library

# Preview only
python transcripto_recursive.py /path/to/library --dry-run

# Several roots at once
python transcripto_recursive.py /path/to/audio /path/to/video

# Override the model and speaker range
python transcripto_recursive.py /path/to/library --model small --min-speakers 1 --max-speakers 4

# Write outputs elsewhere (input tree is mirrored)
python transcripto_recursive.py /path/to/library --output-dir /path/to/transcripts

# Subtitles and speaker-merged transcripts
python transcripto_recursive.py /path/to/library --srt --vtt --merged

# Machine-readable summary on stdout (logs go to stderr)
python transcripto_recursive.py /path/to/library --json > run.json

# Re-transcribe even when outputs exist
python transcripto_recursive.py /path/to/library --force
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `paths...` | `.` | One or more root directories |
| `--dry-run` | off | List matching files without transcribing |
| `--force` | off | Ignore existing outputs and the ledger |
| `--model` | `MODEL_SIZE` or `base` | Whisper model size |
| `--min-speakers` | `MIN_SPEAKERS` or `1` | Minimum speakers |
| `--max-speakers` | `MAX_SPEAKERS` or `2` | Maximum speakers |
| `--max-depth` | unlimited | Limit recursion (`0` = root folder only) |
| `--output-dir` | in place | Mirror the input tree under this folder |
| `--srt` | off | Write SRT subtitles |
| `--vtt` | off | Write WebVTT subtitles |
| `--merged` | off | Write speaker-merged transcripts |
| `--json` | off | Print a JSON run summary to stdout |

## How it works

1. Recursively scans each root, skipping cache and development folders.
2. Computes output names for every media file, including collision handling.
3. Skips files whose outputs already exist, unless `--force` is set.
4. Loads WhisperX once, converts each file to a temporary 16 kHz WAV, and removes it after loading.
5. Writes outputs, updates the ledger (`processed.json` in the app folder), and streams log lines.
6. Prints a text summary (and the JSON summary with `--json`).

## Output files

For each media file:

- `filename.txt` with timestamped lines
- `filename.json` with structured segments
- optional `filename.srt`, `filename.vtt`, `filename.merged.txt`

If two files in the same folder share a name (`talk.mp4` and `talk.mp3`), the extension is kept (`talk.mp4.txt`, `talk.mp3.txt`). With `--output-dir`, files from different input folders are mirrored into separate output subfolders.

Temporary WAV files live in the system temp directory and are always removed after loading.

## Supported formats

Audio: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.opus`, `.aac`, `.wma`, `.aiff`

Video: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`

## Configuration

Environment variables from `.env` provide defaults:

```env
HF_TOKEN=your_huggingface_token
MODEL_SIZE=base
DEVICE=auto
MIN_SPEAKERS=1
MAX_SPEAKERS=2
```

Command-line flags override the model and speaker values. `HF_TOKEN` is required for diarization; without it the transcript is written without speaker labels.

## Skipped directories

`venv`, `.venv`, `env`, `.env`, `node_modules`, `__pycache__`, `.git`, `.hg`, `.svn`, `build`, `dist`, `.eggs`, `_torch_cache`, `_hf_cache`, `.crush`

## Ledger and logs

- `processed.json` in the app folder, keyed by the resolved media path. Bare-name entries from older versions are ignored once.
- `transcripto_recursive.log` rotates at 5 MB with three backups.

## vs. the GUI

| Feature | GUI | Recursive CLI |
| --- | --- | --- |
| Interface | Tkinter | Command line |
| Search | One folder or selected files | Recursive, multiple roots |
| Outputs | In place | In place or `--output-dir` |
| Progress | Progress bar + live log | Streaming log |
| Run summary | Dialog | Text summary + `--json` |
| Formats | Checkboxes | `--srt`, `--vtt`, `--merged` |

`transcripto_batch.py` is a deprecated wrapper that calls this CLI with `--max-depth 0` on the app folder.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
