import subprocess
import sys
import types
from pathlib import Path

import pytest

from transcripto_core import audio


def test_ffmpeg_prefers_path(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    assert audio.ffmpeg_path(Path("/app")) == "/usr/bin/ffmpeg"


def test_ffmpeg_falls_back_to_bundled(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    bundled = tmp_path / "bin" / "ffmpeg.exe"
    bundled.parent.mkdir()
    bundled.write_bytes(b"exe")

    assert audio.ffmpeg_path(tmp_path) == str(bundled)


def test_ffmpeg_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        audio.ffmpeg_path(tmp_path)


def test_convert_builds_16k_mono_command(monkeypatch, tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    out_dir = tmp_path / "out"
    calls = {}

    def fake_run(cmd, check, capture_output, text):
        calls["cmd"] = cmd
        (out_dir / "audio.wav").write_bytes(b"wav")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    result = audio.convert_to_wav_16k(media, out_dir, executable="ffmpeg")

    assert result == out_dir / "audio.wav"
    cmd = calls["cmd"]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert str(media) in cmd


def test_convert_reports_ffmpeg_stderr(monkeypatch, tmp_path):
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")

    def fake_run(cmd, check, capture_output, text):
        raise subprocess.CalledProcessError(
            1, cmd, stderr="line one\nConversion failed!\nline three"
        )

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Conversion failed!"):
        audio.convert_to_wav_16k(media, tmp_path / "out", executable="ffmpeg")


def test_load_audio_cleans_temp_dir(monkeypatch, tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    seen = {}

    def fake_convert(input_path, out_dir, bundled_dir=None):
        seen["out_dir"] = out_dir
        wav = out_dir / "audio.wav"
        wav.write_bytes(b"wav")
        return wav

    fake_whisperx = types.ModuleType("whisperx")
    fake_whisperx.load_audio = lambda path: "audio-buffer"
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(audio, "convert_to_wav_16k", fake_convert)

    assert audio.load_audio_16k(media) == "audio-buffer"
    assert not seen["out_dir"].exists()
