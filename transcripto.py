#!/usr/bin/env python3
"""Transcribe one audio/video file with WhisperX.

Kept as a single-file CLI on top of transcripto_core.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from transcripto_core.env import Settings, app_dir, prepare_environment

APP_FOLDER = prepare_environment(app_dir())

from transcripto_core.logging_utils import setup_logging  # noqa: E402
from transcripto_core.models import (  # noqa: E402
    Transcriber,
    add_safe_globals,
    warn_if_cuda_missing,
)
from transcripto_core.outputs import (  # noqa: E402
    OutputLayout,
    find_collision_stems,
    map_speakers_to_letters,
    render_txt,
    write_outputs,
)

LOG_FILE = "transcription.log"


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe one media file.")
    parser.add_argument("input", help="Audio or video file")
    parser.add_argument("--model", default=settings.model_size, help="Whisper model size")
    parser.add_argument("--min-speakers", type=int, default=settings.min_speakers)
    parser.add_argument("--max-speakers", type=int, default=settings.max_speakers)
    parser.add_argument("--output-dir", type=Path, help="Write outputs to this folder")
    parser.add_argument("--srt", action="store_true", help="Write SRT subtitles")
    parser.add_argument("--vtt", action="store_true", help="Write WebVTT subtitles")
    parser.add_argument("--merged", action="store_true", help="Write a speaker-merged transcript")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = Settings.from_env()
    parser = build_parser(settings)
    args = parser.parse_args(argv)

    if args.min_speakers < 1 or args.max_speakers < args.min_speakers:
        parser.error("--min-speakers must be >= 1 and <= --max-speakers")

    media_path = Path(args.input).expanduser().resolve()
    if not media_path.exists():
        print(f"Error: File does not exist: {media_path}", file=sys.stderr)
        return 1
    if not media_path.is_file():
        print(f"Error: Not a file: {media_path}", file=sys.stderr)
        return 1

    setup_logging(APP_FOLDER / LOG_FILE)
    add_safe_globals()
    warn_if_cuda_missing()

    logging.info("Input: %s", media_path)
    logging.info(
        "Model: %s | speakers: %d-%d",
        args.model,
        args.min_speakers,
        args.max_speakers,
    )

    layout = OutputLayout(
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        roots=(media_path.parent,),
    )
    collision_stems = find_collision_stems([media_path], layout)
    formats = ["txt", "json"]
    if args.srt:
        formats.append("srt")
    if args.vtt:
        formats.append("vtt")

    with Transcriber(
        model_size=args.model,
        hf_token=settings.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=settings.language,
        device=settings.device,
        compute_type=settings.compute_type,
        force_cpu=settings.force_cpu,
        enable_tf32=settings.enable_tf32,
    ) as transcriber:
        result = transcriber.transcribe_file(media_path, bundled_dir=APP_FOLDER)

    segments = map_speakers_to_letters(result["segments"])
    print(render_txt(segments))
    write_outputs(
        media_path,
        segments,
        layout,
        collision_stems,
        formats=formats,
        merged=args.merged,
    )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
