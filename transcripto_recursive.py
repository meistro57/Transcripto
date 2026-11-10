#!/usr/bin/env python3
"""Recursively transcribe audio/video files in place or into an output folder.

Usage:
    python transcripto_recursive.py /path/to/search
    python transcripto_recursive.py /path/to/search --dry-run
    python transcripto_recursive.py /path/to/a /path/to/b --model small
    python transcripto_recursive.py /path/to/search --output-dir /path/to/transcripts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from transcripto_core.env import Settings, app_dir, prepare_environment

APP_FOLDER = prepare_environment(app_dir())

from transcripto_core.ledger import (  # noqa: E402
    already_done,
    load_ledger,
    mark_processed,
    save_ledger,
)
from transcripto_core.logging_utils import setup_logging  # noqa: E402
from transcripto_core.models import (  # noqa: E402
    Transcriber,
    add_safe_globals,
    safe_globals_hint,
    warn_if_cuda_missing,
)
from transcripto_core.outputs import (  # noqa: E402
    OutputLayout,
    SUPPORTED_EXTS,
    find_collision_stems,
    map_speakers_to_letters,
    outputs_exist,
    write_outputs,
)

LOG_FILE = "transcripto_recursive.log"

SKIP_DIRS = {
    "venv", ".venv", "env", ".env",
    "node_modules", "__pycache__", ".git", ".hg", ".svn",
    "build", "dist", ".eggs", "*.egg-info",
    "_torch_cache", "_hf_cache", ".crush",
}


def find_media_files(root_path: Path, max_depth: Optional[int] = None) -> List[Path]:
    """Recursively find all supported media files, optionally limited in depth."""
    found: List[Path] = []
    root_depth = len(root_path.parts)

    for dirpath, dirnames, filenames in os.walk(root_path):
        depth = len(Path(dirpath).parts) - root_depth
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
        if max_depth is not None and depth > max_depth:
            continue

        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        current_dir = Path(dirpath)

        for filename in filenames:
            file_path = current_dir / filename
            if file_path.suffix.lower() in SUPPORTED_EXTS:
                found.append(file_path)

    return sorted(found)


def display_path(path: Path, roots: Sequence[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively transcribe audio and video files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Root directories to search (default: current directory)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without transcribing")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if outputs exist")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit directory depth (0 = only the root folder)",
    )
    parser.add_argument("--model", default=settings.model_size, help="Whisper model size")
    parser.add_argument(
        "--min-speakers", type=int, default=settings.min_speakers, help="Minimum speakers"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=settings.max_speakers, help="Maximum speakers"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write transcripts into this folder, mirroring the input tree",
    )
    parser.add_argument("--srt", action="store_true", help="Write SRT subtitles")
    parser.add_argument("--vtt", action="store_true", help="Write WebVTT subtitles")
    parser.add_argument(
        "--merged", action="store_true", help="Write speaker-merged transcripts"
    )
    parser.add_argument(
        "--json",
        dest="json_summary",
        action="store_true",
        help="Print a machine-readable run summary to stdout",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = Settings.from_env()
    parser = build_parser(settings)
    args = parser.parse_args(argv)

    if args.min_speakers < 1 or args.max_speakers < args.min_speakers:
        parser.error("--min-speakers must be >= 1 and <= --max-speakers")

    roots = [Path(path).resolve() for path in (args.paths or ["."])]
    for root in roots:
        if not root.exists():
            print(f"Error: Path does not exist: {root}", file=sys.stderr)
            return 1
        if not root.is_dir():
            print(f"Error: Path is not a directory: {root}", file=sys.stderr)
            return 1

    log_path = APP_FOLDER / LOG_FILE
    setup_logging(log_path, stream=sys.stderr if args.json_summary else None)

    add_safe_globals()
    warn_if_cuda_missing()

    logging.info("=" * 60)
    logging.info("Transcripto Recursive")
    logging.info("=" * 60)
    logging.info("Search path(s): %s", ", ".join(str(root) for root in roots))
    logging.info("Model: %s", args.model)
    logging.info(
        "Diarization: %s",
        "enabled" if settings.hf_token else "disabled (no HF_TOKEN)",
    )

    all_media = sorted(
        {
            path.resolve()
            for root in roots
            for path in find_media_files(root, max_depth=args.max_depth)
        }
    )
    layout = OutputLayout(
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        roots=tuple(roots),
    )
    collision_stems = find_collision_stems(all_media, layout)

    if args.force:
        media_files = all_media
    else:
        media_files = [
            path for path in all_media if not outputs_exist(path, layout, collision_stems)
        ]

    if not media_files:
        logging.info("No media files found to process.")
        if args.json_summary:
            skipped = 0 if args.force else len(all_media)
            print(
                json.dumps(
                    {
                        "summary": {
                            "roots": [str(root) for root in roots],
                            "output_dir": str(layout.output_dir) if layout.output_dir else None,
                            "total": len(all_media),
                            "success": 0,
                            "failed": 0,
                            "skipped": skipped,
                            "failures": [],
                        },
                        "results": [],
                    },
                    indent=2,
                )
            )
        return 0

    logging.info("\nFound %d file(s) to process:", len(media_files))
    for index, path in enumerate(media_files, 1):
        logging.info("  %3d. %s", index, display_path(path, roots))

    if args.dry_run:
        if args.json_summary:
            print(json.dumps({"dry_run": True, "files": [str(path) for path in media_files]}, indent=2))
        logging.info("\n[DRY RUN] - Not processing files")
        logging.info("Remove --dry-run to actually transcribe")
        return 0

    ledger = load_ledger(APP_FOLDER)
    formats = ["txt", "json"] + [name for name in ("srt", "vtt") if getattr(args, name)]

    stats = {"total": len(media_files), "success": 0, "failed": 0, "skipped": 0}
    failures: List[dict] = []
    results: Optional[List[dict]] = [] if args.json_summary else None

    logging.info("\n" + "=" * 60)
    logging.info("Starting transcription...")
    logging.info("=" * 60)

    try:
        transcriber_context = Transcriber(
            model_size=args.model,
            hf_token=settings.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            language=settings.language,
            device=settings.device,
            compute_type=settings.compute_type,
            force_cpu=settings.force_cpu,
            enable_tf32=settings.enable_tf32,
        )
    except Exception as exc:
        logging.exception("Could not load models: %s", exc)
        if args.json_summary:
            print(
                json.dumps(
                    {
                        "summary": {
                            "roots": [str(root) for root in roots],
                            "output_dir": str(layout.output_dir) if layout.output_dir else None,
                            "total": len(media_files),
                            "success": 0,
                            "failed": len(media_files),
                            "skipped": 0,
                            "failures": [{"file": None, "error": str(exc)}],
                        },
                        "results": [],
                    },
                    indent=2,
                )
            )
        return 1

    with transcriber_context as transcriber:
        for index, media_path in enumerate(media_files, 1):
            logging.info("\n[%d/%d]", index, len(media_files))

            if not args.force and already_done(
                media_path, ledger, layout, collision_stems
            ):
                logging.info("Skipping (already done): %s", media_path.name)
                stats["skipped"] += 1
                if results is not None:
                    results.append(
                        {"file": str(media_path), "status": "skipped", "outputs": {}}
                    )
                continue

            try:
                result = transcriber.transcribe_file(media_path, bundled_dir=APP_FOLDER)
                segments = map_speakers_to_letters(result["segments"])
                paths = write_outputs(
                    media_path,
                    segments,
                    layout,
                    collision_stems,
                    formats=formats,
                    merged=args.merged,
                )
                mark_processed(
                    ledger, media_path, args.model, args.min_speakers, args.max_speakers
                )
                save_ledger(APP_FOLDER, ledger)

                stats["success"] += 1
                if results is not None:
                    results.append(
                        {
                            "file": str(media_path),
                            "status": "ok",
                            "outputs": {name: str(path) for name, path in paths.items()},
                        }
                    )
            except Exception as exc:
                logging.exception("FAILED: %s (%s)", media_path.name, exc)
                hint = safe_globals_hint(exc)
                if hint:
                    logging.error(hint)

                stats["failed"] += 1
                failures.append({"file": str(media_path), "error": str(exc)})
                if results is not None:
                    results.append(
                        {
                            "file": str(media_path),
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    summary = {
        "roots": [str(root) for root in roots],
        "output_dir": str(layout.output_dir) if layout.output_dir else None,
        "total": stats["total"],
        "success": stats["success"],
        "failed": stats["failed"],
        "skipped": stats["skipped"],
        "failures": failures,
    }

    if args.json_summary:
        print(json.dumps({"summary": summary, "results": results or []}, indent=2))

    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info("Total files: %d", stats["total"])
    logging.info("Successful: %d", stats["success"])
    logging.info("Failed: %d", stats["failed"])
    logging.info("Skipped: %d", stats["skipped"])
    logging.info("\nAll done!")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
