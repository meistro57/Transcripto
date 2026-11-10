"""Transcript rendering, output naming, and collision handling."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger(__name__)

AUDIO_EXTS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
}
VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
}
SUPPORTED_EXTS = AUDIO_EXTS | VIDEO_EXTS

Segment = Dict[str, object]
CollisionKey = Tuple[Path, str]
PathPair = Tuple[Path, Path]


@dataclass(frozen=True)
class OutputLayout:
    """Where outputs go and which roots should be mirrored into output_dir."""

    output_dir: Optional[Path] = None
    roots: Tuple[Path, ...] = ()

    def folder(self, media_path: Path) -> Path:
        if self.output_dir is None:
            return media_path.parent

        for root in self.roots:
            try:
                rel_parent = media_path.parent.relative_to(root)
            except ValueError:
                continue
            return self.output_dir if rel_parent == Path(".") else self.output_dir / rel_parent
        return self.output_dir


def find_collision_stems(
    media_files: Iterable[Path], layout: Optional[OutputLayout] = None
) -> Set[CollisionKey]:
    """Find (destination folder, stem) pairs shared by more than one media file."""
    layout = layout or OutputLayout()
    counts = Counter((layout.folder(p), p.stem.lower()) for p in media_files)
    return {(folder, stem) for (folder, stem), count in counts.items() if count > 1}


def collect_folder_media(
    media_files: Iterable[Path], exts: Iterable[str] = SUPPORTED_EXTS
) -> List[Path]:
    """Add unselected siblings so output naming stays stable within a folder."""
    exts = set(exts)
    media = {p.resolve() for p in media_files}
    for parent in {p.parent for p in media}:
        try:
            for child in parent.iterdir():
                if child.is_file() and child.suffix.lower() in exts:
                    media.add(child.resolve())
        except OSError:
            continue
    return sorted(media)


def output_paths(
    media_path: Path, layout: OutputLayout, collision_stems: Set[CollisionKey]
) -> PathPair:
    """Write next to the media file, adding the extension only when stems collide."""
    folder = layout.folder(media_path)
    key = (folder, media_path.stem.lower())
    name = media_path.name if key in collision_stems else media_path.stem
    return folder / f"{name}.txt", folder / f"{name}.json"


def outputs_exist(
    media_path: Path, layout: OutputLayout, collision_stems: Set[CollisionKey]
) -> bool:
    txt_path, json_path = output_paths(media_path, layout, collision_stems)
    return txt_path.exists() and json_path.exists()


def map_speakers_to_letters(segments: Sequence[Segment]) -> Sequence[Segment]:
    """Map SPEAKER_XX values to Speaker A, Speaker B, ... by first appearance."""
    mapping: Dict[str, str] = {}
    next_letter = ord("A")
    for seg in segments:
        raw = str(seg.get("speaker", "Unknown"))
        if raw not in mapping:
            mapping[raw] = f"Speaker {chr(next_letter)}"
            next_letter += 1
        seg["speaker_label"] = mapping[raw]
    return segments


def _fields(seg: Segment) -> Tuple[str, float, float, str]:
    speaker = str(seg.get("speaker_label", "Speaker ?"))
    start = float(seg.get("start", 0.0) or 0.0)
    end = float(seg.get("end", start) or start)
    text = str(seg.get("text") or "").strip()
    return speaker, start, max(start, end), text


def structured_segments(segments: Sequence[Segment]) -> List[Segment]:
    structured: List[Segment] = []
    for seg in segments:
        speaker, start, end, text = _fields(seg)
        structured.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return structured


def render_txt(segments: Sequence[Segment]) -> str:
    return "\n".join(
        f"[{start:0.1f}s - {end:0.1f}s] {speaker}: {text}"
        for speaker, start, end, text in (_fields(seg) for seg in segments)
    )


def render_json(segments: Sequence[Segment]) -> str:
    return json.dumps(structured_segments(segments), indent=2, ensure_ascii=False)


def format_timestamp(seconds: float, separator: str = ",") -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def render_srt(segments: Sequence[Segment]) -> str:
    blocks: List[str] = []
    for index, seg in enumerate(segments, start=1):
        speaker, start, end, text = _fields(seg)
        blocks.append(
            f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{speaker}: {text}"
        )
    return "\n\n".join(blocks)


def render_vtt(segments: Sequence[Segment]) -> str:
    blocks = ["WEBVTT"]
    for seg in segments:
        speaker, start, end, text = _fields(seg)
        blocks.append(
            f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}\n{speaker}: {text}"
        )
    return "\n\n".join(blocks)


def merge_speaker_turns(segments: Sequence[Segment]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    for seg in segments:
        speaker, _, _, text = _fields(seg)
        if not text:
            continue
        if turns and turns[-1][0] == speaker:
            turns[-1] = (speaker, f"{turns[-1][1]} {text}")
        else:
            turns.append((speaker, text))
    return turns


def render_merged(segments: Sequence[Segment]) -> str:
    return "\n\n".join(f"{speaker}: {text}" for speaker, text in merge_speaker_turns(segments))


def write_outputs(
    media_path: Path,
    segments: Sequence[Segment],
    layout: Optional[OutputLayout] = None,
    collision_stems: Optional[Set[CollisionKey]] = None,
    formats: Iterable[str] = ("txt", "json"),
    merged: bool = False,
) -> Dict[str, Path]:
    """Write TXT/JSON plus optional SRT, VTT, and speaker-merged transcripts."""
    layout = layout or OutputLayout()
    collision_stems = collision_stems if collision_stems is not None else set()
    formats = {fmt.lower() for fmt in formats}

    txt_path, json_path = output_paths(media_path, layout, collision_stems)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {"txt": txt_path, "json": json_path}
    txt_path.write_text(render_txt(segments), encoding="utf-8")
    json_path.write_text(render_json(segments), encoding="utf-8")

    if "srt" in formats:
        srt_path = txt_path.with_suffix(".srt")
        srt_path.write_text(render_srt(segments), encoding="utf-8")
        paths["srt"] = srt_path

    if "vtt" in formats:
        vtt_path = txt_path.with_suffix(".vtt")
        vtt_path.write_text(render_vtt(segments), encoding="utf-8")
        paths["vtt"] = vtt_path

    if merged:
        merged_path = txt_path.with_name(f"{txt_path.stem}.merged.txt")
        merged_path.write_text(render_merged(segments), encoding="utf-8")
        paths["merged"] = merged_path

    LOGGER.info("Wrote %s", ", ".join(path.name for path in paths.values()))
    return paths
