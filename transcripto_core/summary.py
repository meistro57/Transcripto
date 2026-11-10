"""Streaming summary writer."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Optional, Sequence

from .outputs import Segment


class SummaryWriter:
    """Append each finished transcript to disk instead of buffering everything."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.written = False
        self._handle: Optional[IO[str]] = None

    def add(self, media_name: str, segments: Sequence[Segment]) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

        text = "\n".join(str(seg.get("text") or "").strip() for seg in segments)
        self._handle.write(f"=== {media_name} ===\n{text.strip()}\n\n")
        self._handle.flush()
        self.written = True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SummaryWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
