"""Processing ledger keyed by resolved media path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .outputs import CollisionKey, OutputLayout, outputs_exist

LEDGER_FILE = "processed.json"


def ledger_path(folder: Path) -> Path:
    return folder / LEDGER_FILE


def load_ledger(folder: Path) -> Dict[str, Any]:
    path = ledger_path(folder)
    if not path.exists():
        return {"processed": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"processed": {}}

    if not isinstance(data, dict):
        return {"processed": {}}
    data.setdefault("processed", {})
    return data


def save_ledger(folder: Path, ledger: Dict[str, Any]) -> None:
    path = ledger_path(folder)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def already_done(
    media_path: Path,
    ledger: Dict[str, Any],
    layout: Optional[OutputLayout] = None,
    collision_stems: Optional[Set[CollisionKey]] = None,
) -> bool:
    """Skip when outputs exist or the resolved path is in the ledger."""
    if (
        layout is not None
        and collision_stems is not None
        and outputs_exist(media_path, layout, collision_stems)
    ):
        return True

    return str(media_path.resolve()) in ledger.get("processed", {})


def mark_processed(
    ledger: Dict[str, Any],
    media_path: Path,
    model_size: str,
    min_speakers: int,
    max_speakers: int,
) -> None:
    ledger.setdefault("processed", {})[str(media_path.resolve())] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_size,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }
