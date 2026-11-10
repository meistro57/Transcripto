"""Logging setup with rotation."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, Optional

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUPS = 3
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(
    log_path: Path,
    *,
    console: bool = True,
    queue_handler: Optional[logging.Handler] = None,
    stream: Optional[IO[str]] = None,
    level: int = logging.INFO,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backups: int = DEFAULT_BACKUPS,
) -> logging.Logger:
    """Configure the root logger with a rotating file and optional sinks."""
    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(LOG_FORMAT)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(stream or sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if queue_handler is not None:
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)

    return logger
