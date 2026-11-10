#!/usr/bin/env python3
"""Deprecated batch entry point.

Processes the app folder (top level only) using the recursive CLI.
Kept for backwards compatibility; new work should use transcripto_recursive.py.
"""

from __future__ import annotations

import sys

from transcripto_core.env import app_dir, prepare_environment

APP_FOLDER = prepare_environment(app_dir())

from transcripto_recursive import main as recursive_main  # noqa: E402


def main() -> int:
    argv = [str(APP_FOLDER), "--max-depth", "0"]
    return recursive_main([*argv, *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
