# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Transcripto GUI.

Build with:
    pyinstaller Transcripto.spec

WhisperX, Pyannote, and Torch load modules and model files dynamically, so
collect_all is used to pull their data and hidden imports into the bundle.
"""

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

PACKAGES = (
    "transcripto_core",
    "whisperx",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "transformers",
)

for package in PACKAGES:
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
    except Exception as exc:  # pragma: no cover - build-time diagnostics
        print(f"WARNING: could not collect {package}: {exc}")
        continue
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["transcripto_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Transcripto",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
