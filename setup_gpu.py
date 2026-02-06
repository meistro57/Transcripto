#!/usr/bin/env python3
"""Windows + WSL2 GPU setup assistant for PyTorch."""
import os
import platform
import subprocess
import sys
from pathlib import Path

SUPPORTED_CUDA = ["12.8", "12.6", "11.8"]


def is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        for path in ("/proc/version", "/proc/sys/kernel/osrelease"):
            text = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
            if "microsoft" in text or "wsl" in text:
                return True
    except Exception:
        pass
    return False


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.returncode, result.stdout.strip()
    except FileNotFoundError:
        return 127, ""


def detect_cuda_version() -> str:
    code, out = run(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
    if code == 0 and out:
        return out.splitlines()[0].strip()

    code, out = run(["nvidia-smi"])
    if code == 0:
        for line in out.splitlines():
            if "CUDA Version" in line:
                parts = line.split("CUDA Version", 1)[-1]
                version = parts.split(":", 1)[-1].strip().split()[0]
                return version
    return ""


def pick_supported_cuda(cuda_version: str) -> str:
    def as_tuple(v: str) -> tuple[int, int]:
        major, minor = v.split(".", 1)
        return int(major), int(minor)

    if not cuda_version:
        return ""
    try:
        target = as_tuple(cuda_version)
    except Exception:
        return ""

    supported = []
    for v in SUPPORTED_CUDA:
        try:
            supported.append((as_tuple(v), v))
        except Exception:
            continue
    supported.sort(reverse=True)

    for (maj_min, v) in supported:
        if maj_min <= target:
            return v
    return ""


def cuda_index_url(cuda_version: str) -> str:
    return f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"


def platform_label() -> str:
    if is_wsl():
        return "WSL2"
    if os.name == "nt":
        return "Windows"
    return platform.system()


def main() -> int:
    print("Transcripto GPU Setup Assistant")
    print("Platform:", platform_label())

    code, out = run(["nvidia-smi", "-L"])
    has_gpu = code == 0 and bool(out)
    if not has_gpu:
        print("GPU: not detected (nvidia-smi not available)")
        print("If you have an NVIDIA GPU, install the NVIDIA driver and retry.")
        return 0

    print("GPU:", out.splitlines()[0].strip())

    cuda_version = detect_cuda_version()
    if cuda_version:
        print("CUDA (driver):", cuda_version)
    else:
        print("CUDA (driver): unknown")

    try:
        import torch  # type: ignore
        cuda_ok = torch.cuda.is_available()
    except Exception:
        torch = None  # type: ignore
        cuda_ok = False

    if cuda_ok:
        name = torch.cuda.get_device_name(0) if torch else "unknown"
        print("PyTorch: CUDA available (", name, ")", sep="")
        return 0

    print("PyTorch: CUDA NOT available")

    supported = pick_supported_cuda(cuda_version)
    if supported:
        index_url = cuda_index_url(supported)
        pip_cmd = f"python -m pip install torch torchvision torchaudio --index-url {index_url}"
    else:
        pip_cmd = "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"

    print("Suggested pip command:")
    print(pip_cmd)
    print("If this fails, use the PyTorch selector for your OS + Pip + CUDA version.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
