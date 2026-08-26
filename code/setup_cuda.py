from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys


TARGET_TORCH = "2.7.1"
TARGET_TORCHAUDIO = "2.7.1"
PYTORCH_INDEX_URL = os.environ.get(
    "BACH_PYTORCH_INDEX_URL",
    "https://download.pytorch.org/whl/cu128",
)


def _cuda_check() -> tuple[bool, str]:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch, torchaudio; "
                "print('TORCH=' + torch.__version__); "
                "print('TORCHAUDIO=' + torchaudio.__version__); "
                "print('CUDA=' + str(torch.version.cuda)); "
                "print('AVAILABLE=' + str(torch.cuda.is_available())); "
                "print('GPU=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU'))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    text = (probe.stdout + "\n" + probe.stderr).strip()
    versions_ok = (
        f"TORCH={TARGET_TORCH}" in text
        and f"TORCHAUDIO={TARGET_TORCHAUDIO}" in text
    )
    cuda_ok = "AVAILABLE=True" in text
    return probe.returncode == 0 and versions_ok and cuda_ok, text


def _driver_cuda_version() -> tuple[float | None, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, "nvidia-smi was not found on PATH"
    result = subprocess.run([nvidia_smi], capture_output=True, text=True)
    text = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return (float(match.group(1)) if match else None), text


def ensure_cuda_pytorch() -> int:
    if os.name != "nt":
        print("[BACH Studio] Non-Windows platform: leaving the existing PyTorch installation unchanged.")
        return 0

    ok, details = _cuda_check()
    if ok:
        print("[BACH Studio] Verified CUDA PyTorch stack is already installed:")
        print(details)
        return 0

    driver_version, driver_details = _driver_cuda_version()
    if driver_version is None:
        print("[BACH Studio] NVIDIA CUDA runtime could not be verified.")
        print(driver_details)
        print("[BACH Studio] A CUDA-capable NVIDIA GPU/driver is required by the current inference backend.")
        return 2

    print(f"[BACH Studio] NVIDIA driver reports CUDA {driver_version}.")
    print(
        f"[BACH Studio] Installing torch {TARGET_TORCH} / torchaudio {TARGET_TORCHAUDIO} "
        f"from {PYTORCH_INDEX_URL}"
    )

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"torch=={TARGET_TORCH}",
        f"torchaudio=={TARGET_TORCHAUDIO}",
        "--index-url",
        PYTORCH_INDEX_URL,
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode

    ok, details = _cuda_check()
    print("[BACH Studio] CUDA verification after installation:")
    print(details)
    if not ok:
        print("[BACH Studio] CUDA PyTorch installation did not pass version/GPU verification.")
        return 3

    print("[BACH Studio] CUDA PyTorch stack is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(ensure_cuda_pytorch())
