from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys


def _cuda_check() -> tuple[bool, str]:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print(torch.__version__); "
                "print(torch.version.cuda); "
                "print(torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
            ),
        ],
        capture_output=True,
        text=True,
    )
    text = (probe.stdout + "\n" + probe.stderr).strip()
    ok = probe.returncode == 0 and "\nTrue\n" in f"\n{text}\n"
    return ok, text


def _driver_cuda_version() -> tuple[float | None, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, "nvidia-smi was not found on PATH"
    result = subprocess.run([nvidia_smi], capture_output=True, text=True)
    text = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return (float(match.group(1)) if match else None), text


def _pytorch_index_for_driver(version: float | None) -> str:
    override = os.environ.get("BACH_PYTORCH_INDEX_URL")
    if override:
        return override
    if version is not None and version >= 13.0:
        return "https://download.pytorch.org/whl/cu130"
    return "https://download.pytorch.org/whl/cu128"


def ensure_cuda_pytorch() -> int:
    if os.name != "nt":
        print("[BACH Studio] Non-Windows platform: leaving the existing PyTorch installation unchanged.")
        return 0

    ok, details = _cuda_check()
    if ok:
        print("[BACH Studio] CUDA-enabled PyTorch is already working:")
        print(details)
        return 0

    driver_version, driver_details = _driver_cuda_version()
    if driver_version is None:
        print("[BACH Studio] NVIDIA CUDA runtime could not be verified.")
        print(driver_details)
        print("[BACH Studio] A CUDA-capable NVIDIA GPU/driver is required by the current inference backend.")
        return 2

    index_url = _pytorch_index_for_driver(driver_version)
    print(f"[BACH Studio] Existing PyTorch is not CUDA-ready. NVIDIA driver reports CUDA {driver_version}.")
    print(f"[BACH Studio] Installing CUDA-enabled torch/torchaudio from {index_url}")

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchaudio",
        "--index-url",
        index_url,
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode

    ok, details = _cuda_check()
    print("[BACH Studio] CUDA verification after installation:")
    print(details)
    if not ok:
        print("[BACH Studio] CUDA-enabled PyTorch installation did not pass verification.")
        return 3

    print("[BACH Studio] CUDA-enabled PyTorch is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(ensure_cuda_pytorch())
